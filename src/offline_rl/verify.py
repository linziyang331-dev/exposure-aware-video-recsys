"""Recompute all saved policy results, trajectories, curves and checkpoint outputs."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

from src.exposure.verify import assert_result_equal
from src.offline_rl.environment import make_world,behavior_policy
from src.offline_rl.data import collect_log,validate_log,action_counts
from src.offline_rl.model import QNetwork,all_q_values,greedy_policy
from src.offline_rl.train import load_checkpoint,log_fingerprint
from src.offline_rl.loss import cql_loss
from src.offline_rl.evaluate import rollout,policy_metrics,mean_interval
from src.offline_rl.experiment import condition_grid,training_config,method_name,baseline_policies,summarize_metrics,paired_comparisons,SELECTION_PROTOCOL,LIMITS


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _npz(path):
    with np.load(path,allow_pickle=False) as saved:return {k:np.array(saved[k]) for k in saved.files}


def _arrays_equal(expected,actual):
    if set(expected)!=set(actual):raise ValueError('Array fields differ')
    for key in expected:
        if not np.array_equal(expected[key],actual[key]):raise ValueError('Array content differs: '+key)


def _csv_equal(path,expected):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);actual=list(reader)
        if reader.fieldnames!=list(expected[0]):raise ValueError('CSV columns differ: '+str(path))
    rebuilt=[{k:'' if v is None else str(v) for k,v in row.items()} for row in expected]
    if actual!=rebuilt:raise ValueError('CSV values differ: '+str(path))


def verify_experiment(config,directory):
    directory=Path(directory);result=_read(directory/'results.json');world=make_world(config['world'])
    assert_result_equal(result['config'],{k:v for k,v in config.items() if k!='output_directory'})
    for key,value in {'status':'SIMULATED','main_method':method_name(config['main_alpha']),
                      'selection':SELECTION_PROTOCOL,'limits':LIMITS}.items():assert_result_equal(value,result[key])
    assert_result_equal({'pearl_installed':importlib.util.find_spec('pearl') is not None,'pearl_install_attempted':False,
        'pearl_runtime_compatibility':'NOT_TESTED','implementation':'independent discrete CQL(H) in existing PyTorch',
        'upstream_code_copied':False,'new_dependencies_installed':False},result['framework_assessment'])
    if result['real_rl_value_claimed'] or result['real_top20_changed'] or result['real_transition_status']!='NOT_ESTABLISHED_FOR_KUAIRAND_PURE':
        raise ValueError('Real RL identification boundary changed')
    _arrays_equal(world,_npz(directory/'artifacts/world.npz'))
    grid=list(condition_grid(config))
    if len(result['conditions'])!=len(grid):raise ValueError('Condition grid is incomplete')
    expected_rows=[];expected_curves=[];models_checked=0;losses_checked=0
    gamma=config['model']['gamma'];cfg=training_config(config,world);s,a=world['reward'].shape;h=world['horizon']
    torch.set_num_threads(1)
    for saved,condition in zip(result['conditions'],grid):
        for key,value in condition.items():assert_result_equal(value,saved[key])
        root=directory/saved['directory']
        expected_root=directory/'artifacts'/condition['id']
        if root.resolve()!=expected_root.resolve():raise ValueError('Condition output path differs')
        mu=behavior_policy(world,condition['scenario']);_arrays_equal({'behavior':mu},{'behavior':np.load(root/'behavior.npy')})
        train=_npz(root/'train_log.npz');valid=_npz(root/'validation_log.npz')
        _arrays_equal(train,collect_log(world,mu,condition['training_episodes'],condition['train_seed']))
        _arrays_equal(valid,collect_log(world,mu,config['validation_episodes'],condition['validation_seed'],
                                      episode_offset=condition['training_episodes']+1000))
        counts=action_counts(train,s,a,h)
        if not np.array_equal(counts,np.load(root/'action_counts.npy')):raise ValueError('Training action counts differ')
        metadata={'train':validate_log(world,mu,train),'validation':validate_log(world,mu,valid),
            'train_log_sha256':log_fingerprint(train),'validation_log_sha256':log_fingerprint(valid),
            'observed_time_state_action_fraction':float((counts>0).mean()),
            'logging_zero_probability_fraction':float((mu==0).mean()),
            'episode_overlap':int(len(np.intersect1d(train['episode_id'],valid['episode_id'])))}
        if metadata['episode_overlap']:raise ValueError('Training and validation overlap')
        assert_result_equal(metadata,saved['data'])
        policies=baseline_policies(world,mu,counts,gamma);q_tables={};initial_hashes=set();batch_hashes=set()
        if set(saved['models'])!={method_name(alpha) for alpha in config['alphas']}:raise ValueError('Model matrix differs')
        for alpha in config['alphas']:
            method=method_name(alpha);entry=saved['models'][method];model_root=root/'models'/method
            if (directory/entry['directory']).resolve()!=model_root.resolve():raise ValueError('Model artifact path differs')
            fitted=_read(model_root/'training.json');assert_result_equal(entry['training'],fitted)
            signature={'config':cfg,'alpha':float(alpha),'seed':condition['seed'],
                'train_log_sha256':metadata['train_log_sha256'],'validation_log_sha256':metadata['validation_log_sha256'],
                'implementation':'discrete_CQL_H_MSE_temperature1'}
            assert_result_equal(signature,fitted['signature'])
            if fitted['oracle_inputs_used_for_training'] or fitted['target_has_gradients']:raise ValueError('Training isolation/target gradient contract changed')
            steps=[0,*range(cfg['trace_interval'],cfg['steps']+1,cfg['trace_interval'])]
            if steps[-1]!=cfg['steps']:steps.append(cfg['steps'])
            if fitted['steps']!=cfg['steps'] or fitted['snapshot_steps']!=steps or len(fitted['history'])!=len(steps):
                raise ValueError('Checkpoint/curve budget differs')
            q_history=np.load(model_root/'q_history.npy');target_history=np.load(model_root/'target_q_history.npy')
            if q_history.shape!=(len(steps),h,s,a) or target_history.shape!=q_history.shape or not np.isfinite(q_history).all() or not np.isfinite(target_history).all():
                raise ValueError('Invalid saved Q snapshots')
            model=load_checkpoint(model_root/'model.pt',signature)
            if not np.array_equal(all_q_values(model),q_history[-1]):raise ValueError('Reloaded checkpoint Q values differ')
            with torch.random.fork_rng():
                torch.manual_seed(condition['seed']);initial=QNetwork(cfg['states'],cfg['horizon'],cfg['actions'],cfg['hidden'])
            if not np.array_equal(all_q_values(initial),q_history[0]):raise ValueError('Initial Q snapshot differs')
            initial_vector=torch.cat([p.detach().ravel() for p in initial.parameters()])
            final_vector=torch.cat([p.detach().ravel() for p in model.parameters()])
            expected_initial=hashlib.sha256(initial_vector.numpy().tobytes()).hexdigest()
            if expected_initial!=fitted['initial_parameter_sha256']:raise ValueError('Initial parameter hash differs')
            assert_result_equal(float(torch.linalg.vector_norm(final_vector-initial_vector)),fitted['parameter_change_norm'])
            digest=hashlib.sha256();rng=np.random.default_rng(condition['seed']+101)
            for _ in range(cfg['steps']):digest.update(rng.integers(0,len(train['action']),size=cfg['batch_size']).tobytes())
            if digest.hexdigest()!=fitted['batch_sequence_sha256']:raise ValueError('Batch sequence differs')
            initial_hashes.add(expected_initial);batch_hashes.add(digest.hexdigest())
            previous_gradient=0.
            for step,q,target,trace in zip(steps,q_history,target_history,fitted['history']):
                if (trace['step']!=step or not np.isfinite(trace['max_gradient_norm_so_far'])
                        or trace['max_gradient_norm_so_far']<previous_gradient
                        or trace['train_diagnostic_rows']!=min(2048,len(train['action']))):
                    raise ValueError('Training curve order/recorded metadata differs')
                previous_gradient=trace['max_gradient_norm_so_far']
                for label,log in [('train',train),('validation',valid)]:
                    subset={k:v[:2048] for k,v in log.items()} if label=='train' else log
                    st,t,act=subset['state'],subset['time'],subset['action']
                    next_q=target[np.minimum(t+1,h-1),subset['next_state']]
                    loss=cql_loss(torch.from_numpy(q[t,st]),torch.from_numpy(act),
                        torch.from_numpy(subset['reward']).float(),torch.from_numpy(next_q),torch.from_numpy(subset['terminated']),gamma,alpha)
                    fields=['bellman','conservative','loss'] if label=='train' else ['bellman']
                    for field in fields:
                        # Batched neural GEMM and table lookup can differ at float32 roundoff.
                        if not np.isclose(float(loss[field]),trace[label+'_'+field],rtol=2e-5,atol=2e-6):
                            raise ValueError('Training/validation diagnostic loss differs')
                        losses_checked+=1
                value=policy_metrics(world,greedy_policy(q),mu,counts,gamma,q)
                expected_curves.append({'condition_id':condition['id'],'scenario':condition['scenario'],
                    'training_episodes':condition['training_episodes'],'seed':condition['seed'],'method':method,
                    'step':step,'discounted_return':value['discounted_return'],'q_start_bias':value['q_start_bias'],
                    'zero_behavior_support_fraction':value['zero_behavior_support_fraction'],**trace})
            q_tables[method]=q_history[-1];policies[method]=greedy_policy(q_history[-1]);models_checked+=1
        if len(initial_hashes)!=1 or len(batch_hashes)!=1:raise ValueError('DQN/CQL paired initialization or batches differ')
        if set(saved['metrics'])!=set(policies):raise ValueError('Policy metric set differs')
        rebuilt={};samples={}
        for method,policy in policies.items():
            policy_path=root/'policies'/(method+'_policy.npy');return_path=root/'policies'/(method+'_returns.npz')
            if not np.array_equal(policy,np.load(policy_path)):raise ValueError('Saved policy differs from its model/baseline')
            metric=policy_metrics(world,policy,mu,counts,gamma,q_tables.get(method))
            sampled=rollout(world,policy,config['rollout_episodes'],condition['rollout_seed'],gamma)
            _arrays_equal(sampled,_npz(return_path));samples[method]=sampled
            interval=mean_interval(sampled['returns'],config['bootstrap_repeats'],config['bootstrap_seed'])
            row={k:condition[k] for k in ['scenario','training_episodes','seed']}
            row.update(condition_id=condition['id'],method=method,metrics=metric,rollout=interval,
                policy_artifact=policy_path.relative_to(directory).as_posix(),returns_artifact=return_path.relative_to(directory).as_posix())
            assert_result_equal(row,saved['metrics'][method]);rebuilt[method]=row;expected_rows.append(row)
            if abs(interval['mean']-metric['discounted_return'])>6*interval['standard_error']+.03:
                raise ValueError('Monte Carlo return disagrees with exact DP beyond diagnostic tolerance')
        assert_result_equal(paired_comparisons(rebuilt,samples,config),saved['comparisons'])
    assert_result_equal(expected_rows,result['metrics']);assert_result_equal(expected_curves,result['curves'])
    aggregate=summarize_metrics(expected_rows);assert_result_equal(aggregate,result['aggregate'])
    _csv_equal(directory/'aggregate.csv',aggregate);_csv_equal(directory/'learning_curves.csv',expected_curves)
    flat=[{**{k:r[k] for k in ['condition_id','scenario','training_episodes','seed','method']},
        **{key:r['metrics'].get(key) for key in ['discounted_return','undiscounted_return','zero_behavior_support_fraction',
            'unobserved_training_action_fraction','action_catalog_coverage','predicted_start_value','q_start_bias']},
        **{'rollout_'+key:r['rollout'][key] for key in ['mean','standard_error','low','high']}} for r in expected_rows]
    _csv_equal(directory/'metrics.csv',flat)
    return {'status':'PASS','conditions_replayed':len(grid),'models_reloaded':models_checked,
        'policy_metrics_recomputed':len(expected_rows),'curve_rows_recomputed':len(expected_curves),
        'diagnostic_loss_fields_recomputed':losses_checked,'all_training_validation_logs_replayed':True,
        'recorded_gradient_norm_independently_recomputed':False,
        'all_policy_rollouts_replayed':True,'all_csv_values_checked':True,'paired_initialization_and_batches_verified':True,
        'real_rl_value_claimed':False,'numeric_tolerance':{'reported_metrics_rtol':1e-8,'reported_metrics_atol':1e-10,
            'float32_diagnostic_loss_rtol':2e-5,'float32_diagnostic_loss_atol':2e-6}}
