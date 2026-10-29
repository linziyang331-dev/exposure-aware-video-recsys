"""Predeclared paired offline training matrix and post-training simulator evaluation."""
import json
import importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np

from src.exposure.synthetic import write_csv
from src.offline_rl.environment import make_world,behavior_policy,optimal_policy
from src.offline_rl.data import collect_log,validate_log,action_counts
from src.offline_rl.model import greedy_policy
from src.offline_rl.train import fit_model,log_fingerprint
from src.offline_rl.evaluate import empirical_bc,policy_metrics,rollout,mean_interval,compare_returns


SELECTION_PROTOCOL='predeclared alpha and final checkpoint; exact return curves calculated after training'
LIMITS=['Synthetic fully observed finite-horizon MDP; not a fitted real user simulator',
    'Returns obtained from known simulator, not identifiable real-log OPE',
    'Oracle baselines use privileged transition/reward information',
    'Episode intervals condition on fixed trained policies; three runs vary both logs and initialization',
    'Neural CQL does not automatically guarantee lower bounds or policy improvement']


def method_name(alpha):return 'offline_dqn' if alpha==0 else 'cql_alpha'+str(float(alpha))


def condition_grid(config):
    for si,scenario in enumerate(config['scenarios']):
        for ni,episodes in enumerate(config['training_episodes']):
            for seed in config['seeds']:
                seeds=[int(np.random.SeedSequence([config['world']['world_seed'],si,episodes,seed,domain])
                           .generate_state(1,dtype=np.uint64)[0]) for domain in [1,2,3]]
                yield {'id':f'{scenario}_episodes{episodes}_seed{seed}','scenario':scenario,
                       'training_episodes':episodes,'seed':seed,'train_seed':seeds[0],
                       'validation_seed':seeds[1],'rollout_seed':seeds[2]}


def training_config(config,world):
    s,a=world['reward'].shape
    return {'states':s,'actions':a,'horizon':world['horizon'],**config['model']}


def baseline_policies(world,mu,counts,gamma):
    s,a=world['reward'].shape;h=world['horizon']
    myopic=greedy_policy(np.broadcast_to(world['reward'],(h,s,a)))
    return {'uniform':np.full((s,a),1/a),'logging_policy':mu,'empirical_bc':empirical_bc(counts),
            'myopic_oracle':myopic,'optimal_oracle':optimal_policy(world,gamma)}


def summarize_metrics(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['scenario'],row['training_episodes'],row['method'])].append(row)
    result=[]
    for (scenario,n,method),group in grouped.items():
        values=np.array([r['metrics']['discounted_return'] for r in group])
        dqn={r['seed']:r['metrics']['discounted_return'] for r in rows if r['scenario']==scenario
             and r['training_episodes']==n and r['method']=='offline_dqn'}
        diff=values-np.array([dqn[r['seed']] for r in group])
        result.append({'scenario':scenario,'training_episodes':n,'method':method,'runs':len(group),
            'discounted_return_mean':float(values.mean()),
            'discounted_return_sample_std':float(values.std(ddof=1)) if len(values)>1 else None,
            'undiscounted_return_mean':float(np.mean([r['metrics']['undiscounted_return'] for r in group])),
            'paired_exact_difference_vs_dqn_mean':float(diff.mean()),
            'paired_exact_difference_vs_dqn_sample_std':float(diff.std(ddof=1)) if len(diff)>1 else None,
            'zero_support_fraction_mean':float(np.mean([r['metrics']['zero_behavior_support_fraction'] for r in group])),
            'unobserved_action_fraction_mean':float(np.mean([r['metrics']['unobserved_training_action_fraction'] for r in group])),
            'q_start_bias_mean':float(np.mean([r['metrics']['q_start_bias'] for r in group]))
                if 'q_start_bias' in group[0]['metrics'] else None})
    return result


def paired_comparisons(metrics,samples,config):
    result={}
    for subject in [method_name(a) for a in config['alphas']]:
        for reference in ['logging_policy','empirical_bc','offline_dqn']:
            if subject==reference:continue
            result[subject+'_minus_'+reference]={
                'subject':subject,'reference':reference,
                'exact_discounted_difference':metrics[subject]['metrics']['discounted_return']-metrics[reference]['metrics']['discounted_return'],
                'paired_episode_difference':compare_returns(samples[reference]['returns'],samples[subject]['returns'],
                    config['bootstrap_repeats'],config['bootstrap_seed'])}
    return result


def run_experiment(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);artifact=output/'artifacts';artifact.mkdir()
    world=make_world(config['world']);np.savez_compressed(artifact/'world.npz',**world)
    gamma=config['model']['gamma'];model_cfg=training_config(config,world);s,a=world['reward'].shape
    conditions=[];rows=[];curves=[]
    for condition in condition_grid(config):
        directory=artifact/condition['id'];directory.mkdir();mu=behavior_policy(world,condition['scenario'])
        train=collect_log(world,mu,condition['training_episodes'],condition['train_seed'])
        validation=collect_log(world,mu,config['validation_episodes'],condition['validation_seed'],
                               episode_offset=condition['training_episodes']+1000)
        np.savez_compressed(directory/'train_log.npz',**train);np.savez_compressed(directory/'validation_log.npz',**validation)
        counts=action_counts(train,s,a,world['horizon']);np.save(directory/'action_counts.npy',counts);np.save(directory/'behavior.npy',mu)
        models={};q_tables={};policies=baseline_policies(world,mu,counts,gamma)
        for alpha in config['alphas']:
            method=method_name(alpha);model_dir=directory/'models'/method
            fitted=fit_model(train,validation,model_cfg,alpha,condition['seed'],model_dir)
            q_history=np.load(model_dir/'q_history.npy');q_tables[method]=q_history[-1];policies[method]=greedy_policy(q_history[-1])
            models[method]={'directory':model_dir.relative_to(output).as_posix(),'training':fitted}
            for snapshot,q,trace in zip(fitted['snapshot_steps'],q_history,fitted['history']):
                value=policy_metrics(world,greedy_policy(q),mu,counts,gamma,q_prediction=q)
                curves.append({'condition_id':condition['id'],'scenario':condition['scenario'],
                    'training_episodes':condition['training_episodes'],'seed':condition['seed'],'method':method,
                    'step':snapshot,'discounted_return':value['discounted_return'],'q_start_bias':value['q_start_bias'],
                    'zero_behavior_support_fraction':value['zero_behavior_support_fraction'],**trace})
            print(f'trained {condition["id"]} {method} steps={config["model"]["steps"]}',flush=True)
        initials={m['training']['initial_parameter_sha256'] for m in models.values()}
        batches={m['training']['batch_sequence_sha256'] for m in models.values()}
        if len(initials)!=1 or len(batches)!=1:raise ValueError('Paired training initialization/batches differ')
        metrics={};samples={};policy_dir=directory/'policies';policy_dir.mkdir()
        for method,policy in policies.items():
            np.save(policy_dir/(method+'_policy.npy'),policy)
            value=policy_metrics(world,policy,mu,counts,gamma,q_tables.get(method))
            sampled=rollout(world,policy,config['rollout_episodes'],condition['rollout_seed'],gamma)
            np.savez_compressed(policy_dir/(method+'_returns.npz'),**sampled);samples[method]=sampled
            interval=mean_interval(sampled['returns'],config['bootstrap_repeats'],config['bootstrap_seed'])
            row={k:condition[k] for k in ['scenario','training_episodes','seed']}
            row.update(condition_id=condition['id'],method=method,metrics=value,rollout=interval,
                policy_artifact=(policy_dir/(method+'_policy.npy')).relative_to(output).as_posix(),
                returns_artifact=(policy_dir/(method+'_returns.npz')).relative_to(output).as_posix())
            metrics[method]=row;rows.append(row)
        metadata={'train':validate_log(world,mu,train),'validation':validate_log(world,mu,validation),
            'train_log_sha256':log_fingerprint(train),'validation_log_sha256':log_fingerprint(validation),
            'observed_time_state_action_fraction':float((counts>0).mean()),
            'logging_zero_probability_fraction':float((mu==0).mean()),
            'episode_overlap':int(len(np.intersect1d(train['episode_id'],validation['episode_id'])))}
        conditions.append({**condition,'directory':directory.relative_to(output).as_posix(),'data':metadata,
            'models':models,'metrics':metrics,'comparisons':paired_comparisons(metrics,samples,config)})
    result={'status':'SIMULATED','real_rl_value_claimed':False,'real_top20_changed':False,
        'real_transition_status':'NOT_ESTABLISHED_FOR_KUAIRAND_PURE',
        'config':{k:v for k,v in config.items() if k!='output_directory'},'conditions':conditions,
        'metrics':rows,'aggregate':summarize_metrics(rows),'curves':curves,
        'main_method':method_name(config['main_alpha']),
        'framework_assessment':{'pearl_installed':importlib.util.find_spec('pearl') is not None,'pearl_install_attempted':False,
            'pearl_runtime_compatibility':'NOT_TESTED','implementation':'independent discrete CQL(H) in existing PyTorch',
            'upstream_code_copied':False,'new_dependencies_installed':False},
        'selection':SELECTION_PROTOCOL,'limits':LIMITS}
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(output/'metrics.csv',[{**{k:r[k] for k in ['condition_id','scenario','training_episodes','seed','method']},
        **{key:r['metrics'].get(key) for key in ['discounted_return','undiscounted_return','zero_behavior_support_fraction',
            'unobserved_training_action_fraction','action_catalog_coverage','predicted_start_value','q_start_bias']},
        **{'rollout_'+key:r['rollout'][key] for key in ['mean','standard_error','low','high']}} for r in rows])
    write_csv(output/'aggregate.csv',result['aggregate']);write_csv(output/'learning_curves.csv',curves)
    return result
