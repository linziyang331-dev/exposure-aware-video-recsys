"""Recompute comparable real metrics, chain losses, and separate simulation evidence."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.metrics import binary_metrics
from src.exposure.verify import assert_result_equal
from src.exposure.uncertainty import paired_cluster_means
from src.retrieval.data import load_partition,training_counts
from src.retrieval.provenance import read_verified_vocabulary
from src.system_evaluation.registry import load_registry
from src.system_evaluation.contracts import check_alignment,rank_metrics,check_saved_metrics,cohort_masks,candidate_funnel,transition_counts

PARTS=['standard/validation','standard/test','random/validation','random/test']


def summarize_runs(rows):
    groups=defaultdict(list)
    for row in rows:
        family,_,seed=row['model'].rpartition('_seed')
        if family in ['two_tower_v2','din_v2','sasrec','din_v3_experimental'] and seed in ['42','43','44']:
            groups[(family,row['space'],row['partition'])].append(row)
    result=[]
    for (family,space,part),group in groups.items():
        if len(group)!=3 or len({r['model'] for r in group})!=3:raise ValueError('Three-run summary matrix differs')
        metrics={}
        for name in ['recall@20','ndcg@20','recall@100']:
            values=[r['metrics'][name] for r in group]
            metrics[name]={'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
        result.append({'family':family,'space':space,'partition':part,'runs':3,'metrics':metrics,
                       'shared_initial_backbone':family=='din_v3_experimental'})
    return result


def audit_modules(assets,deep):
    from src.exposure import run as exposure
    from src.exposure.verify import verify_synthetic,verify_real
    from src.offline_rl import run as rl
    from src.offline_rl.report import require_verified_experiment
    from src.offline_rl.verify import verify_experiment
    ec,eo=exposure.get_config();reuse=assets.read('reports/exposure_component_reuse.json')
    for phase in ['synthetic','real']:
        exposure.validate_component(eo/phase,exposure.signature(ec,phase))
        assets.check(eo/phase/'component.json',reuse['component_file_sha256'][phase])
    ev=assets.read(eo/'verification.json')
    if ev['status']!='PASS':raise ValueError('Exposure module verification missing')
    expected={p:assets.read(eo/p/'component.json')['fingerprint'] for p in ['synthetic','real']}
    if ev['component_fingerprints']!=expected:raise ValueError('Exposure verification is stale')
    rc,ro=rl.get_config();rl.validate_component(ro/'experiment',rl.signature(rc));require_verified_experiment(ro)
    assets.check(ro/'experiment/component.json');assets.check(ro/'verification.json')
    simulated=assets.read(eo/'synthetic/synthetic_results.json');real=assets.read(eo/'real/real_results.json')
    rl_result=assets.read(ro/'experiment/results.json')
    audits={'exposure':{'artifact_checks':'PASS','recomputed_in_this_run':deep},
            'offline_rl':{'artifact_checks':'PASS','recomputed_in_this_run':deep}}
    if deep:
        audits['exposure'].update(synthetic=verify_synthetic(ec,eo/'synthetic'),real=verify_real(ec,eo/'real'))
        audits['offline_rl']['numerical']=verify_experiment(rc,ro/'experiment')
    separate={'exposure_calibration':real['calibration_results'],
        'synthetic_ope':{'known_values':simulated['known_values'],'metrics':simulated['metrics'],
                         'weighted_learning':simulated['weighted_learning']},
        'synthetic_rl':{'aggregate':rl_result['aggregate'],'main_method':rl_result['main_method']}}
    return audits,separate


def context_caches(config):
    return {p:load_partition(ROOT/config['prepared_directory'],p,ROOT/config['cache_directory'],50) for p in PARTS}


def analyze(config,registry=None):
    reg=registry or load_registry();assets=reg['assets'];caches=context_caches(config)
    try:
        vocabulary=read_verified_vocabulary(ROOT/config['prepared_directory']);counts=training_counts(vocabulary)
        records=[];groups=[];pointwise=[];raw={};funnels={};paired={};transitions={}
        for part in PARTS:
            c=caches[part];idx=np.flatnonzero(c['label']==1)
            masks=cohort_masks(c['item_index'][idx],c['user_index'][idx],c['history'][idx],counts)
            for spec in [r for r in reg['ranks'] if r['partition']==part]:
                with np.load(spec['path']) as saved:
                    check_alignment(c['event_id'][idx],c['user_id'][idx],saved['event_ids'],saved['users'])
                    ranks=np.array(saved['ranks']);users=np.array(saved['users'])
                metrics=rank_metrics(ranks,users,spec['space']);check_saved_metrics(metrics,spec['expected'])
                if (ranks>len(counts)-1).any():raise ValueError('Rank exceeds known catalog')
                raw[(part,spec['model'],spec['space'])]=ranks
                row={k:spec[k] for k in ['model','partition','space','primary','experimental']}
                row.update(task='logged_positive_ranking',metrics=metrics,artifact=Path(spec['path']).relative_to(ROOT).as_posix())
                records.append(row)
                if spec['primary']:
                    for name,mask in masks.items():
                        m=rank_metrics(ranks[mask],users[mask],spec['space']) if mask.any() else {'queries':0}
                        groups.append({k:row[k] for k in ['model','partition','space','primary','experimental','task']}|{'cohort':name,'metrics':m})
            base=raw[(part,'two_tower_v2_seed42','catalog')];din=raw[(part,'din_v2_seed42','fixed_v2_top100')]
            for spec in [s for s in reg['ranks'] if s['partition']==part and s['space']=='fixed_v2_top100']:
                candidate_funnel(base,raw[(part,spec['model'],spec['space'])])
            funnels[part]={name:candidate_funnel(base[mask],din[mask]) if mask.any() else {'queries':0} for name,mask in masks.items()}
            transitions[part]=transition_counts(base,din)
            def terms(r):
                hit=(r>0)&(r<=20);gain=np.zeros(len(r));gain[hit]=1/np.log2(r[hit]+1)
                return np.column_stack([hit,gain])
            paired[part]={'metrics':['recall@20','ndcg@20'],'direction':'DIN minus full-catalog V2',
                **paired_cluster_means(terms(base),c['user_id'][idx],terms(din),c['user_id'][idx],
                    config['bootstrap_repeats'],config['bootstrap_seed'])}
            for spec in [p for p in reg['predictions'] if p['partition']==part]:
                with np.load(spec['path']) as saved:
                    if not np.array_equal(saved['event_ids'],c['event_id']) or not np.array_equal(saved['labels'],c['label']):
                        raise ValueError('Exposure event/label alignment differs')
                    metrics=binary_metrics(saved['labels'],saved['logits'],bins=15)
                for name in ['auc','logloss','positives','positive_rate']:
                    assert_result_equal(metrics[name],spec['expected'][name])
                assert_result_equal(metrics['rows'],spec['expected']['queries'])
                pointwise.append({'model':spec['model'],'partition':part,'task':'observed_feedback_prediction','metrics':metrics})
            print('Unified metrics '+part+' exposures='+str(len(c))+' positive_queries='+str(len(idx)),flush=True)
        audits,separate=audit_modules(assets,config['deep_audit'])
        return {'status':'ANALYZED','main_pipeline':'two_tower_v2_seed42 -> fixed Top100 -> din_v2_seed42 -> Top20',
            'rank_results':records,'subgroups':groups,'pointwise_results':pointwise,'seed_summary':summarize_runs(records),
            'candidate_funnel':funnels,'top20_transitions':transitions,'paired_user_intervals':paired,
            'other_task_results':separate,'module_audits':audits,'input_hashes':dict(sorted(assets.hashes.items())),
            'catalog_size':len(counts)-1,'test_is_iterative_reassessment':True,'experimental_v3_adopted':False,
            'real_causal_value_claimed':False,'new_model_trained':False,
            'scope':'frozen artifact recomputation and sampled offline pipeline; separate task domains',
            'limits':['Single-positive observed-log proxy; no unexposed relevance labels',
                'Catalog retrieval and fixed-pool ordering shown separately; MRR cutoffs explicit',
                'Calibration/synthetic OPE/synthetic RL are not combined into an overall score',
                'No new environment, online deployment or causal real-policy gain claim']}
    finally:
        for cache in caches.values():cache.close()
