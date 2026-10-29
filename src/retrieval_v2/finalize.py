"""Freeze validation winner, replicate, then reassess and export exactly once."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.data import load_partition
from src.retrieval.metrics import summarize_ranks
from src.retrieval.candidates import export_snapshot,export_event_candidates
from src.retrieval_v2.run import setup,write_json
from src.retrieval_v2.train import train_run,load_model,implementation_hashes
from src.retrieval_v2.evaluate import Scorer,evaluate
from src.retrieval_v2.provenance import freeze_evidence,verify_evidence


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_completed_trial(trial,train,validation):
    result=read(trial['result_path']);signature=result['signature'];cfg=trial['config']
    expected={'config':cfg,'seed':42,'train':train.metadata['signature'],
              'validation':validation.metadata['signature'],'code':implementation_hashes()}
    fingerprint=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    if signature!=expected or result['fingerprint']!=fingerprint or result['config']!=cfg or result['seed']!=42:
        raise ValueError('Training config/code/data signature differs from current trial')
    if trial['best_validation']!=result['best_validation'] or trial['trained_parameter_sha256']!=result['trained_parameter_sha256']:
        raise ValueError('Trial ledger metrics/parameters differ from training evidence')
    if trial['checkpoint_sha256']!=result['checkpoint_sha256'] or digest_file(result['checkpoint'])!=result['checkpoint_sha256']:
        raise ValueError('Trial checkpoint changed')
    best=result['curve'][result['best_epoch']]
    if best['validation']!=result['best_validation'] or best['validation']['ndcg@20']!=max(r['validation']['ndcg@20'] for r in result['curve']):
        raise ValueError('Trial best epoch/metrics disagree')
    return result


def freeze():
    cfg,output,prepared,vocabulary,counts,parts=setup()
    path=output/'selection.json'
    if path.exists():
        selection=read(path);verify_evidence(ROOT,selection['evidence'],selection['config'])
        for c in parts.values():c.close()
        return selection
    ledger=read(output/'tuning.json');original=read(ROOT/'experiments/retrieval_selected/selection.json')
    caches={cfg['history_length']:parts}
    for trial in ledger['trials'].values():
        length=trial['config']['history_length']
        if length not in caches:
            caches[length]={p:load_partition(prepared,p,ROOT/cfg['cache_directory'],length)
                            for p in ['standard/train','standard/validation']}
        verify_completed_trial(trial,caches[length]['standard/train'],caches[length]['standard/validation'])
    baseline=original['validation_baselines']['popularity']
    qualified=[(name,t) for name,t in ledger['trials'].items()
               if t['best_validation']['recall@20']>baseline['recall@20']
               and t['best_validation']['ndcg@20']>baseline['ndcg@20']]
    if not qualified:raise ValueError('No trial beats popularity on both validation metrics')
    name,trial=max(qualified,key=lambda pair:pair[1]['best_validation']['ndcg@20'])
    chosen=verify_completed_trial(trial,caches[trial['config']['history_length']]['standard/train'],
                                  caches[trial['config']['history_length']]['standard/validation']);config=trial['config']
    if digest_file(chosen['checkpoint'])!=trial['checkpoint_sha256']:raise ValueError('Trial checkpoint changed')
    files=list((ROOT/'src/retrieval_v2').glob('*.py'))+list((ROOT/'src/retrieval').glob('*.py'))
    files += [ROOT/'configs/retrieval_v2.yaml',prepared/'manifest.json',prepared/'vocabulary.json',
              output/'tuning.json',Path(trial['result_path']),Path(chosen['checkpoint']),
              ROOT/'experiments/retrieval_selected/selection.json',ROOT/'experiments/retrieval_selected/evaluation.json',
              ROOT/'experiments/retrieval_selected/artifacts/itemcf_selected.npz']
    for baseline_name in ['popularity','itemcf','two_tower_seed42']:
        files.extend((ROOT/'experiments/retrieval_selected/artifacts/ranks'/baseline_name).glob('*.npz'))
    selection={'status':'FROZEN_BEFORE_ITERATIVE_TEST_REASSESSMENT','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
               'test_already_seen_in_previous_stage':True,'selection_uses':'standard validation NDCG@20, Recall/NDCG both above popularity',
               'trial':name,'config':config,'primary_seed':42,'primary_training':chosen,
               'evidence':freeze_evidence(ROOT,config,files),'original_validation_baselines':original['validation_baselines']}
    write_json(path,selection)
    for cached in caches.values():
        for c in cached.values():c.close()
    print('FROZEN '+name+' before test reassessment',flush=True);return selection


def replicate():
    selection=freeze();cfg=selection['config'];_,output,prepared,vocabulary,counts,parts=setup(cfg['history_length'])
    path=output/'replication.json'
    if path.exists():
        result=read(path);verify_evidence(ROOT,result['evidence'],cfg);return result
    runs={'two_tower_v2_seed42':selection['primary_training']}
    for seed in [43,44]:
        runs[f'two_tower_v2_seed{seed}']=train_run(parts['standard/train'],parts['standard/validation'],counts,cfg,
                                               output/'artifacts/replication'/f'seed{seed}',seed)
    ablation=cfg|{'history_mode':'none' if cfg['history_mode']!='none' else 'mean'}
    runs['two_tower_v2_history_ablation_seed42']=train_run(parts['standard/train'],parts['standard/validation'],counts,ablation,
                                                        output/'artifacts/replication/history_ablation_seed42',42)
    pop=selection['original_validation_baselines']['popularity']
    mean={key:float(np.mean([runs[f'two_tower_v2_seed{s}']['best_validation'][key] for s in [42,43,44]]))
          for key in ['recall@20','ndcg@20']}
    passed=all(mean[k]>pop[k] for k in mean)
    files=[Path(r['checkpoint']) for r in runs.values()]+[output/'selection.json']
    result={'status':'REPLICATED','runs':runs,'validation_three_seed_mean':mean,'first_quality_goal_validation':passed,
            'evidence':freeze_evidence(ROOT,cfg,files)}
    write_json(path,result)
    for c in parts.values():c.close()
    print('REPLICATED validation mean '+str(mean)+' quality_goal='+str(passed),flush=True);return result


def reassess():
    cfg0,output,prepared,vocabulary,counts,initial=setup()
    for c in initial.values():c.close()
    selection=read(output/'selection.json');cfg=selection['config'];replication=read(output/'replication.json')
    verify_evidence(ROOT,selection['evidence'],cfg);verify_evidence(ROOT,replication['evidence'],cfg)
    if not replication['first_quality_goal_validation']:raise ValueError('Replicated validation goal failed; report before test')
    path=output/'evaluation.json'
    if path.exists():raise ValueError('Test reassessment already completed; do not rerun or tune against it')
    old=read(ROOT/'experiments/retrieval_selected/evaluation.json')['results']
    parts={p:load_partition(prepared,p,ROOT/cfg['cache_directory'],cfg['history_length'])
           for p in ['standard/validation','standard/test','random/validation','random/test']}
    pop=np.log1p(counts[1:]).astype(np.float32);results={}
    # Baselines use the immutable original full-catalog ranks, with matching event IDs.
    for name in ['popularity','itemcf','two_tower_seed42']:
        results[name]={}
        for key,cache in parts.items():
            rank_path=ROOT/'experiments/retrieval_selected/artifacts/ranks'/name/(key.replace('/','_')+'.npz')
            with np.load(rank_path) as data:
                eligible=np.flatnonzero(cache['label']==1)
                if not np.array_equal(data['event_ids'],cache['event_id'][eligible]):raise ValueError('Baseline query events differ')
                if not np.array_equal(data['users'],cache['user_id'][eligible]):raise ValueError('Baseline query users differ')
                result=old[name][key].copy();extra=summarize_ranks(data['ranks'],data['users'],ks=(100,))
                result.update({k:extra[k] for k in ['recall@100','ndcg@100']});result['rank_artifact']=str(rank_path)
            results[name][key]=result
    for name,run in replication['runs'].items():
        model,_=load_model(run['checkpoint'],cfg['device']);scorer=Scorer('two_tower',pop,cfg['device'],model=model)
        results[name]={}
        for key,cache in parts.items():
            artifact=output/'artifacts/ranks'/name/(key.replace('/','_')+'.npz')
            value=evaluate(scorer,cache,counts,batch_size=cfg['eval_batch_size'],artifact=artifact)
            value['rank_artifact']=str(artifact.resolve());results[name][key]=value
            print(name+' '+key+' R20='+str(value['recall@20'])+' N20='+str(value['ndcg@20']),flush=True)
        write_json(output/'evaluation_partial.json',results)
    result={'status':'ITERATIVE_REASSESSMENT_COMPLETE','selection_sha256':digest_file(output/'selection.json'),
            'replication_sha256':digest_file(output/'replication.json'),'test_already_seen_in_previous_stage':True,
            'results':results}
    write_json(path,result)
    for c in parts.values():c.close()
    return result


def export():
    _,output,prepared,vocabulary,counts,initial=setup()
    for c in initial.values():c.close()
    selection=read(output/'selection.json');cfg=selection['config'];replication=read(output/'replication.json')
    verify_evidence(ROOT,selection['evidence'],cfg);verify_evidence(ROOT,replication['evidence'],cfg)
    model,_=load_model(replication['runs']['two_tower_v2_seed42']['checkpoint'],cfg['device'])
    train=load_partition(prepared,'standard/train',ROOT/cfg['cache_directory'],cfg['history_length'])
    manifest=read(prepared/'manifest.json')
    result=export_snapshot(model,train,vocabulary,manifest['signature']['protocol']['train_end_ms'],
                           output/'artifacts/snapshot',100,cfg['eval_batch_size']);train.close()
    validation=load_partition(prepared,'standard/validation',ROOT/cfg['cache_directory'],cfg['history_length'])
    scorer=Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),cfg['device'],model=model)
    result['events']=export_event_candidates(scorer,validation,vocabulary,output/'artifacts/candidates/standard_validation.parquet',100,cfg['eval_batch_size'])
    # Actual target/label independence check on fixed random exposure rows.
    random=np.random.default_rng(20261028);idx=random.choice(len(validation),256,replace=False)
    original=scorer.scores(validation,idx)
    class Mutated:
        def __getitem__(self,name):
            if name in {'item_index','label'}:return np.zeros(len(validation),dtype=validation[name].dtype)
            return validation[name]
    changed=scorer.scores(Mutated(),idx)
    torch.testing.assert_close(original,changed,rtol=0,atol=0)
    result['target_label_independence_queries']=256;result['target_label_independence']=True
    write_json(output/'exports.json',result);validation.close();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['replicate','evaluate','export'],required=True)
    args=parser.parse_args();{'replicate':replicate,'evaluate':reassess,'export':export}[args.phase]()


if __name__=='__main__':main()
