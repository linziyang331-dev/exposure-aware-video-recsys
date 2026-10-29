"""Bind validation evidence, freeze, replicate, and reassess the existing tests."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.data import write_json
from src.ranking.evaluate import rank_scores
from src.ranking.finalize import export_sorted
from src.ranking.metrics import exposure_metrics
from src.retrieval_v2.provenance import freeze_evidence,verify_evidence
from src.ranking_v2.run import setup
from src.ranking_v2.train import code_hashes,load_model,train_run
from src.ranking_v2.inference import deterministic,candidate_scores,exposure_logits,history_rule_scores


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_training(trial,train,val,train_candidates,val_candidates,seed=42):
    run=read(trial['training_result_path']);signature=run['signature'];cfg=trial['config']
    expected={'config':cfg,'seed':seed,'train':train.metadata['signature'],'validation':val.metadata['signature'],
              'train_candidates':train_candidates.metadata['signature'],
              'validation_candidates':val_candidates.metadata['signature'],
              'teacher_parameters':run['teacher_parameter_sha256'],'code':code_hashes()}
    if signature!=expected or hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()!=run['fingerprint']:
        raise ValueError('Training source/config/data signature differs')
    if cfg!=run['config'] or run['seed']!=seed or digest_file(run['checkpoint'])!=run['checkpoint_sha256']:
        raise ValueError('Training checkpoint binding differs')
    if trial['checkpoint_sha256']!=run['checkpoint_sha256'] or trial['best_validation']!=run['best_validation']:
        raise ValueError('Trial ledger differs from training')
    def value(row):
        m=row['validation'];return m['ranking']['ndcg@20'] if cfg['select_metric']=='ndcg@20' else -m['pointwise']['logloss']
    selected=max(run['curve'],key=value)
    if selected['epoch']!=run['best_epoch'] or selected['validation']!=run['best_validation']:
        raise ValueError('Selected checkpoint does not match early stopping')
    model,cp=load_model(run['checkpoint'],cfg['device'])
    from src.retrieval_v2.train import parameter_digest,load_model as load_teacher
    from src.ranking.provenance import verify_v2
    if cp['architecture']['config']!=cfg or cp['fingerprint']!=run['fingerprint'] or cp['epoch']!=run['best_epoch'] or parameter_digest(model)!=run['trained_parameter_sha256']:
        raise ValueError('Checkpoint contents differ')
    if model.teacher is not None:
        frozen=verify_v2();teacher,_=load_teacher(ROOT/frozen['checkpoint'],cfg['device'])
        if parameter_digest(teacher)!=run['teacher_parameter_sha256'] or parameter_digest(model.teacher)!=parameter_digest(teacher):
            raise ValueError('Frozen teacher differs')
        del teacher
    scores=candidate_scores(model,val,val_candidates,cfg['candidate_batch_size'])
    directory=Path(run['checkpoint']).parent;rank_path=directory/'validation_rechecked.npz'
    metrics=rank_scores(val_candidates,val,scores,rank_path)
    if metrics!=run['best_validation']['ranking']:raise ValueError('Actual checkpoint ranking differs from validation ledger')
    original=directory/('baseline_validation_ranks.npz' if run['best_epoch']==-1 else f'validation_epoch{run["best_epoch"]}.npz')
    with np.load(original) as a,np.load(rank_path) as b:
        for field in ['ranks','users','event_ids']:np.testing.assert_array_equal(a[field],b[field])
    logits=exposure_logits(model,val,cfg['eval_batch_size'])
    if exposure_metrics(val['label'],logits)!=run['best_validation']['pointwise']:
        raise ValueError('Actual checkpoint exposure predictions differ')
    np.save(directory/'validation_rechecked_scores.npy',scores)
    np.savez_compressed(directory/'validation_rechecked_exposures.npz',logits=logits,labels=np.array(val['label']),event_ids=np.array(val['event_id']))
    binding={'status':'PASS','checkpoint_sha256':digest_file(run['checkpoint']),
             'full_validation_rescored':True,'queries':len(val_candidates),'code':code_hashes(),
             'candidate_signature':val_candidates.metadata['signature'],
             'scores_sha256':digest_file(directory/'validation_rechecked_scores.npy'),
             'exposures_sha256':digest_file(directory/'validation_rechecked_exposures.npz'),
             'ranks_sha256':digest_file(rank_path),'original_ranks_sha256':digest_file(original)}
    write_json(directory/'validation_binding.json',binding)
    print('BOUND '+directory.name+' N20='+str(metrics['ndcg@20']),flush=True)
    return run


def close(caches,candidates):
    for c in candidates.values():c.close()
    for c in caches.values():c.close()


def freeze():
    deterministic();cfg,paths,vocab,counts,caches,candidates,frozen=setup();out=paths['output'];path=out/'selection.json'
    if path.exists():
        selected=read(path);verify_evidence(ROOT,selected['evidence'],selected['config']);close(caches,candidates);return selected
    ledger=read(out/'tuning.json')
    required={'rich_bce','choice_residual','direct_choice','no_attention','linear_only','no_repeat_feature'}
    if set(ledger['trials'])!=required or ledger['test_partitions_loaded']:raise ValueError('Complete six validation-only trials required')
    runs={name:verify_training(t,caches['standard/train'],caches['standard/validation'],candidates['standard/train'],candidates['standard/validation'])
          for name,t in ledger['trials'].items()}
    # Component ablations describe attribution and are never silently substituted for DIN.
    eligible=['choice_residual','direct_choice']
    name=max(eligible,key=lambda n:runs[n]['best_validation']['ranking']['ndcg@20']);run=runs[name]
    baseline=rank_scores(candidates['standard/validation'],caches['standard/validation'],candidates['standard/validation']['scores'])
    if run['selected_baseline'] or run['best_validation']['ranking']['ndcg@20']<=baseline['ndcg@20']:
        raise ValueError('No trained DIN improvement; do not claim success')
    files=list((ROOT/'src/ranking_v2').glob('*.py'))+[ROOT/'scripts/train_ranking_v2.py',ROOT/'scripts/train_retrieval_v2.py',ROOT/'configs/ranking_v2.yaml',out/'tuning.json',ROOT/'reports/ranking_v2_history_diagnostic.json',
          ROOT/'experiments/ranking/selection.json',ROOT/'reports/retrieval_v2_freeze.json']
    files.extend((ROOT/'tests').glob('*.py'))
    files.append(ROOT/'reports/environment_logs/all_tests_ranking_v2.txt')
    old=read(ROOT/'experiments/ranking/selection.json')
    files.extend(ROOT/p for p in old['evidence']['files'])
    for r in runs.values():
        d=Path(r['checkpoint']).parent
        files.extend([d/'training.json',Path(r['checkpoint']),d/'validation_binding.json',d/'validation_rechecked_scores.npy',
                      d/'validation_rechecked_exposures.npz',d/'validation_rechecked.npz'])
        files.extend(d.glob('validation_epoch*.npz'));files.append(d/'baseline_validation_ranks.npz')
    for c in candidates.values():files.extend(p for p in c.path.iterdir() if p.is_file())
    selected={'status':'FROZEN_BEFORE_REVISION_TEST','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'test_is_iterative_reassessment':True,'trial':name,'config':run['config'],'primary_seed':42,
              'selection_rule':'Maximum standard validation NDCG20 among attention-enabled choice_residual/direct_choice; controls reported separately',
              'primary_training':run,'baseline_validation':baseline,'history_rule_penalty':4.,
              'training_candidate_coverage':{'positive_queries':len(candidates['standard/train']),
                  'target_present_queries':int((candidates['standard/train']['v2_ranks']>0).sum()),
                  'fraction':float((candidates['standard/train']['v2_ranks']>0).mean())},
              'all_validation_trials':{n:r['best_validation'] for n,r in runs.items()},
              'evidence':freeze_evidence(ROOT,run['config'],files)}
    write_json(path,selected);close(caches,candidates)
    print('FROZEN '+name,flush=True);return selected


def replicate():
    selected=freeze();cfg=selected['config'];_,paths,vocab,counts,caches,candidates,_=setup();out=paths['output'];path=out/'replication.json'
    if path.exists():
        result=read(path);verify_evidence(ROOT,result['evidence'],cfg);close(caches,candidates);return result
    runs={'din_seed42':selected['primary_training']}
    for seed in [43,44]:
        run=train_run(caches['standard/train'],caches['standard/validation'],candidates['standard/train'],candidates['standard/validation'],
                      len(vocab['user_raw_to_index']),len(counts)-1,cfg,out/'artifacts/replication'/f'seed{seed}',seed)
        trial={'config':cfg,'training_result_path':str(Path(run['checkpoint']).parent/'training.json'),
               'checkpoint_sha256':run['checkpoint_sha256'],'best_validation':run['best_validation']}
        runs[f'din_seed{seed}']=verify_training(trial,caches['standard/train'],caches['standard/validation'],candidates['standard/train'],candidates['standard/validation'],seed)
    files=[out/'selection.json']
    for r in runs.values():
        d=Path(r['checkpoint']).parent
        files.extend([Path(r['checkpoint']),d/'training.json',d/'validation_binding.json',d/'validation_rechecked.npz',
                      d/'validation_rechecked_scores.npy',d/'validation_rechecked_exposures.npz'])
    result={'status':'REPLICATED','runs':runs,'evidence':freeze_evidence(ROOT,cfg,files)}
    write_json(path,result);close(caches,candidates);return result


def evaluate():
    deterministic();out=ROOT/'experiments/ranking_v2';selected=read(out/'selection.json');cfg=selected['config'];rep=read(out/'replication.json')
    verify_evidence(ROOT,selected['evidence'],cfg);verify_evidence(ROOT,rep['evidence'],cfg)
    if (out/'evaluation.json').exists():raise ValueError('Already evaluated; use verifier/report instead')
    parts=['standard/validation','standard/test','random/validation','random/test']
    _,_,_,_,caches,candidates,_=setup(parts)
    ledger=read(out/'tuning.json');runs=dict(rep['runs'])
    for name,t in ledger['trials'].items():
        if name!=selected['trial']:runs[name]=read(t['training_result_path'])
    partial=out/'evaluation_partial.json'
    result=read(partial) if partial.exists() else {'status':'EVALUATING','results':{},'baseline':{},'history_rule':{},'exports':{},
              'selection_sha256':digest_file(out/'selection.json'),'replication_sha256':digest_file(out/'replication.json'),
              'test_is_iterative_reassessment':True}
    if result['selection_sha256']!=digest_file(out/'selection.json') or result['replication_sha256']!=digest_file(out/'replication.json'):
        raise ValueError('Partial evaluation is bound to another freeze')
    for part,cache in caches.items():
        cand=candidates[part];stem=part.replace('/','_')
        if part not in result['baseline']:
            for name,scores,key in [('v2',cand['scores'],'baseline'),('history_rule',history_rule_scores(cache,cand,selected['history_rule_penalty']),'history_rule')]:
                path=out/'artifacts/ranks'/name/(stem+'.npz');metrics=rank_scores(cand,cache,scores,path)
                metrics['artifact']=str(path.resolve());result[key][part]=metrics
            write_json(partial,result)
        for name,run in runs.items():
            if part in result['results'].get(name,{}):continue
            if digest_file(run['checkpoint'])!=run['checkpoint_sha256']:raise ValueError('Evaluation checkpoint changed')
            model,_=load_model(run['checkpoint'],cfg['device']);scores=candidate_scores(model,cache,cand,cfg['candidate_batch_size'])
            rank_path=out/'artifacts/ranks'/name/(stem+'.npz');ranking=rank_scores(cand,cache,scores,rank_path)
            logits=exposure_logits(model,cache,cfg['eval_batch_size']);pointwise=exposure_metrics(cache['label'],logits)
            pred=out/'artifacts/predictions'/name/(stem+'.npz');pred.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(pred,logits=logits,labels=np.array(cache['label']),event_ids=np.array(cache['event_id']))
            row={'ranking':ranking,'pointwise':pointwise,'rank_artifact':str(rank_path.resolve()),
                 'prediction_artifact':str(pred.resolve()),'checkpoint_sha256':run['checkpoint_sha256']}
            if name=='din_seed42':
                score_path=out/'artifacts/scores'/(stem+'.npy');score_path.parent.mkdir(parents=True,exist_ok=True);np.save(score_path,scores)
                row['scores_artifact']=str(score_path.resolve())
                result['exports'][part]=export_sorted(out/'artifacts/ranked_candidates'/(stem+'.parquet'),cand,scores,cache)
            result['results'].setdefault(name,{})[part]=row;write_json(partial,result)
            print(name+' '+part+' N20='+str(ranking['ndcg@20'])+' AUC='+str(pointwise['auc']),flush=True)
            del model,scores,logits
    result['status']='EVALUATED';write_json(out/'evaluation.json',result);close(caches,candidates);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['freeze','replicate','evaluate'],required=True)
    a=p.parse_args();{'freeze':freeze,'replicate':replicate,'evaluate':evaluate}[a.phase]()


if __name__=='__main__':main()
