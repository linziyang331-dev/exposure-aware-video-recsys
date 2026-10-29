"""Freeze validation-selected sequence model, replicate, and compare actual outputs."""
import argparse
import hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.data import write_json
from src.ranking.candidates import load_candidates
from src.ranking.evaluate import rank_scores
from src.ranking.finalize import export_sorted
from src.retrieval.metrics import stable_topk
from src.retrieval_v2.provenance import freeze_evidence,verify_evidence
from src.retrieval_v2.train import parameter_digest
from src.ranking_v2.inference import deterministic
from src.sequential.run import setup
from src.sequential.provenance import read,code_hashes
from src.sequential.train import load_model,train_run
from src.sequential.evaluate import catalog_evaluate,pool_scores,metric_identity,SequenceScorer


class CandidateView(dict):
    def __len__(self):return len(self['indices'])


def verify_run(trial,train,val,counts,seed=42):
    run=read(trial['training_result_path']);cfg=trial['config']
    signature={'config':cfg,'seed':seed,'train':train.metadata['signature'],'validation':val.metadata['signature'],
               'counts_sha256':hashlib.sha256(np.asarray(counts,dtype=np.int64).tobytes()).hexdigest(),'code':code_hashes()}
    fp=hashlib.sha256(__import__('json').dumps(signature,sort_keys=True).encode()).hexdigest()
    if signature!=run['signature'] or fp!=run['fingerprint']:raise ValueError('Sequence training signature changed')
    if run['checkpoint_sha256']!=trial['checkpoint_sha256'] or digest_file(run['checkpoint'])!=run['checkpoint_sha256']:
        raise ValueError('Sequence checkpoint changed')
    best=max((r for r in run['curve'] if r['epoch']>=0),key=lambda r:r['validation']['ndcg@20'])
    if best['epoch']!=run['best_epoch'] or best['validation']!=run['best_validation'] or trial['best_validation']!=run['best_validation']:
        raise ValueError('Sequence checkpoint selection differs')
    model,cp=load_model(run['checkpoint'],cfg['device'])
    if cp['config']!=cfg or cp['fingerprint']!=fp or cp['seed']!=seed or cp['epoch']!=run['best_epoch']:
        raise ValueError('Checkpoint signature differs')
    if parameter_digest(model)!=run['trained_parameter_sha256']:raise ValueError('Sequence parameters differ')
    np.testing.assert_array_equal(model.counts.cpu().numpy(),counts)
    directory=Path(run['checkpoint']).parent;path=directory/'validation_rechecked.npz'
    metrics=catalog_evaluate(model,val,counts,cfg['eval_batch_size'],path)
    if metric_identity(metrics)!=metric_identity(run['best_validation']):raise ValueError('Actual sequence validation metrics differ')
    with np.load(path) as a,np.load(directory/f'validation_epoch{run["best_epoch"]}.npz') as b:
        for field in ['ranks','users','event_ids']:np.testing.assert_array_equal(a[field],b[field])
    binding={'status':'PASS','full_validation_rescore':True,'queries':metrics['queries'],'code':code_hashes(),
             'checkpoint_sha256':digest_file(run['checkpoint']),'ranks_sha256':digest_file(path),
             'original_validation_scores_not_stored':True,'original_validation_ranks_reproduced_exactly':True}
    write_json(directory/'validation_binding.json',binding)
    print('BOUND '+directory.name+' N20='+str(metrics['ndcg@20']),flush=True);return run


def freeze():
    deterministic();cfg,paths,counts,caches,frozen,din=setup();out=paths['output'];path=out/'selection.json'
    if path.exists():
        sel=read(path);verify_evidence(ROOT,sel['evidence'],sel['config'])
        for c in caches.values():c.close()
        return sel
    ledger=read(out/'tuning.json')
    if set(ledger['trials'])!={'history20','history50','no_position'} or ledger['test_partitions_loaded']:raise ValueError('Exactly3 validation-only configurations required')
    runs={n:verify_run(t,caches['standard/train'],caches['standard/validation'],counts) for n,t in ledger['trials'].items()}
    name=max(['history20','history50'],key=lambda n:runs[n]['best_validation']['ndcg@20']);run=runs[name]
    if runs['no_position']['config']!=run['config']|{'position_embeddings':False}:raise ValueError('Positional ablation must match selected configuration')
    files=list((ROOT/'src/sequential').glob('*.py'))+[ROOT/'scripts/train_sequential.py',ROOT/'scripts/train_retrieval_v2.py',
             ROOT/'configs/sequential.yaml',out/'tuning.json',ROOT/'reports/retrieval_v2_freeze.json',
             ROOT/'reports/sequential_review.md',ROOT/'reports/sequential_throughput.json',
             ROOT/'experiments/ranking_v2/selection.json',ROOT/'experiments/ranking_v2/replication.json',
             ROOT/'experiments/retrieval_v2/evaluation.json',ROOT/'reports/retrieval_optimization_results.json',
             ROOT/'experiments/ranking_v2/evaluation.json',ROOT/'reports/ranking_v2_artifact_validation.json',
             ROOT/'experiments/ranking/evaluation.json']
    files.extend(ROOT/p for p in din['evidence']['files'])
    rep=read(ROOT/'experiments/ranking_v2/replication.json');files.extend(ROOT/p for p in rep['evidence']['files'])
    prior=read(ROOT/'reports/ranking_v2_artifact_validation.json')
    for p,h in prior['artifact_hashes'].items():
        if digest_file(ROOT/p)!=h:raise ValueError('Prior DIN artifact changed')
        files.append(ROOT/p)
    retrieval=read(ROOT/'reports/retrieval_optimization_results.json')
    for p,h in retrieval['rank_artifact_hashes'].items():
        if digest_file(ROOT/p)!=h:raise ValueError('Prior retrieval rank changed')
        files.append(ROOT/p)
    import recbole
    upstream=Path(recbole.__file__).parent;files.extend(upstream/n for n in code_hashes()['upstream'])
    files.extend((ROOT/'tests').glob('*.py'));files.append(ROOT/'reports/environment_logs/all_tests_sequential.txt')
    for r in runs.values():
        d=Path(r['checkpoint']).parent
        files.extend([Path(r['checkpoint']),d/'initial.pt',d/'training.json',d/'validation_rechecked.npz',d/'validation_binding.json'])
        files.extend(d.glob('validation_epoch*.npz'))
    sel={'status':'FROZEN_BEFORE_SEQUENTIAL_TEST','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
         'test_is_iterative_reassessment':True,'trial':name,'config':run['config'],'primary_seed':42,
         'primary_training':run,'position_ablation':runs['no_position'],
         'selection_rule':'Maximum full standard-validation NDCG20 among history20/history50; matched ablation excluded',
         'evidence':freeze_evidence(ROOT,run['config'],files)}
    write_json(path,sel)
    for c in caches.values():c.close()
    print('FROZEN SASRec '+name,flush=True);return sel


def replicate():
    sel=freeze();cfg=sel['config'];_,paths,counts,caches,_,_=setup();out=paths['output'];path=out/'replication.json'
    if path.exists():
        result=read(path);verify_evidence(ROOT,result['evidence'],cfg)
        for c in caches.values():c.close()
        return result
    runs={'sasrec_seed42':sel['primary_training']}
    for seed in [43,44]:
        run=train_run(caches['standard/train'],caches['standard/validation'],counts,cfg,out/'artifacts/replication'/f'seed{seed}',seed)
        t={'config':cfg,'training_result_path':str(Path(run['checkpoint']).parent/'training.json'),
           'best_validation':run['best_validation'],'checkpoint_sha256':run['checkpoint_sha256']}
        runs[f'sasrec_seed{seed}']=verify_run(t,caches['standard/train'],caches['standard/validation'],counts,seed)
    files=[out/'selection.json']
    for r in runs.values():
        d=Path(r['checkpoint']).parent;files.extend([Path(r['checkpoint']),d/'training.json',d/'validation_rechecked.npz',d/'validation_binding.json'])
    result={'status':'REPLICATED','runs':runs,'evidence':freeze_evidence(ROOT,cfg,files)};write_json(path,result)
    for c in caches.values():c.close()
    return result


@torch.no_grad()
def export_catalog(model,cache,counts,directory,batch_size):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);eligible=np.flatnonzero(cache['label']==1)
    ids=np.lib.format.open_memmap(directory/'ids.npy',mode='w+',dtype=np.int32,shape=(len(eligible),100))
    scores=np.lib.format.open_memmap(directory/'scores.npy',mode='w+',dtype=np.float32,shape=ids.shape)
    scorer=SequenceScorer(model,counts)
    for start in range(0,len(eligible),batch_size):
        stop=min(start+batch_size,len(eligible));values=scorer.scores(cache,eligible[start:stop])
        top,value=stable_topk(values,scorer.candidate_ids,100);ids[start:stop]=top.cpu().numpy();scores[start:stop]=value.cpu().numpy()
    ids.flush();scores.flush()
    view=CandidateView(ids=ids,indices=eligible,event_ids=np.array(cache['event_id'][eligible]),user_ids=np.array(cache['user_id'][eligible]))
    exported=export_sorted(directory/'top100.parquet',view,scores,cache)
    ids._mmap.close();scores._mmap.close();return exported


def evaluate():
    deterministic();out=ROOT/'experiments/sequential';sel=read(out/'selection.json');cfg=sel['config'];rep=read(out/'replication.json')
    verify_evidence(ROOT,sel['evidence'],cfg);verify_evidence(ROOT,rep['evidence'],cfg)
    if (out/'evaluation.json').exists():raise ValueError('Already evaluated; use verify/report')
    parts=['standard/validation','standard/test','random/validation','random/test']
    _,_,counts,caches,frozen,_=setup(parts);runs=rep['runs']|{'no_position':sel['position_ablation']}
    partial=out/'evaluation_partial.json';result=read(partial) if partial.exists() else {'status':'EVALUATING','results':{},'exports':{},
            'selection_sha256':digest_file(out/'selection.json'),'replication_sha256':digest_file(out/'replication.json'),
            'test_is_iterative_reassessment':True}
    if result['selection_sha256']!=digest_file(out/'selection.json') or result['replication_sha256']!=digest_file(out/'replication.json'):
        raise ValueError('Partial evaluation has a different freeze')
    for part,cache in caches.items():
        cand=load_candidates(part,cache,counts,ROOT/'experiments/ranking',frozen,cfg['device']);stem=part.replace('/','_')
        for name,run in runs.items():
            if part in result['results'].get(name,{}):continue
            if digest_file(run['checkpoint'])!=run['checkpoint_sha256']:raise ValueError('Changed checkpoint')
            model,_=load_model(run['checkpoint'],cfg['device']);d=out/'artifacts/evaluation'/name/stem;d.mkdir(parents=True,exist_ok=True)
            cat=catalog_evaluate(model,cache,counts,cfg['eval_batch_size'],d/'catalog_ranks.npz')
            values=pool_scores(model,cache,cand,cfg['eval_batch_size']);pool=rank_scores(cand,cache,values,d/'pool_ranks.npz')
            row={'catalog':cat,'fixed_v2_pool':pool,'catalog_rank_artifact':str((d/'catalog_ranks.npz').resolve()),
                 'pool_rank_artifact':str((d/'pool_ranks.npz').resolve()),'checkpoint_sha256':run['checkpoint_sha256']}
            if name=='sasrec_seed42':
                np.save(d/'pool_scores.npy',values);row['pool_scores_artifact']=str((d/'pool_scores.npy').resolve())
                exports={'fixed_v2_pool':export_sorted(d/'pool_top100.parquet',cand,values,cache),
                         'catalog':export_catalog(model,cache,counts,d/'catalog_top100',cfg['eval_batch_size'])}
                result['exports'][part]=exports
                item_path=out/'artifacts/item_embeddings.npy'
                if not item_path.exists():np.save(item_path,model.core.item_embedding.weight[1:].detach().cpu().numpy())
            result['results'].setdefault(name,{})[part]=row;write_json(partial,result)
            print(name+' '+part+' catalogN20='+str(cat['ndcg@20'])+' poolN20='+str(pool['ndcg@20']),flush=True)
            del model,values
        cand.close()
    result['status']='EVALUATED';write_json(out/'evaluation.json',result)
    for c in caches.values():c.close()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['freeze','replicate','evaluate'],required=True)
    a=p.parse_args();{'freeze':freeze,'replicate':replicate,'evaluate':evaluate}[a.phase]()
