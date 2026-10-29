"""Freeze validation-selected upstream DIN, replicate, then evaluate once."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import recbole
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.metrics import summarize_ranks
from src.retrieval_v2.provenance import freeze_evidence,verify_evidence
from src.ranking.data import setup,write_json
from src.ranking.provenance import implementation_hashes
from src.ranking.train import train_run,load_model
from src.ranking.candidates import load_candidates
from src.ranking.evaluate import pointwise,evaluate_candidates,rank_scores,candidate_logits


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validation_signature(trial,candidates,logits_path):
    files=['src/ranking/evaluate.py','src/ranking/candidates.py','src/ranking/model.py',
           'src/ranking/metrics.py','src/retrieval/metrics.py']
    return {'checkpoint_sha256':trial['checkpoint_sha256'],'candidate_signature':candidates.metadata['signature'],
            'logits_sha256':digest_file(logits_path),'evaluator_files':{name:digest_file(ROOT/name) for name in files}}


def validate_saved_predictions(trial,cache,candidates):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    directory=Path(trial['training_result_path']).parent;cfg=trial['config'];run=read(trial['training_result_path'])
    if digest_file(run['checkpoint'])!=trial['checkpoint_sha256']:raise ValueError('Validation checkpoint changed')
    path=directory/'validation_candidate_logits.npy';saved=np.load(path,mmap_mode='r')
    signature=validation_signature(trial,candidates,path)
    if 'validation_evidence' in trial and trial['validation_evidence']!=signature:
        raise ValueError('Saved validation evaluator/logits signature differs')
    model,_=load_model(run['checkpoint'],cfg['device'])
    actual=candidate_logits(model,cache,candidates,cfg['candidate_query_batch_size'])
    if not np.array_equal(saved,actual):raise ValueError('Saved validation predictions differ from actual checkpoint')
    files={}
    for mode in trial['candidate_validation']['results']:
        scores=saved if mode=='pure' else candidates['scores']+float(mode.removeprefix('alpha_'))*saved
        current=directory/'validation_rechecked_ranks'/(mode+'.npz')
        rank_scores(candidates,cache,scores,current)
        original=directory/'validation_ranks'/(mode+'.npz')
        with np.load(current) as a,np.load(original) as b:
            for field in ['ranks','users','event_ids']:np.testing.assert_array_equal(a[field],b[field])
        files[mode]=digest_file(original)
    record={'status':'PASS','full_validation_rescore':True,'rows':len(candidates),
            'original_evaluator_hash_recorded':'validation_evidence' in trial,
            'signature':signature,'rank_hashes':files}
    write_json(directory/'validation_binding.json',record);return record


def verify_training_signature(trial,train,validation,frozen):
    run=read(trial['training_result_path']);cfg=trial['config']
    expected={'config':cfg,'seed':42,'train':train.metadata['signature'],
              'validation':validation.metadata['signature'],'implementation':implementation_hashes()}
    if cfg['pretrained_items']:expected['v2_checkpoint_sha256']=frozen['checkpoint_sha256']
    fingerprint=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    if run['signature']!=expected or run['fingerprint']!=fingerprint or run['config']!=cfg:
        raise ValueError('Trial code/data/config differs from actual training')
    if run['checkpoint_sha256']!=trial['checkpoint_sha256'] or digest_file(run['checkpoint'])!=run['checkpoint_sha256']:
        raise ValueError('Trial checkpoint changed')
    if trial['pointwise_validation']!=run['best_validation']:
        raise ValueError('Trial validation evidence changed')
    best=run['curve'][run['best_epoch']]
    if best['validation']!=run['best_validation'] or best['validation']['logloss']!=min(r['validation']['logloss'] for r in run['curve']):
        raise ValueError('Trial early stopping mismatch')
    return run


def verify_trial(trial,train,validation,candidates,frozen):
    run=verify_training_signature(trial,train,validation,frozen)
    if candidates.metadata['signature']!=trial['candidate_signature']:raise ValueError('Trial candidates changed')
    validate_saved_predictions(trial,validation,candidates)
    directory=Path(trial['training_result_path']).parent/'validation_ranks'
    for mode,metrics in trial['candidate_validation']['results'].items():
        with np.load(directory/(mode+'.npz')) as z:
            np.testing.assert_array_equal(z['event_ids'],candidates['event_ids'])
            np.testing.assert_array_equal(z['users'],candidates['user_ids'])
            if not np.array_equal(z['ranks']>0,candidates['v2_ranks']>0):raise ValueError('Candidate membership differs')
            derived=summarize_ranks(z['ranks'],z['users'],ks=(10,20,100))
        for key in ['recall@10','recall@20','ndcg@10','ndcg@20','recall@100']:
            if not np.isclose(derived[key],metrics[key],rtol=0,atol=1e-12):raise ValueError('Ledger rank metrics differ')
    return run


def freeze():
    cfg,paths,vocabulary,counts,caches,frozen=setup();output=paths['output'];path=output/'selection.json'
    if path.exists():
        selected=read(path);verify_evidence(ROOT,selected['evidence'],selected['config'])
        for cache in caches.values():cache.close()
        return selected
    ledger=read(output/'tuning.json');candidates=load_candidates('standard/validation',caches['standard/validation'],counts,output,frozen,cfg['device'])
    runs={name:verify_trial(trial,caches['standard/train'],caches['standard/validation'],candidates,frozen)
          for name,trial in ledger['trials'].items()}
    options=[]
    for name,trial in ledger['trials'].items():
        if trial['config']['kind']!='din':continue
        for mode,value in trial['candidate_validation']['results'].items():
            options.append((value['ndcg@20'],trial['candidate_validation']['results']['pure']['ndcg@20'],
                            -trial['pointwise_validation']['logloss'],name,mode))
    if not options:raise ValueError('No actual DIN trial available')
    choice=max(options,key=lambda x:x[:3]);name,mode=choice[-2:]
    baseline_ndcg=summarize_ranks(candidates['v2_ranks'],candidates['user_ids'])['ndcg@20']
    ranking_improved=choice[0]>baseline_ndcg
    if not ranking_improved:
        # No model is deployed into ranking; preserve the useful exposure predictor.
        name=min((n for n,t in ledger['trials'].items() if t['config']['kind']=='din'),
                 key=lambda n:ledger['trials'][n]['pointwise_validation']['logloss'])
        mode='alpha_0'
    alpha=None if mode=='pure' else float(mode.removeprefix('alpha_'))
    primary=runs[name];selected_config=ledger['trials'][name]['config']
    lr_name=next(n for n,t in ledger['trials'].items() if t['config']['kind']=='lr')
    lr_mode=max(ledger['trials'][lr_name]['candidate_validation']['results'].items(),key=lambda pair:pair[1]['ndcg@20'])[0]
    lr_alpha=None if lr_mode=='pure' else float(lr_mode.removeprefix('alpha_'))
    files=list((ROOT/'src/ranking').glob('*.py'))+[ROOT/'configs/ranking.yaml',output/'tuning.json',
          ROOT/'reports/retrieval_v2_freeze.json',paths['prepared']/'manifest.json',paths['prepared']/'vocabulary.json']
    files.extend(ROOT/name for name in frozen['evidence']['files'])
    upstream=Path(recbole.__file__).parent
    files.extend(upstream/name for name in implementation_hashes()['upstream'])
    for trial in ledger['trials'].values():
        run=read(trial['training_result_path'])
        files.extend([Path(trial['training_result_path']),Path(run['checkpoint'])])
        files.extend([Path(trial['training_result_path']).parent/'validation_candidate_logits.npy',
                      Path(trial['training_result_path']).parent/'validation_binding.json'])
        files.extend((Path(trial['training_result_path']).parent/'validation_ranks').glob('*.npz'))
    files.extend(candidates.path.glob('*'))
    selected={'status':'FROZEN_BEFORE_DIN_TEST','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'test_previously_seen_by_retrieval':True,'trial':name,'config':selected_config,
              'selection_metric':'validation Top100 NDCG20; if no gain retain V2 ordering and select exposure DIN by LogLoss',
              'validation_reranking_improved':ranking_improved,
              'mode':mode,'alpha':alpha,'primary_training':primary,'primary_seed':42,
              'lr_training':runs[lr_name],'lr_mode':lr_mode,'lr_alpha':lr_alpha,
              'evidence':freeze_evidence(ROOT,selected_config,files)}
    write_json(path,selected);candidates.close()
    for cache in caches.values():cache.close()
    print('FROZEN DIN '+name+' mode='+mode,flush=True);return selected


def replicate():
    selection=freeze();cfg=selection['config'];_,paths,vocabulary,counts,caches,frozen=setup()
    path=paths['output']/'replication.json'
    if path.exists():
        old=read(path);verify_evidence(ROOT,old['evidence'],cfg);return old
    runs={'din_seed42':selection['primary_training']}
    for seed in [43,44]:
        runs[f'din_seed{seed}']=train_run(caches['standard/train'],caches['standard/validation'],
                                        len(vocabulary['user_raw_to_index']),len(counts)-1,cfg,
                                        paths['output']/'artifacts/replication'/f'seed{seed}',seed)
    files=[Path(run['checkpoint']) for run in runs.values()]+[paths['output']/'selection.json']
    result={'status':'REPLICATED','runs':runs,'evidence':freeze_evidence(ROOT,cfg,files)}
    write_json(path,result)
    for cache in caches.values():cache.close()
    return result


def export_sorted(path,candidates,scores,cache):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('event_id',pa.int64()),('user_id',pa.int64()),('time_ms',pa.int64()),
                      ('candidate_item_indices',pa.list_(pa.int32())),('ranking_scores',pa.list_(pa.float32()))])
    with pq.ParquetWriter(path,schema,compression='zstd') as writer:
        for start in range(0,len(candidates),512):
            stop=min(start+512,len(candidates));ids=np.array(candidates['ids'][start:stop]);values=np.array(scores[start:stop])
            order=np.lexsort((ids,-values),axis=1);ids=np.take_along_axis(ids,order,axis=1);values=np.take_along_axis(values,order,axis=1)
            indices=candidates['indices'][start:stop]
            rows=[{'event_id':int(candidates['event_ids'][start+j]),'user_id':int(candidates['user_ids'][start+j]),
                   'time_ms':int(cache['time_ms'][indices[j]]),'candidate_item_indices':ids[j].tolist(),'ranking_scores':values[j].tolist()}
                  for j in range(len(ids))]
            writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    return {'path':str(path.resolve()),'rows':len(candidates),'topk':100,'sha256':digest_file(path)}


def evaluate():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    _,paths,vocabulary,counts,_,frozen=setup(partitions=())
    selection=read(paths['output']/'selection.json');replication=read(paths['output']/'replication.json');cfg=selection['config']
    verify_evidence(ROOT,selection['evidence'],cfg);verify_evidence(ROOT,replication['evidence'],cfg)
    result_path=paths['output']/'evaluation.json'
    if result_path.exists():raise ValueError('DIN test already evaluated; no silent repeated test')
    _,_,_,_,caches,_=setup(partitions=['standard/validation','standard/test','random/validation','random/test'])
    runs={'lr':selection['lr_training']}|replication['runs'];results={};exports={};baseline={}
    for partition,cache in caches.items():
        candidates=load_candidates(partition,cache,counts,paths['output'],frozen,cfg['device'])
        base_path=paths['output']/'artifacts/ranks/v2'/(partition.replace('/','_')+'.npz')
        baseline[partition]=rank_scores(candidates,cache,candidates['scores'],base_path)
        baseline[partition]['artifact']=str(base_path.resolve())
        for name,run in runs.items():
            model,_=load_model(run['checkpoint'],cfg['device'])
            prefix=paths['output']/'artifacts/predictions'/name/partition.replace('/','_')
            pw=pointwise(model,cache,cfg['eval_batch_size'],prefix.with_suffix('.npz'))
            rank_dir=paths['output']/'artifacts/ranks'/name/partition.replace('/','_')
            # Alpha selected on seed42 validation is fixed for every test and replica.
            alpha=selection['lr_alpha'] if name=='lr' else selection['alpha']
            alphas=[] if alpha is None else [alpha]
            ranks,logits=evaluate_candidates(model,cache,candidates,alphas,cfg['candidate_query_batch_size'],rank_dir)
            results.setdefault(name,{})[partition]={'pointwise':pw,'reranking':ranks,
                                                  'rank_directory':str(rank_dir.resolve()),'prediction_artifact':str(prefix.with_suffix('.npz').resolve())}
            if name=='din_seed42':
                selected_score=logits if selection['alpha'] is None else candidates['scores']+selection['alpha']*logits
                exports[partition]=export_sorted(paths['output']/'artifacts/ranked_candidates'/(partition.replace('/','_')+'.parquet'),candidates,selected_score,cache)
            print(name+' '+partition+' AUC='+str(pw['auc'])+' pureN20='+str(ranks['results']['pure']['ndcg@20']),flush=True)
        candidates.close();write_json(paths['output']/'evaluation_partial.json',{'results':results,'baseline':baseline,'exports':exports})
    result={'status':'EVALUATED','test_is_iterative_reassessment':True,'results':results,'baseline':baseline,'exports':exports,
            'selection_sha256':digest_file(paths['output']/'selection.json'),'replication_sha256':digest_file(paths['output']/'replication.json')}
    write_json(result_path,result)
    for cache in caches.values():cache.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['replicate','evaluate'],required=True)
    args=parser.parse_args();{'replicate':replicate,'evaluate':evaluate}[args.phase]()


if __name__=='__main__':main()
