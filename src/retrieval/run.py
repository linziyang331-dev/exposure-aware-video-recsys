"""Fit on train/validation first, then evaluate frozen selected retrieval models."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sparse
import torch
import yaml
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.baselines import popularity_scores,fit_itemcf,truncate_neighbors
from src.retrieval.data import load_partition,training_counts
from src.retrieval.evaluate import Scorer,evaluate_partition
from src.retrieval.train import train_run,load_model
from src.retrieval.provenance import read_verified_vocabulary,code_hashes,verify_frozen_inputs


def load_setup(partitions,smoke=False):
    config=yaml.safe_load((ROOT/'configs/retrieval.yaml').read_text(encoding='utf-8'))
    paths={key:(ROOT/config[name]).resolve() for key,name in [('prepared','prepared_directory'),('cache','cache_directory'),('output','output_directory')]}
    if smoke:paths['output']=paths['output'].with_name(paths['output'].name+'_smoke')
    for path in paths.values():
        if not path.is_relative_to(ROOT.resolve()) or path.drive.upper()!='D:':raise ValueError('All paths must stay in D: project')
    paths['artifacts']=paths['output']/'artifacts';paths['artifacts'].mkdir(parents=True,exist_ok=True)
    vocabulary=read_verified_vocabulary(paths['prepared'])
    counts=training_counts(vocabulary)
    caches={key:load_partition(paths['prepared'],key,paths['cache'],config['max_history']) for key in partitions}
    return config,paths,vocabulary,counts,caches


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')


def fit(config,paths,vocabulary,counts,caches,smoke=False):
    train=caches['standard/train'];validation=caches['standard/validation'];start=time.perf_counter()
    pop=popularity_scores(counts);device=config['device'];results={}
    results['popularity']=evaluate_partition(Scorer('popularity',pop,device),validation,counts,query_limit=128 if smoke else None)
    positive=np.flatnonzero(train['label']==1)
    cf_start=time.perf_counter()
    full=fit_itemcf(np.array(train['user_index'][positive]),np.array(train['item_index'][positive]),
                    len(vocabulary['user_raw_to_index']),len(counts)-1,max(config['itemcf_neighbors']))
    cf_fit_seconds=time.perf_counter()-cf_start
    best_score=-1;selected=None
    for neighbors in config['itemcf_neighbors']:
        weights=truncate_neighbors(full,neighbors)
        value=evaluate_partition(Scorer('itemcf',pop,device,weights=weights),validation,counts,
                                 batch_size=config['eval_batch_size'],query_limit=128 if smoke else None)
        results[f'itemcf_k{neighbors}']=value
        print(f'ItemCF k={neighbors} validation NDCG20={value["ndcg@20"]:.6f}',flush=True)
        if value['ndcg@20']>best_score:
            best_score=value['ndcg@20'];selected=neighbors
            sparse.save_npz(paths['artifacts']/'itemcf_selected.npz',weights)
    runs={}
    seeds=[config['primary_seed']] if smoke else config['seeds']
    for seed in seeds:
        name=f'two_tower_seed{seed}'
        runs[name]=train_run(train,validation,counts,config,paths['artifacts']/name,seed,True,smoke=smoke)
    name=f'two_tower_no_history_seed{config["ablation_seed"]}'
    runs[name]=train_run(train,validation,counts,config,paths['artifacts']/name,config['ablation_seed'],False,smoke=smoke)
    prepared=json.loads((paths['prepared']/'manifest.json').read_text(encoding='utf-8'))
    selection={'status':'FROZEN_BEFORE_TEST','smoke_only':smoke,'config':config,'data_fingerprint':prepared['fingerprint'],
               'primary_seed':config['primary_seed'],'selected_itemcf_neighbors':selected,'itemcf_fit_seconds':cf_fit_seconds,
               'itemcf_sha256':digest_file(paths['artifacts']/'itemcf_selected.npz'),
               'validation_baselines':results,'training_runs':runs,'fit_elapsed_seconds':time.perf_counter()-start,
               'torch':torch.__version__,'recbole':metadata.version('recbole'),
               'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
               'code_hashes':code_hashes(),'prepared_manifest_sha256':digest_file(paths['prepared']/'manifest.json'),
               'vocabulary_sha256':digest_file(paths['prepared']/'vocabulary.json')}
    write_json(paths['output']/'selection.json',selection)
    print('FROZEN: all configurations selected using standard validation; no test files loaded in fit',flush=True)
    return selection


def evaluate(config,paths,vocabulary,counts,smoke=False):
    selection=json.loads((paths['output']/'selection.json').read_text())
    verify_frozen_inputs(selection,paths['prepared'])
    if selection['config']!=config or selection['smoke_only']!=smoke:raise ValueError('Frozen configuration changed')
    if digest_file(paths['artifacts']/'itemcf_selected.npz')!=selection['itemcf_sha256']:raise ValueError('ItemCF weights changed')
    prepared=json.loads((paths['prepared']/'manifest.json').read_text())
    if prepared['fingerprint']!=selection['data_fingerprint']:raise ValueError('Data changed after selection')
    parts=['standard/validation','standard/test','random/validation','random/test']
    caches={key:load_partition(paths['prepared'],key,paths['cache'],config['max_history']) for key in parts}
    pop=popularity_scores(counts);weights=sparse.load_npz(paths['artifacts']/'itemcf_selected.npz')
    scorers={'popularity':Scorer('popularity',pop,config['device']),
             'itemcf':Scorer('itemcf',pop,config['device'],weights=weights)}
    results={}
    for name in list(scorers)+list(selection['training_runs']):
        if name in scorers:scorer=scorers[name]
        else:
            info=selection['training_runs'][name];path=Path(info['checkpoint'])
            if digest_file(path)!=info['checkpoint_sha256']:raise ValueError('Checkpoint changed after selection')
            model,checkpoint=load_model(path,config['device'])
            if checkpoint['smoke_only']!=smoke:raise ValueError('Smoke/full model mismatch')
            scorer=Scorer('two_tower',pop,config['device'],model=model)
        results[name]={}
        for key,cache in caches.items():
            artifact=paths['artifacts']/'ranks'/name/(key.replace('/','_')+'.npz')
            value=evaluate_partition(scorer,cache,counts,batch_size=config['eval_batch_size'],artifact=artifact,query_limit=128 if smoke else None)
            results[name][key]=value
            print(f'{name} {key}: recall20={value["recall@20"]:.6f} ndcg20={value["ndcg@20"]:.6f} coverage20={value["coverage_at20"]:.4f}',flush=True)
        write_json(paths['output']/'evaluation_partial.json',results)
    summary={'status':'EVALUATED','smoke_only':smoke,'protocol':'positive-event,repeat-allowed,full-train-catalog,cold-target-miss,float32-ID-ties',
             'selection_sha256':digest_file(paths['output']/'selection.json'),'evaluation_code_hashes':code_hashes(),'results':results}
    write_json(paths['output']/'evaluation.json',summary)
    for cache in caches.values():cache.close()
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['fit','evaluate','all'],default='all')
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    parts=['standard/train','standard/validation'] if args.phase!='evaluate' else []
    config,paths,vocabulary,counts,caches=load_setup(parts,args.smoke)
    if args.phase in ['fit','all']:fit(config,paths,vocabulary,counts,caches,args.smoke)
    if args.phase in ['evaluate','all']:evaluate(config,paths,vocabulary,counts,args.smoke)
    for cache in caches.values():cache.close()


if __name__=='__main__':main()
