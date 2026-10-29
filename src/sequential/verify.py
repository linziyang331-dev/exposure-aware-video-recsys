"""Recompute primary SASRec recommendations and validate all saved rank metrics."""
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.data import write_json
from src.ranking.candidates import load_candidates
from src.ranking.verify import verify_rank_file
from src.retrieval.metrics import summarize_ranks,stable_topk,target_ranks
from src.retrieval_v2.provenance import verify_evidence
from src.ranking_v2.inference import deterministic
from src.sequential.provenance import read
from src.sequential.run import setup
from src.sequential.train import load_model
from src.sequential.evaluate import pool_scores,SequenceScorer
from src.sequential.finalize import CandidateView


def check_catalog_rank(path,cache,metrics,counts):
    positive=np.flatnonzero(cache['label']==1)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['event_ids'],cache['event_id'][positive]);np.testing.assert_array_equal(z['users'],cache['user_id'][positive])
        ranks=z['ranks'].copy()
        if ((ranks<0)|(ranks>=len(counts))).any():raise ValueError('Invalid catalog rank')
        np.testing.assert_array_equal(ranks==0,cache['item_index'][positive]==0)
        derived=summarize_ranks(ranks,z['users'],ks=(10,20,100))
    for key in ['queries','users','recall@10','ndcg@10','recall@20','ndcg@20','recall@100','ndcg@100','mrr','user_macro']:
        if derived[key]!=metrics[key]:raise ValueError('Catalog aggregate differs: '+key)
    users=np.array(cache['user_id'][positive]);items=np.array(cache['item_index'][positive]);warm=cache['user_index'][positive]>0
    head=np.zeros(len(counts),dtype=bool);ordering=np.argsort(-counts[1:],kind='stable')+1
    head[ordering[:int(np.ceil(.2*len(ordering)))]]=True
    empty=(cache['history'][positive]>0).sum(1)==0
    masks={'warm_user_and_item':warm&(items>0),'head_target':(items>0)&head[items],
           'tail_target':(items>0)&~head[items],'cold_target':items==0,'cold_user':~warm,
           'empty_history':empty,'nonempty_history':~empty}
    for name,mask in masks.items():
        ks=(10,20,100) if name in {'empty_history','nonempty_history'} else (10,20)
        expected=summarize_ranks(ranks[mask],users[mask],ks=ks) if mask.any() else {'queries':0}
        if expected!=metrics['groups'][name]:raise ValueError('Catalog group aggregate differs: '+name)
    bounded=summarize_ranks(np.where(ranks<=100,ranks,0),users)['mrr']
    if bounded!=metrics['mrr@100']:raise ValueError('Catalog MRR100 differs')
    for key,value in {'known_item_queries':int((items>0).sum()),'cold_item_queries':int((items==0).sum()),
                      'cold_user_queries':int((~warm).sum()),'candidate_count':len(counts)-1}.items():
        if metrics[key]!=value:raise ValueError('Catalog denominator differs: '+key)
    return ranks


def check_pool_rank(path,candidates,metrics):
    verify_rank_file(path,candidates,metrics)
    with np.load(path) as z:
        ranks=z['ranks'];users=z['users'];present=candidates['v2_ranks']>0
        derived=summarize_ranks(ranks,users,ks=(10,20,100))
        derived['mrr@100']=derived.pop('mrr');derived['user_macro']['mrr@100']=derived['user_macro'].pop('mrr')
        derived['retrieval_miss_queries']=int((~present).sum())
        derived['conditional_hit_metrics']=summarize_ranks(ranks[present],users[present],ks=(10,20)) if present.any() else {'queries':0}
    if derived!=metrics:raise ValueError('Fixed pool aggregate differs')
    return derived


def main():
    deterministic();out=ROOT/'experiments/sequential';sel=read(out/'selection.json');cfg=sel['config'];rep=read(out/'replication.json');ev=read(out/'evaluation.json')
    verify_evidence(ROOT,sel['evidence'],cfg);verify_evidence(ROOT,rep['evidence'],cfg)
    if ev['status']!='EVALUATED' or ev['selection_sha256']!=digest_file(out/'selection.json') or ev['replication_sha256']!=digest_file(out/'replication.json'):
        raise ValueError('Sequence evaluation freeze differs')
    _,_,counts,caches,frozen,_=setup(list(ev['exports']))
    model,_=load_model(rep['runs']['sasrec_seed42']['checkpoint'],cfg['device']);scorer=SequenceScorer(model,counts);hashes={};checks={}
    def record(path):hashes[Path(path).relative_to(ROOT).as_posix()]=digest_file(path)
    for name in ['selection.json','replication.json','evaluation.json']:record(out/name)
    for part,cache in caches.items():
        cand=load_candidates(part,cache,counts,ROOT/'experiments/ranking',frozen,cfg['device']);positive=np.flatnonzero(cache['label']==1)
        np.testing.assert_array_equal(cand['indices'],positive);np.testing.assert_array_equal(cand['event_ids'],cache['event_id'][positive])
        for name,parts in ev['results'].items():
            row=parts[part];check_catalog_rank(row['catalog_rank_artifact'],cache,row['catalog'],counts)
            check_pool_rank(row['pool_rank_artifact'],cand,row['fixed_v2_pool'])
            if row['fixed_v2_pool']['recall@100']!=float((cand['v2_ranks']>0).mean()):raise ValueError('Fixed pool Recall100 changed')
            record(row['catalog_rank_artifact']);record(row['pool_rank_artifact'])
        primary=ev['results']['sasrec_seed42'][part]
        with np.load(primary['catalog_rank_artifact']) as z:ranks=z['ranks'].copy()
        values=pool_scores(model,cache,cand,cfg['eval_batch_size']);saved=np.load(primary['pool_scores_artifact'],mmap_mode='r')
        np.testing.assert_array_equal(values,saved);record(primary['pool_scores_artifact'])
        from src.ranking.evaluate import rank_scores
        if rank_scores(cand,cache,values)!=primary['fixed_v2_pool']:raise ValueError('Actual pool metrics differ')
        for kind in ['catalog','fixed_v2_pool']:
            exported=ev['exports'][part][kind];path=Path(exported['path'])
            if digest_file(path)!=exported['sha256']:raise ValueError('Export changed')
            offset=0;covered=np.zeros(len(counts),dtype=bool);tail_count=0;recommendation_count=0
            ordering=np.argsort(-counts[1:],kind='stable')+1;head=np.zeros(len(counts),dtype=bool)
            head[ordering[:int(np.ceil(.2*len(ordering)))]]=True
            with pq.ParquetFile(path) as parquet:
                for block in parquet.iter_batches(batch_size=cfg['eval_batch_size']):
                    size=len(block);stop=offset+size;indices=positive[offset:stop]
                    def col(name):return block.column(block.schema.get_field_index(name))
                    ids=pc.list_flatten(col('candidate_item_indices')).to_numpy().reshape(size,100)
                    scores=pc.list_flatten(col('ranking_scores')).to_numpy().reshape(size,100)
                    if kind=='catalog':
                        all_scores=scorer.scores(cache,indices);top,value=stable_topk(all_scores,scorer.candidate_ids,100)
                        expected_ids=top.cpu().numpy();expected_scores=value.cpu().numpy()
                        covered[expected_ids[:,:20].ravel()]=True
                        tail_count+=int((~head[expected_ids[:,:20]]).sum());recommendation_count+=expected_ids[:,:20].size
                        targets=torch.as_tensor(np.array(cache['item_index'][indices],copy=True),device=scorer.device,dtype=torch.long)
                        np.testing.assert_array_equal(ranks[offset:stop],target_ranks(all_scores,scorer.candidate_ids,targets).cpu().numpy())
                    else:
                        source=np.array(cand['ids'][offset:stop]);v=values[offset:stop];order=np.lexsort((source,-v),axis=1)
                        expected_ids=np.take_along_axis(source,order,1);expected_scores=np.take_along_axis(v,order,1)
                    np.testing.assert_array_equal(ids,expected_ids);np.testing.assert_array_equal(scores,expected_scores)
                    np.testing.assert_array_equal(col('event_id').to_numpy(),cache['event_id'][indices]);np.testing.assert_array_equal(col('user_id').to_numpy(),cache['user_id'][indices])
                    np.testing.assert_array_equal(col('time_ms').to_numpy(),cache['time_ms'][indices]);offset=stop
            if offset!=len(positive):raise ValueError('Missing exported positive events')
            if kind=='catalog':
                expected={'coverage_at20':float(covered[1:].mean()),'tail_coverage_at20':float(covered[1:][~head[1:]].mean()),
                          'tail_recommendation_fraction_at20':tail_count/recommendation_count}
                for key,value in expected.items():
                    if primary['catalog'][key]!=value:raise ValueError('Primary recommendation coverage differs: '+key)
            record(path)
        n=min(cfg['eval_batch_size'],len(cand));subset=CandidateView({k:np.array(cand[k][:n]) for k in ['ids','indices']})
        class Changed:
            def __getitem__(self,key):
                if key in {'item_index','label'}:return np.zeros(len(cache),dtype=cache[key].dtype)
                return cache[key]
        np.testing.assert_array_equal(values[:n],pool_scores(model,Changed(),subset,cfg['eval_batch_size']))
        checks[part]={'positive_queries':len(positive),'full_primary_catalog_predictions_and_top100_recomputed':True,
                      'full_primary_pool_predictions_and_export_recomputed':True,'all_model_rank_aggregates_checked':True,
                      'all_model_subgroups_and_macro_metrics_checked':True,'primary_catalog_coverage_recomputed':True,
                      'current_target_label_mutation_invariant_rows':n,'pool_recall100_unchanged':True}
        cand.close();print('VERIFIED SASRec '+part+' rows='+str(len(positive)),flush=True)
    np.testing.assert_array_equal(np.load(out/'artifacts/item_embeddings.npy'),model.core.item_embedding.weight[1:].detach().cpu().numpy())
    record(out/'artifacts/item_embeddings.npy')
    for c in caches.values():c.close()
    result={'status':'PASS','checks':checks,'artifact_hashes':hashes,'verifier_sha256':digest_file(__file__)}
    write_json(ROOT/'reports/sequential_artifact_validation.json',result);return result


if __name__=='__main__':main()
