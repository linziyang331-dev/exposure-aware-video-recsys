"""Check actual exposure predictions, fixed candidate membership and ranked exports."""
import json
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.metrics import summarize_ranks
from src.retrieval_v2.provenance import verify_evidence
from src.ranking.data import setup,write_json
from src.ranking.candidates import load_candidates
from src.ranking.train import load_model
from src.ranking.evaluate import candidate_logits
from src.ranking.metrics import exposure_metrics


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_rank_file(path,candidates,metrics,expected_ranks=None):
    with np.load(path) as z:
        np.testing.assert_array_equal(z['event_ids'],candidates['event_ids'])
        np.testing.assert_array_equal(z['users'],candidates['user_ids'])
        ranks=z['ranks']
        if ((ranks<0)|(ranks>100)).any():raise ValueError('Candidate ranks must be within0..100')
        np.testing.assert_array_equal(ranks>0,candidates['v2_ranks']>0)
        if expected_ranks is not None:np.testing.assert_array_equal(ranks,expected_ranks)
        rebuilt=summarize_ranks(ranks,z['users'],ks=(10,20,100))
    for key in ['recall@10','recall@20','ndcg@10','ndcg@20','recall@100']:
        if not np.isclose(metrics[key],rebuilt[key],rtol=0,atol=1e-12):raise ValueError('Ranking aggregate mismatch')
    if not np.isclose(metrics['mrr@100'],rebuilt['mrr'],rtol=0,atol=1e-12):raise ValueError('MRR100 aggregate mismatch')
    return digest_file(path)


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    cfg0,paths,vocabulary,counts,_,frozen=setup(partitions=());output=paths['output']
    selection=read(output/'selection.json');cfg=selection['config'];replication=read(output/'replication.json')
    evaluation=read(output/'evaluation.json')
    verify_evidence(ROOT,selection['evidence'],cfg);verify_evidence(ROOT,replication['evidence'],cfg)
    if evaluation['selection_sha256']!=digest_file(output/'selection.json') or evaluation['replication_sha256']!=digest_file(output/'replication.json'):
        raise ValueError('Evaluation freeze binding changed')
    model,_=load_model(replication['runs']['din_seed42']['checkpoint'],cfg['device'])
    _,_,_,_,caches,_=setup(partitions=list(evaluation['baseline']))
    checks={};hashes={};rng=np.random.default_rng(20261028)
    for partition,cache in caches.items():
        candidates=load_candidates(partition,cache,counts,output,frozen,cfg['device'])
        eligible=np.flatnonzero(cache['label']==1)
        np.testing.assert_array_equal(candidates['indices'],eligible)
        np.testing.assert_array_equal(candidates['event_ids'],cache['event_id'][eligible])
        np.testing.assert_array_equal(candidates['user_ids'],cache['user_id'][eligible])
        for start in range(0,len(candidates),8192):
            ids=candidates['ids'][start:start+8192];scores=candidates['scores'][start:start+8192]
            if not ((ids>0)&(ids<len(counts))).all() or not np.isfinite(scores).all():raise ValueError('Invalid raw candidates')
            if not (np.diff(np.sort(ids,axis=1),axis=1)>0).all():raise ValueError('Duplicate raw candidates')
            diff=np.diff(scores,axis=1)
            if (diff>0).any() or not (np.diff(ids,axis=1)[diff==0]>0).all():raise ValueError('Raw score/tie order differs')
        reference=evaluation['baseline'][partition]
        path=Path(reference['artifact'])
        hashes[str(path.relative_to(ROOT))]=verify_rank_file(path,candidates,reference,expected_ranks=candidates['v2_ranks'])
        for name,parts in evaluation['results'].items():
            result=parts[partition]
            pred_path=Path(result['prediction_artifact']);hashes[str(pred_path.relative_to(ROOT))]=digest_file(pred_path)
            with np.load(pred_path) as z:
                np.testing.assert_array_equal(z['event_ids'],cache['event_id']);np.testing.assert_array_equal(z['labels'],cache['label'])
                actual=exposure_metrics(z['labels'],z['logits'])
            for key in ['auc','logloss']:
                if not np.isclose(actual[key],result['pointwise'][key],rtol=0,atol=1e-12):raise ValueError('Exposure aggregate mismatch')
            for mode,metrics in result['reranking']['results'].items():
                path=Path(result['rank_directory'])/(mode+'.npz')
                hashes[str(path.relative_to(ROOT))]=verify_rank_file(path,candidates,metrics)
                if metrics['recall@100']!=reference['recall@100']:raise ValueError('Recall100 changed after reranking')
        exported=evaluation['exports'][partition];path=Path(exported['path'])
        if digest_file(path)!=exported['sha256']:raise ValueError('Ranked export hash mismatch')
        width=cfg['candidate_query_batch_size'];blocks=int(np.ceil(len(candidates)/width))
        selected_blocks=np.sort(rng.choice(blocks,min(2,blocks),replace=False))
        sampled=np.concatenate([np.arange(b*width,min((b+1)*width,len(candidates))) for b in selected_blocks])
        sample_ids=[];sample_scores=[];offset=0
        with pq.ParquetFile(path) as parquet:
            for table in parquet.iter_batches(batch_size=512):
                count=len(table)
                ids=pc.list_flatten(table.column(table.schema.get_field_index('candidate_item_indices'))).to_numpy().reshape(count,100)
                scores=pc.list_flatten(table.column(table.schema.get_field_index('ranking_scores'))).to_numpy().reshape(count,100)
                np.testing.assert_array_equal(np.sort(ids,axis=1),np.sort(candidates['ids'][offset:offset+count],axis=1))
                diff=np.diff(scores,axis=1)
                if not np.isfinite(scores).all() or (diff>0).any() or not (np.diff(ids,axis=1)[diff==0]>0).all():raise ValueError('Ranked order invalid')
                np.testing.assert_array_equal(table.column(table.schema.get_field_index('event_id')).to_numpy(),candidates['event_ids'][offset:offset+count])
                np.testing.assert_array_equal(table.column(table.schema.get_field_index('user_id')).to_numpy(),candidates['user_ids'][offset:offset+count])
                source_rows=candidates['indices'][offset:offset+count]
                np.testing.assert_array_equal(table.column(table.schema.get_field_index('time_ms')).to_numpy(),cache['time_ms'][source_rows])
                local=sampled[(sampled>=offset)&(sampled<offset+count)]-offset
                sample_ids.extend(ids[local]);sample_scores.extend(scores[local]);offset+=count
        if offset!=len(candidates):raise ValueError('Ranked row count differs')
        class Subset(dict):
            def __len__(self):return len(self['indices'])
        subset=Subset(ids=np.array(candidates['ids'][sampled]),indices=np.array(candidates['indices'][sampled]))
        logits=candidate_logits(model,cache,subset,width)
        class Changed:
            def __getitem__(self,key):
                if key in {'item_index','label'}:return np.zeros(len(cache),dtype=cache[key].dtype)
                return cache[key]
        np.testing.assert_array_equal(logits,candidate_logits(model,Changed(),subset,width))
        expected=logits if selection['alpha'] is None else candidates['scores'][sampled]+selection['alpha']*logits
        ids=subset['ids'];order=np.lexsort((ids,-expected),axis=1)
        np.testing.assert_array_equal(np.asarray(sample_ids),np.take_along_axis(ids,order,axis=1))
        np.testing.assert_allclose(np.asarray(sample_scores),np.take_along_axis(expected,order,axis=1),rtol=1e-5,atol=1e-5)
        checks[partition]={'all_candidate_rows_checked':len(candidates),'ranked_rows_checked':offset,
                           'queries_rescored_and_target_label_mutated':len(sampled),'recall100_unchanged':True}
        hashes[str(path.relative_to(ROOT))]=digest_file(path);candidates.close()
    for cache in caches.values():cache.close()
    result={'status':'PASS','checks':checks,'artifact_hashes':hashes,'verifier_sha256':digest_file(__file__)}
    write_json(ROOT/'reports/ranking_artifact_validation.json',result)
    print(json.dumps({'status':'PASS','checks':checks},indent=2),flush=True)


if __name__=='__main__':main()
