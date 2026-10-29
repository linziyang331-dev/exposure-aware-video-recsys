"""Independent checks on actual v2 vector and candidate artifacts, including bias."""
import json
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.data import load_partition
from src.retrieval.candidates import inverse_ids
from src.retrieval.metrics import stable_topk
from src.retrieval_v2.run import setup,write_json
from src.retrieval_v2.train import load_model
from src.retrieval_v2.evaluate import Scorer
from src.retrieval_v2.provenance import verify_evidence


def validate_candidates(indices,videos,scores,raw):
    if indices.ndim!=2 or videos.shape!=indices.shape or scores.shape!=indices.shape:raise ValueError('Invalid candidate shapes')
    if not ((indices>0)&(indices<len(raw))).all():raise ValueError('Invalid candidate IDs')
    if not np.isfinite(scores).all() or not np.array_equal(videos,raw[indices]):raise ValueError('Invalid scores/raw IDs')
    if not (np.diff(np.sort(indices,axis=1),axis=1)>0).all():raise ValueError('Duplicate candidates')
    diff=np.diff(scores,axis=1)
    if (diff>0).any() or not (np.diff(indices,axis=1)[diff==0]>0).all():raise ValueError('Invalid score/tie ordering')


def main():
    _,output,prepared,vocabulary,counts,initial=setup()
    for c in initial.values():c.close()
    selection=json.loads((output/'selection.json').read_text());cfg=selection['config']
    replication=json.loads((output/'replication.json').read_text());exports=json.loads((output/'exports.json').read_text())
    verify_evidence(ROOT,selection['evidence'],cfg);verify_evidence(ROOT,replication['evidence'],cfg)
    root=output/'artifacts/snapshot';raw=inverse_ids(vocabulary['item_raw_to_index'])
    for name,expected in exports['files'].items():
        if digest_file(root/name)!=expected:raise ValueError('Snapshot hash mismatch')
    event_path=output/'artifacts/candidates/standard_validation.parquet'
    if digest_file(event_path)!=exports['events']['sha256']:raise ValueError('Event export hash mismatch')
    with np.load(root/'user_vectors.npz') as z:users={k:z[k] for k in z.files}
    with np.load(root/'item_vectors.npz') as z:items={k:z[k] for k in z.files}
    model,_=load_model(replication['runs']['two_tower_v2_seed42']['checkpoint'],cfg['device'])
    scorer=Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),cfg['device'],model=model)
    # Vectors may include a constant/bias coordinate; do not assume full norm=1.
    cache=load_partition(prepared,'standard/validation',ROOT/cfg['cache_directory'],cfg['history_length'])
    totals={};max_error=0.;sampled=0;random=np.random.default_rng(20261028)
    for name,path in [('snapshot',root/'snapshot_candidates.parquet'),('validation_events',event_path)]:
        offset=0
        with pq.ParquetFile(path) as parquet:
            groups=set(random.choice(parquet.num_row_groups,min(5,parquet.num_row_groups),replace=False).tolist())
            for group in range(parquet.num_row_groups):
                table=parquet.read_row_group(group)
                indices=pc.list_flatten(table['candidate_item_indices']).to_numpy().reshape(len(table),-1)
                videos=pc.list_flatten(table['candidate_video_ids']).to_numpy().reshape(indices.shape)
                scores=pc.list_flatten(table['scores']).to_numpy().reshape(indices.shape)
                validate_candidates(indices,videos,scores,raw)
                if indices.shape[1]!=100:raise ValueError('Expected Top100')
                if name=='snapshot':
                    np.testing.assert_array_equal(table['user_index'].to_numpy(),users['user_index'][offset:offset+len(table)])
                    np.testing.assert_array_equal(table['user_id'].to_numpy(),users['user_id'][offset:offset+len(table)])
                    dot=np.einsum('bd,bkd->bk',users['vectors'][offset:offset+len(table)],items['vectors'][indices-1])/cfg['temperature']
                    max_error=max(max_error,float(np.abs(dot-scores).max()))
                    np.testing.assert_allclose(scores,dot,rtol=1e-5,atol=1e-5)
                    if group in groups:
                        uid=torch.as_tensor(users['user_index'][offset:offset+len(table)],device=cfg['device'])
                        tab=torch.as_tensor(users['tabs'][offset:offset+len(table)],device=cfg['device'])
                        hist=torch.as_tensor(users['history'][offset:offset+len(table)],device=cfg['device'],dtype=torch.long)
                        with torch.no_grad():vec=model.encode_user(uid,tab,hist).cpu().numpy()
                        np.testing.assert_allclose(vec,users['vectors'][offset:offset+len(table)],rtol=1e-5,atol=1e-5)
                else:
                    rows=np.arange(offset,offset+len(table))
                    np.testing.assert_array_equal(table['event_id'].to_numpy(),cache['event_id'][rows])
                    if group in groups:
                        expected=scorer.scores(cache,rows);top,values=stable_topk(expected,scorer.candidate_ids,100)
                        np.testing.assert_array_equal(indices,top.cpu().numpy())
                        np.testing.assert_allclose(scores,values.cpu().numpy(),rtol=1e-5,atol=1e-5);sampled+=len(rows)
                offset+=len(table)
        totals[name]=offset
    if totals!={'snapshot':exports['users'],'validation_events':exports['events']['rows']}:raise ValueError('Row count mismatch')
    if not exports['target_label_independence']:raise ValueError('Target isolation failed')
    result={'status':'PASS','candidate_rows_checked':totals,'all_snapshot_topk_dot_max_error':max_error,
            'event_queries_independently_rescored':sampled,'target_label_independence_queries':exports['target_label_independence_queries'],
            'verifier_sha256':digest_file(__file__),'files':exports['files'],'event_file_sha256':exports['events']['sha256']}
    write_json(ROOT/'reports/retrieval_optimization_artifact_validation.json',result)
    cache.close();print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
