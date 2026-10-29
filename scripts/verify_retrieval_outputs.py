"""Verify real vector/candidate outputs; requires completed retrieval experiment."""
import json
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.run import load_setup,write_json
from src.retrieval.provenance import verify_frozen_inputs
from src.retrieval.train import load_model
from src.retrieval.evaluate import Scorer
from src.retrieval.baselines import popularity_scores
from src.retrieval.candidates import inverse_ids
from src.retrieval.metrics import stable_topk


def main():
    config,paths,vocabulary,counts,caches=load_setup(['standard/validation'])
    selection=json.loads((paths['output']/'selection.json').read_text());verify_frozen_inputs(selection,paths['prepared'])
    exports=json.loads((paths['output']/'exports.json').read_text());events=json.loads((paths['output']/'candidates_standard_validation.json').read_text())
    root=paths['artifacts']/'snapshot';raw=inverse_ids(vocabulary['item_raw_to_index'])
    for filename,expected in exports['files'].items():assert digest_file(root/filename)==expected
    assert digest_file(events['path'])==events['sha256']
    with np.load(root/'user_vectors.npz') as data:users={key:data[key] for key in data.files}
    with np.load(root/'item_vectors.npz') as data:items={key:data[key] for key in data.files}
    np.testing.assert_allclose(np.linalg.norm(users['vectors'],axis=1),1.,atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(items['vectors'],axis=1),1.,atol=1e-5)
    totals={};max_dot_error=0.
    for name,path in [('snapshot',root/'snapshot_candidates.parquet'),('validation_events',Path(events['path']))]:
        total=0
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=512):
                def col(key):return batch.column(batch.schema.get_field_index(key))
                lengths=pc.list_value_length(col('candidate_item_indices')).to_numpy()
                assert (lengths==config['candidate_topk']).all()
                indices=pc.list_flatten(col('candidate_item_indices')).to_numpy().reshape(len(batch),-1)
                videos=pc.list_flatten(col('candidate_video_ids')).to_numpy().reshape(indices.shape)
                scores=pc.list_flatten(col('scores')).to_numpy().reshape(indices.shape)
                assert ((indices>0)&(indices<len(raw))).all() and np.isfinite(scores).all()
                assert (np.diff(np.sort(indices,axis=1),axis=1)>0).all()
                assert np.array_equal(videos,raw[indices])
                differences=np.diff(scores,axis=1)
                assert (differences<=0).all()
                assert (np.diff(indices,axis=1)[differences==0]>0).all()
                if name=='snapshot':
                    np.testing.assert_array_equal(col('user_index').to_numpy(),users['user_index'][total:total+len(batch)])
                    np.testing.assert_array_equal(col('user_id').to_numpy(),users['user_id'][total:total+len(batch)])
                    dot=np.einsum('bd,bkd->bk',users['vectors'][total:total+len(batch)],items['vectors'][indices-1])/config['temperature']
                    max_dot_error=max(max_dot_error,float(np.max(np.abs(scores-dot))))
                    np.testing.assert_allclose(scores,dot,rtol=1e-5,atol=1e-5)
                total+=len(batch)
        totals[name]=total
    assert totals['snapshot']==exports['users'] and totals['validation_events']==events['rows']
    info=selection['training_runs'][f'two_tower_seed{selection["primary_seed"]}']
    assert digest_file(info['checkpoint'])==info['checkpoint_sha256']
    model,_=load_model(info['checkpoint'],config['device'])
    scorer=Scorer('two_tower',popularity_scores(counts),config['device'],model=model)
    cache=caches['standard/validation'];sampled=0
    class ChangedTarget:
        def __getitem__(self,key):
            if key in ['item_index','label']:return np.zeros_like(cache[key])
            return cache[key]
    with pq.ParquetFile(events['path']) as parquet:
        sizes=np.array([parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)])
        offsets=np.r_[0,np.cumsum(sizes)]
        groups=np.random.default_rng(42).choice(parquet.num_row_groups,size=min(5,parquet.num_row_groups),replace=False)
        for group in groups:
            indices=np.arange(offsets[group],offsets[group+1]);table=parquet.read_row_group(group)
            expected=scorer.scores(cache,indices)
            changed=scorer.scores(ChangedTarget(),indices)
            torch.testing.assert_close(expected,changed,rtol=0,atol=0)
            top,scores=stable_topk(expected,scorer.candidate_ids,config['candidate_topk'])
            stored_ids=np.array(table['candidate_item_indices'].to_pylist())
            stored_scores=np.array(table['scores'].to_pylist())
            np.testing.assert_array_equal(stored_ids,top.cpu().numpy())
            np.testing.assert_allclose(stored_scores,scores.cpu().numpy(),rtol=1e-5,atol=1e-5)
            np.testing.assert_array_equal(table['event_id'].to_numpy(),cache['event_id'][indices]);sampled+=len(indices)
    result={'status':'PASS','candidate_rows_checked':totals,'snapshot_all_topk_dot_max_error':max_dot_error,
            'event_queries_independently_rescored':sampled,'logged_target_and_label_mutation_does_not_change_scores':True,
            'checks':['file hashes','ID/raw-ID mapping','unique valid nonpadding candidates','score/tie ordering',
                      'all snapshot candidate dot products','sampled event candidate exact recomputation'],
            'verifier_sha256':digest_file(__file__)}
    write_json(ROOT/'reports/retrieval_artifact_validation.json',result)
    cache.close();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
