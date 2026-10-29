"""Verify full fixed-pool artifacts against actual checkpoint predictions."""
import json
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.data import write_json
from src.ranking.evaluate import rank_scores
from src.ranking.metrics import exposure_metrics
from src.ranking.verify import verify_rank_file
from src.retrieval_v2.provenance import verify_evidence
from src.ranking_v2.run import setup
from src.ranking_v2.train import load_model
from src.ranking_v2.inference import deterministic,candidate_scores,exposure_logits,history_rule_scores


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    deterministic();out=ROOT/'experiments/ranking_v2';sel=read(out/'selection.json');cfg=sel['config'];rep=read(out/'replication.json');ev=read(out/'evaluation.json')
    verify_evidence(ROOT,sel['evidence'],cfg);verify_evidence(ROOT,rep['evidence'],cfg)
    if ev['status']!='EVALUATED' or ev['selection_sha256']!=digest_file(out/'selection.json') or ev['replication_sha256']!=digest_file(out/'replication.json'):
        raise ValueError('Evaluation freeze differs')
    _,_,_,counts,caches,candidates,_=setup(list(ev['baseline']))
    model,_=load_model(rep['runs']['din_seed42']['checkpoint'],cfg['device']);checks={};hashes={}
    def record(path):hashes[Path(path).relative_to(ROOT).as_posix()]=digest_file(path)
    for p in ['selection.json','replication.json','evaluation.json']:record(out/p)
    for part,cache in caches.items():
        cand=candidates[part];idx=np.flatnonzero(cache['label']==1)
        np.testing.assert_array_equal(cand['indices'],idx)
        np.testing.assert_array_equal(cand['event_ids'],cache['event_id'][idx])
        np.testing.assert_array_equal(cand['user_ids'],cache['user_id'][idx])
        for start in range(0,len(cand),8192):
            ids=cand['ids'][start:start+8192];scores=cand['scores'][start:start+8192]
            if not ((ids>0)&(ids<len(counts))).all() or not np.isfinite(scores).all():raise ValueError('Invalid candidate IDs/scores')
            if not (np.diff(np.sort(ids,axis=1),axis=1)>0).all():raise ValueError('Duplicate candidates')
            diff=np.diff(scores,axis=1)
            if (diff>0).any() or not (np.diff(ids,axis=1)[diff==0]>0).all():raise ValueError('Original candidate ordering changed')
        base=ev['baseline'][part]
        verify_rank_file(base['artifact'],cand,base,cand['v2_ranks']);record(base['artifact'])
        rule=ev['history_rule'][part];verify_rank_file(rule['artifact'],cand,rule);record(rule['artifact'])
        rebuilt_rule=rank_scores(cand,cache,history_rule_scores(cache,cand,sel['history_rule_penalty']))
        if rebuilt_rule!={k:v for k,v in rule.items() if k!='artifact'}:raise ValueError('History rule differs from frozen penalty')
        for name,parts in ev['results'].items():
            row=parts[part];verify_rank_file(row['rank_artifact'],cand,row['ranking']);record(row['rank_artifact'])
            if row['ranking']['recall@100']!=base['recall@100']:raise ValueError('Reranking changed Recall100')
            with np.load(row['prediction_artifact']) as z:
                np.testing.assert_array_equal(z['event_ids'],cache['event_id']);np.testing.assert_array_equal(z['labels'],cache['label'])
                if exposure_metrics(z['labels'],z['logits'])!=row['pointwise']:raise ValueError('Exposure metrics differ')
            record(row['prediction_artifact'])
        primary=ev['results']['din_seed42'][part];saved=np.load(primary['scores_artifact'],mmap_mode='r')
        actual=candidate_scores(model,cache,cand,cfg['candidate_batch_size'])
        np.testing.assert_array_equal(saved,actual)
        rescored=out/'artifacts/verification'/(part.replace('/','_')+'.npz')
        if rank_scores(cand,cache,actual,rescored)!=primary['ranking']:raise ValueError('Actual checkpoint ranks differ')
        with np.load(rescored) as a,np.load(primary['rank_artifact']) as b:
            for field in ['ranks','users','event_ids']:np.testing.assert_array_equal(a[field],b[field])
        with np.load(primary['prediction_artifact']) as z:
            np.testing.assert_array_equal(z['logits'],exposure_logits(model,cache,cfg['eval_batch_size']))
        record(primary['scores_artifact']);export=ev['exports'][part];path=Path(export['path'])
        if digest_file(path)!=export['sha256']:raise ValueError('Export hash changed')
        offset=0
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=512):
                count=len(batch);ids=np.array(cand['ids'][offset:offset+count]);values=actual[offset:offset+count]
                order=np.lexsort((ids,-values),axis=1)
                def column(name):return batch.column(batch.schema.get_field_index(name))
                expected_ids=np.take_along_axis(ids,order,axis=1);expected_scores=np.take_along_axis(values,order,axis=1)
                np.testing.assert_array_equal(pc.list_flatten(column('candidate_item_indices')).to_numpy().reshape(count,100),expected_ids)
                np.testing.assert_array_equal(pc.list_flatten(column('ranking_scores')).to_numpy().reshape(count,100),expected_scores)
                np.testing.assert_array_equal(column('event_id').to_numpy(),cand['event_ids'][offset:offset+count])
                np.testing.assert_array_equal(column('user_id').to_numpy(),cand['user_ids'][offset:offset+count])
                np.testing.assert_array_equal(column('time_ms').to_numpy(),cache['time_ms'][cand['indices'][offset:offset+count]])
                offset+=count
        if offset!=len(cand):raise ValueError('Export row count differs')
        record(path)
        class Subset(dict):
            def __len__(self):return len(self['indices'])
        n=min(cfg['candidate_batch_size'],len(cand))
        subset=Subset({k:np.array(cand[k][:n]) for k in ['ids','scores','indices']})
        class Changed:
            def __getitem__(self,key):
                if key in {'item_index','label'}:return np.zeros(len(cache),dtype=cache[key].dtype)
                return cache[key]
        np.testing.assert_array_equal(candidate_scores(model,Changed(),subset,cfg['candidate_batch_size']),actual[:n])
        checks[part]={'all_candidate_rows':len(cand),'all_primary_predictions_rescored':True,
                      'all_exports_equal_checkpoint_scores':True,'target_label_mutation_invariant_rows':n,'recall100_unchanged':True}
        print('VERIFIED '+part+' rows='+str(len(cand)),flush=True)
    for c in candidates.values():c.close()
    for c in caches.values():c.close()
    result={'status':'PASS','checks':checks,'artifact_hashes':hashes,'verifier_sha256':digest_file(__file__)}
    write_json(ROOT/'reports/ranking_v2_artifact_validation.json',result)
    return result


if __name__=='__main__':main()
