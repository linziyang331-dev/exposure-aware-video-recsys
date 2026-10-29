import unittest
import numpy as np
import torch


class SequenceEvidenceTests(unittest.TestCase):
    def test_catalog_verifier_rejects_modified_subgroups_macro_and_query_counts(self):
        import copy,tempfile
        from pathlib import Path
        from src.sequential.model import SequentialModel
        from src.sequential.evaluate import catalog_evaluate
        from src.sequential.verify import check_catalog_rank
        from tests.test_sequential import config,COUNTS
        class Cache(dict):
            def __len__(self):return len(self['label'])
        c=Cache(history=np.array([[1,2,0,0],[0,0,0,0]]),item_index=np.array([3,0]),label=np.ones(2),
                user_index=np.array([1,0]),user_id=np.array([101,102]),event_id=np.array([1,2]))
        model=SequentialModel(config(),COUNTS).eval()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rank.npz';metrics=catalog_evaluate(model,c,COUNTS,2,path)
            check_catalog_rank(path,c,metrics,COUNTS)
            for key in ['subgroup','macro','queries','mrr100']:
                broken=copy.deepcopy(metrics)
                if key=='subgroup':broken['groups']['cold_target']['ndcg@20']=1.
                elif key=='macro':broken['user_macro']['ndcg@100']+=1.
                elif key=='queries':broken['queries']+=1
                else:broken['mrr@100']+=1.
                with self.subTest(key=key),self.assertRaises(ValueError):check_catalog_rank(path,c,broken,COUNTS)

    def test_pool_verifier_rejects_modified_macro_and_ndcg100(self):
        import copy,tempfile
        from pathlib import Path
        from src.ranking.evaluate import rank_scores
        from src.sequential.verify import check_pool_rank
        class C(dict):
            def __len__(self):return len(self['indices'])
        cand=C(indices=np.arange(2),ids=np.array([[1,2,3],[1,2,3]]),v2_ranks=np.array([1,0]),
               user_ids=np.array([10,11]),event_ids=np.array([1,2]))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ranks.npz';metrics=rank_scores(cand,{'item_index':np.array([1,4])},np.array([[3.,2.,1.],[3.,2.,1.]]),path)
            check_pool_rank(path,cand,metrics)
            for key in ['ndcg@100','macro']:
                broken=copy.deepcopy(metrics)
                if key=='macro':broken['user_macro']['ndcg@100']+=1.
                else:broken[key]+=1.
                with self.subTest(key=key),self.assertRaises(ValueError):check_pool_rank(path,cand,broken)

    def test_optional_artifact_does_not_change_evaluation_metrics(self):
        import tempfile
        from pathlib import Path
        from src.sequential.model import SequentialModel
        from src.sequential.evaluate import catalog_evaluate,metric_identity
        from tests.test_sequential import config,COUNTS
        class Cache(dict):
            def __len__(self):return len(self['label'])
        c=Cache(history=np.array([[1,2,0,0],[0,0,0,0]]),item_index=np.array([3,0]),label=np.ones(2),
                user_index=np.array([1,0]),user_id=np.array([101,102]),event_id=np.array([1,2]))
        m=SequentialModel(config(),COUNTS).eval()
        with tempfile.TemporaryDirectory() as d:
            saved=catalog_evaluate(m,c,COUNTS,2,Path(d)/'ranks.npz')
            transient=catalog_evaluate(m,c,COUNTS,2)
        self.assertEqual(metric_identity(saved),metric_identity(transient))

    def test_trial_output_is_restricted_to_the_sequential_experiment(self):
        from src.sequential.run import output_path
        from src.data.common import ROOT
        for p in ['data/processed','experiments/ranking_v2','experiments/retrieval_v2','experiments/sequential/x','../outside','C:/tmp/x']:
            with self.subTest(path=p),self.assertRaises(ValueError):output_path(p)
        self.assertEqual(output_path('experiments/sequential'),(ROOT/'experiments/sequential').resolve())

    def test_fixed_pool_matches_catalog_scores_and_keeps_retrieval_misses(self):
        from src.sequential.model import SequentialModel
        from src.sequential.evaluate import pool_scores
        from src.ranking.evaluate import rank_scores
        from tests.test_sequential import config,COUNTS
        class Cache(dict):
            def __len__(self):return len(self['label'])
        class Candidates(dict):
            def __len__(self):return len(self['indices'])
        cache=Cache(history=np.array([[1,2,0,0],[0,0,0,0]]),item_index=np.array([3,6]),label=np.ones(2))
        candidates=Candidates(ids=np.array([[1,3,4],[1,2,3]]),indices=np.arange(2),v2_ranks=np.array([2,0]),
                              user_ids=np.array([101,102]),event_ids=np.array([10,11]))
        m=SequentialModel(config(),COUNTS).eval();scores=pool_scores(m,cache,candidates,2)
        with torch.no_grad():expected=m.scores(torch.as_tensor(cache['history'])).numpy()
        np.testing.assert_array_equal(scores,np.take_along_axis(expected,candidates['ids']-1,axis=1))
        metrics=rank_scores(candidates,cache,scores)
        self.assertEqual(metrics['recall@100'],.5);self.assertEqual(metrics['retrieval_miss_queries'],1)
        cache['label'][:]=0;cache['item_index'][:]=0
        np.testing.assert_array_equal(scores,pool_scores(m,cache,candidates,2))

    def test_reproduction_output_cannot_overwrite_existing_frozen_modules(self):
        from scripts.train_sequential import reproduction_config
        for path in ['experiments/sequential','experiments/sequential/sub','experiments/ranking_v2/sub',
                     'experiments/ranking/x','experiments/retrieval_v2/x','experiments','../outside','C:/tmp/x']:
            with self.subTest(path=path),self.assertRaises(ValueError):reproduction_config({},path)
        cfg,p=reproduction_config({},'experiments/sequential_reproduction/seed42')
        self.assertEqual(cfg['output_directory'],'experiments/sequential_reproduction/seed42')
        self.assertEqual(p.drive.upper(),'D:')


if __name__=='__main__':unittest.main()
