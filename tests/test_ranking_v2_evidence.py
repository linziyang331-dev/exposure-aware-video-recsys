import unittest
import numpy as np
import torch
from tests.test_ranking_v2 import config


class EvidenceTests(unittest.TestCase):
    def test_candidate_scores_ignore_current_item_and_label_and_match_training_evaluator(self):
        from src.ranking_v2.model import RichDIN
        from src.ranking_v2.inference import candidate_scores,deterministic
        from src.ranking_v2.evaluate import assess
        from src.retrieval_v2.model import TwoTowerV2
        class Cache(dict):
            def __len__(self):return len(self['label'])
        class Candidates(dict):
            def __len__(self):return len(self['indices'])
        deterministic();torch.manual_seed(21)
        cache=Cache(user_index=np.array([1,2]),item_index=np.array([1,3]),tab=np.ones(2,dtype=np.int64),
                    history=np.array([[2,0,0,0],[1,0,0,0]]),label=np.array([1,0]),event_id=np.array([10,11]))
        cand=Candidates(indices=np.array([0]),ids=np.array([[1,2,3,4,5,6]]),scores=np.array([[6.,5.,4.,3.,2.,1.]],dtype=np.float32),
                        v2_ranks=np.array([1]),user_ids=np.array([1]),event_ids=np.array([10]))
        cfg=config(residual=True,teacher_features=True,pretrained=True,eval_batch_size=2,candidate_batch_size=1)
        model=RichDIN(3,6,cfg,TwoTowerV2(3,6,embedding_dim=8)).eval()
        with torch.no_grad():model.rank_delta.weight.fill_(.1)
        scores=candidate_scores(model,cache,cand,1)
        self.assertGreater(float(np.abs(scores-cand['scores']).max()),0.)
        changed=Cache(cache|{'item_index':np.zeros(2,dtype=np.int64),'label':np.zeros(2)})
        np.testing.assert_array_equal(scores,candidate_scores(model,changed,cand,1))
        from src.ranking.evaluate import rank_scores
        self.assertEqual(rank_scores(cand,cache,scores),assess(model,cache,cand,cfg)['ranking'])

    def test_reproduction_protects_frozen_directories_and_requires_D_project(self):
        from scripts.train_ranking_v2 import reproduction_config
        for path in ['experiments/ranking','experiments/ranking/sub','experiments/ranking_v2',
                     'experiments/ranking_v2/sub','experiments/retrieval_v2/x','experiments','../outside','C:/tmp/x']:
            with self.subTest(path=path),self.assertRaises(ValueError):reproduction_config({},path)
        cfg,path=reproduction_config({},'experiments/ranking_v2_reproduction/test')
        self.assertEqual(path.drive.upper(),'D:')
        self.assertEqual(cfg['output_directory'],'experiments/ranking_v2_reproduction/test')


if __name__=='__main__':unittest.main()
