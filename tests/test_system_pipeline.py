import importlib.util
import inspect
import unittest
import numpy as np
import torch
from torch import nn


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.system_evaluation.pipeline'),'Frozen pipeline missing')

    def test_top20_stable_ties_and_invalid_candidates_are_rejected(self):
        from src.system_evaluation.pipeline import order_candidates
        ids=np.array([[3,1,2]]);scores=np.array([[.5,.5,.7]],dtype=np.float32)
        got,value=order_candidates(ids,scores,2)
        self.assertEqual(got.tolist(),[[2,1]]);np.testing.assert_array_equal(value,np.array([[.7,.5]],dtype=np.float32))
        for bad_ids,bad_scores in [(np.array([[1,1,2]]),scores),(ids,np.full_like(scores,np.nan))]:
            with self.assertRaises(ValueError):order_candidates(bad_ids,bad_scores,2)

    def test_context_only_interface_returns_twenty_unique_valid_items(self):
        from src.system_evaluation.pipeline import FrozenPipeline
        class Teacher(nn.Module):
            def __init__(self):
                super().__init__();self.user_embedding=nn.Embedding(4,4);self.item_embedding=nn.Embedding(111,4);self.temperature=1.
            def encode_user(self,u,t,h):return self.user_embedding(u)
            def encode_item(self,i):return self.item_embedding(i)
        class Model(nn.Module):
            def __init__(self):super().__init__();self.teacher=Teacher()
            def forward(self,users,items,tabs,history,base_scores):return {'rank':base_scores+.01*items.float()}
        torch.manual_seed(1);pipe=FrozenPipeline(Model(),retrieval_batch=512,rerank_batch=256)
        users=np.array([0,1]);tabs=np.array([1,1]);history=np.zeros((2,50),dtype=np.int64)
        got=pipe.recommend(users,tabs,history)
        self.assertEqual(got['top20_ids'].shape,(2,20));self.assertEqual(got['candidate_ids'].shape,(2,100))
        for row in got['top20_ids']:self.assertEqual(len(np.unique(row)),20);self.assertTrue(((row>0)&(row<=110)).all())
        self.assertNotIn('targets',inspect.signature(pipe.recommend).parameters)
        with self.assertRaises(TypeError):pipe.recommend(users,tabs,history,labels=np.ones(2))
        with self.assertRaises(ValueError):pipe.recommend(users,tabs,np.full((2,50),999))


if __name__=='__main__':unittest.main()
