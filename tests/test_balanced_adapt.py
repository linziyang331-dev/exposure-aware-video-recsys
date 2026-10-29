import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from src.data.common import ROOT
from src.retrieval_v2.train import parameter_digest
from src.retrieval_balanced.model import SharedTower
from src.retrieval_balanced.adapt import TeacherDIN,load_model,candidate_cache,train_ranker
from test_balanced_train import SmallCache


class BalancedAdaptTests(unittest.TestCase):
    def teacher(self):
        torch.manual_seed(42);m=SharedTower(3,4,embedding_dim=32,graph_alpha=.5,neighbors=2)
        m.set_neighbors(np.array([[0,0],[2,0],[1,3],[2,0],[0,0]]),np.array([[0,0],[1,0],[.5,.5],[1,0],[0,0]],np.float32))
        return m

    def test_din_graph_teacher_reload_and_frozen_gradients(self):
        cfg=yaml.safe_load((ROOT/'configs/ranking_v2.yaml').read_text('utf-8'))|{'device':'cpu','dropout':0.}
        teacher=self.teacher();before=parameter_digest(teacher);m=TeacherDIN(3,4,cfg,teacher);m.eval()
        u=torch.tensor([1,2]);i=torch.tensor([2,3]);t=torch.tensor([0,1]);h=torch.tensor([[1,2,0],[2,1,0]])
        out=m(u,i,t,h);self.assertEqual(float(out['delta'].abs().max()),0.)
        out['ctr'].sum().backward();self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertEqual(parameter_digest(teacher),before)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d)/'model.pt'
            torch.save({'architecture':{'num_users':3,'num_items':4,'config':cfg,'teacher_spec':m.teacher_spec()},'state':m.state_dict()},p)
            restored,_=load_model(p,'cpu')
            torch.testing.assert_close(restored(u,i,t,h)['rank'],out['rank'],rtol=0,atol=0)
            self.assertIsInstance(restored.teacher,SharedTower)
            self.assertEqual(parameter_digest(restored.teacher),before)

    def test_candidates_ignore_target_values_without_injection(self):
        c=SmallCache();counts=np.array([0,4,4,4,0]);teacher=self.teacher();teacher.eval()
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            a=candidate_cache(c,counts,teacher,Path(d)/'a','fixed-checkpoint',topk=2)
            original=np.array(a['ids']);scores=np.array(a['scores']);missing=next(i for i in range(1,5) if i not in original[0])
            c.a['item_index'][0]=missing;c.metadata={'signature':{'fixture':'mutated target only'}}
            b=candidate_cache(c,counts,teacher,Path(d)/'b','fixed-checkpoint',topk=2)
            np.testing.assert_array_equal(original,b['ids']);np.testing.assert_array_equal(scores,b['scores'])
            self.assertEqual(int(b['v2_ranks'][0]),0)
            a.close();b.close()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_actual_cuda_din_training_reload_probe_device(self):
        c=SmallCache();counts=np.array([0,4,4,4,0]);teacher=self.teacher().to('cuda')
        cfg=yaml.safe_load((ROOT/'configs/ranking_v2.yaml').read_text('utf-8'))|{
            'device':'cuda','dropout':0.,'epochs':1,'queries_per_epoch':6,'query_batch_size':2,
            'competitors':2,'aux_batch_size':4,'eval_batch_size':4,'candidate_batch_size':2}
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d);candidates=candidate_cache(c,counts,teacher,p/'candidates','fixture-sha')
            try:
                result=train_ranker(c,c,candidates,candidates,teacher,cfg,p/'run',42)
                self.assertTrue(result['reload_identical']);self.assertTrue(result['full_validation_rank_replay_identical'])
            finally:candidates.close()


if __name__=='__main__':unittest.main()
