import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.retrieval_balanced.graph import build_neighbors
from src.retrieval_balanced.train import train_run,load_model


class SmallCache:
    def __init__(self):
        self.a={'user_index':np.tile([1,2,3],4),'item_index':np.tile([1,2,3],4),
            'user_id':np.tile([11,12,13],4),'label':np.ones(12,dtype=np.int8),'event_id':np.arange(12),
            'tab':np.zeros(12,dtype=np.int8),'history':np.tile([[1,2,0],[2,1,0],[3,1,0]],(4,1))}
        self.metadata={'signature':{'fixture':'balanced12'}}
    def __getitem__(self,k):return self.a[k]
    def __len__(self):return 12


class BalancedTrainTests(unittest.TestCase):
    def config(self,device='cpu'):
        return dict(device=device,embedding_dim=4,temperature=.1,history_decay=.9,graph_alpha=.5,neighbors=2,
            rho=.1,tau=.1,learning_rate=.001,weight_decay=1e-5,batch_size=4,eval_batch_size=4,epochs=2)
    def baseline(self):return {'recall@20':.1,'ndcg@20':.04,'groups':{'head_target':{'recall@20':.1},'tail_target':{'recall@20':.01,'recall@100':.05}}}
    def test_real_training_selection_reload_and_reuse_guards(self):
        c=SmallCache();counts=np.array([0,4,4,4,0]);ids,w=build_neighbors(c['user_index'],c['item_index'],3,4,2)
        gate={'retention':.98,'tail20_delta':.002,'tail100_delta':.01}
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d)/'run';cfg=self.config();r=train_run(c,c,counts,ids,w,cfg,self.baseline(),gate,p,42)
            self.assertTrue(r['qualified']);self.assertTrue(r['reload_identical'])
            self.assertNotEqual(r['initial_parameter_sha256'],r['trained_parameter_sha256'])
            m,_=load_model(r['checkpoint'],'cpu');self.assertEqual(m.graph_alpha,.5)
            self.assertEqual(train_run(c,c,counts,ids,w,cfg,self.baseline(),gate,p,42),r)
            with self.assertRaises(ValueError):train_run(c,c,counts,ids,w,cfg|{'rho':.25},self.baseline(),gate,p,42)
            (p/'best.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):train_run(c,c,counts,ids,w,cfg,self.baseline(),gate,p,42)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_graph_backward_deterministic_cuda_path(self):
        c=SmallCache();c.a['user_index']=np.array([1,1,2,2,3,3,1,2,3,1,2,3]);c.a['user_id']=c.a['user_index']+10
        counts=np.array([0,4,4,4,0]);ids,w=build_neighbors(c['user_index'],c['item_index'],3,4,2)
        self.assertGreater(int(np.count_nonzero(w)),0,'CUDA test must exercise real shared edges')
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            r=train_run(c,c,counts,ids,w,self.config('cuda')|{'epochs':1},self.baseline(),
                {'retention':.98,'tail20_delta':.002,'tail100_delta':.01},Path(d)/'run',42)
            self.assertTrue(r['reload_identical'])


if __name__=='__main__':unittest.main()
