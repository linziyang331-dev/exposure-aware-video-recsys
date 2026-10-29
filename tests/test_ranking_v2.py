import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch


def config(**changes):
    return {'embedding_dim':8,'hidden_sizes':[16,8],'history_length':4,'dropout':0.,'device':'cpu',
            'residual':False,'teacher_features':False,'pretrained':False,
            'user_features':True,'attention':True}|changes


class RichDINTests(unittest.TestCase):
    def test_user_embedding_gets_gradient_and_changes_candidate_order_preferences(self):
        from src.ranking_v2.model import RichDIN
        torch.manual_seed(17);m=RichDIN(3,6,config()).eval()
        users=torch.tensor([1,1,2,2]);items=torch.tensor([2,3,2,3]);tab=torch.ones(4,dtype=torch.long)
        history=torch.tensor([[1,4,0,0]]*4)
        out=m(users,items,tab,history)
        self.assertGreater(abs(float((out['rank'][0]-out['rank'][1])-(out['rank'][2]-out['rank'][3]))),1e-7)
        torch.nn.functional.binary_cross_entropy_with_logits(out['ctr'],torch.tensor([1.,0.,0.,1.])).backward()
        grad=m.core.embedding_layer.token_embedding_table['user'].embedding.weight.grad
        self.assertIsNotNone(grad);self.assertGreater(float(grad.abs().sum()),0.)

    def test_zero_residual_uses_canonical_scores_exactly_and_teacher_stays_frozen(self):
        from src.ranking_v2.model import RichDIN
        from src.retrieval_v2.model import TwoTowerV2
        torch.manual_seed(5);teacher=TwoTowerV2(3,6,embedding_dim=8)
        initial={k:v.detach().clone() for k,v in teacher.state_dict().items()}
        m=RichDIN(3,6,config(residual=True,teacher_features=True,pretrained=True),teacher).eval()
        users=torch.tensor([1,2]);items=torch.tensor([2,3]);tab=torch.ones(2,dtype=torch.long);h=torch.tensor([[1,0,0,0],[2,0,0,0]])
        canonical=torch.tensor([2.125,3.75])
        out=m(users,items,tab,h,base_scores=canonical)
        torch.testing.assert_close(out['rank'],canonical,rtol=0,atol=0)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(out['ctr'],torch.tensor([1.,0.]))+out['rank'].sum()
        loss.backward();optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01);optimizer.step()
        for name,value in teacher.state_dict().items():torch.testing.assert_close(value,initial[name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))

    def test_empty_history_is_finite_padding_invariant_and_attention_is_upstream(self):
        from src.ranking_v2.model import RichDIN
        from recbole.model.layers import SequenceAttLayer
        m=RichDIN(3,6,config()).eval();self.assertIsInstance(m.core.attention,SequenceAttLayer)
        u=torch.tensor([1,0]);i=torch.tensor([2,0]);t=torch.tensor([1,1]);h=torch.tensor([[1,2],[0,0]])
        a=m(u,i,t,h)['rank'];b=m(u,i,t,torch.cat([h,torch.zeros(2,4,dtype=torch.long)],1))['rank']
        self.assertTrue(torch.isfinite(a).all());torch.testing.assert_close(a,b,rtol=0,atol=0)


class ChoiceDataTests(unittest.TestCase):
    def test_unknown_competitors_unique_past_positives_excluded_target_preserved(self):
        from src.ranking_v2.data import choice_batch
        cache={'item_index':np.array([1,2]),'label':np.ones(2,dtype=np.int8),
               'history':np.array([[2,0],[2,3]])}
        candidates={'ids':np.array([[1,2,3,4],[1,2,3,4]]),'scores':np.array([[4.,3.,2.,1.]]*2),
                    'indices':np.array([0,1]),'v2_ranks':np.array([1,2])}
        result=choice_batch(cache,candidates,np.array([0,1]),np.random.default_rng(42),2)
        self.assertEqual(result['items'][:,0].tolist(),[1,2])
        self.assertEqual(np.sort(result['items'][:,1:],axis=1).tolist(),[[3,4],[1,4]])
        self.assertEqual(result['base_scores'][:,0].tolist(),[4.,3.])
        all_alternatives=choice_batch(cache,candidates,np.array([0]),np.random.default_rng(42),3,mask_past=False)
        self.assertEqual(np.sort(all_alternatives['items'][0,1:]).tolist(),[2,3,4])
        candidates['v2_ranks'][1]=0
        with self.assertRaises(ValueError):choice_batch(cache,candidates,np.array([1]),np.random.default_rng(42),2)


class TrainingTests(unittest.TestCase):
    def test_real_choice_training_checkpoint_and_frozen_teacher(self):
        from src.ranking_v2.train import train_run,load_model
        from src.retrieval_v2.model import TwoTowerV2
        class Cache(dict):
            def __len__(self):return len(self['label'])
        class Candidates(dict):
            def __len__(self):return len(self['indices'])
        t=Cache(user_index=np.array([1,1,2,2]*8),item_index=np.array([1,3,2,4]*8),
                tab=np.ones(32,dtype=np.int64),history=np.array([[2,0,0,0],[2,0,0,0],[1,0,0,0],[1,0,0,0]]*8),
                label=np.array([1,0,1,0]*8),user_id=np.array([1,1,2,2]*8),event_id=np.arange(32))
        t.metadata={'signature':{'fixture':True}}
        idx=np.flatnonzero(t['label']==1)
        c=Candidates(indices=idx,ids=np.tile(np.arange(1,7),(len(idx),1)),scores=np.tile(np.arange(6,0,-1,dtype=np.float32),(len(idx),1)),
                     user_ids=t['user_id'][idx],event_ids=t['event_id'][idx],v2_ranks=t['item_index'][idx])
        c.metadata={'signature':{'fixture':True}}
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            cfg=config(residual=True,teacher_features=True,pretrained=True,device=device,
                       objective='choice',learning_rate=.01,weight_decay=0.,epochs=2,patience=2,
                       query_batch_size=4,competitors=2,aux_batch_size=4,aux_weight=.1,delta_reg=.01,
                       queries_per_epoch=16,mask_past=False,eval_batch_size=8,candidate_batch_size=4,
                       select_metric='ndcg@20',batch_size=8)
            teacher=TwoTowerV2(3,6,embedding_dim=8).to(device)
            with tempfile.TemporaryDirectory() as d:
                result=train_run(t,t,c,c,3,6,cfg,Path(d),42,teacher=teacher)
                self.assertGreater(result['optimizer_updates'],0);self.assertTrue(result['teacher_unchanged'])
                self.assertTrue(result['reload_identical']);model,_=load_model(Path(d)/'best.pt',device)
                self.assertIsNotNone(model.teacher)
                with self.assertRaises(ValueError):train_run(t,t,c,c,3,6,cfg|{'delta_reg':1.},Path(d),42,teacher=teacher)


if __name__=='__main__':unittest.main()
