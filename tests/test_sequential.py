import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch


def config(**changes):
    return {'history_length':4,'hidden_size':8,'n_layers':1,'n_heads':2,'inner_size':16,
            'dropout':0.,'position_embeddings':True,'device':'cpu','learning_rate':.01,
            'weight_decay':0.,'batch_size':8,'eval_batch_size':8,'epochs':2,'patience':2}|changes


COUNTS=np.array([0,10,5,3,1,0,2])


class SequenceModelTests(unittest.TestCase):
    def test_latest_history_preserves_order_and_right_padding(self):
        from src.sequential.model import prepare_history
        h=torch.tensor([[1,2,3,4,5,0],[2,3,0,0,0,0],[0,0,0,0,0,0]])
        seq,length=prepare_history(h,3)
        self.assertEqual(seq.tolist(),[[3,4,5],[2,3,0],[0,0,0]])
        self.assertEqual(length.tolist(),[3,2,0])

    def test_native_model_causal_mask_prevents_future_token_influence(self):
        from src.sequential.model import SequentialModel
        from recbole.model.sequential_recommender.sasrec import SASRec
        torch.manual_seed(11);m=SequentialModel(config(),COUNTS).eval();self.assertIsInstance(m.core,SASRec)
        a=torch.tensor([[1,2,3,0]]);b=torch.tensor([[1,2,4,5]]);length=torch.tensor([2])
        torch.testing.assert_close(m.core(a,length),m.core(b,length),rtol=0,atol=0)

    def test_empty_history_fallback_is_training_popularity_and_padding_invariant(self):
        from src.sequential.model import SequentialModel
        m=SequentialModel(config(),COUNTS).eval()
        h=torch.tensor([[1,2,0,0],[0,0,0,0]])
        a=m.scores(h);b=m.scores(torch.cat([h,torch.zeros(2,3,dtype=torch.long)],1))
        self.assertTrue(torch.isfinite(a).all());self.assertEqual(a.shape,(2,6))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(a[1],torch.log1p(torch.tensor(COUNTS[1:],dtype=torch.float32)),rtol=0,atol=0)

    def test_padding_class_excluded_and_embedding_stays_zero(self):
        from src.sequential.model import SequentialModel,KnownItemCrossEntropy
        ce=KnownItemCrossEntropy();logits=torch.randn(2,7);targets=torch.tensor([1,6])
        expected=torch.nn.functional.cross_entropy(logits[:,1:],targets-1)
        torch.testing.assert_close(ce(logits,targets),expected)
        logits[:,0]=1e9;torch.testing.assert_close(ce(logits,targets),expected)
        m=SequentialModel(config(),COUNTS);loss=m.loss(torch.tensor([[1,2,0,0],[3,2,1,0]]),torch.tensor([3,4]))
        loss.backward();self.assertGreater(float(m.core.item_embedding.weight.grad[1:].abs().sum()),0.)
        torch.testing.assert_close(m.core.item_embedding.weight.grad[0],torch.zeros(8),rtol=0,atol=0)
        torch.optim.Adam(m.parameters(),lr=.01,weight_decay=.01).step()
        torch.testing.assert_close(m.core.item_embedding.weight[0],torch.zeros(8),rtol=0,atol=0)
        with self.assertRaises(ValueError):m.loss(torch.zeros(1,4,dtype=torch.long),torch.tensor([1]))
        with self.assertRaises(ValueError):m.loss(torch.tensor([[1,0,0,0]]),torch.tensor([0]))

    def test_position_ablation_is_fixed_zero_but_native_causality_remains(self):
        from src.sequential.model import SequentialModel
        m=SequentialModel(config(position_embeddings=False),COUNTS)
        self.assertFalse(m.core.position_embedding.weight.requires_grad)
        self.assertEqual(float(m.core.position_embedding.weight.abs().sum()),0.)
        mask=m.core.get_attention_mask(torch.tensor([[1,2,3,0]]))
        self.assertLess(float(mask[0,0,0,1]),-1000.)


class SequentialIntegrationTests(unittest.TestCase):
    def test_real_training_exact_reload_and_mutated_config_rejected(self):
        from src.sequential.train import train_run,load_model
        class Cache(dict):
            def __len__(self):return len(self['label'])
        cache=Cache(user_index=np.array([1,2]*16),item_index=np.array([3,4]*16),
                    user_id=np.array([101,102]*16),event_id=np.arange(32),label=np.ones(32,dtype=np.int8),
                    history=np.array([[1,2,0,0],[3,2,1,0]]*16),history_length=np.array([2,3]*16))
        cache.metadata={'signature':{'fixture':True}}
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with tempfile.TemporaryDirectory() as d:
                cfg=config(device=device);r=train_run(cache,cache,COUNTS,cfg,Path(d),42)
                self.assertTrue(r['parameters_changed']);self.assertTrue(r['reload_identical'])
                self.assertGreater(r['optimizer_updates'],0)
                model,_=load_model(r['checkpoint'],device)
                self.assertTrue(torch.isfinite(model.scores(torch.tensor([[1,2,0,0]],device=device))).all())
                with self.assertRaises(ValueError):train_run(cache,cache,COUNTS,cfg|{'learning_rate':.02},Path(d),42)

    def test_scoring_ignores_current_target_and_label(self):
        from src.sequential.model import SequentialModel
        from src.sequential.evaluate import SequenceScorer
        m=SequentialModel(config(),COUNTS).eval();s=SequenceScorer(m,COUNTS)
        h=np.array([[1,2,0,0],[0,0,0,0]])
        cache={'history':h,'item_index':np.array([1,2]),'label':np.array([1,0])}
        a=s.scores(cache,np.arange(2));cache['item_index'][:]=0;cache['label'][:]=0
        torch.testing.assert_close(a,s.scores(cache,np.arange(2)),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
