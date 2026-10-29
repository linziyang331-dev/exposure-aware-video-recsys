import unittest
import numpy as np
import torch
from src.retrieval_v2.model import TwoTowerV2
from src.retrieval_balanced.graph import build_neighbors
from src.retrieval_balanced.model import SharedTower
from src.retrieval_balanced.objectives import mixture_weights,choice_loss
from src.retrieval_balanced.selection import qualifies,utility


class BalancedGraphTests(unittest.TestCase):
    def test_binary_edges_self_exclusion_ties_and_empty_items(self):
        ids,w=build_neighbors(np.array([1,1,1,2,2]),np.array([1,1,2,2,3]),2,4,2)
        np.testing.assert_array_equal(ids,[[0,0],[2,0],[1,3],[2,0],[0,0]])
        np.testing.assert_allclose(w,[[0,0],[1,0],[.5,.5],[1,0],[0,0]],atol=1e-7)
        z,v=build_neighbors(np.array([1]),np.array([1]),1,2,2)
        self.assertEqual(int(z.sum()),0);self.assertEqual(float(v.sum()),0.)
        with self.assertRaises(ValueError):build_neighbors(np.array([1]),np.array([0]),1,2,2)

    def test_zero_graph_exact_original_and_shared_padding_safe(self):
        torch.manual_seed(42);a=TwoTowerV2(3,4,embedding_dim=4,history_mode='recency')
        torch.manual_seed(42);b=SharedTower(3,4,embedding_dim=4,graph_alpha=0,neighbors=2)
        u=torch.tensor([1,0]);t=torch.tensor([0,1]);h=torch.tensor([[1,2,0],[0,0,0]])
        torch.testing.assert_close(a.encode_user(u,t,h),b.encode_user(u,t,h),rtol=0,atol=0)
        torch.testing.assert_close(a.encode_item(torch.arange(5)),b.encode_item(torch.arange(5)),rtol=0,atol=0)
        b.graph_alpha=.5
        ids,w=build_neighbors(np.array([1,1,2,2]),np.array([1,2,2,3]),3,4,2)
        b.set_neighbors(ids,w)
        x=b.encode_user(u,t,h)
        torch.testing.assert_close(x,b.encode_user(u,t,torch.tensor([[0,1,0,2],[0,0,0,0]])))
        self.assertTrue(torch.isfinite(x).all())
        torch.testing.assert_close(b.encode_item(torch.tensor([0])),torch.zeros(1,4))
        # The final score is still an independently encoded dot product.
        scores=x@b.encode_item(torch.arange(1,5)).T/.1
        torch.testing.assert_close(scores[:,0],b(u,torch.tensor([1,1]),t,h))
        scores.sum().backward();self.assertGreater(b.item_embedding.weight.grad.abs().sum().item(),0)

    def test_eval_bank_is_invalidated_on_training(self):
        b=SharedTower(2,3,embedding_dim=4,graph_alpha=.5,neighbors=1)
        b.eval()
        with torch.no_grad():
            a=b.encode_item(torch.tensor([1])).clone()
            b.train();b.item_embedding.weight[1].add_(torch.tensor([1.,0,0,0]));b.eval()
            self.assertFalse(torch.equal(a,b.encode_item(torch.tensor([1]))))


class BalancedObjectiveTests(unittest.TestCase):
    def test_mixture_mass_is_normalized_and_monotone(self):
        c=np.array([0,9,1,0]);mass=[]
        for rho in [0,.25,.5,1.]:
            w=mixture_weights(c,rho);self.assertAlmostEqual(float(np.dot(c,w)),10,places=5)
            mass.append(float(w[2]/10));self.assertEqual(w[3],0)
        np.testing.assert_allclose(mass,[.1,.2,.3,.5],rtol=1e-6)
        with self.assertRaises(ValueError):mixture_weights(c,1.1)

    def test_logit_adjustment_and_weight_gradient(self):
        s=torch.zeros(2,2,requires_grad=True);targets=torch.tensor([1,2])
        w=torch.tensor([0.,.5,1.5]);prior=torch.log(torch.tensor([.25,.75]))
        loss=choice_loss(s,targets,w,prior,1.)
        self.assertAlmostEqual(loss.item(),-.25*np.log(.25)-.75*np.log(.75),places=6)
        loss.backward();torch.testing.assert_close(s.grad,torch.tensor([[-.1875,.1875],[.1875,-.1875]]))
        with self.assertRaises(ValueError):choice_loss(s,torch.tensor([0,2]),w,prior,0)

    def test_selection_requires_actual_effect_not_one_hit(self):
        base={'recall@20':.1,'ndcg@20':.04,'groups':{'head_target':{'recall@20':.13},'tail_target':{'recall@20':.014,'recall@100':.064}}}
        gate={'retention':.98,'tail20_delta':.002,'tail100_delta':.01}
        tiny=base|{'groups':base['groups']|{'tail_target':{'recall@20':.0141,'recall@100':.065}}}
        good=base|{'groups':base['groups']|{'tail_target':{'recall@20':.02,'recall@100':.09}}}
        self.assertFalse(qualifies(tiny,base,gate));self.assertTrue(qualifies(good,base,gate))
        self.assertGreater(utility(good,base),utility(base,base))
        self.assertFalse(qualifies(good|{'ndcg@20':.03},base,gate))


if __name__=='__main__':unittest.main()
