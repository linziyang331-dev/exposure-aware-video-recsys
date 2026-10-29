import unittest
import numpy as np
import torch
from src.retrieval_longtail.contracts import (frequency_weights, weighted_choice_loss,
    summarize, selection_passes, choose_trial)
from src.retrieval_longtail.model import GatedTwoTower


class LongtailContractsTests(unittest.TestCase):
    def test_weights_use_global_event_mean_not_catalog_mean(self):
        # counts 9:1, raw weights 1:3 -> event mean 1.2.
        weights=frequency_weights(np.array([0,9,1,0]),1.,3.)
        np.testing.assert_allclose(weights,[0,5/6,2.5,2.5],rtol=1e-6)
        np.testing.assert_allclose(frequency_weights([0,9,1,0],0.,3.),[0,1,1,1])
        for counts,beta,cap in [([0,0],.5,3),([0,1],-.1,3),([0,1],.5,0),([0,-1],.5,3),([0,1],float('nan'),3)]:
            with self.assertRaises(ValueError):frequency_weights(counts,beta,cap)

    def test_weighted_ce_hand_value_gradient_and_invalid_targets(self):
        scores=torch.zeros(2,2,requires_grad=True)
        weights=torch.tensor([0.,.5,1.5])
        loss=weighted_choice_loss(scores,torch.tensor([1,2]),weights)
        self.assertAlmostEqual(loss.item(),np.log(2),places=6)
        loss.backward()
        torch.testing.assert_close(scores.grad,torch.tensor([[-.125,.125],[.375,-.375]]))
        with self.assertRaises(ValueError):weighted_choice_loss(scores,torch.tensor([0,2]),weights)
        with self.assertRaises(ValueError):weighted_choice_loss(scores,torch.tensor([1,3]),weights)

    def test_groups_keep_cold_misses_and_zero_positive_catalog_items(self):
        result=summarize(np.array([1,25,0,200]),np.array([1,1,2,3]),
            np.array([1,2,0,3]),np.array([[0,0],[2,0],[0,0],[1,0]]),np.array([0,9,1,0]))
        self.assertEqual(result['queries'],4)
        self.assertEqual(result['recall@20'],.25)
        self.assertEqual(result['recall@100'],.5)
        self.assertEqual(result['groups']['cold_target']['queries'],1)
        self.assertEqual(result['groups']['freq_zero']['queries'],1)
        self.assertEqual(result['groups']['history_target_seen']['recall@100'],1.)
        self.assertEqual(result['groups']['tail_target']['recall@100'],.5)

    def test_selection_requires_tail_gains_and_total_head_guards(self):
        def m(r=.1,n=.04,h=.13,t=.02,p=.06):
            return {'recall@20':r,'ndcg@20':n,'groups':{'head_target':{'recall@20':h},
                'tail_target':{'recall@20':t,'recall@100':p}}}
        gate={'overall_retention':.99,'head_retention':.98}
        old=m(); good=m(t=.03,p=.07)
        self.assertTrue(selection_passes(good,old,gate))
        self.assertFalse(selection_passes(m(r=.098,t=.03,p=.07),old,gate))
        self.assertFalse(selection_passes(m(h=.12,t=.03,p=.07),old,gate))
        self.assertFalse(selection_passes(old,old,gate))
        self.assertIsNone(choose_trial({'bad':old},old,gate))
        self.assertEqual(choose_trial({'z':good,'a':good},old,gate),'a')


class GatedHistoryTests(unittest.TestCase):
    def test_padding_and_empty_history_are_safe_and_candidates_independent(self):
        torch.manual_seed(42)
        model=GatedTwoTower(4,6,embedding_dim=4)
        users=torch.tensor([1,0]);tabs=torch.tensor([0,1])
        a=torch.tensor([[1,2,0],[0,0,0]])
        b=torch.tensor([[0,1,0,2,0],[0,0,0,0,0]])
        torch.testing.assert_close(model.encode_user(users,tabs,a),model.encode_user(users,tabs,b))
        self.assertTrue(torch.isfinite(model.encode_user(users,tabs,a)).all())
        torch.testing.assert_close(model.encode_user(users,tabs,a)[1],
            torch.nn.functional.normalize(model.user_embedding(users)+model.tab_embedding(tabs),dim=-1)[1])
        items=torch.arange(1,7)
        before=model.encode_item(items).detach().clone()
        score=model.encode_user(users,tabs,a)@model.encode_item(items).T/model.temperature
        torch.testing.assert_close(score[:,0],model(users,torch.ones(2,dtype=torch.long),tabs,a))
        torch.testing.assert_close(before,model.encode_item(items))

    def test_gate_and_attention_learn_from_context(self):
        torch.manual_seed(7)
        model=GatedTwoTower(4,6,embedding_dim=4)
        users=torch.tensor([1,2]);tabs=torch.tensor([0,1]);h=torch.tensor([[1,2,3],[2,4,0]])
        scores=model.encode_user(users,tabs,h)@model.encode_item(torch.arange(1,7)).T/.1
        weighted_choice_loss(scores,torch.tensor([4,5]),torch.tensor([0.]+[1.]*6)).backward()
        self.assertGreater(model.attention_query.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.history_gate.weight.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
