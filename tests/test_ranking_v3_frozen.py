"""The second-stage structural regularization must preserve the old predictor."""
import importlib.util
import unittest
import torch


class ProtectedBackboneTests(unittest.TestCase):
    def test_backbone_parameters_and_batchnorm_state_remain_exact_after_updates(self):
        self.assertIsNotNone(importlib.util.find_spec('src.ranking_v3_frozen'), 'Protected backbone is not implemented')
        from src.ranking_v3_frozen.model import build_model
        from src.ranking_v2.model import RichDIN
        from src.retrieval_v2.model import TwoTowerV2
        from tests.test_ranking_v2 import config
        torch.set_num_threads(2);torch.manual_seed(13)
        cfg=config(residual=True,teacher_features=True,pretrained=True,variant='gated',short_history=1,branch_hidden=16)
        teacher=TwoTowerV2(3,6,embedding_dim=8)
        old=RichDIN(3,6,cfg,teacher).eval()
        initial={k:v.detach().clone() for k,v in old.state_dict().items()}
        new=build_model(3,6,cfg,teacher,initial).train()
        u=torch.tensor([1,1,2]);i=torch.tensor([2,3,4]);t=torch.ones(3,dtype=torch.long)
        h=torch.tensor([[1,3,0,0],[2,4,0,0],[1,2,0,0]])
        with torch.no_grad():
            expected=old(u,i,t,h)
            actual=new(u,i,t,h)
        torch.testing.assert_close(expected['rank'],actual['rank'],rtol=0,atol=0)
        opt=torch.optim.Adam([p for p in new.parameters() if p.requires_grad],lr=.01)
        for _ in range(3):
            opt.zero_grad();new(u,i,t,h)['rank'].square().mean().backward();opt.step()
        for name,value in initial.items():
            torch.testing.assert_close(new.state_dict()[name],value,rtol=0,atol=0)
        self.assertGreater(new.interest_output.weight.abs().sum().item(),0.)
        self.assertGreater(new.interest_gate.weight.grad.abs().sum().item(),0.)
        self.assertFalse(new.core.training)
        self.assertTrue(new.interest_network.training)


if __name__=='__main__':unittest.main()
