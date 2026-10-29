import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch


class OfflineLearningTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.loss'),'Conservative loss is missing')
        torch.set_num_threads(1)

    def test_hand_computed_bellman_and_conservative_loss_and_detached_target(self):
        from src.offline_rl.loss import cql_loss
        q=torch.tensor([[.2,.8],[.4,.1]],dtype=torch.float64,requires_grad=True)
        next_q=torch.tensor([[1.,2.],[8.,9.]],dtype=torch.float64,requires_grad=True)
        a=torch.tensor([0,1]);r=torch.tensor([1.,0.],dtype=torch.float64);done=torch.tensor([False,True])
        result=cql_loss(q,a,r,next_q,done,.5,.2)
        expected=(np.logaddexp(.2,.8)-.2+np.logaddexp(.4,.1)-.1)/2
        self.assertAlmostEqual(result['bellman'].item(),1.625)
        self.assertAlmostEqual(result['conservative'].item(),expected)
        self.assertAlmostEqual(result['loss'].item(),1.625+.2*expected)
        result['loss'].backward();self.assertIsNone(next_q.grad)
        self.assertIsNotNone(q.grad)

    def test_alpha_zero_is_exact_dqn_and_regularizer_pushes_unlogged_actions_down(self):
        from src.offline_rl.loss import cql_loss
        q=torch.zeros((1,3),requires_grad=True)
        args=(q,torch.tensor([0]),torch.zeros(1),torch.zeros((1,3)),torch.ones(1,dtype=torch.bool),.95)
        zero=cql_loss(*args,0.);self.assertEqual(zero['loss'].item(),zero['bellman'].item())
        penalized=cql_loss(*args,1.);penalized['loss'].backward()
        self.assertLess(q.grad[0,0].item(),0.)
        self.assertTrue((q.grad[0,1:]>0).all())

    def test_loss_rejects_bad_actions_flags_rewards_and_nonfinite_values(self):
        from src.offline_rl.loss import cql_loss
        q=torch.zeros((2,3));a=torch.tensor([0,1]);r=torch.zeros(2);done=torch.tensor([False,True])
        for mutation in ['action','terminal','reward','next','alpha','gamma']:
            args=[q,a,r,q,done,.95,.2]
            if mutation=='action':args[1]=torch.tensor([0,3])
            elif mutation=='terminal':args[4]=torch.tensor([0.,.5])
            elif mutation=='reward':args[2]=torch.tensor([float('nan'),0.])
            elif mutation=='next':args[3]=torch.full_like(q,float('inf'))
            elif mutation=='alpha':args[6]=-1.
            else:args[5]=1.1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):cql_loss(*args)

    def test_network_encodes_time_and_supports_terminal_next_time(self):
        from src.offline_rl.model import QNetwork
        model=QNetwork(12,3,8,[16,16])
        features=model.encode(torch.tensor([0,0,0]),torch.tensor([0,1,3]))
        self.assertFalse(torch.equal(features[0],features[1]))
        self.assertEqual(model(torch.tensor([0,0]),torch.tensor([0,3])).shape,(2,8))
        with self.assertRaises(ValueError):model(torch.tensor([0]),torch.tensor([4]))

    def training_case(self):
        from src.offline_rl.environment import make_world,behavior_policy
        from src.offline_rl.data import collect_log
        world=make_world(dict(world_seed=77,topics=4,fatigue_levels=3,horizon=3))
        mu=behavior_policy(world,'narrow')
        train=collect_log(world,mu,30,41);valid=collect_log(world,mu,15,51,episode_offset=1000)
        cfg=dict(states=12,actions=8,horizon=3,hidden=[16,16],steps=80,batch_size=32,
                 learning_rate=.001,gamma=.95,target_update=10,trace_interval=40,gradient_clip=10.)
        return train,valid,cfg

    def test_actual_training_exact_reload_and_data_signature_rejection(self):
        from src.offline_rl.train import fit_model,load_checkpoint,log_fingerprint
        from src.offline_rl.model import all_q_values
        train,valid,cfg=self.training_case()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'model';result=fit_model(train,valid,cfg,.2,42,out)
            self.assertGreater(result['parameter_change_norm'],0.)
            self.assertEqual(result['steps'],80)
            self.assertEqual(result['snapshot_steps'],[0,40,80])
            model=load_checkpoint(out/'model.pt',result['signature'])
            np.testing.assert_array_equal(all_q_values(model),np.load(out/'q_history.npy')[-1])
            wrong=dict(result['signature']);wrong['train_log_sha256']='changed'
            with self.assertRaises(ValueError):load_checkpoint(out/'model.pt',wrong)
            changed={k:v.copy() for k,v in train.items()};changed['reward'][0]=1.-changed['reward'][0]
            self.assertNotEqual(log_fingerprint(train),log_fingerprint(changed))
            with self.assertRaises(FileExistsError):fit_model(train,valid,cfg,.2,42,out)

    def test_training_rejects_overlapping_episodes_and_oracle_inputs(self):
        from src.offline_rl.train import fit_model
        train,valid,cfg=self.training_case()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):fit_model(train,train,cfg,.2,42,Path(d)/'overlap')
            leaked={**train,'oracle_reward':np.zeros(len(train['action']))}
            with self.assertRaises(ValueError):fit_model(leaked,valid,cfg,.2,42,Path(d)/'oracle')


if __name__=='__main__':unittest.main()
