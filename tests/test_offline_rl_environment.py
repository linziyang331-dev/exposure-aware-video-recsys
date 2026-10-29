import importlib.util
import unittest
import numpy as np


class OfflineEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl'), 'Offline RL module is missing')

    def config(self):
        return dict(world_seed=77,topics=4,fatigue_levels=3,horizon=5)

    def toy(self):
        return dict(reward=np.array([[.4,0.],[1.,1.]]),
            transition=np.array([[[1.,0.],[0.,1.]],[[0.,1.],[0.,1.]]]),
            initial=np.array([1.,0.]),horizon=2)

    def test_exact_finite_horizon_value_and_delayed_reward_optimum(self):
        from src.offline_rl.environment import evaluate_policy,optimal_policy
        world=self.toy();uniform=np.full((2,2),.5)
        value=evaluate_policy(world,uniform,.9)
        self.assertAlmostEqual(value['discounted_return'],.74)
        self.assertAlmostEqual(value['undiscounted_return'],.8)
        optimal=optimal_policy(world,.9)
        self.assertEqual(optimal[0,0].argmax(),1)
        self.assertEqual(optimal[1,0].argmax(),0)
        self.assertAlmostEqual(evaluate_policy(world,optimal,.9)['discounted_return'],.9)
        self.assertTrue(np.all(value['state_values'][-1]==0))

    def test_world_and_logging_support_are_reproducible_and_normalized(self):
        from src.offline_rl.environment import make_world,behavior_policy,validate_world
        a=make_world(self.config());b=make_world(self.config());validate_world(a)
        for key in ['reward','transition','initial']:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(a['reward'].shape,(12,8))
        np.testing.assert_allclose(a['transition'].sum(-1),1.)
        broad=behavior_policy(a,'exploratory');narrow=behavior_policy(a,'narrow')
        self.assertTrue((broad>0).all())
        np.testing.assert_array_equal((narrow>0).sum(1),2)
        np.testing.assert_allclose(narrow.sum(1),1.)

    def test_policy_or_world_with_invalid_probabilities_is_rejected(self):
        from src.offline_rl.environment import evaluate_policy,validate_world
        world=self.toy()
        for policy in [np.zeros((2,2)),np.full((2,2),np.nan),np.array([[-.1,1.1],[.5,.5]])]:
            with self.assertRaises(ValueError):evaluate_policy(world,policy,.95)
        broken=dict(world);broken['transition']=world['transition']*.5
        with self.assertRaises(ValueError):validate_world(broken)

    def test_logged_trajectories_are_contiguous_reproducible_and_separate_by_episode(self):
        from src.offline_rl.environment import make_world,behavior_policy
        from src.offline_rl.data import collect_log,validate_log
        w=make_world(self.config());mu=behavior_policy(w,'exploratory')
        a=collect_log(w,mu,20,42);b=collect_log(w,mu,20,42)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        validate_log(w,mu,a)
        self.assertEqual(len(a['action']),100)
        np.testing.assert_array_equal(a['next_state'].reshape(20,5)[:,:-1],a['state'].reshape(20,5)[:,1:])
        self.assertEqual(int(a['terminated'].sum()),20)
        validation=collect_log(w,mu,10,43,episode_offset=1000)
        self.assertFalse(np.intersect1d(a['episode_id'],validation['episode_id']).size)

    def test_protocol_rejects_broken_time_terminal_pscore_and_next_state_alignment(self):
        from src.offline_rl.environment import make_world,behavior_policy
        from src.offline_rl.data import collect_log,validate_log
        w=make_world(self.config());mu=behavior_policy(w,'narrow');log=collect_log(w,mu,5,42)
        for field in ['time','terminated','pscore','next_state','episode_id']:
            changed={k:v.copy() for k,v in log.items()}
            if field=='time':changed[field][0]=1
            elif field=='terminated':changed[field][0]=True
            elif field=='pscore':changed[field][0]=.123
            elif field=='next_state':changed[field][0]=(changed['state'][1]+1)%12
            else:changed[field][1]=999
            with self.subTest(field=field),self.assertRaises(ValueError):validate_log(w,mu,changed)

    def test_rollout_agrees_with_exact_value_on_independent_episodes(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.evaluate'),'Policy evaluation is missing')
        from src.offline_rl.evaluate import rollout
        from src.offline_rl.environment import evaluate_policy
        w=self.toy();pi=np.full((2,2),.5)
        samples=rollout(w,pi,10000,91,.9)
        self.assertAlmostEqual(samples['returns'].mean(),evaluate_policy(w,pi,.9)['discounted_return'],delta=.025)
        again=rollout(w,pi,10000,91,.9)
        np.testing.assert_array_equal(samples['returns'],again['returns'])

    def test_policy_metrics_separate_zero_support_and_unseen_training_actions(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.evaluate'),'Policy evaluation is missing')
        from src.offline_rl.evaluate import policy_metrics
        w=self.toy();mu=np.array([[1.,0.],[1.,0.]])
        pi=np.array([[0.,1.],[1.,0.]]);counts=np.ones((2,2,2),dtype=int)
        m=policy_metrics(w,pi,mu,counts,.9)
        self.assertAlmostEqual(m['zero_behavior_support_fraction'],.5)
        self.assertEqual(m['unobserved_training_action_fraction'],0.)
        counts[:,:,0]=0
        m=policy_metrics(w,pi,mu,counts,.9)
        self.assertAlmostEqual(m['unobserved_training_action_fraction'],.5)

    def test_paired_episode_bootstrap_matches_reference_and_has_correct_direction(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.evaluate'),'Policy evaluation is missing')
        from src.offline_rl.evaluate import compare_returns
        a=np.array([0.,2.,4.]);b=np.array([1.,1.,6.]);rng=np.random.default_rng(13)
        reference=[(b-a)[rng.integers(0,3,size=3)].mean() for _ in range(100)]
        result=compare_returns(a,b,100,13)
        self.assertAlmostEqual(result['mean'],(b-a).mean())
        np.testing.assert_allclose([result['low'],result['high']],np.quantile(reference,[.025,.975]))
        with self.assertRaises(ValueError):compare_returns(a,b[:2],100,13)

    def test_empirical_behavior_clone_uses_logged_actions_and_global_fallback(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.evaluate'),'Policy evaluation is missing')
        from src.offline_rl.evaluate import empirical_bc
        counts=np.zeros((2,2,2),dtype=int);counts[0,0]=[3,1]
        pi=empirical_bc(counts)
        np.testing.assert_allclose(pi,np.broadcast_to([.75,.25],(2,2,2)))
        self.assertEqual(pi.shape,counts.shape)


if __name__=='__main__':unittest.main()
