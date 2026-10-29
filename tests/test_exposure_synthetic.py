import importlib.util
import unittest
import numpy as np


class SyntheticEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.exposure.synthetic'), 'Synthetic experiment is not implemented')

    def config(self):
        return dict(n_contexts=128,n_actions=4,context_dim=3,world_seed=77,exploration=.05)

    def test_reproducible_world_logs_and_known_finite_population_value(self):
        from src.exposure.synthetic import make_world,sample_log,policy_value
        a=make_world(self.config());b=make_world(self.config())
        np.testing.assert_array_equal(a['expected_reward'],b['expected_reward'])
        log=sample_log(a,1.5,200,42,.05);again=sample_log(a,1.5,200,42,.05)
        for key in ['context_ids','action','reward','pscore']:np.testing.assert_array_equal(log[key],again[key])
        mu=log['behavior_policy'];rows=np.arange(200)
        np.testing.assert_allclose(log['pscore'],mu[rows,log['action']])
        self.assertTrue((mu>=.05/4-1e-12).all())
        self.assertAlmostEqual(policy_value(a,a['targets']['uniform']),a['expected_reward'].mean())

    def test_reward_model_uses_only_separate_logged_feedback_not_oracle_rewards(self):
        from src.exposure.synthetic import make_world,sample_log,fit_reward_prediction
        w=make_world(self.config());train=sample_log(w,1.5,1000,91,.05)
        first=fit_reward_prediction(w,train)
        changed=dict(w);changed['expected_reward']=np.zeros_like(w['expected_reward'])
        second=fit_reward_prediction(changed,train)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(first.shape,w['expected_reward'].shape)
        self.assertTrue(((first>=0)&(first<=1)).all())

    def test_known_weighted_learning_really_updates_and_reports_true_target_risk(self):
        from src.exposure.synthetic import make_world
        from src.exposure.learning import fit_weighted_logistic
        w=make_world(self.config())
        result=fit_weighted_logistic(w,rows=500,seed=17,strength=2.,exploration=.05)
        self.assertEqual(set(result['models']),{'unweighted','ips','ips_clip5'})
        for row in result['models'].values():
            self.assertTrue(np.isfinite(row['true_uniform_logloss']))
            self.assertGreater(row['parameter_norm'],0.)
            self.assertGreater(row['closure_evaluations'],0)
        self.assertTrue(result['evaluation_uses_finite_population_expected_rewards'])


if __name__=='__main__':unittest.main()
