import importlib.util
import unittest
import numpy as np


class OPEMathTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.exposure'), 'Exposure module is not implemented')

    def fixture(self, **extra):
        from src.exposure.ope import ope_terms
        args=dict(action=np.array([0,1]), reward=np.array([1.,0.]), pscore=np.array([.5,.25]),
                  target_policy=np.array([[.25,.75],[.5,.5]]),
                  behavior_policy=np.array([[.5,.5],[.75,.25]]),
                  reward_prediction=np.array([[.2,.8],[.4,.6]]))
        args.update(extra);return ope_terms(**args)

    def test_hand_computed_ips_snips_dr_dm_and_ess(self):
        from src.exposure.ope import summarize_ope
        terms=self.fixture();result=summarize_ope(terms)
        for name,value in [('ips',.25),('snips',.2),('dm',.575),('dr',.175),('naive',.5)]:
            self.assertAlmostEqual(result['estimates'][name],value,places=12)
        self.assertAlmostEqual(result['weight_diagnostics']['raw']['ess'],6.25/4.25)
        np.testing.assert_allclose(terms['dr'],[1.05,-.7])

    def test_clipping_reports_changed_weights_and_does_not_clip_dr_to_unit_interval(self):
        from src.exposure.ope import summarize_ope
        result=summarize_ope(self.fixture(clip=1.))
        self.assertAlmostEqual(result['estimates']['snips'],1/3)
        self.assertAlmostEqual(result['estimates']['dr'],.475)
        self.assertEqual(result['weight_diagnostics']['clipped_fraction'],.5)

    def test_exact_finite_population_double_robustness(self):
        from src.exposure.ope import ope_terms
        mu=np.array([.2,.8]);pi=np.array([.7,.3]);q=np.array([.9,.1]);wrong_mu=np.array([.5,.5])
        # Enumerate action and binary reward, then integrate using TRUE joint masses.
        action=np.array([0,0,1,1]);reward=np.array([0.,1.,0.,1.])
        mass=np.array([mu[0]*(1-q[0]),mu[0]*q[0],mu[1]*(1-q[1]),mu[1]*q[1]])
        truth=pi@q
        for supplied_mu,prediction in [(mu,np.array([.4,.4])),(wrong_mu,q),(mu,q)]:
            t=ope_terms(action,reward,supplied_mu[action],np.tile(pi,(4,1)),
                        reward_prediction=np.tile(prediction,(4,1)),behavior_policy=np.tile(supplied_mu,(4,1)))
            self.assertAlmostEqual(float(mass@t['dr']),truth,places=12)
        t=ope_terms(action,reward,wrong_mu[action],np.tile(pi,(4,1)),reward_prediction=np.full((4,2),.4))
        self.assertGreater(abs(float(mass@t['dr'])-truth),.01)

    def test_invalid_probabilities_actions_and_unsupported_policy_rejected(self):
        for changes in [dict(pscore=np.array([0.,.25])),dict(pscore=np.array([np.nan,.25])),
                        dict(action=np.array([0,2])),dict(action=np.array([0.,1.])),
                        dict(target_policy=np.ones((2,2))),dict(reward=np.array([1.,np.nan])),
                        dict(behavior_policy=np.array([[1.,0.],[.75,.25]]),pscore=np.array([1.,.25])),
                        dict(clip=0.),dict(reward_prediction=np.array([[.2,np.nan],[.4,.6]]))]:
            with self.subTest(changes=list(changes)),self.assertRaises(ValueError):self.fixture(**changes)

    def test_zero_target_weights_have_explicit_undefined_snips(self):
        from src.exposure.ope import ope_terms,summarize_ope
        t=ope_terms(np.array([0,0]),np.array([1.,0.]),np.array([.5,.5]),np.tile([0.,1.],(2,1)))
        result=summarize_ope(t)
        self.assertIsNone(result['estimates']['snips'])
        self.assertEqual(result['snips_status'],'UNDEFINED_ZERO_WEIGHT')
        self.assertEqual(result['support_status'],'NOT_VERIFIABLE_FROM_LOGGED_PSCORE_ONLY')
        self.assertEqual(result['weight_diagnostics']['raw']['ess'],0.)

    def test_bootstrap_recomputes_ratio_and_resamples_complete_clusters(self):
        from src.exposure.uncertainty import bootstrap_ope
        t=self.fixture();a=bootstrap_ope(t,repeats=80,seed=17)
        b=bootstrap_ope(t,repeats=80,seed=17)
        self.assertEqual(a,b)
        self.assertLessEqual(a['intervals']['snips']['low'],.2)
        self.assertGreaterEqual(a['intervals']['snips']['high'],.2)
        fixed=bootstrap_ope(t,clusters=np.array([4,4]),repeats=30,seed=17)
        for key,value in [('snips',.2),('ips',.25),('dr',.175)]:
            self.assertAlmostEqual(fixed['intervals'][key]['low'],value)
            self.assertAlmostEqual(fixed['intervals'][key]['high'],value)

    def test_weighted_bce_loss_and_gradient(self):
        import torch
        from src.exposure.learning import weighted_bce
        x=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        r=torch.tensor([1.,0.],dtype=torch.float64);w=torch.tensor([.5,2.],dtype=torch.float64)
        loss=weighted_bce(x,r,w,'ips');loss.backward()
        self.assertAlmostEqual(loss.item(),1.25*np.log(2))
        np.testing.assert_allclose(x.grad.numpy(),[-.125,.5])
        self.assertAlmostEqual(weighted_bce(x,r,w,'snips').item(),np.log(2))
        with self.assertRaises(ValueError):weighted_bce(x,r,torch.zeros_like(w),'snips')

    def test_unequal_cluster_sizes_keep_event_denominator_and_count_undefined_ratios(self):
        from src.exposure.ope import ope_terms
        from src.exposure.uncertainty import bootstrap_ope
        t=ope_terms(np.zeros(4,dtype=int),np.array([0.,1.,1.,1.]),np.ones(4),np.ones((4,1)))
        actual=bootstrap_ope(t,clusters=np.array([1,2,2,2]),repeats=80,seed=11,confidence=.5)
        rng=np.random.default_rng(11);reference=[]
        for _ in range(80):
            sample=rng.integers(0,2,size=2)
            reference.append(sum(3 if x else 0 for x in sample)/sum(3 if x else 1 for x in sample))
        low,high=np.quantile(reference,[.25,.75])
        self.assertAlmostEqual(actual['intervals']['ips']['low'],low)
        self.assertAlmostEqual(actual['intervals']['ips']['high'],high)
        zero=self.fixture(target_policy=np.array([[0.,1.],[.5,.5]]))
        result=bootstrap_ope(zero,repeats=80,seed=11)
        rng=np.random.default_rng(11);undefined=sum((rng.integers(0,2,size=2)==0).all() for _ in range(80))
        self.assertEqual(result['intervals']['snips']['undefined_replicates'],undefined)

    def test_ess_is_stable_for_large_finite_weights(self):
        from src.exposure.ope import weight_diagnostics
        with np.errstate(over='raise'):
            result=weight_diagnostics(np.array([1e200,2e200]))
        self.assertAlmostEqual(result['ess'],1.8)
        self.assertAlmostEqual(result['ess_fraction'],.9)


if __name__=='__main__':unittest.main()
