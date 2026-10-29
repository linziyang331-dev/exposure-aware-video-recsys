import importlib.util
import unittest
import numpy as np


class ExposureMetricTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.exposure.metrics'), 'Real diagnostic metrics are not implemented')

    def test_binary_losses_calibration_and_weighted_groups(self):
        from src.exposure.metrics import binary_metrics
        m=binary_metrics(np.array([0,1]),np.zeros(2),bins=2)
        self.assertAlmostEqual(m['logloss'],np.log(2))
        self.assertAlmostEqual(m['brier'],.25);self.assertAlmostEqual(m['ece'],0.)
        self.assertAlmostEqual(m['auc'],.5)
        w=binary_metrics(np.array([0,1]),np.zeros(2),weights=np.array([1.,3.]),bins=2)
        self.assertAlmostEqual(w['positive_rate'],.75);self.assertAlmostEqual(w['ece'],.25)
        self.assertIsNone(binary_metrics(np.ones(2),np.zeros(2))['auc'])
        self.assertIsNone(binary_metrics(np.ones(6),np.zeros(6))['auc'])

    def test_platt_fits_validation_only_and_preserves_order(self):
        from src.exposure.metrics import fit_platt,apply_platt,binary_metrics
        x=np.repeat(np.array([-2.,-1.,0.,1.,2.]),20)
        y=np.concatenate([np.r_[np.ones(n),np.zeros(20-n)] for n in [2,4,7,11,16]])
        model=fit_platt(x,y)
        adjusted=apply_platt(x,model)
        self.assertGreater(model['scale'],0)
        self.assertLessEqual(binary_metrics(y,adjusted)['logloss'],binary_metrics(y,x)['logloss']+1e-9)
        np.testing.assert_array_equal(np.argsort(x,kind='stable'),np.argsort(adjusted,kind='stable'))
        self.assertAlmostEqual(binary_metrics(y,x)['auc'],binary_metrics(y,adjusted)['auc'])
        with self.assertRaises(ValueError):fit_platt(x,np.ones_like(y))

    def test_shared_strata_weights_balance_both_sources_without_using_labels(self):
        from src.exposure.metrics import shared_stratum_weights
        a,b,meta=shared_stratum_weights(np.array([1,1,2,3]),np.array([1,1,1,2]),
                                      np.array([1,2,2,4]),np.array([1,1,1,1]))
        np.testing.assert_allclose(a,[.25,.25,.5,0])
        np.testing.assert_allclose(b,[.5,.25,.25,0])
        self.assertEqual(meta['shared_strata'],2)
        self.assertAlmostEqual(a.sum(),1.);self.assertAlmostEqual(b.sum(),1.)

    def test_js_divergence_and_user_cluster_denominators(self):
        from src.exposure.metrics import js_divergence_counts
        from src.exposure.uncertainty import paired_cluster_means
        self.assertAlmostEqual(js_divergence_counts(np.array([1,0]),np.array([0,1])),1.)
        self.assertAlmostEqual(js_divergence_counts(np.array([1,2]),np.array([2,4])),0.)
        result=paired_cluster_means(np.array([0.,1.]),np.array([7,7]),
                                    np.array([1.,1.,1.]),np.array([7,7,7]),repeats=20)
        self.assertAlmostEqual(result['difference'][0],.5)
        self.assertAlmostEqual(result['low'][0],.5);self.assertAlmostEqual(result['high'][0],.5)

    def test_temporal_calibration_contract_rejects_future_fit(self):
        self.assertIsNotNone(importlib.util.find_spec('src.exposure.real'), 'Real diagnostic runner is not implemented')
        from src.exposure.real import check_temporal_fit
        check_temporal_fit(np.array([1,2]),np.array([3,4]))
        for fit,evaluation in [(np.array([1,3]),np.array([3,4])),(np.array([1,5]),np.array([3,4]))]:
            with self.assertRaises(ValueError):check_temporal_fit(fit,evaluation)


if __name__=='__main__':unittest.main()
