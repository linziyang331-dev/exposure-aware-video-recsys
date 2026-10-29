import unittest
import numpy as np


class ClusterBootstrapTests(unittest.TestCase):
    def test_event_weighted_difference_resamples_whole_users(self):
        from src.retrieval.statistics import paired_user_bootstrap
        result=paired_user_bootstrap(np.array([1,0,1]),np.array([0,0,1]),np.array([1,1,2]),repeats=500,seed=42)
        recall=result['recall@20']
        self.assertAlmostEqual(recall['difference'],1/3)
        self.assertGreaterEqual(recall['ci95_low'],0.)
        self.assertLessEqual(recall['ci95_high'],.5)
        self.assertLessEqual(recall['ci95_low'],recall['difference'])
        self.assertGreaterEqual(recall['ci95_high'],recall['difference'])

    def test_one_user_fixed_gain_has_degenerate_interval(self):
        from src.retrieval.statistics import paired_user_bootstrap
        result=paired_user_bootstrap(np.array([1,1]),np.array([0,0]),np.array([7,7]),repeats=50,seed=2)
        self.assertEqual(result['recall@20']['difference'],1.)
        self.assertEqual(result['recall@20']['ci95_low'],1.)
        self.assertEqual(result['recall@20']['ci95_high'],1.)
