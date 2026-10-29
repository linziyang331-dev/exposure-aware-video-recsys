"""Hand-checked partitions and weighting accounting for the diagnostic audit."""
import unittest
import numpy as np
from scripts.audit_longtail_root_cause import competition_breakdown,paired_outcomes,weighting_audit,choice_mass


class LongtailRootCauseTests(unittest.TestCase):
    def test_head_competition_and_within_tail_failure_partition(self):
        r=competition_breakdown(np.array([5,101,201,202]),np.array([1,50,101,150]),100)
        self.assertEqual(r['queries'],4)
        self.assertEqual(r['global_hits'],1)
        self.assertEqual(r['head_crowding_losses'],1)
        self.assertEqual(r['within_tail_failures'],2)
        for a,b in [([0],[0]),([5],[6])]:
            with self.assertRaises(ValueError):competition_breakdown(np.array(a),np.array(b),20)

    def test_comparator_rescue_counts_do_not_drop_misses(self):
        r=paired_outcomes(np.array([1,0,101,5]),np.array([0,1,2,3]),20)
        self.assertEqual(r,{'queries':4,'both_hit':1,'tower_only':1,'itemcf_only':2,'both_miss':0})

    def test_event_mass_not_item_fraction_and_zero_positive_has_no_mass(self):
        r=weighting_audit(np.array([0,9,1,0]))
        self.assertAlmostEqual(r['0.0']['groups']['tail_target']['weighted_event_share'],.1)
        self.assertAlmostEqual(r['0.5']['groups']['tail_target']['weighted_event_share'],.25)
        self.assertEqual(r['0.5']['groups']['freq_zero']['weighted_event_share'],0)
        self.assertAlmostEqual(r['0.5']['global_event_mean_weight'],1.,places=6)

    def test_full_catalog_choice_mass_includes_non_target_columns(self):
        r=choice_mass(np.zeros((2,3)),np.array([1,1]))
        np.testing.assert_allclose(r['attraction_mass'],[1,0,0])
        np.testing.assert_allclose(r['competition_mass'],[1/3]*3)
        np.testing.assert_allclose(r['logit_shift_derivative'],[-2/3,1/3,1/3])
        with self.assertRaises(ValueError):choice_mass(np.zeros((2,3)),np.array([0,1]))


if __name__=='__main__':unittest.main()
