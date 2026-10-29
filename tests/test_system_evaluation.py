import importlib.util
import unittest
import numpy as np


class SystemMetricTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.system_evaluation'),'Unified evaluation is missing')

    def test_catalog_and_pool_mrr_keep_different_meanings(self):
        from src.system_evaluation.contracts import rank_metrics
        users=np.array([1,1,2]);ranks=np.array([1,101,0])
        m=rank_metrics(ranks,users,'catalog')
        self.assertAlmostEqual(m['mrr'],(1+1/101)/3)
        self.assertEqual(m['mrr_cutoff'],'full_catalog')
        with self.assertRaises(ValueError):rank_metrics(ranks,users,'fixed_v2_top100')
        pool=rank_metrics(np.array([1,0,0]),users,'fixed_v2_top100')
        self.assertAlmostEqual(pool['mrr'],1/3);self.assertEqual(pool['mrr_cutoff'],'100')
        self.assertAlmostEqual(pool['user_macro']['recall@20'],.25)

    def test_stage_funnel_counts_all_queries_and_conditional_denominator(self):
        from src.system_evaluation.contracts import candidate_funnel
        m=candidate_funnel(np.array([1,25,80,101,0]),np.array([30,2,90,0,0]))
        self.assertEqual([m[k] for k in ['queries','top20_hits','pool_not_top20','outside_pool']],[5,1,2,2])
        self.assertAlmostEqual(m['recall@20'],.2);self.assertAlmostEqual(m['recall@100'],.6)
        self.assertAlmostEqual(m['conditional_top20_given_pool'],1/3)
        self.assertAlmostEqual(m['recall@100']*m['conditional_top20_given_pool'],m['recall@20'])
        zero=candidate_funnel(np.array([101,0]),np.array([0,0]))
        self.assertIsNone(zero['conditional_top20_given_pool'])
        with self.assertRaises(ValueError):candidate_funnel(np.array([101,0]),np.array([1,0]))

    def test_alignment_rejects_event_reordering_user_changes_and_duplicate_events(self):
        from src.system_evaluation.contracts import check_alignment
        events=np.array([11,12]);users=np.array([1,2])
        check_alignment(events,users,events.copy(),users.copy())
        for e,u in [(events[::-1],users),(events,users[::-1]),(np.array([11,11]),users)]:
            with self.assertRaises(ValueError):check_alignment(events,users,e,u)

    def test_reranking_wins_losses_are_paired_on_original_denominator(self):
        from src.system_evaluation.contracts import transition_counts
        m=transition_counts(np.array([1,30,2,90,0]),np.array([25,1,2,99,0]))
        self.assertEqual(m['new_top20_hits'],1);self.assertEqual(m['lost_top20_hits'],1)
        self.assertEqual(m['retained_top20_hits'],1);self.assertEqual(m['neither_top20'],2)
        self.assertEqual(m['net_recall_change'],0.)

    def test_cohorts_use_training_popularity_and_keep_cold_targets(self):
        from src.system_evaluation.contracts import cohort_masks
        masks=cohort_masks(np.array([1,2,0]),np.array([1,0,1]),np.array([[1,0],[1,0],[0,0]]),np.array([0,100,10,5,1,1]))
        self.assertEqual(masks['head_target'].tolist(),[True,False,False])
        self.assertEqual(masks['tail_target'].tolist(),[False,True,False])
        self.assertEqual(masks['cold_target'].tolist(),[False,False,True])
        self.assertFalse(masks['history_target_seen'][2]);self.assertTrue(masks['empty_history'][2])

    def test_saved_metrics_cannot_change_denominators_macro_or_mrr_cutoff(self):
        from src.system_evaluation import contracts
        self.assertTrue(hasattr(contracts,'check_saved_metrics'),'Saved metric verification missing')
        m=contracts.rank_metrics(np.array([1,25,0]),np.array([1,1,2]),'fixed_v2_top100')
        expected={k:m[k] for k in ['queries','users','recall@20','ndcg@20','recall@100','ndcg@100']}
        expected['mrr@100']=m['mrr'];expected['user_macro']=dict(m['user_macro'])
        expected['user_macro']['mrr@100']=expected['user_macro'].pop('mrr')
        contracts.check_saved_metrics(m,expected)
        for field in ['queries','recall@20','mrr@100']:
            wrong=dict(expected);wrong[field]=999
            with self.subTest(field=field),self.assertRaises(ValueError):contracts.check_saved_metrics(m,wrong)
        wrong={**expected,'user_macro':{**expected['user_macro'],'ndcg@20':.99}}
        with self.assertRaises(ValueError):contracts.check_saved_metrics(m,wrong)

    def test_legacy_missing_macro100_is_recomputed_without_inventing_saved_values(self):
        from src.system_evaluation.contracts import rank_metrics,check_saved_metrics
        m=rank_metrics(np.array([1,101,0]),np.array([1,1,2]),'catalog')
        expected={k:m[k] for k in ['queries','users','recall@20','ndcg@20','recall@100','ndcg@100','mrr']}
        expected['user_macro']={k:m['user_macro'][k] for k in ['recall@20','ndcg@20','mrr']}
        try:check_saved_metrics(m,expected)
        except KeyError:self.fail('Legacy artifacts did not save macro100; verify available fields and recompute it')
        self.assertAlmostEqual(m['user_macro']['recall@100'],.25)

    def test_seed_summary_keeps_catalog_and_fixed_pool_separate(self):
        self.assertIsNotNone(importlib.util.find_spec('src.system_evaluation.analysis'),'Whole-system analysis missing')
        from src.system_evaluation.analysis import summarize_runs
        rows=[]
        for space,scale in [('catalog',1.),('fixed_v2_top100',.1)]:
            for seed,value in zip([42,43,44],[1.,2.,3.]):
                rows.append({'model':'sasrec_seed'+str(seed),'space':space,'partition':'standard/test',
                    'metrics':{'recall@20':value*scale,'ndcg@20':value*scale,'recall@100':value*scale}})
        got=summarize_runs(rows)
        self.assertEqual(len(got),2)
        means={row['space']:row['metrics']['ndcg@20']['mean'] for row in got}
        self.assertAlmostEqual(means['catalog'],2.);self.assertAlmostEqual(means['fixed_v2_top100'],.2)


if __name__=='__main__':unittest.main()
