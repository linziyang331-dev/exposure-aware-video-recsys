import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class SystemRunTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.system_evaluation.run'),'Unified CLI missing')

    def test_output_does_not_replace_old_experiments_or_leave_d_project(self):
        from src.system_evaluation.run import output_path
        for path in ['experiments/ranking_v2','experiments/offline_rl_cql','experiments/exposure_evaluation',
                     'data/system_evaluation_x','C:/tmp/system_evaluation_x','../system_evaluation_x']:
            with self.subTest(path=path),self.assertRaises(ValueError):output_path(path)
        self.assertEqual(output_path('experiments/system_evaluation_reproduction').drive.upper(),'D:')

    def test_report_rejects_new_outputs_with_same_input_fingerprint(self):
        from src.system_evaluation.report import require_verified
        from src.data.download import digest_file
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'evaluation').mkdir();path=out/'evaluation/component.json'
            path.write_text(json.dumps({'fingerprint':'input','output_hashes':{'result':'old'},'signature':{'config':{'deep_audit':False}}}))
            (out/'evaluation/analysis.json').write_text(json.dumps({'rank_results':[1,2],'pointwise_results':[1],'subgroups':[1,2,3]}))
            (out/'evaluation/pipeline.json').write_text(json.dumps({'checks':{'part':{'queries':512}}}))
            verification={'status':'PASS','rank_artifacts_recomputed':2,'exposure_predictions_recomputed':1,
                'subgroup_rows_recomputed':3,'pipeline_queries_replayed':512,'deep_module_recomputation':False,
                'latency_observations_replayed':False,'latency_validity_and_summary_checked':True,
                'component_fingerprint':'input','manifest_sha256':digest_file(path)}
            (out/'verification.json').write_text(json.dumps(verification))
            require_verified(out)
            for key,value in [('rank_artifacts_recomputed',-1),('pipeline_queries_replayed',999999),('deep_module_recomputation',True)]:
                wrong={**verification,key:value};(out/'verification.json').write_text(json.dumps(wrong))
                with self.subTest(key=key),self.assertRaises(ValueError):require_verified(out)
            (out/'verification.json').write_text(json.dumps(verification))
            path.write_text(json.dumps({'fingerprint':'input','output_hashes':{'result':'new'}}))
            with self.assertRaises(ValueError):require_verified(out)

    def test_recomputation_rejects_macro_funnel_and_scope_changes(self):
        from src.system_evaluation.run import compare_analysis
        expected={'rank_results':[{'space':'catalog','metrics':{'user_macro':{'ndcg@20':.2}}}],
                  'funnel':{'queries':10,'outside_pool':7},'real_causal_value_claimed':False}
        compare_analysis(expected,expected)
        for wrong in [{'rank_results':[],'funnel':expected['funnel'],'real_causal_value_claimed':False},
            {'rank_results':[{'space':'fixed_v2_top100','metrics':{'user_macro':{'ndcg@20':.2}}}],
             'funnel':expected['funnel'],'real_causal_value_claimed':False},
            {**expected,'funnel':{'queries':9,'outside_pool':7}},
            {**expected,'real_causal_value_claimed':True}]:
            with self.assertRaises(ValueError):compare_analysis(expected,wrong)

    def test_direct_report_checks_actual_outputs_before_rendering(self):
        from unittest.mock import patch
        from src.system_evaluation.run import run_component
        from src.system_evaluation.report import create_report
        from src.data.download import digest_file
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);folder=out/'evaluation';sig={'test':'fixed'}
            def worker():
                folder.mkdir();(folder/'analysis.json').write_text('{"value": 1}')
            component=run_component(folder,sig,worker)
            (out/'verification.json').write_text(json.dumps({'status':'PASS','component_fingerprint':component['fingerprint'],
                'manifest_sha256':digest_file(folder/'component.json')}))
            (folder/'analysis.json').write_text('{"value": 999}')
            with patch('src.system_evaluation.run.signature',return_value=sig),patch('src.system_evaluation.report._plots') as render:
                with self.assertRaises(ValueError):create_report({},out)
                render.assert_not_called()

    def test_every_csv_field_is_recomputed_including_empty_values(self):
        import csv
        from src.system_evaluation import run
        self.assertTrue(hasattr(run,'check_csv'),'Numerical CSV verification missing')
        rows=[{'model':'a','queries':3,'value':None},{'model':'b','queries':5,'value':.2}]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'table.csv'
            with path.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            run.check_csv(path,rows)
            path.write_text(path.read_text(encoding='utf-8-sig').replace('b,5,0.2','b,50,0.2'),encoding='utf-8')
            with self.assertRaises(ValueError):run.check_csv(path,rows)

    def test_timing_validation_preserves_descriptive_scope_and_observation_count(self):
        import copy
        from src.system_evaluation import run
        from src.system_evaluation.pipeline import latency_summary
        self.assertTrue(hasattr(run,'check_latency'),'Timing scope/metadata verification missing')
        payload={'latency':{'1':latency_summary([.001,.002],1)},
            'latency_scope':'preconstructed contexts to Top20, includes host/device transfer; excludes history lookup, network and concurrency; descriptive offline timings, not online SLO'}
        cfg={'benchmark_batches':[1],'measurement_batches':2}
        run.check_latency(payload,cfg)
        bad=copy.deepcopy(payload);bad['latency_scope']='online SLO guaranteed'
        with self.assertRaises(ValueError):run.check_latency(bad,cfg)
        bad=copy.deepcopy(payload);bad['latency']['1']['p99_ms']=1234.
        with self.assertRaises(ValueError):run.check_latency(bad,cfg)

    def test_report_budget_does_not_claim_deep_replay_when_disabled(self):
        from src.system_evaluation import report
        self.assertTrue(hasattr(report,'budget_description'),'Deep-audit budget description missing')
        self.assertIn('未深度重放',report.budget_description({'smoke':False,'deep_audit':False}))

    def test_delivery_manifest_detects_modified_summary_and_report(self):
        from src.system_evaluation import report
        from src.data.common import ROOT
        self.assertTrue(hasattr(report,'write_report_delivery'),'Report output manifest missing')
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            out=Path(d);(out/'evaluation').mkdir();(out/'figures').mkdir()
            (out/'evaluation/component.json').write_text('{}');(out/'verification.json').write_text('{}')
            for name in ['report.md','summary.json','figures/ranking_comparison.png','figures/ranking_comparison.svg',
                         'figures/candidate_funnel.png','figures/candidate_funnel.svg']:(out/name).write_text('original')
            cfg={'smoke':True,'output_directory':out.relative_to(ROOT).as_posix()}
            report.write_report_delivery(cfg,out);report.validate_report_delivery(cfg,out)
            for name in ['report.md','summary.json']:
                (out/name).write_text('fabricated improvement')
                with self.subTest(name=name),self.assertRaises(ValueError):report.validate_report_delivery(cfg,out)
                (out/name).write_text('original')


if __name__=='__main__':unittest.main()
