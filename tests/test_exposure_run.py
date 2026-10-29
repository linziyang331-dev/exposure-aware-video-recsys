import importlib.util
import unittest
import tempfile
from pathlib import Path


class ExposureRunTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.exposure.run'),'Reproducible runner is not implemented')

    def test_outputs_cannot_replace_existing_model_data_or_leave_d_project(self):
        from src.exposure.run import output_path
        for path in ['experiments/ranking_v2','experiments/retrieval_v2','data/processed/kuairand_pure',
                     '.runtime/cache','../outside','C:/tmp/exposure','experiments/exposure/../../src']:
            with self.subTest(path=path),self.assertRaises(ValueError):output_path(path)
        self.assertEqual(output_path('experiments/exposure_reproduction').drive.upper(),'D:')

    def test_result_comparison_rejects_missing_rows_macro_and_nonfinite_values(self):
        from src.exposure.run import assert_result_equal
        value={'count':3,'groups':[{'auc':None,'value':.25}]}
        assert_result_equal(value,{'count':3,'groups':[{'auc':None,'value':.25+1e-12}]})
        for wrong in [{'count':2,'groups':value['groups']},{'count':3,'groups':[]},
                      {'count':3,'groups':[{'auc':None,'value':float('nan')}]},
                      {'count':3,'groups':[{'auc':None,'value':.4}]}]:
            with self.assertRaises(ValueError):assert_result_equal(value,wrong)

    def test_completed_component_reuse_and_tampering_rejection(self):
        from src.exposure.run import run_component
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'phase';calls=[]
            def worker():
                calls.append(1);output.mkdir()
                (output/'result.json').write_text('{"ok": true}',encoding='utf-8')
            signature={'configuration':1}
            run_component(output,signature,worker)
            run_component(output,signature,worker)
            self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):run_component(output,{'configuration':2},worker)
            (output/'result.json').write_text('changed')
            with self.assertRaises(ValueError):run_component(output,signature,worker)

    def test_configuration_rejects_missing_or_duplicate_target_policies(self):
        import copy
        from unittest.mock import patch
        from src.exposure import run
        config,_=run.get_config()
        for policies in [['uniform'],['reward_softmax'],['uniform','uniform','reward_softmax']]:
            changed=copy.deepcopy(config);changed['synthetic']['target_policies']=policies
            with self.subTest(policies=policies),patch.object(run.yaml,'safe_load',return_value=changed):
                with self.assertRaises(ValueError):run.get_config()

    def test_coverage_verification_checks_full_metadata_and_condition_repeat_grid(self):
        import copy
        from src.exposure import verify
        from src.exposure.synthetic import wilson_interval
        self.assertTrue(hasattr(verify,'verify_coverage_rows'),'Full coverage validation is missing')
        cfg={'coverage_rows':7,'coverage_repeats':2,'coverage_bootstrap_repeats':5,'world_seed':99}
        cases=['correct_pscore_oracle_q','correct_pscore_wrong_q','wrong_pscore_oracle_q','wrong_pscore_wrong_q']
        intervals=[];aggregates=[]
        for case in cases:
            for estimator in ['naive','ips','snips','dm','dr']:
                for repeat in range(2):
                    intervals.append(dict(repeat=str(repeat),seed=str(99+900000+repeat),case=case,
                        estimator=estimator,truth='0.5',low='0.2',high='0.8',defined='True',covered='True'))
                aggregates.append(dict(case=case,estimator=estimator,target='uniform',rows=7,covered=2,
                    valid_intervals=2,repeats=2,coverage_over_all_repeats=1.,coverage_given_defined=1.,
                    coverage_wilson95=wilson_interval(2,2),nominal_confidence=.95,bootstrap_repeats=5))
        verify.verify_coverage_rows(cfg,.5,intervals,aggregates)
        for field in ['coverage_given_defined','nominal_confidence','rows','target','bootstrap_repeats']:
            changed=copy.deepcopy(aggregates);changed[0][field]=-99
            with self.subTest(field=field),self.assertRaises(ValueError):
                verify.verify_coverage_rows(cfg,.5,intervals,changed)
        for mutation in ['duplicate','seed','truth','undefined_bounds']:
            changed=copy.deepcopy(intervals)
            if mutation=='duplicate':changed[-1]=changed[0]
            elif mutation=='seed':changed[0]['seed']='0'
            elif mutation=='truth':changed[0]['truth']='0.8'
            else:changed[0]['defined']='False'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                verify.verify_coverage_rows(cfg,.5,changed,aggregates)

    def test_report_requires_verification_for_the_exact_completed_components(self):
        import json
        from src.exposure import report
        self.assertTrue(hasattr(report,'require_verified_components'),'Report verification binding is missing')
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            for phase in ['synthetic','real']:
                folder=output/phase;folder.mkdir()
                (folder/'component.json').write_text(json.dumps({'fingerprint':phase+'-current'}))
            verification={'status':'PASS','component_fingerprints':{'synthetic':'synthetic-old','real':'real-current'}}
            (output/'verification.json').write_text(json.dumps(verification))
            with self.assertRaises(ValueError):report.require_verified_components(output)
            verification['component_fingerprints']['synthetic']='synthetic-current'
            (output/'verification.json').write_text(json.dumps(verification))
            self.assertEqual(report.require_verified_components(output),verification)


if __name__=='__main__':unittest.main()
