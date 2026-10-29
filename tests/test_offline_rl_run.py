import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class OfflineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.run'),'Offline experiment runner is missing')

    def test_output_is_a_separate_d_project_experiment(self):
        from src.offline_rl.run import output_path
        for path in ['experiments/ranking_v2','experiments/exposure_evaluation','experiments/sequential',
                     'data/offline_rl_cql','C:/tmp/offline_rl_cql','../offline_rl_cql']:
            with self.subTest(path=path),self.assertRaises(ValueError):output_path(path)
        self.assertEqual(output_path('experiments/offline_rl_reproduction').drive.upper(),'D:')

    def test_invalid_or_unpaired_predeclared_matrix_is_rejected(self):
        from unittest.mock import patch
        from src.offline_rl import run
        cfg,_=run.get_config()
        for field,value in [('alphas',[.2,1.]),('alphas',[0.,.2,.2]),('seeds',[42,42]),
                            ('main_alpha',0.),('training_episodes',[0])]:
            changed=copy.deepcopy(cfg);changed[field]=value
            with self.subTest(field=field,value=value),patch.object(run.yaml,'safe_load',return_value=changed):
                with self.assertRaises(ValueError):run.get_config()

    def test_distinct_condition_and_role_random_streams_do_not_collide(self):
        from src.offline_rl.run import get_config
        from src.offline_rl.experiment import condition_grid
        cfg,_=get_config();cfg['seeds']=[42,1042]
        values=[row[key] for row in condition_grid(cfg) for key in ['train_seed','validation_seed','rollout_seed']]
        self.assertEqual(len(values),len(set(values)),'Independent logs must not reuse a random stream')

    def test_invalid_world_and_network_dimensions_are_rejected_before_experiment(self):
        from unittest.mock import patch
        from src.offline_rl import run
        cfg,_=run.get_config()
        for group,key,value in [('world','horizon',True),('world','topics',4.5),
                                ('model','hidden',[0]),('model','hidden',[16.5])]:
            changed=copy.deepcopy(cfg);changed[group][key]=value
            with self.subTest(group=group,key=key,value=value),patch.object(run.yaml,'safe_load',return_value=changed):
                with self.assertRaises(ValueError):run.get_config()

    def test_small_training_matrix_full_recomputation_and_metric_curve_tampering_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.experiment'),'Experiment matrix is missing')
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.verify'),'Full numerical verification is missing')
        from src.offline_rl.run import get_config
        from src.offline_rl.experiment import run_experiment
        from src.offline_rl.verify import verify_experiment
        cfg,_=get_config(smoke=True)
        cfg.update(training_episodes=[8],validation_episodes=4,alphas=[0.,.2],rollout_episodes=40,bootstrap_repeats=10)
        cfg['world']['horizon']=3;cfg['model'].update(steps=8,trace_interval=4,target_update=4)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'experiment';result=run_experiment(cfg,out)
            check=verify_experiment(cfg,out)
            self.assertEqual(check['status'],'PASS')
            self.assertEqual(check['models_reloaded'],4)
            self.assertEqual(len(result['conditions']),2)
            path=out/'results.json';original=path.read_text(encoding='utf-8')
            for field in ['aggregate','curves']:
                changed=json.loads(original)
                changed[field][0]['discounted_return_mean' if field=='aggregate' else 'discounted_return']+=.1
                path.write_text(json.dumps(changed),encoding='utf-8')
                with self.subTest(field=field),self.assertRaises(ValueError):verify_experiment(cfg,out)
                path.write_text(original,encoding='utf-8')
            for field,value in [('main_method','optimal_oracle'),('status','REAL_RL_SUCCESS'),
                                ('selection','selected best evaluator return'),('limits',[])]:
                changed=json.loads(original);changed[field]=value
                path.write_text(json.dumps(changed),encoding='utf-8')
                with self.subTest(field=field),self.assertRaises(ValueError):verify_experiment(cfg,out)
                path.write_text(original,encoding='utf-8')
            changed=json.loads(original);changed['conditions'].pop()
            path.write_text(json.dumps(changed),encoding='utf-8')
            with self.assertRaises(ValueError):verify_experiment(cfg,out)

    def test_report_verification_must_match_this_exact_experiment(self):
        self.assertIsNotNone(importlib.util.find_spec('src.offline_rl.report'),'Report verification binding is missing')
        from src.offline_rl.report import require_verified_experiment
        from src.data.download import digest_file
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'experiment').mkdir()
            (out/'experiment/component.json').write_text(json.dumps({'fingerprint':'current'}))
            path=out/'verification.json';path.write_text(json.dumps({'status':'PASS','experiment_fingerprint':'old'}))
            with self.assertRaises(ValueError):require_verified_experiment(out)
            path.write_text(json.dumps({'status':'PASS','experiment_fingerprint':'current',
                'experiment_manifest_sha256':digest_file(out/'experiment/component.json')}))
            self.assertEqual(require_verified_experiment(out)['status'],'PASS')
            # A new completion may retain the same input fingerprint but have different outputs.
            (out/'experiment/component.json').write_text(json.dumps({'fingerprint':'current','output_hashes':{'results.json':'new'}}))
            with self.assertRaises(ValueError):require_verified_experiment(out)

    def test_direct_report_rejects_changed_results_before_rendering(self):
        from unittest.mock import patch
        from src.data.common import ROOT
        from src.offline_rl.run import get_config,signature,run_component,verify
        from src.offline_rl.experiment import run_experiment
        from src.offline_rl.report import create_report
        cfg,_=get_config(smoke=True)
        cfg.update(training_episodes=[4],validation_episodes=3,alphas=[0.,.2],rollout_episodes=20,bootstrap_repeats=5)
        cfg['world']['horizon']=2;cfg['model'].update(steps=2,trace_interval=1,target_update=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            out=Path(d);run_component(out/'experiment',signature(cfg),lambda:run_experiment(cfg,out/'experiment'))
            verify(cfg,out)
            path=out/'experiment/results.json';changed=json.loads(path.read_text(encoding='utf-8'))
            changed['aggregate'][0]['discounted_return_mean']+=100.
            path.write_text(json.dumps(changed),encoding='utf-8')
            with patch('src.offline_rl.report._plots') as render:
                with self.assertRaises(ValueError):create_report(cfg,out)
                render.assert_not_called()


if __name__=='__main__':unittest.main()
