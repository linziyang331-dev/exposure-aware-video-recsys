import importlib.util
import unittest
import numpy as np
import torch


class NetworkStructureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('src.ranking_v3'), 'Top20 network module is not implemented')
        torch.set_num_threads(2)

    def config(self, **changes):
        from tests.test_ranking_v2 import config
        return config(residual=True, teacher_features=True, pretrained=True,
                      variant='gated', short_history=2, branch_hidden=16) | changes

    def models(self, variant='gated'):
        from src.retrieval_v2.model import TwoTowerV2
        from src.ranking_v2.model import RichDIN
        from src.ranking_v3.model import build_model
        torch.manual_seed(19)
        teacher = TwoTowerV2(3, 6, embedding_dim=8)
        cfg = self.config(variant=variant)
        old = RichDIN(3, 6, cfg, teacher).eval()
        with torch.no_grad():
            old.rank_delta.weight.fill_(.05)
            old.repeat_linear.weight.fill_(-.4)
        new = build_model(3, 6, cfg, teacher, old.state_dict()).eval()
        return old, new

    def batch(self):
        return (torch.tensor([1, 2, 0]), torch.tensor([2, 3, 4]), torch.tensor([1, 1, 2]),
                torch.tensor([[1, 2, 0, 0], [3, 3, 0, 0], [0, 0, 0, 0]]))

    def test_trainable_dice_registers_alpha_and_receives_gradient(self):
        from src.ranking_v3.model import TrainableDice
        layer = TrainableDice(8)
        self.assertIn('alpha', dict(layer.named_parameters()))
        layer(torch.randn(4, 8)).sum().backward()
        self.assertGreater(layer.alpha.grad.abs().sum().item(), 0.)
        self.assertIn('alpha', layer.state_dict())

    def test_warm_start_preserves_old_scores_exactly_for_all_variants(self):
        for variant in ['v2_continued', 'dice', 'gated', 'gated_copy_safe']:
            old, new = self.models(variant)
            args = self.batch(); canonical = torch.tensor([2., 1., .5])
            with torch.no_grad():
                a = old(*args, base_scores=canonical)
                b = new(*args, base_scores=canonical)
            torch.testing.assert_close(a['rank'], b['rank'], rtol=0, atol=0)
            torch.testing.assert_close(a['ctr'], b['ctr'], rtol=0, atol=0)

    def test_attention_is_masked_normalized_and_safe_for_empty_or_copy_only_history(self):
        _, model = self.models('gated_copy_safe')
        args = self.batch(); details = model.interest_details(*args)
        for key in ['long_weights', 'short_weights']:
            weights = details[key]
            self.assertTrue(torch.isfinite(weights).all())
            self.assertAlmostEqual(weights[0].sum().item(), 1., places=6)
            self.assertEqual(weights[1:].abs().sum().item(), 0.)
            self.assertEqual(weights[0, 1:].abs().sum().item(), 0.)
        self.assertTrue(((details['gate'] >= 0) & (details['gate'] <= 1)).all())

    def test_branch_is_padding_invariant_and_candidate_permutation_equivariant(self):
        _, model = self.models()
        with torch.no_grad(): model.interest_output.weight.fill_(.1)
        u, i, t, h = self.batch()
        a = model(u, i, t, h)['rank']
        b = model(u, i, t, torch.cat([h, torch.zeros(3, 3, dtype=torch.long)], 1))['rank']
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        order = torch.tensor([2, 0, 1])
        c = model(u[order], i[order], t[order], h[order])['rank']
        torch.testing.assert_close(a[order], c, rtol=1e-6, atol=1e-6)

    def test_interest_and_gate_learn_and_teacher_remains_frozen(self):
        _, model = self.models()
        # At least one history must exceed the short window; otherwise the two
        # interests coincide and the gate is correctly unidentifiable.
        args = self.batch()
        args = (*args[:3], torch.tensor([[1, 2, 5, 0], [3, 3, 0, 0], [0, 0, 0, 0]]))
        model.train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            out = model(*args)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out['rank'], torch.tensor([1., 0., 1.]))
            loss.backward(); optimizer.step()
        self.assertGreater(model.interest_gate.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.interest_key.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.teacher.parameters()))
        self.assertFalse(model.teacher.training)


class NetworkTrainingTests(unittest.TestCase):
    def test_actual_training_reload_and_changed_config_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('src.ranking_v3.train'), 'Network training is not implemented')
        import tempfile
        from pathlib import Path
        from src.ranking_v3.train import train_run, load_model
        from src.retrieval_v2.model import TwoTowerV2
        from src.ranking_v2.model import RichDIN
        from tests.test_ranking_v2 import config
        class Cache(dict):
            def __len__(self): return len(self['label'])
        class Candidates(dict):
            def __len__(self): return len(self['indices'])
        t = Cache(user_index=np.array([1,1,2,2]*4), item_index=np.array([1,3,2,4]*4),
                  tab=np.ones(16,dtype=np.int64), history=np.array([[2,0,0,0],[2,0,0,0],[1,0,0,0],[1,0,0,0]]*4),
                  label=np.array([1,0,1,0]*4), user_id=np.array([1,1,2,2]*4), event_id=np.arange(16))
        t.metadata={'signature':{'fixture':True}}
        idx=np.flatnonzero(t['label']==1)
        c=Candidates(indices=idx, ids=np.tile(np.arange(1,7),(len(idx),1)),
                     scores=np.tile(np.arange(6,0,-1,dtype=np.float32),(len(idx),1)),
                     user_ids=t['user_id'][idx], event_ids=t['event_id'][idx], v2_ranks=t['item_index'][idx])
        c.metadata={'signature':{'fixture':True}}
        cfg=config(residual=True, teacher_features=True, pretrained=True, variant='gated', short_history=2,
                   objective='choice', learning_rate=.01, repeat_lr_multiplier=1., weight_decay=0., epochs=2, patience=2,
                   query_batch_size=4, competitors=2, aux_batch_size=4, aux_weight=.1, delta_reg=.01,
                   queries_per_epoch=8, mask_past=False, eval_batch_size=8, candidate_batch_size=4)
        teacher=TwoTowerV2(3,6,embedding_dim=8)
        initial={k:v.detach().clone() for k,v in RichDIN(3,6,cfg,teacher).state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            result=train_run(t,t,c,c,3,6,cfg,Path(d),42,teacher,initial)
            self.assertGreater(result['optimizer_updates'],0)
            self.assertTrue(result['optimization_changed_parameters'])
            self.assertTrue(result['teacher_unchanged'])
            self.assertTrue(result['reload_identical'])
            m,_=load_model(Path(d)/'best.pt','cpu')
            self.assertIn('interest_output.weight',m.state_dict())
            with self.assertRaises(ValueError):
                train_run(t,t,c,c,3,6,cfg|{'delta_reg':1.},Path(d),42,teacher,initial)


class NetworkProtocolTests(unittest.TestCase):
    def test_previous_comparison_metrics_are_bound_to_verified_artifact_hashes(self):
        from src.ranking_v3 import run
        self.assertTrue(hasattr(run, 'verify_previous_results'), 'Old comparison evidence is not verified')
        import tempfile, json, hashlib
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); hashes={}
            names=['experiments/ranking_v2/evaluation.json']+[
                'experiments/ranking_v2/artifacts/ranks/din_seed42/'+p+'.npz'
                for p in ['standard_validation','standard_test','random_validation','random_test']]
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original')
                hashes[name]=hashlib.sha256(b'original').hexdigest()
            report=root/'reports/ranking_v2_artifact_validation.json';report.parent.mkdir()
            report.write_text(json.dumps({'status':'PASS','artifact_hashes':hashes}))
            self.assertEqual(len(run.verify_previous_results(root)),6)
            (root/names[0]).write_bytes(b'changed')
            with self.assertRaises(ValueError):run.verify_previous_results(root)

    def test_selection_requires_improvement_over_continuation_and_original_recall(self):
        from src.ranking_v3 import run
        self.assertTrue(hasattr(run, 'choose_structure'), 'Explicit structural gain selection is missing')
        def trial(n, r):
            return {'baseline_validation': {'ranking': {'ndcg@20': .05, 'recall@20': .13}},
                    'best_validation': {'ranking': {'ndcg@20': n, 'recall@20': r}}}
        trials={'v2_continued':trial(.055,.14), 'dice':trial(.054,.14), 'gated':trial(.056,.12)}
        self.assertIsNone(run.choose_structure(trials))
        trials['gated']=trial(.056,.13)
        self.assertEqual(run.choose_structure(trials)[0], 'gated')

    def test_saved_rank_verification_rejects_macro_and_conditional_metric_changes(self):
        from src.ranking_v3 import run
        self.assertTrue(hasattr(run, 'validate_rank_artifact'), 'Full saved-rank metric validation is missing')
        import tempfile, copy
        from pathlib import Path
        from src.ranking.evaluate import rank_scores
        class Cache(dict):
            def __len__(self): return len(self['item_index'])
        class Candidates(dict):
            def __len__(self): return len(self['indices'])
        cache=Cache(item_index=np.array([1,3]))
        cand=Candidates(ids=np.array([[1,2],[1,2]]), indices=np.array([0,1]),
                        v2_ranks=np.array([1,0]), user_ids=np.array([1,2]), event_ids=np.array([10,11]))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ranks.npz'
            metrics=rank_scores(cand,cache,np.array([[2.,1.],[2.,1.]]),path)
            run.validate_rank_artifact(path,cand,metrics)
            for key in ['user_macro','conditional_hit_metrics']:
                broken=copy.deepcopy(metrics);broken[key]['ndcg@20']+=.1
                with self.assertRaises(ValueError):run.validate_rank_artifact(path,cand,broken)


if __name__ == '__main__': unittest.main()
