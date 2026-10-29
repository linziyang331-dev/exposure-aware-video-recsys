import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch


class ObjectiveV2Tests(unittest.TestCase):
    def test_full_catalog_loss_uses_one_based_targets_and_all_candidates(self):
        from src.retrieval_v2.objectives import full_catalog_loss
        scores=torch.tensor([[1.,0.,0.],[0.,1.,0.]],requires_grad=True)
        loss=full_catalog_loss(scores,torch.tensor([1,2]))
        self.assertAlmostEqual(loss.item(),math.log(math.e+2)-1,places=6)
        loss.backward()
        self.assertLess(scores.grad[0,0].item(),0)
        self.assertGreater(scores.grad[0,2].item(),0)
        for target in [0,4]:
            with self.assertRaises(ValueError):full_catalog_loss(scores,torch.tensor([target,2]))


class ModelV2Tests(unittest.TestCase):
    def test_padding_empty_history_and_biased_export_score(self):
        from src.retrieval_v2.model import TwoTowerV2
        torch.manual_seed(8)
        model=TwoTowerV2(3,5,embedding_dim=8,history_mode='recency',item_bias=True)
        with torch.no_grad():model.item_bias.weight[3]=.7
        users=torch.tensor([1,0]);tabs=torch.tensor([1,2]);items=torch.tensor([3,4])
        a=model.encode_user(users,tabs,torch.tensor([[1,2,0],[0,0,0]]))
        b=model.encode_user(users,tabs,torch.tensor([[1,2,0,0,0],[0,0,0,0,0]]))
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all())
        direct=model(users,items,tabs,torch.tensor([[1,2,0],[0,0,0]]))
        dot=(a*model.encode_item(items)).sum(1)/model.temperature
        torch.testing.assert_close(direct,dot)
        self.assertEqual(a.shape[1],9)

    def test_recency_is_order_sensitive_and_none_ignores_history(self):
        from src.retrieval_v2.model import TwoTowerV2
        torch.manual_seed(3);m=TwoTowerV2(2,4,embedding_dim=8,history_mode='recency')
        uid=torch.tensor([1]);tab=torch.tensor([1]);a=torch.tensor([[1,2,0]]);b=torch.tensor([[2,1,0]])
        self.assertFalse(torch.allclose(m.encode_user(uid,tab,a),m.encode_user(uid,tab,b)))
        n=TwoTowerV2(2,4,embedding_dim=8,history_mode='none')
        torch.testing.assert_close(n.encode_user(uid,tab,a),n.encode_user(uid,tab,b),rtol=0,atol=0)
        with self.assertRaises(ValueError):TwoTowerV2(2,4,history_mode='invalid')


class TrainingV2Tests(unittest.TestCase):
    def test_real_training_reload_and_input_config_mutation_rejected(self):
        from src.retrieval_v2.train import train_run,load_model
        class Fixture(dict):
            def __len__(self):return len(self['label'])
        t=Fixture(user_index=np.array([1,1,2,2]*8),item_index=np.array([1,3,2,4]*8),
                  tab=np.ones(32,dtype=np.int64),label=np.array([1,0,1,0]*8),
                  history=np.array([[2,0],[2,0],[1,0],[1,0]]*8),
                  user_id=np.array([1,1,2,2]*8),event_id=np.arange(32))
        t.metadata={'signature':{'fixture':True}}
        cfg={'model':'residual','history_mode':'recency','item_bias':True,'embedding_dim':8,
             'output_dim':8,'hidden_dim':16,'temperature':.1,'device':'cpu','objective':'full',
             'batch_size':8,'eval_batch_size':8,'epochs':2,'patience':3,'learning_rate':.01,
             'weight_decay':0.,'contrastive_weight':1.,'contrastive_max_queries':8,'history_length':2}
        with tempfile.TemporaryDirectory() as d:
            result=train_run(t,t,np.array([0,8,8,0,0]),cfg,Path(d),42)
            self.assertTrue(result['parameters_changed']);self.assertTrue(result['checkpoint_reload_identical'])
            model,_=load_model(Path(d)/'best.pt','cpu')
            self.assertTrue(torch.isfinite(model.encode_item(torch.tensor([1,2]))).all())
            from src.retrieval_v2.finalize import verify_completed_trial
            trial={'config':cfg,'result_path':str(Path(d)/'training.json'),
                   'best_validation':result['best_validation'],'checkpoint_sha256':result['checkpoint_sha256'],
                   'trained_parameter_sha256':result['trained_parameter_sha256']}
            verify_completed_trial(trial,t,t)
            with self.assertRaises(ValueError):verify_completed_trial(trial|{'config':cfg|{'temperature':1.}},t,t)
            with self.assertRaises(ValueError):verify_completed_trial(trial|{'best_validation':{'ndcg@20':1.}},t,t)
            t.metadata={'signature':{'fixture':'changed'}}
            with self.assertRaises(ValueError):verify_completed_trial(trial,t,t)
            t.metadata={'signature':{'fixture':True}}
            import json
            metadata=Path(d)/'training.json';saved=metadata.read_text()
            wrong=json.loads(saved);wrong['signature']['code']['v2']['model.py']='changed'
            metadata.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):verify_completed_trial(trial,t,t)
            metadata.write_text(saved)
            with self.assertRaises(ValueError):train_run(t,t,np.array([0,8,8,0,0]),cfg|{'temperature':1.},Path(d),42)
            with self.assertRaises(ValueError):train_run(t,t,np.array([0,8,8,0,0]),cfg|{'objective':'bad'},Path(d)/'bad',42)


class FrozenV2Tests(unittest.TestCase):
    def test_frozen_evidence_rejects_changed_model_input_and_config(self):
        from src.retrieval_v2.provenance import freeze_evidence,verify_evidence
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code=root/'model.py';data=root/'data.json';weight=root/'best.pt'
            code.write_text('model-v1');data.write_text('input-v1');weight.write_bytes(b'weights')
            config={'temperature':.1}
            frozen=freeze_evidence(root,config,[code,data,weight])
            verify_evidence(root,frozen,config)
            with self.assertRaises(ValueError):verify_evidence(root,frozen,{'temperature':1.})
            for path in [code,data,weight]:
                old=path.read_bytes();path.write_bytes(b'changed')
                with self.assertRaises(ValueError):verify_evidence(root,frozen,config)
                path.write_bytes(old)


class ExportV2Tests(unittest.TestCase):
    def test_candidate_validation_rejects_padding_duplicates_and_wrong_ties(self):
        from src.retrieval_v2.verify import validate_candidates
        raw=np.array([-1,101,103,107])
        ids=np.array([[2,1,3]]);scores=np.array([[2.,1.,1.]],dtype=np.float32)
        validate_candidates(ids,raw[ids],scores,raw)
        for broken,values in [(np.array([[0,1,3]]),scores),(np.array([[2,2,3]]),scores),
                              (np.array([[2,3,1]]),scores)]:
            with self.assertRaises(ValueError):validate_candidates(broken,raw[broken],values,raw)


class ReproductionPathTests(unittest.TestCase):
    def test_path_spellings_have_one_config_and_frozen_outputs_are_rejected(self):
        from scripts.train_retrieval_v2 import reproduction_config
        base={'temperature':.1}
        a,path_a=reproduction_config(base,'experiments/reproduce_fixture/seed42')
        b,path_b=reproduction_config(base,r'experiments\reproduce_fixture\seed42')
        self.assertEqual(a,b);self.assertEqual(path_a,path_b)
        self.assertEqual(a['output_directory'],'experiments/reproduce_fixture/seed42')
        self.assertEqual(base,{'temperature':.1})
        for value in ['experiments/retrieval_v2/seed42','experiments/retrieval_selected',
                      'experiments/retrieval','data/processed/kuairand_pure','../outside']:
            with self.assertRaises(ValueError):reproduction_config(base,value)


if __name__=='__main__':unittest.main()
