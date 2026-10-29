import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.retrieval_longtail.train import train_run, load_model
from src.retrieval_longtail.run import trial_matrix, freeze_selection, load_selection


class ToyCache:
    def __init__(self):
        self.arrays={'label':np.ones(12,dtype=np.int8),'user_index':np.tile([1,2,3],4),
            'user_id':np.tile([11,12,13],4),'item_index':np.tile([1,2,3],4),
            'tab':np.zeros(12,dtype=np.int8),'history':np.tile([[1,2,0],[2,1,0],[3,1,0]],(4,1)),
            'event_id':np.arange(12)}
        self.metadata={'signature':{'fixture':'12 independent event IDs'}}
    def __getitem__(self,key):return self.arrays[key]
    def __len__(self):return 12


class LongtailRunTests(unittest.TestCase):
    def test_training_reload_and_changed_config_refused(self):
        cache=ToyCache();counts=np.array([0,4,4,4,0])
        config={'model':'gated','embedding_dim':4,'temperature':.1,'history_decay':.9,
            'beta':.25,'weight_cap':3.,'device':'cpu','learning_rate':.001,'weight_decay':1e-5,
            'batch_size':4,'eval_batch_size':4,'epochs':2,'patience':2}
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            run=train_run(cache,cache,counts,config,Path(d)/'run',42)
            self.assertTrue(run['checkpoint_reload_identical'])
            self.assertNotEqual(run['initial_parameter_sha256'],run['trained_parameter_sha256'])
            self.assertEqual(run['best_validation']['queries'],12)
            model,_=load_model(run['checkpoint'],'cpu')
            self.assertTrue(model is not None)
            again=train_run(cache,cache,counts,config,Path(d)/'run',42)
            self.assertEqual(run,again)
            with self.assertRaises(ValueError):train_run(cache,cache,counts,config|{'beta':.5},Path(d)/'run',42)
            checkpoint=Path(run['checkpoint']);checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):train_run(cache,cache,counts,config,Path(d)/'run',42)

    def test_selection_freeze_rejects_further_tuning_and_modified_trials(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d);trial=p/'trials.json';trial.write_text('{"selected_trial":null}',encoding='utf-8')
            freeze_selection(p,{'selected_trial':None},[trial])
            with self.assertRaises(ValueError):trial_matrix({'trials':[]},p)
            self.assertIsNone(load_selection(p)['selected_trial'])
            trial.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):load_selection(p)

    def test_selection_payload_cannot_diverge_from_frozen_trials(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d);trial=p/'trials.json'
            trial.write_text('{"selected_trial":null}',encoding='utf-8')
            freeze_selection(p,{'selected_trial':None},[trial])
            selection=json.loads((p/'selection.json').read_text('utf-8'))
            selection['selected_trial']='invented'
            (p/'selection.json').write_text(json.dumps(selection),encoding='utf-8')
            with self.assertRaises(ValueError):load_selection(p)

    def test_matrix_rejects_unbounded_or_duplicate_trial_names(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            for trials in [[{'name':'x'}]*2,[{'name':'../escape'}],[{'name':str(i)} for i in range(9)]]:
                with self.assertRaises(ValueError):trial_matrix({'trials':trials},Path(d))

    def test_actual_reproduction_required_and_corrupted_summary_never_overwritten(self):
        from src.retrieval_longtail.verify import _reproduction_record,verify_reproduction,reproduce
        from src.retrieval_longtail.train import write_json
        cache=ToyCache();counts=np.array([0,4,4,4,0])
        config={'model':'recency','embedding_dim':4,'temperature':.1,'history_decay':.9,
            'beta':0.,'weight_cap':3.,'device':'cpu','learning_rate':.001,'weight_decay':1e-5,
            'batch_size':4,'eval_batch_size':4,'epochs':1,'patience':1}
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d)
            old=train_run(cache,cache,counts,config,p/'original',42)
            train_run(cache,cache,counts,config,p/'reproduction/control',42)
            payload={'selected_trial':None,'config':{},'runs':{'control':old}}
            write_json(p/'trials.json',payload)
            selection=freeze_selection(p,payload,[p/'trials.json',*list((p/'original').glob('*'))])
            record=_reproduction_record(p,selection);write_json(p/'reproduction.json',record)
            self.assertEqual(verify_reproduction(p,selection),record)
            write_json(p/'reproduction.json',record|{'models':{}})
            before=(p/'reproduction.json').read_bytes()
            with self.assertRaises(ValueError):reproduce({},p)
            self.assertEqual((p/'reproduction.json').read_bytes(),before)
            changed=json.loads(json.dumps(record));changed['models']['control']['parameters_identical']=False
            write_json(p/'reproduction.json',changed)
            with self.assertRaises(ValueError):verify_reproduction(p,selection)


if __name__=='__main__':unittest.main()
