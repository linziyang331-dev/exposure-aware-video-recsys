import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.data.common import ROOT
from test_balanced_train import SmallCache
from src.retrieval_balanced.run import select_replicated


class BalancedReplicationTests(unittest.TestCase):
    def metric(self,tail20,tail100):return {'recall@20':.1,'ndcg@20':.04,'groups':{
        'head_target':{'recall@20':.13,'ndcg@20':.05,'recall@100':.4},
        'tail_target':{'recall@20':tail20,'ndcg@20':.005,'recall@100':tail100}},'recall@100':.3}
    def test_replicated_gate_uses_actual_equal_budget_control(self):
        groups={name:{str(s):{'best_validation':self.metric(*v)} for s in [42,43,44]}
            for name,v in [('control',(.025,.15)),('candidate',(.02,.1))]}
        _,eligible,chosen=select_replicated(groups,{'retention':.98,'tail20_delta':.002,'tail100_delta':.01})
        self.assertEqual(eligible,[]);self.assertIsNone(chosen)

    def test_graph_cache_rejects_changed_label_filter_wrapper(self):
        from src.retrieval_balanced.run import graph
        c=SmallCache();counts=np.array([0,4,4,4,0])
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            graph({},Path(d),counts,c)
            with patch('src.retrieval_balanced.run.inspect.getsource',return_value='changed graph label filtering'):
                with self.assertRaises(ValueError):graph({},Path(d),counts,c)
    def test_candidate_requires_every_seed_tail_nonregression(self):
        groups={name:{str(s):{'best_validation':self.metric(*v)} for s in [42,43,44]}
            for name,v in [('control',(.01,.05)),('candidate',(.02,.1))]}
        _,eligible,chosen=select_replicated(groups,{'retention':.98,'tail20_delta':.002,'tail100_delta':.01})
        self.assertEqual(eligible,['candidate']);self.assertEqual(chosen,'candidate')
        groups['candidate']['44']['best_validation']=self.metric(.009,.2)
        _,eligible,chosen=select_replicated(groups,{'retention':.98,'tail20_delta':.002,'tail100_delta':.01})
        self.assertEqual(eligible,[]);self.assertIsNone(chosen)
