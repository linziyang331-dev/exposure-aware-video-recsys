import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.retrieval_longtail.train import write_json
from src.retrieval_longtail.verify import verify_rank_file,validate_outputs,verify_reproduction


class LongtailVerifyTests(unittest.TestCase):
    def test_rank_event_mismatch_and_cold_hit_rejected(self):
        cache={'label':np.array([1,0,1]),'event_id':np.array([10,11,12]),'user_id':np.array([3,4,5]),
               'item_index':np.array([1,2,0]),'history':np.zeros((3,2),dtype=int)}
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d)/'ranks.npz'
            np.savez(p,ranks=[1,0],users=[3,5],event_ids=[10,12])
            self.assertEqual(verify_rank_file(p,cache,np.array([0,2,1]))['recall@20'],.5)
            np.savez(p,ranks=[1,0],users=[3,5],event_ids=[12,10])
            with self.assertRaises(ValueError):verify_rank_file(p,cache,np.array([0,2,1]))
            np.savez(p,ranks=[1,1],users=[3,5],event_ids=[10,12])
            with self.assertRaises(ValueError):verify_rank_file(p,cache,np.array([0,2,1]))

    def test_output_manifest_rejects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d);(p/'data.json').write_text('{}',encoding='utf-8')
            from src.data.download import digest_file
            write_json(p/'manifest.json',{'output_hashes':{'data.json':digest_file(p/'data.json')}})
            self.assertEqual(validate_outputs(p,p/'manifest.json'),1)
            (p/'data.json').write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):validate_outputs(p,p/'manifest.json')

    def test_empty_reproduction_claim_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d);write_json(p/'reproduction.json',{'status':'PASS','models':{}})
            with self.assertRaises(ValueError):verify_reproduction(p,{'selected_trial':None,'runs':{}})
            write_json(p/'manifest.json',{'output_hashes':{'../escape':'no'}})
            with self.assertRaises(ValueError):validate_outputs(p,p/'manifest.json')


if __name__=='__main__':unittest.main()
