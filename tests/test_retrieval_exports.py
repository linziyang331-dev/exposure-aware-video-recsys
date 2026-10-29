import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch


class ProvenanceTests(unittest.TestCase):
    def test_vocabulary_mutation_rejected(self):
        from src.retrieval.provenance import read_verified_vocabulary
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'vocabulary.json';path.write_text('{"count":1}')
            (root/'manifest.json').write_text(json.dumps({'vocabulary_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
            self.assertEqual(read_verified_vocabulary(root),{'count':1})
            path.write_text('{"count":2}')
            with self.assertRaisesRegex(ValueError,'vocabulary hash'):read_verified_vocabulary(root)

    def test_frozen_code_mutation_rejected(self):
        from src.retrieval.provenance import verify_code_hashes
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'model.py';path.write_text('x=1\n')
            expected={'model.py':hashlib.sha256(path.read_bytes()).hexdigest()}
            verify_code_hashes(expected,root)
            path.write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'code changed'):verify_code_hashes(expected,root)


class VectorExportTests(unittest.TestCase):
    def test_snapshot_uses_train_end_history_and_scores_exported_vectors(self):
        from src.retrieval.candidates import export_snapshot
        from src.retrieval.model import TwoTower
        import pyarrow.parquet as pq
        class Fixture(dict):
            def __len__(self):return len(self['label'])
        train=Fixture(user_index=np.array([1,1,1,2,2]),item_index=np.array([1,3,2,3,1]),
                      label=np.array([1,0,1,1,0]),time_ms=np.array([100,110,120,100,110]),
                      tab=np.array([1,1,2,1,1]),history=np.zeros((5,2),dtype=np.int32))
        vocabulary={'user_raw_to_index':{'100':1,'200':2},'item_raw_to_index':{'50':1,'60':2,'70':3}}
        model=TwoTower(2,3,embedding_dim=8,output_dim=4);model.eval()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=export_snapshot(model,train,vocabulary,200,root,topk=2,batch_size=2)
            with np.load(root/'user_vectors.npz') as users,np.load(root/'item_vectors.npz') as items:
                self.assertEqual(users['history'].tolist(),[[1,2],[3,0]])
                self.assertEqual(users['tabs'].tolist(),[2,1])
                scores=users['vectors']@items['vectors'].T/model.temperature
            rows=pq.read_table(root/'snapshot_candidates.parquet').to_pylist()
            for i,row in enumerate(rows):
                indices=np.array(row['candidate_item_indices'])
                self.assertNotIn(0,indices.tolist())
                np.testing.assert_allclose(row['scores'],scores[i,indices-1],rtol=1e-5,atol=1e-6)
            self.assertEqual(result['users'],2)
            with self.assertRaisesRegex(ValueError,'cutoff'):
                export_snapshot(model,train,vocabulary,110,root/'bad',topk=2)
