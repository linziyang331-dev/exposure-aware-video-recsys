"""Real boundary tests for NumPy 2 configuration and trusted checkpoint loading."""
import tempfile
import unittest
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def apply_project_compatibility():
    # The first RED run deliberately exercises the unadapted upstream library.
    if Path('src/compat/recbole.py').exists():
        from src.compat.recbole import apply_compatibility
        apply_compatibility()


class RecBoleCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        apply_project_compatibility()

    def test_all_target_models_construct_config_with_numpy2(self):
        from recbole.config import Config
        for name in ['DSSM', 'DIN', 'SASRec']:
            with self.subTest(model=name):
                try:
                    options = {'use_gpu': False}
                    if name == 'SASRec':
                        # Full-softmax CE uses the catalog, not sampled negatives.
                        options['train_neg_sample_args'] = None
                    config = Config(model=name, dataset='ml-100k', config_dict=options)
                except Exception as error:
                    self.fail(f'{name} configuration failed on current packages: {error}')
                self.assertEqual(config['model'], name)

    def test_checkpoint_preserves_config_and_tensor(self):
        from recbole.config import Config
        from src.compat.recbole import load_run_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            config = Config(model='BPR', dataset='ml-100k', config_dict={'use_gpu': False})
            torch.save({'config': config, 'state_dict': {'weight': torch.tensor([1., 2.])}}, path)
            actual = load_run_checkpoint(path, Path(directory), device='cpu')
            self.assertEqual(actual['config']['model'], 'BPR')
            torch.testing.assert_close(actual['state_dict']['weight'], torch.tensor([1., 2.]))

    def test_checkpoint_outside_run_directory_is_rejected(self):
        from src.compat.recbole import load_run_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            path = root / 'outside.pth'
            torch.save({'weight': torch.tensor([1.])}, path)
            with self.assertRaisesRegex(ValueError, 'run directory'):
                load_run_checkpoint(path, run, device='cpu')

    def test_missing_features_are_filled_in_original_dataframe(self):
        from recbole.data.dataset import Dataset
        from recbole.utils import FeatureType
        dataset = object.__new__(Dataset)
        dataset.logger = logging.getLogger(__name__)
        dataset.feat_name_list = ['inter_feat']
        dataset.field2type = {'user_id': FeatureType.TOKEN, 'score': FeatureType.FLOAT}
        dataset.inter_feat = pd.DataFrame({'user_id': [1., np.nan, 3.], 'score': [2., np.nan, 4.]})
        dataset._fill_nan()
        self.assertEqual(dataset.inter_feat['user_id'].tolist(), [1., 0., 3.])
        self.assertEqual(dataset.inter_feat['score'].tolist(), [2., 3., 4.])

    def test_interaction_tensor_does_not_mutate_source_dataframe(self):
        from recbole.data.dataset import Dataset
        from recbole.utils import FeatureType
        dataset = object.__new__(Dataset)
        dataset.field2type = {'item_id': FeatureType.TOKEN}
        frame = pd.DataFrame({'item_id': np.array([1, 2, 3], dtype=np.int64)})
        interaction = dataset._dataframe_to_interaction(frame)
        interaction['item_id'][0] = 99
        self.assertEqual(frame['item_id'].tolist(), [1, 2, 3])

    def test_blank_sequence_feature_is_loaded_as_empty_sequence(self):
        from recbole.data.dataset import Dataset
        from recbole.utils import FeatureSource
        dataset = object.__new__(Dataset)
        dataset.logger = logging.getLogger(__name__)
        dataset.config = {'load_col': {'item': ['item_id', 'genres']}, 'unload_col': None,
                          'field_separator': '\t', 'seq_separator': ' ', 'encoding': 'utf-8', 'seq_len': None}
        dataset.field2type, dataset.field2source = {}, {}
        dataset.field2seqlen, dataset.field2bucketnum = {}, {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.item'
            path.write_text('item_id:token\tgenres:token_seq\nv1\ta b\nv2\t\n', encoding='utf-8')
            frame = dataset._load_feat(str(path), FeatureSource.ITEM)
        self.assertEqual(frame['genres'].iloc[0].tolist(), ['a', 'b'])
        self.assertEqual(frame['genres'].iloc[1].tolist(), [])


if __name__ == '__main__':
    unittest.main()
