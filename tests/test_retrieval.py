import math
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch


class RetrievalMetricTests(unittest.TestCase):
    def test_float32_ranking_ties_negative_zero_and_unknown_target(self):
        from src.retrieval.metrics import stable_topk, target_ranks
        scores=torch.tensor([[0.,-0.,-1.,-float('inf')],[.4,.4,.1,-.1]])
        ids=torch.tensor([4,2,3,1])
        self.assertEqual(stable_topk(scores,ids,3)[0].tolist(),[[2,4,3],[2,4,3]])
        self.assertEqual(target_ranks(scores,ids,torch.tensor([2,1])).tolist(),[1,4])
        self.assertEqual(target_ranks(scores[:1],ids,torch.tensor([0])).tolist(),[0])

    def test_single_positive_metrics_keep_cold_miss_in_denominator(self):
        from src.retrieval.metrics import summarize_ranks
        result=summarize_ranks(np.array([1,3,0]),np.array([1,1,2]),ks=(1,3))
        self.assertAlmostEqual(result['recall@1'],1/3)
        self.assertAlmostEqual(result['recall@3'],2/3)
        self.assertAlmostEqual(result['ndcg@3'],.5)
        self.assertAlmostEqual(result['mrr'],4/9)
        self.assertAlmostEqual(result['user_macro']['recall@3'],.5)

    def test_nan_scores_rejected(self):
        from src.retrieval.metrics import stable_topk
        with self.assertRaises(ValueError):stable_topk(torch.tensor([[float('nan'),1.]]),torch.tensor([1,2]),1)


class TwoTowerTests(unittest.TestCase):
    def test_padding_does_not_change_user_vector_and_pair_score_equals_dot(self):
        from src.retrieval.model import TwoTower
        torch.manual_seed(7);model=TwoTower(3,5,embedding_dim=8,output_dim=4)
        model.eval();users=torch.tensor([1]);tabs=torch.tensor([1])
        short=torch.tensor([[1,2]]);padded=torch.tensor([[1,2,0,0]])
        torch.testing.assert_close(model.encode_user(users,tabs,short),model.encode_user(users,tabs,padded))
        target=torch.tensor([3]);user=model.encode_user(users,tabs,padded);item=model.encode_item(target)
        torch.testing.assert_close(model(users,target,tabs,padded),(user*item).sum(-1)/model.temperature)

    def test_history_off_ablation_is_independent_of_history_contents(self):
        from src.retrieval.model import TwoTower
        model=TwoTower(3,5,embedding_dim=8,output_dim=4,use_history=False);model.eval()
        a=model.encode_user(torch.tensor([1]),torch.tensor([1]),torch.tensor([[1,2]]))
        b=model.encode_user(torch.tensor([1]),torch.tensor([1]),torch.tensor([[3,4]]))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_contrastive_pool_deduplicates_items_and_masks_past_positives(self):
        from src.retrieval.model import contrastive_loss
        user=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],requires_grad=True)
        unique_items=torch.tensor([[1.,0.],[0.,1.]])
        targets=torch.tensor([1,1,2]);pool=torch.tensor([1,2])
        loss=contrastive_loss(user,unique_items,targets,pool,torch.zeros((3,2),dtype=torch.long),1.)
        self.assertAlmostEqual(loss.item(),math.log1p(math.exp(-1)),places=6)
        masked=contrastive_loss(user,unique_items,targets,pool,torch.tensor([[2,0],[2,0],[1,0]]),1.)
        self.assertAlmostEqual(masked.item(),0.,places=6)
        loss.backward();self.assertTrue(torch.isfinite(user.grad).all())

    def test_weights_only_checkpoint_round_trip_preserves_vectors(self):
        from src.retrieval.model import TwoTower
        model=TwoTower(3,5,embedding_dim=8,output_dim=4);model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';torch.save(model.state_dict(),path)
            other=TwoTower(3,5,embedding_dim=8,output_dim=4)
            other.load_state_dict(torch.load(path,weights_only=True));other.eval()
            torch.testing.assert_close(model.encode_item(torch.tensor([1,4])),other.encode_item(torch.tensor([1,4])),rtol=0,atol=0)


class ItemCFTests(unittest.TestCase):
    def test_binary_fit_cosine_and_popularity_backfill(self):
        from src.retrieval.baselines import fit_itemcf, score_itemcf, popularity_scores
        users=np.array([1,1,1,2,2,3]);items=np.array([1,1,2,1,3,3])
        weights=fit_itemcf(users,items,3,3,neighbors=3)
        self.assertAlmostEqual(float(weights[1,2]),1/math.sqrt(2),places=5)
        self.assertAlmostEqual(float(weights[1,3]),.5,places=5)
        self.assertEqual(float(weights[1,1]),0.)
        pop=popularity_scores(np.array([0,2,1,2]))
        from src.retrieval.metrics import stable_topk
        scores=score_itemcf(np.array([[2,0],[0,0]]),weights,pop)
        ranked=stable_topk(torch.from_numpy(scores),torch.arange(1,4),3)[0]
        self.assertEqual(ranked[0].tolist(),[1,3,2])
        self.assertEqual(ranked[1].tolist(),[1,3,2])


class RetrievalDataTests(unittest.TestCase):
    def test_verified_parquet_cache_clips_oldest_tokens_and_reuses(self):
        from src.retrieval.data import load_partition
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared';(prepared/'standard').mkdir(parents=True)
            path=prepared/'standard/train.parquet'
            rows=[{'event_id':1,'user_id':0,'video_id':5,'user_index':1,'item_index':2,'time_ms':500,
                   'label':1,'tab':1,'history_item_indices':[1,2,3],'history_time_ms':[100,200,300]},
                  {'event_id':2,'user_id':0,'video_id':6,'user_index':1,'item_index':3,'time_ms':600,
                   'label':0,'tab':1,'history_item_indices':[],'history_time_ms':[]}]
            pq.write_table(pa.Table.from_pylist(rows),path)
            manifest={'fingerprint':'fixture','partitions':{'standard/train':{'path':'standard/train.parquet','rows':2,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            (prepared/'manifest.json').write_text(json.dumps(manifest))
            cache=load_partition(prepared,'standard/train',root/'cache',max_history=2)
            self.assertEqual(cache['history'].tolist(),[[2,3],[0,0]])
            self.assertEqual(cache['label'].tolist(),[1,0])
            self.assertEqual(load_partition(prepared,'standard/train',root/'cache',max_history=2).root,cache.root)
            cache.close()
            path.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'hash'):
                load_partition(prepared,'standard/train',root/'cache',max_history=2)


class TrainingWorkflowTests(unittest.TestCase):
    def test_optimizer_changes_weights_and_selected_checkpoint_reloads(self):
        from src.retrieval.train import train_run
        from src.retrieval.model import TwoTower
        class Fixture(dict):
            def __len__(self):return len(self['label'])
        train=Fixture(user_index=np.array([1,1,2,2]*8),item_index=np.array([1,3,2,4]*8),
                      tab=np.ones(32,dtype=np.int64),label=np.array([1,0,1,0]*8),
                      history=np.array([[1,0],[1,0],[2,0],[2,0]]*8),
                      user_id=np.array([1,1,2,2]*8),event_id=np.arange(32))
        train.metadata={'signature':{'fixture':True}}
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device),tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(7);initial=TwoTower(2,4,embedding_dim=8,output_dim=4,hidden_dim=16,temperature=1.)
                config={'device':device,'embedding_dim':8,'output_dim':4,'hidden_dim':16,'temperature':1.,
                        'learning_rate':.01,'weight_decay':0.,'batch_size':8,'eval_batch_size':8,'epochs':1,
                        'patience':2,'contrastive_max_queries':8,'contrastive_weight':.2}
                result=train_run(train,train,np.array([0,8,8,0,0]),config,Path(directory)/'run',7)
                checkpoint=torch.load(Path(directory)/'run/best.pt',map_location='cpu',weights_only=True)
                self.assertFalse(torch.equal(initial.item_embedding.weight,checkpoint['model_state']['item_embedding.weight']))
                self.assertTrue(result['checkpoint_reload_identical'])
