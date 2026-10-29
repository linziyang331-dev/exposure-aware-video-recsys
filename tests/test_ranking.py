import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch


class RankingMetricTests(unittest.TestCase):
    def test_baseline_rank_artifact_and_mrr_are_verified(self):
        from src.ranking.verify import verify_rank_file
        class Candidates(dict):
            def __len__(self):return len(self['v2_ranks'])
        c=Candidates(v2_ranks=np.array([1,0]),event_ids=np.array([11,12]),user_ids=np.array([8,9]))
        metrics={'recall@10':.5,'recall@20':.5,'ndcg@10':.5,'ndcg@20':.5,'recall@100':.5,'mrr@100':.5}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'baseline.npz';np.savez(path,ranks=np.array([1,0]),users=np.array([8,9]),event_ids=np.array([11,12]))
            self.assertEqual(len(verify_rank_file(path,c,metrics,expected_ranks=c['v2_ranks'])),64)
            with self.assertRaises(ValueError):verify_rank_file(path,c,metrics|{'mrr@100':.1},expected_ranks=c['v2_ranks'])
            np.savez(path,ranks=np.array([0,0]),users=np.array([8,9]),event_ids=np.array([11,12]))
            with self.assertRaises((ValueError,AssertionError)):verify_rank_file(path,c,metrics,expected_ranks=c['v2_ranks'])

    def test_pointwise_auc_and_logloss_hand_example(self):
        from src.ranking.metrics import exposure_metrics
        labels=np.array([0,1,1,0]);logits=np.array([-math.log(9),math.log(9),0.,0.])
        m=exposure_metrics(labels,logits)
        self.assertAlmostEqual(m['auc'],.875)
        self.assertAlmostEqual(m['logloss'],.3992538481088858)
        with self.assertRaises(ValueError):exposure_metrics(labels,np.array([0.,np.nan,0.,0.]))
        self.assertIsNone(exposure_metrics(np.array([0,0]),np.array([0.,0.]))['auc'])

    def test_missing_target_stays_zero_and_ties_use_video_id(self):
        from src.ranking.metrics import candidate_ranks
        ids=torch.tensor([[3,2,1],[1,4,5],[1,2,3]])
        scores=torch.tensor([[1.,2.,2.],[3.,1.,0.],[2.,1.,0.]])
        ranks=candidate_ranks(ids,scores,torch.tensor([1,2,0]))
        self.assertEqual(ranks.tolist(),[1,0,0])
        with self.assertRaises(ValueError):candidate_ranks(torch.tensor([[1,1]]),torch.zeros(1,2),torch.tensor([1]))

    def test_full_evaluator_keeps_membership_and_ignores_logged_target_and_label(self):
        from src.ranking.evaluate import candidate_logits,rank_scores
        from src.ranking.model import RankingModel
        class Cache(dict):
            def __len__(self):return len(self['label'])
        class Candidates(dict):
            def __len__(self):return len(self['indices'])
        cache=Cache(user_index=np.array([1,2]),item_index=np.array([1,140]),tab=np.array([1,1]),
                    history=np.array([[2,3,0,0],[1,0,0,0]]),label=np.ones(2,dtype=np.int8))
        ids=np.tile(np.arange(1,101,dtype=np.int32),(2,1))
        candidates=Candidates(ids=ids,scores=np.tile(-np.arange(100,dtype=np.float32),(2,1)),
                              indices=np.array([0,1]),event_ids=np.array([11,12]),user_ids=np.array([8,9]),v2_ranks=np.array([1,0]))
        cfg={'kind':'din','embedding_dim':8,'hidden_sizes':[16,8],'history_length':4,'dropout':0.,
             'context_bias':True,'device':'cpu','pretrained_items':False}
        model=RankingModel(2,150,cfg).eval()
        logits=candidate_logits(model,cache,candidates,batch_size=1)
        changed=Cache(cache|{'item_index':np.zeros(2,dtype=np.int64),'label':np.zeros(2,dtype=np.int8)})
        np.testing.assert_array_equal(logits,candidate_logits(model,changed,candidates,batch_size=1))
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);(directory/'validation_ranks').mkdir()
            result=rank_scores(candidates,cache,logits,directory/'validation_ranks/pure.npz')
            self.assertEqual(result['recall@100'],.5);self.assertEqual(result['queries'],2)
            self.assertEqual(result['retrieval_miss_queries'],1)
            with np.load(directory/'validation_ranks/pure.npz') as z:
                np.testing.assert_array_equal(z['event_ids'],[11,12])
                self.assertGreater(z['ranks'][0],0);self.assertEqual(z['ranks'][1],0)
            from src.ranking.finalize import validate_saved_predictions
            from src.data.download import digest_file
            import json
            cfg=cfg|{'candidate_query_batch_size':1}
            path=directory/'best.pt'
            torch.save({'architecture':{'num_users':2,'num_items':150,'config':cfg},'state':model.state_dict()},path)
            (directory/'training.json').write_text(json.dumps({'checkpoint':str(path)}))
            np.save(directory/'validation_candidate_logits.npy',logits)
            candidates.metadata={'signature':{'fixture':True}}
            trial={'training_result_path':str(directory/'training.json'),'config':cfg,'checkpoint_sha256':digest_file(path),
                   'candidate_validation':{'results':{'pure':result}}}
            validate_saved_predictions(trial,cache,candidates)
            corrupt=logits.copy();corrupt[0,0]+=1;np.save(directory/'validation_candidate_logits.npy',corrupt)
            with self.assertRaises(ValueError):validate_saved_predictions(trial,cache,candidates)


class UpstreamRankingTests(unittest.TestCase):
    def config(self,kind='din'):
        return {'kind':kind,'embedding_dim':8,'hidden_sizes':[16,8],'history_length':4,
                'dropout':0.,'context_bias':True,'device':'cpu','pretrained_items':False}

    def test_calls_actual_upstream_models_and_prediction_ignores_labels(self):
        from src.ranking.model import RankingModel
        from recbole.model.sequential_recommender.din import DIN
        from recbole.model.context_aware_recommender.lr import LR
        users=torch.tensor([1,2]);items=torch.tensor([2,3]);tabs=torch.tensor([1,2]);h=torch.tensor([[1,4,0,0],[2,0,0,0]])
        for kind,cls in [('din',DIN),('lr',LR)]:
            model=RankingModel(3,6,self.config(kind)).eval()
            self.assertIsInstance(model.core,cls)
            output=model(users,items,tabs,h)
            self.assertEqual(tuple(output.shape),(2,));self.assertTrue(torch.isfinite(output).all())

    def test_candidate_conditioned_attention_empty_history_and_padding(self):
        from src.ranking.model import RankingModel
        torch.manual_seed(7);model=RankingModel(3,6,self.config()).eval()
        users=torch.tensor([1,1]);tabs=torch.tensor([1,1]);history=torch.tensor([[1,4,0,0],[1,4,0,0]])
        captured=[]
        hook=model.core.attention.register_forward_hook(lambda module,args,out:captured.append(out.detach().clone()))
        model(users,torch.tensor([2,3]),tabs,history)
        self.assertFalse(torch.allclose(captured[-1][0],captured[-1][1]))
        model(users,torch.tensor([2,3]),tabs,torch.zeros_like(history))
        torch.testing.assert_close(captured[-1],torch.zeros_like(captured[-1]),rtol=0,atol=0)
        a=model(users,torch.tensor([2,3]),tabs,history)
        b=model(users,torch.tensor([2,3]),tabs,torch.cat([history,torch.zeros(2,3,dtype=torch.long)],1))
        torch.testing.assert_close(a,b,rtol=0,atol=0);hook.remove()

    def test_history_adapter_keeps_latest_not_first_and_pads_empty(self):
        from src.ranking.model import recent_history
        h=torch.tensor([[1,2,3,4,0],[5,0,0,0,0],[0,0,0,0,0]])
        self.assertEqual(recent_history(h,2).tolist(),[[3,4],[5,0],[0,0]])
        self.assertEqual(recent_history(torch.tensor([[1,2]]),4).tolist(),[[1,2,0,0]])


class RankingTrainingTests(unittest.TestCase):
    def test_real_training_reload_and_config_mutation(self):
        from src.ranking.train import train_run,load_model
        class Fixture(dict):
            def __len__(self):return len(self['label'])
        train=Fixture(user_index=np.array([1,1,2,2]*8),item_index=np.array([1,3,2,4]*8),
                      tab=np.array([1,1,2,2]*8),label=np.array([1,0,1,0]*8),
                      history=np.array([[2,0,0,0],[2,0,0,0],[1,0,0,0],[1,0,0,0]]*8),
                      user_id=np.array([1,1,2,2]*8),event_id=np.arange(32))
        train.metadata={'signature':{'fixture':True}}
        cases=[('lr','cpu'),('din','cpu')]+([('din','cuda')] if torch.cuda.is_available() else [])
        for kind,device in cases:
            cfg={'kind':kind,'embedding_dim':8,'hidden_sizes':[16,8],'history_length':4,'dropout':0.,
                 'context_bias':True,'device':device,'pretrained_items':False,'epochs':2,'patience':2,
                 'learning_rate':.01,'weight_decay':0.,'batch_size':8,'eval_batch_size':16}
            with tempfile.TemporaryDirectory() as d:
                result=train_run(train,train,3,4,cfg,Path(d),42)
                self.assertTrue(result['parameters_changed']);self.assertTrue(result['reload_identical'])
                from src.ranking.finalize import verify_training_signature
                trial={'config':cfg,'training_result_path':str(Path(d)/'training.json'),
                       'pointwise_validation':result['best_validation'],'checkpoint_sha256':result['checkpoint_sha256']}
                verify_training_signature(trial,train,train,{'checkpoint_sha256':'fixture'})
                with self.assertRaises(ValueError):verify_training_signature(trial|{'config':cfg|{'dropout':.2}},train,train,{'checkpoint_sha256':'fixture'})
                model,_=load_model(Path(d)/'best.pt','cpu')
                self.assertTrue(torch.isfinite(model(torch.tensor([1]),torch.tensor([2]),torch.tensor([1]),torch.tensor([[1,0,0,0]]))).all())
                with self.assertRaises(ValueError):train_run(train,train,3,4,cfg|{'history_length':2},Path(d),42)


if __name__=='__main__':unittest.main()
