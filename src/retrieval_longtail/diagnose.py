"""Train/validation-only frequency, positive-user support and score-gap audit."""
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval_v2.train import load_model
from src.retrieval.evaluate import Scorer
from src.retrieval_longtail.contracts import group_masks,summarize
from src.retrieval_longtail.train import read_json,write_json
from src.retrieval_longtail.run import setup,old_integrity


def diagnose(cfg,output):
    if (output/'diagnosis.json').exists():raise ValueError('Diagnosis already saved; preserve existing evidence')
    counts,caches=setup(cfg,['standard/train','standard/validation'])
    try:
        train=caches['standard/train'];val=caches['standard/validation'];idx=np.flatnonzero(val['label']==1)
        path=ROOT/'experiments/retrieval_v2/artifacts/ranks/two_tower_v2_seed42/standard_validation.npz'
        with np.load(path) as a:
            if not np.array_equal(a['event_ids'],val['event_id'][idx]):raise ValueError('Old validation event alignment differs')
            ranks=np.array(a['ranks']);users=np.array(a['users'])
        metrics=summarize(ranks,users,val['item_index'][idx],val['history'][idx],counts)
        masks=group_masks(val['item_index'][idx],val['history'][idx],counts)
        positive=np.flatnonzero(train['label']==1)
        pairs=np.unique(np.column_stack([train['item_index'][positive],train['user_index'][positive]]),axis=0)
        unique_users=np.bincount(pairs[:,0],minlength=len(counts))
        item_masks=group_masks(np.arange(1,len(counts)),np.zeros((len(counts)-1,1),dtype=int),counts)
        support={}
        for name in ['head_target','tail_target','freq_zero','freq_1_5','freq_6_20','freq_21_100','freq_101_plus']:
            c=counts[1:][item_masks[name]];u=unique_users[1:][item_masks[name]]
            support[name]={'catalog_items':len(c),'training_positive_events':int(c.sum()),
                'median_positive_count':float(np.median(c)) if len(c) else None,
                'median_distinct_positive_users':float(np.median(u)) if len(u) else None}
        fixed=old_integrity();model,_=load_model(ROOT/fixed['checkpoint'],cfg['device'])
        scorer=Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),cfg['device'],model=model)
        rng=np.random.default_rng(20261029);gaps={};probe_artifacts={}
        for name in ['head_target','tail_target','freq_zero','freq_1_5','freq_6_20','freq_21_100','freq_101_plus']:
            chosen=np.flatnonzero(masks[name]);chosen=np.sort(rng.choice(chosen,min(512,len(chosen)),replace=False))
            if not len(chosen):continue
            rows=idx[chosen];scores=scorer.scores(val,rows)
            target=torch.tensor(np.array(val['item_index'][rows]),device=scorer.device,dtype=torch.long)
            actual=scores.gather(1,(target-1)[:,None]).squeeze(1)
            threshold=torch.topk(scores,100,dim=1).values
            gap20=(actual-threshold[:,19]).cpu().numpy();gap100=(actual-threshold[:,99]).cpu().numpy()
            gaps[name]={'queries':len(rows),'sampling':'fixed-seed within train-defined group, at most512',
                'score_gap20_quantiles':np.quantile(gap20,[0,.25,.5,.75,1]).tolist(),
                'score_gap100_quantiles':np.quantile(gap100,[0,.25,.5,.75,1]).tolist()}
            probe_artifacts[name]={'event_ids':val['event_id'][rows].tolist(),'gap20':gap20.tolist(),'gap100':gap100.tolist()}
        result={'scope':'standard train and validation only','test_partitions_loaded':False,
            'catalog_items':len(counts)-1,'positive_training_events':int(counts.sum()),
            'training_positive_count_quantiles':np.quantile(counts[1:],[0,.2,.5,.8,.95,1]).tolist(),
            'catalog_support':support,'baseline_validation':metrics,'sample_score_gaps':gaps,'score_gap_probes':probe_artifacts,
            'validation_rank_sha256':digest_file(path),'validation_cache_signature':val.metadata['signature'],
            'training_cache_signature':train.metadata['signature'],
            'interpretation':'Sparsity is observed; score gaps and frequency associations alone do not establish causality.'}
        write_json(output/'diagnosis.json',result)
        print('DIAGNOSED train/validation only: '+str(support),flush=True)
        return result
    finally:
        for c in caches.values():c.close()
