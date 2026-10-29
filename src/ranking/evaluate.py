"""Observed exposure prediction and separate fixed-candidate ranking diagnostics."""
import time
from pathlib import Path
import numpy as np
import torch
from src.retrieval.data import tensor_batch
from src.retrieval.metrics import summarize_ranks
from src.ranking.metrics import candidate_ranks,exposure_metrics
from src.ranking.train import predict


def pointwise(model,cache,batch_size=2048,artifact=None):
    start=time.perf_counter();logits=predict(model,cache,batch_size)
    result=exposure_metrics(cache['label'],logits);result['seconds']=time.perf_counter()-start
    masks={'cold_user':cache['user_index']==0,'cold_item':cache['item_index']==0,
           'warm_user_and_item':(cache['user_index']>0)&(cache['item_index']>0)}
    result['groups']={name:exposure_metrics(cache['label'][mask],logits[mask]) if mask.any() else {'queries':0}
                      for name,mask in masks.items()}
    if artifact:
        artifact=Path(artifact);artifact.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(artifact,logits=logits,labels=np.array(cache['label']),event_ids=np.array(cache['event_id']))
    return result


@torch.no_grad()
def candidate_logits(model,cache,candidates,batch_size=32):
    model.eval();device=next(model.parameters()).device
    result=np.empty((len(candidates),100),dtype=np.float32)
    for start in range(0,len(candidates),batch_size):
        stop=min(start+batch_size,len(candidates));rows=candidates['indices'][start:stop]
        b=tensor_batch(cache,rows,device);ids=torch.as_tensor(np.array(candidates['ids'][start:stop],copy=True),device=device,dtype=torch.long)
        batch=len(rows);count=ids.shape[1]
        output=model(b['users'].repeat_interleave(count),ids.flatten(),b['tabs'].repeat_interleave(count),
                     b['history'].repeat_interleave(count,dim=0))
        result[start:stop]=output.reshape(batch,count).cpu().numpy()
    return result


def rank_scores(candidates,cache,scores,artifact=None):
    ranks=np.zeros(len(candidates),dtype=np.int32)
    for start in range(0,len(candidates),512):
        stop=min(start+512,len(candidates));rows=candidates['indices'][start:stop]
        ids=torch.from_numpy(np.array(candidates['ids'][start:stop],copy=True)).long()
        value=torch.from_numpy(np.array(scores[start:stop],copy=True))
        target=torch.from_numpy(np.array(cache['item_index'][rows],copy=True)).long()
        ranks[start:stop]=candidate_ranks(ids,value,target).numpy()
    present=candidates['v2_ranks']>0
    if not np.array_equal(ranks>0,present):raise ValueError('Reranking changed candidate membership')
    result=summarize_ranks(ranks,candidates['user_ids'],ks=(10,20,100))
    result['mrr@100']=result.pop('mrr');result['user_macro']['mrr@100']=result['user_macro'].pop('mrr')
    result['retrieval_miss_queries']=int((~present).sum())
    result['conditional_hit_metrics']=summarize_ranks(ranks[present],candidates['user_ids'][present],ks=(10,20)) if present.any() else {'queries':0}
    if artifact:
        artifact=Path(artifact);artifact.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(artifact,ranks=ranks,users=np.array(candidates['user_ids']),event_ids=np.array(candidates['event_ids']))
    return result


def evaluate_candidates(model,cache,candidates,alphas,batch_size=32,artifact_directory=None):
    start=time.perf_counter();logits=candidate_logits(model,cache,candidates,batch_size);results={}
    def choices():
        yield 'pure',logits
        for a in alphas:yield f'alpha_{a:g}',candidates['scores']+float(a)*logits
    for name,scores in choices():
        artifact=Path(artifact_directory)/(name+'.npz') if artifact_directory else None
        results[name]=rank_scores(candidates,cache,scores,artifact)
    return {'results':results,'scoring_seconds':time.perf_counter()-start},logits
