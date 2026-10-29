"""One positive-event/full-catalog protocol for every retrieval model."""
import time
from pathlib import Path
import numpy as np
import torch
from src.retrieval.baselines import score_itemcf
from src.retrieval.data import tensor_batch
from src.retrieval.metrics import stable_topk,target_ranks,summarize_ranks


class Scorer:
    def __init__(self,kind,popularity,device='cuda',model=None,weights=None):
        self.kind=kind;self.popularity=popularity;self.device=torch.device(device)
        self.model=model;self.weights=weights
        self.candidate_ids=torch.arange(1,len(popularity)+1,device=self.device)
        self.item_vectors=None
        if model is not None:
            model.eval()
            with torch.no_grad():self.item_vectors=model.encode_item(self.candidate_ids)
    @torch.no_grad()
    def scores(self,cache,indices):
        if self.kind=='two_tower':
            batch=tensor_batch(cache,indices,self.device)
            return self.model.encode_user(batch['users'],batch['tabs'],batch['history'])@self.item_vectors.T/self.model.temperature
        if self.kind=='itemcf':
            value=score_itemcf(cache['history'][indices],self.weights,self.popularity)
        else:value=np.broadcast_to(self.popularity,(len(indices),len(self.popularity))).copy()
        return torch.from_numpy(value).to(self.device)


def evaluate_partition(scorer,cache,positive_counts,*,batch_size=512,artifact=None,query_limit=None):
    eligible=np.flatnonzero(cache['label']==1)
    if query_limit is not None:eligible=eligible[:query_limit]
    if not len(eligible):raise ValueError('No positive retrieval queries')
    users=np.array(cache['user_id'][eligible]);items=np.array(cache['item_index'][eligible])
    warm=np.asarray(cache['user_index'][eligible])>0
    ranks=np.zeros(len(eligible),dtype=np.int32);covered=np.zeros(len(positive_counts),dtype=bool)
    ordering=np.argsort(-positive_counts[1:],kind='stable')+1
    head=np.zeros(len(positive_counts),dtype=bool);head[ordering[:int(np.ceil(.2*len(ordering)))]]=True
    tail_recommendations=0;recommendations=0
    start=time.perf_counter()
    if scorer.kind=='popularity':
        inverse=np.zeros(len(positive_counts),dtype=np.int32);inverse[ordering]=np.arange(1,len(ordering)+1)
        ranks=inverse[items];chosen=ordering[:20]
        covered[chosen]=True;tail_recommendations=int((~head[chosen]).sum())*len(eligible);recommendations=len(chosen)*len(eligible)
    else:
        for offset in range(0,len(eligible),batch_size):
            indices=eligible[offset:offset+batch_size]
            scores=scorer.scores(cache,indices)
            targets=torch.from_numpy(np.array(cache['item_index'][indices],copy=True)).to(scorer.device,dtype=torch.long)
            ranks[offset:offset+len(indices)]=target_ranks(scores,scorer.candidate_ids,targets).cpu().numpy()
            top,_=stable_topk(scores,scorer.candidate_ids,20);top=top.cpu().numpy()
            covered[top.ravel()]=True;tail_recommendations+=int((~head[top]).sum());recommendations+=top.size
        if scorer.device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    metrics=summarize_ranks(ranks,users)
    metrics.update(known_item_queries=int((items>0).sum()),cold_item_queries=int((items==0).sum()),
                   cold_user_queries=int((~warm).sum()),candidate_count=len(ordering),
                   coverage_at20=float(covered[1:].mean()),
                   tail_coverage_at20=float(covered[1:][~head[1:]].mean()),
                   tail_recommendation_fraction_at20=tail_recommendations/recommendations,
                   evaluation_seconds=elapsed,queries_per_second=len(eligible)/max(elapsed,1e-9))
    masks={'warm_user_and_item':warm&(items>0),'head_target':(items>0)&head[items],
           'tail_target':(items>0)&~head[items],'cold_target':items==0,'cold_user':~warm}
    metrics['groups']={name:summarize_ranks(ranks[mask],users[mask]) if mask.any() else {'queries':0}
                       for name,mask in masks.items()}
    if artifact is not None:
        artifact=Path(artifact);artifact.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(artifact,ranks=ranks,users=users,event_ids=np.array(cache['event_id'][eligible]))
    return metrics
