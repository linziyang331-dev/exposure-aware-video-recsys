"""Same logged-query protocol, with catalog and fixed-pool metrics kept distinct."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from src.retrieval_v2.evaluate import evaluate as original_catalog_evaluate
from src.retrieval.metrics import summarize_ranks
from src.ranking.evaluate import rank_scores


class SequenceScorer:
    def __init__(self,model,counts):
        self.model=model.eval();self.device=next(model.parameters()).device;self.kind='sasrec'
        self.candidate_ids=torch.arange(1,len(counts),device=self.device)

    @torch.no_grad()
    def scores(self,cache,indices):
        history=torch.as_tensor(np.array(cache['history'][indices],copy=True),device=self.device,dtype=torch.long)
        return self.model.scores(history)


def catalog_evaluate(model,cache,counts,batch_size=512,artifact=None):
    if artifact is None:
        with tempfile.TemporaryDirectory() as directory:
            return catalog_evaluate(model,cache,counts,batch_size,Path(directory)/'ranks.npz')
    result=original_catalog_evaluate(SequenceScorer(model,counts),cache,counts,batch_size=batch_size,artifact=artifact)
    if artifact is not None:
        with np.load(artifact) as z:
            ranks=z['ranks'];users=z['users'];positive=np.flatnonzero(cache['label']==1)
            empty=(cache['history'][positive]>0).sum(1)==0
            for name,mask in [('empty_history',empty),('nonempty_history',~empty)]:
                result['groups'][name]=summarize_ranks(ranks[mask],users[mask],ks=(10,20,100)) if mask.any() else {'queries':0}
            bounded=np.where(ranks<=100,ranks,0)
            result['mrr@100']=summarize_ranks(bounded,users)['mrr']
    return result


@torch.no_grad()
def pool_scores(model,cache,candidates,batch_size=512):
    scorer=SequenceScorer(model,model.counts);scores=np.empty(candidates['ids'].shape,dtype=np.float32)
    for start in range(0,len(candidates),batch_size):
        stop=min(start+batch_size,len(candidates));indices=candidates['indices'][start:stop]
        all_scores=scorer.scores(cache,indices)
        ids=torch.as_tensor(np.array(candidates['ids'][start:stop],copy=True),device=scorer.device,dtype=torch.long)
        scores[start:stop]=all_scores.gather(1,ids-1).cpu().numpy()
    if not np.isfinite(scores).all():raise ValueError('Nonfinite pool scores')
    return scores


def metric_identity(metrics):
    return {k:v for k,v in metrics.items() if k not in {'evaluation_seconds','queries_per_second'}}
