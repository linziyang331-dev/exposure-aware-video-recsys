"""Standard exposure metrics and exact per-query fixed-candidate ranks."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def exposure_metrics(labels,logits):
    labels=np.asarray(labels);logits=np.asarray(logits,dtype=np.float64)
    if labels.shape!=logits.shape or not len(labels) or not np.isin(labels,[0,1]).all() or not np.isfinite(logits).all():
        raise ValueError('Aligned finite logits and binary observed labels required')
    return {'queries':len(labels),'positives':int(labels.sum()),'positive_rate':float(labels.mean()),
            'auc':float(roc_auc_score(labels,logits)) if len(np.unique(labels))==2 else None,
            'logloss':float((np.logaddexp(0,logits)-labels*logits).mean())}


def candidate_ranks(ids,scores,targets):
    if ids.ndim!=2 or ids.shape!=scores.shape or len(targets)!=len(ids):raise ValueError('Invalid candidate shapes')
    if not torch.isfinite(scores).all() or (ids<=0).any():raise ValueError('Finite scores and positive candidate IDs required')
    if (ids.sort(1).values.diff(dim=1)==0).any():raise ValueError('Duplicate candidates')
    match=ids==targets[:,None];present=match.any(1)
    target_score=scores.gather(1,match.to(torch.long).argmax(1)[:,None])
    ahead=(scores>target_score)|((scores==target_score)&(ids<targets[:,None]))
    return torch.where(present,ahead.sum(1)+1,torch.zeros_like(targets))
