"""Single-positive full-catalog ranks with exact float32/ID tie ordering."""
import numpy as np
import torch


def stable_topk(scores, candidate_ids, k):
    if scores.ndim!=2 or scores.shape[1]!=len(candidate_ids) or k<=0:
        raise ValueError('Invalid scores/candidates/topk')
    if torch.isnan(scores).any():
        raise ValueError('NaN retrieval scores')
    if (candidate_ids<=0).any() or len(torch.unique(candidate_ids))!=len(candidate_ids):
        raise ValueError('Candidates must be unique positive encoded IDs')
    scores=scores.float().contiguous()
    bits=scores.view(torch.int32).to(torch.int64)
    ordered=torch.where(bits<0,-(bits & 0x7fffffff),bits)
    maximum=int(candidate_ids.max());stride=1<<maximum.bit_length()
    if maximum>=2**31:raise ValueError('Candidate ID exceeds tie-key capacity')
    keys=ordered*stride+(maximum-candidate_ids.to(torch.int64))[None,:]
    indices=keys.topk(min(k,len(candidate_ids)),dim=1,sorted=True).indices
    return candidate_ids[indices],scores.gather(1,indices)


def target_ranks(scores,candidate_ids,targets):
    matches=candidate_ids[None,:]==targets[:,None]
    present=matches.any(dim=1)
    target_score=scores.gather(1,matches.to(torch.int64).argmax(dim=1)[:,None])
    ahead=(scores>target_score)|((scores==target_score)&(candidate_ids[None,:]<targets[:,None]))
    return torch.where(present,ahead.sum(dim=1)+1,torch.zeros_like(targets))


def summarize_ranks(ranks,users,ks=(10,20)):
    ranks=np.asarray(ranks);users=np.asarray(users)
    if len(ranks)==0 or ranks.shape!=users.shape or (ranks<0).any():
        raise ValueError('Nonempty aligned nonnegative ranks/users required')
    _,inverse=np.unique(users,return_inverse=True);counts=np.bincount(inverse)
    values={}
    valid=ranks>0
    for k in ks:
        hit=valid&(ranks<=k)
        values[f'recall@{k}']=hit.astype(float)
        gain=np.zeros(len(ranks));gain[hit]=1/np.log2(ranks[hit]+1)
        values[f'ndcg@{k}']=gain
    reciprocal=np.zeros(len(ranks));reciprocal[valid]=1/ranks[valid]
    values['mrr']=reciprocal
    return {'queries':len(ranks),'users':len(counts),
            **{name:float(value.mean()) for name,value in values.items()},
            'user_macro':{name:float((np.bincount(inverse,weights=value)/counts).mean()) for name,value in values.items()}}
