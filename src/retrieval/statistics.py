"""Paired user-cluster bootstrap for event-weighted logged-query metrics."""
import numpy as np


def metric_values(ranks):
    ranks=np.asarray(ranks);valid=ranks>0;hit=valid&(ranks<=20)
    ndcg=np.zeros(len(ranks));ndcg[hit]=1/np.log2(ranks[hit]+1)
    mrr=np.zeros(len(ranks));mrr[valid]=1/ranks[valid]
    return np.column_stack([hit.astype(float),ndcg,mrr])


def paired_user_bootstrap(model_ranks,baseline_ranks,users,repeats=500,seed=42):
    if len(model_ranks)!=len(baseline_ranks) or len(users)!=len(model_ranks) or not len(users):
        raise ValueError('Aligned nonempty paired ranks/users required')
    if repeats<=0:raise ValueError('Bootstrap repeats must be positive')
    _,inverse=np.unique(users,return_inverse=True);counts=np.bincount(inverse)
    differences=metric_values(model_ranks)-metric_values(baseline_ranks)
    grouped=np.column_stack([np.bincount(inverse,weights=differences[:,i]) for i in range(3)])
    rng=np.random.default_rng(seed);draws=np.empty((repeats,3))
    for repeat in range(repeats):
        sample=rng.integers(0,len(counts),size=len(counts))
        draws[repeat]=grouped[sample].sum(axis=0)/counts[sample].sum()
    result={}
    for i,name in enumerate(['recall@20','ndcg@20','mrr']):
        lower,upper=np.quantile(draws[:,i],[.025,.975])
        result[name]={'difference':float(differences[:,i].mean()),'ci95_low':float(lower),'ci95_high':float(upper)}
    return result
