"""Observed-choice weighting, full-denominator metrics and validation gates."""
import numpy as np
import torch
from torch.nn import functional as F
from src.retrieval.metrics import summarize_ranks


def frequency_weights(counts,beta=0.,cap=3.):
    c=np.asarray(counts,dtype=np.float64)
    if (c.ndim!=1 or len(c)<2 or not np.isfinite(c).all() or (c<0).any()
            or c[0]!=0 or c.sum()<=0 or not np.isfinite(beta) or beta<0
            or not np.isfinite(cap) or cap<1):
        raise ValueError('Nonnegative training counts, beta>=0 and cap>=1 required')
    raw=np.minimum((c.max()/np.maximum(c[1:],1))**beta,cap)
    weights=np.zeros(len(c),dtype=np.float32)
    weights[1:]=raw/(np.dot(c[1:],raw)/c.sum())
    return weights


def weighted_choice_loss(scores,targets,weights):
    if (scores.ndim!=2 or targets.ndim!=1 or len(scores)!=len(targets) or not len(targets)
            or weights.ndim!=1 or len(weights)!=scores.shape[1]+1
            or (targets<=0).any() or (targets>scores.shape[1]).any()
            or not torch.isfinite(weights).all() or (weights<0).any()):
        raise ValueError('Aligned known targets, finite nonnegative global weights required')
    return (F.cross_entropy(scores,targets-1,reduction='none')*weights[targets]).mean()


def group_masks(items,history,counts):
    items=np.asarray(items);counts=np.asarray(counts);history=np.asarray(history)
    if (items.ndim!=1 or history.ndim!=2 or len(history)!=len(items)
            or (items<0).any() or (items>=len(counts)).any()):
        raise ValueError('Aligned catalog targets/history required')
    known=items>0; head=np.zeros(len(counts),dtype=bool)
    order=np.argsort(-counts[1:],kind='stable')+1
    head[order[:int(np.ceil(.2*len(order)))]]=True
    seen=known&((history==items[:,None])&(history>0)).any(1)
    n=counts[items]
    return {'head_target':known&head[items],'tail_target':known&~head[items],
        'cold_target':~known,'freq_zero':known&(n==0),'freq_1_5':known&(n>=1)&(n<=5),
        'freq_6_20':known&(n>=6)&(n<=20),'freq_21_100':known&(n>=21)&(n<=100),
        'freq_101_plus':known&(n>100),'history_target_seen':seen,
        'history_target_unseen':known&~seen,'empty_history':(history==0).all(1)}


def summarize(ranks,users,items,history,counts):
    ranks=np.asarray(ranks);users=np.asarray(users)
    if (ranks.ndim!=1 or users.shape!=ranks.shape or len(items)!=len(ranks)
            or ranks.dtype.kind not in 'iu' or (ranks<0).any() or not len(ranks)):
        raise ValueError('Nonempty aligned integer ranks required')
    if ((np.asarray(items)==0)&(ranks!=0)).any():raise ValueError('Cold targets must remain misses')
    result=summarize_ranks(ranks,users,ks=(20,100))
    result['groups']={name:summarize_ranks(ranks[m],users[m],ks=(20,100)) if m.any() else {'queries':0}
        for name,m in group_masks(items,history,counts).items()}
    return result


def selection_passes(candidate,baseline,gate):
    keys=('recall@20','ndcg@20')
    return bool(all(np.isfinite(candidate[k]) and candidate[k]>=baseline[k]*gate['overall_retention'] for k in keys)
        and candidate['groups']['head_target']['recall@20']>=baseline['groups']['head_target']['recall@20']*gate['head_retention']
        and all(candidate['groups']['tail_target'][k]>baseline['groups']['tail_target'][k] for k in ['recall@20','recall@100']))


def choose_trial(trials,baseline,gate):
    eligible=[name for name,m in trials.items() if selection_passes(m,baseline,gate)]
    return min(eligible,key=lambda n:(-trials[n]['ndcg@20'],-trials[n]['groups']['tail_target']['recall@100'],n)) if eligible else None


def mean_metrics(metrics):
    keys=['recall@20','ndcg@20','recall@100']
    return {**{k:float(np.mean([m[k] for m in metrics])) for k in keys},
        'groups':{g:{k:float(np.mean([m['groups'][g][k] for m in metrics])) for k in keys}
                  for g in ['head_target','tail_target']}}
