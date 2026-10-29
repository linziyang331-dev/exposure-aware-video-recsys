"""Explicit natural-event/uniform-item risk mixture and train-only logit adjustment."""
import numpy as np
import torch
from torch.nn import functional as F


def mixture_weights(counts,rho=0.):
    c=np.asarray(counts,dtype=np.float64)
    if (c.ndim!=1 or len(c)<2 or c[0]!=0 or (c<0).any() or not np.isfinite(c).all()
            or c.sum()<=0 or not np.isfinite(rho) or not 0<=rho<=1):raise ValueError('Training counts and rho in[0,1] required')
    known=c>0;w=np.zeros(len(c),dtype=np.float32)
    w[known]=(1-rho)+rho*c.sum()/(known.sum()*c[known])
    return w


def choice_loss(scores,targets,weights,logprior,tau=0.):
    if (scores.ndim!=2 or targets.shape!=(len(scores),) or not len(scores) or weights.shape!=(scores.shape[1]+1,)
            or logprior.shape!=(scores.shape[1],) or (targets<=0).any() or (targets>scores.shape[1]).any()
            or tau<0 or not torch.isfinite(weights).all() or (weights<0).any() or not torch.isfinite(logprior).all()):
        raise ValueError('Finite aligned full-choice inputs required')
    loss=F.cross_entropy(scores+tau*logprior[None,:],targets-1,reduction='none')
    return (loss*weights[targets]).mean()
