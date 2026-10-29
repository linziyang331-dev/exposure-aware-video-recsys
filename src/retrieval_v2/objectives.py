"""Observed-choice objectives; unobserved alternatives are not feedback labels."""
import torch
from torch.nn import functional as F


def full_catalog_loss(scores,targets):
    if scores.ndim!=2 or targets.ndim!=1 or len(scores)!=len(targets):
        raise ValueError('Aligned scores and targets required')
    if not len(targets) or (targets<=0).any() or (targets>scores.shape[1]).any():
        raise ValueError('Targets must be known one-based catalog IDs')
    return F.cross_entropy(scores,targets-1)
