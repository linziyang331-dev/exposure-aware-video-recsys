"""Frozen candidate conditional-choice batches; alternatives have unknown feedback."""
import numpy as np


def choice_batch(cache,candidates,query_rows,rng,negatives,mask_past=True):
    rows=np.asarray(query_rows);indices=np.array(candidates['indices'][rows],copy=True)
    targets=np.array(cache['item_index'][indices],copy=True);ids=np.array(candidates['ids'][rows],copy=True)
    if (candidates['v2_ranks'][rows]<=0).any() or not (cache['label'][indices]==1).all():
        raise ValueError('Choice training requires an observed positive already in frozen candidates')
    match=ids==targets[:,None]
    if not (match.sum(1)==1).all():raise ValueError('Positive target must occur exactly once')
    history=cache['history'][indices]
    prior=(ids[:,:,None]==history[:,None,:]).any(2)
    available=(ids>0)&~match
    if mask_past:available &= ~prior
    if (available.sum(1)<negatives).any():raise ValueError('Insufficient distinct unknown competitors')
    noise=rng.random(ids.shape);noise[~available]=-1
    positions=np.argpartition(-noise,negatives-1,axis=1)[:,:negatives]
    positions=np.concatenate([match.argmax(1)[:,None],positions],axis=1)
    return {'indices':indices,'items':np.take_along_axis(ids,positions,1),
            'base_scores':np.take_along_axis(np.asarray(candidates['scores'][rows]),positions,1),
            'positive_slot':0,'competitor_semantics':'unobserved alternatives, not feedback label0'}
