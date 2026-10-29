"""Auditable scoring: candidate IDs and past context only, never current feedback."""
import numpy as np
import torch
from src.retrieval.data import tensor_batch


def deterministic():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def candidate_scores(model, cache, candidates, batch_size=256):
    model.eval(); device=next(model.parameters()).device
    scores=np.empty(candidates['ids'].shape,dtype=np.float32)
    for start in range(0,len(candidates),batch_size):
        stop=min(start+batch_size,len(candidates)); rows=candidates['indices'][start:stop]
        b=tensor_batch(cache,rows,device); ids=np.array(candidates['ids'][start:stop],copy=True)
        k=ids.shape[1]; canonical=np.array(candidates['scores'][start:stop],copy=True)
        out=model(b['users'].repeat_interleave(k),torch.as_tensor(ids.ravel(),device=device,dtype=torch.long),
                  b['tabs'].repeat_interleave(k),b['history'].repeat_interleave(k,dim=0),
                  base_scores=torch.as_tensor(canonical.ravel(),device=device) if model.teacher is not None else None)
        scores[start:stop]=out['rank'].reshape(len(rows),k).cpu().numpy()
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite candidate score')
    return scores


@torch.no_grad()
def exposure_logits(model,cache,batch_size=8192):
    model.eval();device=next(model.parameters()).device
    values=np.empty(len(cache),dtype=np.float32)
    for start in range(0,len(cache),batch_size):
        rows=np.arange(start,min(start+batch_size,len(cache)));b=tensor_batch(cache,rows,device)
        values[rows]=model(b['users'],b['items'],b['tabs'],b['history'])['ctr'].cpu().numpy()
    return values


def history_rule_scores(cache,candidates,penalty=4.):
    scores=np.array(candidates['scores'],copy=True)
    for start in range(0,len(candidates),2048):
        stop=min(start+2048,len(candidates));h=cache['history'][candidates['indices'][start:stop]]
        seen=((candidates['ids'][start:stop,:,None]==h[:,None,:]) & (h[:,None,:]>0)).any(2)
        scores[start:stop]-=float(penalty)*seen
    return scores
