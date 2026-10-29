"""Full positive-event denominator, unchanged frozen candidates, distinct CTR head."""
import time
import numpy as np
import torch
from src.retrieval.data import tensor_batch
from src.ranking.metrics import exposure_metrics
from src.ranking.evaluate import rank_scores


@torch.no_grad()
def assess(model,cache,candidates,config,artifact=None):
    model.eval();device=next(model.parameters()).device;start=time.perf_counter()
    ctr=np.empty(len(cache),dtype=np.float32)
    for start_row in range(0,len(cache),config['eval_batch_size']):
        idx=np.arange(start_row,min(start_row+config['eval_batch_size'],len(cache)));b=tensor_batch(cache,idx,device)
        ctr[idx]=model(b['users'],b['items'],b['tabs'],b['history'])['ctr'].cpu().numpy()
    pointwise=exposure_metrics(cache['label'],ctr)
    scores=np.empty(candidates['ids'].shape,dtype=np.float32);correction_sum=0.;maximum=0.
    changed=0;changed_set=0
    for start_row in range(0,len(candidates),config['candidate_batch_size']):
        stop=min(start_row+config['candidate_batch_size'],len(candidates));indices=candidates['indices'][start_row:stop]
        b=tensor_batch(cache,indices,device);ids=np.array(candidates['ids'][start_row:stop],copy=True);k=ids.shape[1]
        canon=np.array(candidates['scores'][start_row:stop],copy=True)
        output=model(b['users'].repeat_interleave(k),torch.as_tensor(ids.flatten(),device=device,dtype=torch.long),
                     b['tabs'].repeat_interleave(k),b['history'].repeat_interleave(k,dim=0),
                     base_scores=torch.as_tensor(canon.flatten(),device=device) if model.teacher is not None else None)
        value=output['rank'].reshape(len(indices),k).cpu().numpy();scores[start_row:stop]=value
        correction=np.abs(value-canon);correction_sum+=float(correction.sum());maximum=max(maximum,float(correction.max()))
        order=np.lexsort((ids,-value),axis=1);top=np.take_along_axis(ids,order,axis=1)[:,:min(20,k)]
        changed+=int((top!=ids[:,:min(20,k)]).any(1).sum())
        changed_set+=int((np.sort(top,axis=1)!=np.sort(ids[:,:min(20,k)],axis=1)).any(1).sum())
    ranking=rank_scores(candidates,cache,scores,artifact)
    return {'pointwise':pointwise,'ranking':ranking,'seconds':time.perf_counter()-start,
            'mean_abs_score_correction':correction_sum/scores.size,'max_abs_score_correction':maximum,
            'top20_order_changed_fraction':changed/len(candidates),'top20_set_changed_fraction':changed_set/len(candidates)}
