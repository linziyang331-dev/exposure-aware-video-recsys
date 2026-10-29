"""Context-only, frozen V2 -> Top100 -> DIN -> Top20 offline recommendation."""
import time
import hashlib
import json
import numpy as np
import torch
from src.retrieval.metrics import stable_topk

LATENCY_SCOPE='preconstructed contexts to Top20, includes host/device transfer; excludes history lookup, network and concurrency; descriptive offline timings, not online SLO'


def order_candidates(ids,scores,k):
    ids=np.asarray(ids);scores=np.asarray(scores)
    if (ids.ndim!=2 or ids.shape!=scores.shape or ids.dtype.kind not in 'iu' or not 0<k<=ids.shape[1]
            or (ids<=0).any() or not np.isfinite(scores).all() or (np.diff(np.sort(ids,axis=1),axis=1)==0).any()):
        raise ValueError('Unique positive candidates and finite aligned scores required')
    order=np.lexsort((ids,-scores),axis=1)[:,:k]
    return np.take_along_axis(ids,order,1),np.take_along_axis(scores,order,1)


class FrozenPipeline:
    def __init__(self,model,retrieval_batch=512,rerank_batch=256):
        self.model=model.eval();self.teacher=model.teacher
        self.device=next(model.parameters()).device
        self.items=self.teacher.item_embedding.num_embeddings-1
        if self.items<100 or min(retrieval_batch,rerank_batch)<1:raise ValueError('At least100 catalog items and positive batches required')
        self.retrieval_batch=retrieval_batch;self.rerank_batch=rerank_batch
        self.ids=torch.arange(1,self.items+1,device=self.device)
        with torch.no_grad():self.item_vectors=self.teacher.encode_item(self.ids)

    @torch.no_grad()
    def recommend(self,users,tabs,history):
        users=np.asarray(users);tabs=np.asarray(tabs);history=np.asarray(history)
        if (users.ndim!=1 or not len(users) or tabs.shape!=users.shape or history.shape!=(len(users),50)
                or any(a.dtype.kind not in 'iu' for a in [users,tabs,history])
                or ((users<0)|(users>=self.teacher.user_embedding.num_embeddings)).any()
                or ((tabs<0)|(tabs>=15)).any() or ((history<0)|(history>self.items)).any()):
            raise ValueError('Encoded users/tabs/strict-past history50 outside frozen vocabulary')
        candidate_ids=[];base=[]
        for start in range(0,len(users),self.retrieval_batch):
            stop=min(start+self.retrieval_batch,len(users))
            u,t,h=[torch.as_tensor(a[start:stop].copy(),device=self.device,dtype=torch.long) for a in [users,tabs,history]]
            values=self.teacher.encode_user(u,t,h)@self.item_vectors.T/self.teacher.temperature
            ids,scores=stable_topk(values,self.ids,100)
            candidate_ids.append(ids.cpu().numpy());base.append(scores.cpu().numpy())
        ids=np.concatenate(candidate_ids);base=np.concatenate(base);rank=np.empty_like(base)
        for start in range(0,len(users),self.rerank_batch):
            stop=min(start+self.rerank_batch,len(users))
            u,t,h=[torch.as_tensor(a[start:stop].copy(),device=self.device,dtype=torch.long) for a in [users,tabs,history]]
            result=self.model(u.repeat_interleave(100),torch.as_tensor(ids[start:stop].ravel(),device=self.device,dtype=torch.long),
                t.repeat_interleave(100),h.repeat_interleave(100,dim=0),
                base_scores=torch.as_tensor(base[start:stop].ravel(),device=self.device))
            rank[start:stop]=result['rank'].reshape(stop-start,100).cpu().numpy()
        top,values=order_candidates(ids,rank,20)
        return {'candidate_ids':ids,'candidate_scores':base,'rank_scores':rank,'top20_ids':top,'top20_scores':values}


def synchronized(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def latency_summary(samples,batch_size):
    a=np.asarray(samples,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or (a<=0).any():raise ValueError('Positive finite latency observations required')
    return {'measured_batches':len(a),'batch_size':batch_size,'p50_ms':float(np.quantile(a,.5)*1000),
        'p95_ms':float(np.quantile(a,.95)*1000),'p99_ms':float(np.quantile(a,.99)*1000),
        'requests_per_second':float(len(a)*batch_size/a.sum()),'raw_seconds':a.tolist()}


def benchmark(pipe,cache,config):
    result={}
    for batch in config['benchmark_batches']:
        if len(cache)<batch:raise ValueError('Insufficient benchmark context rows')
        samples=[]
        for step in range(config['warmup_batches']+config['measurement_batches']):
            rows=np.arange(step*batch,(step+1)*batch)%len(cache)
            inputs=[np.array(cache[key][rows],copy=True) for key in ['user_index','tab','history']]
            synchronized(pipe.device);start=time.perf_counter();pipe.recommend(*inputs);synchronized(pipe.device)
            elapsed=time.perf_counter()-start
            if step>=config['warmup_batches']:samples.append(elapsed)
        result[str(batch)]=latency_summary(samples,batch)
    return result


def probe_pipeline(config,directory,verify=False):
    from pathlib import Path
    from src.data.common import ROOT
    from src.ranking_v2.train import load_model
    from src.ranking_v2.inference import deterministic
    from src.ranking.candidates import load_candidates
    from src.retrieval.provenance import read_verified_vocabulary
    from src.retrieval.data import training_counts
    from src.retrieval_v2.train import parameter_digest
    from src.system_evaluation.analysis import context_caches
    from src.system_evaluation.registry import load_registry
    reg=load_registry();deterministic();directory=Path(directory)
    selection=reg['assets'].read('experiments/ranking_v2/selection.json');run=selection['primary_training']
    model,_=load_model(run['checkpoint'],config['device'])
    if parameter_digest(model.teacher)!=run['teacher_parameter_sha256']:raise ValueError('Loaded teacher differs from frozen V2')
    pipe=FrozenPipeline(model);caches=context_caches(config);checks={}
    counts=training_counts(read_verified_vocabulary(ROOT/config['prepared_directory']))
    try:
        artifact=directory/'artifacts';artifact.mkdir(exist_ok=True,parents=True)
        for part,c in caches.items():
            old=ROOT/'experiments/ranking/artifacts/candidates'/part.replace('/','_')
            if not (old/'metadata.json').is_file():raise FileNotFoundError('Frozen candidates must already exist')
            candidate=load_candidates(part,c,counts,ROOT/'experiments/ranking',reg['frozen'],config['device'])
            try:
                n=min(config['probe_queries_per_partition'],len(candidate));idx=np.array(candidate['indices'][:n])
                inputs={key:np.array(c[source][idx],copy=True) for key,source in [('users','user_index'),('tabs','tab'),('history','history')]}
                actual=pipe.recommend(**inputs)
                original_ids=np.array(candidate['ids'][:n]);original_base=np.array(candidate['scores'][:n])
                original_scores=np.array(np.load(reg['din']['results']['din_seed42'][part]['scores_artifact'],mmap_mode='r')[:n])
                if not np.array_equal(actual['candidate_ids'],original_ids):raise ValueError('Fresh full-catalog candidates differ from frozen pool')
                for name,expected in [('candidate_scores',original_base),('rank_scores',original_scores)]:
                    if not np.allclose(actual[name],expected,rtol=1e-5,atol=1e-5):raise ValueError('Pipeline scores differ from frozen scores')
                expected_top,_=order_candidates(original_ids,original_scores,20)
                if not np.array_equal(expected_top,actual['top20_ids']):raise ValueError('Fresh pipeline Top20 differs')
                path=artifact/(part.replace('/','_')+'_probe.npz')
                if verify:
                    with np.load(path) as saved:
                        if set(saved.files)!=set(inputs)|set(actual):raise ValueError('Probe fields differ')
                        for name,value in (inputs|actual).items():
                            if not np.array_equal(value,saved[name]):raise ValueError('Saved pipeline response differs: '+name)
                else:np.savez_compressed(path,**inputs,**actual)
                checks[part]={'queries':n,'candidate_membership_identical':True,'top20_ids_identical':True,
                    'max_base_score_abs_difference':float(np.max(np.abs(actual['candidate_scores']-original_base))),
                    'max_rank_score_abs_difference':float(np.max(np.abs(actual['rank_scores']-original_scores))),
                    'top20_id_sha256':hashlib.sha256(actual['top20_ids'].tobytes()).hexdigest()}
            finally:candidate.close()
        metadata={'status':'PASS','checks':checks,'input_fields':['encoded_user','tab','strict_past_history50'],
            'no_target_or_feedback_input':True,'device':str(pipe.device),'torch_version':torch.__version__,
            'hardware':torch.cuda.get_device_name(pipe.device) if pipe.device.type=='cuda' else 'CPU',
            'torch_threads':torch.get_num_threads(),'catalog_size':pipe.items,'candidate_count':100,'output_count':20,
            'scope':'sampled checkpoint-to-response replay; original full-model verification remains separate'}
        if verify:return metadata
        metadata['latency']=benchmark(pipe,caches['standard/test'],config)
        metadata['latency_scope']=LATENCY_SCOPE
        return metadata
    finally:
        for c in caches.values():c.close()
