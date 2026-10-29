"""Matched-budget DIN adaptation with genuinely new frozen teacher/candidates."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.run import run_component
from src.retrieval.data import tensor_batch
from src.retrieval.evaluate import Scorer
from src.retrieval.metrics import stable_topk
from src.retrieval_v2.model import TwoTowerV2
from src.retrieval_v2.train import parameter_digest
from src.ranking.candidates import CandidateCache
from src.ranking.metrics import candidate_ranks
from src.ranking_v2.model import RichDIN
from src.ranking_v2.data import choice_batch
from src.ranking_v2.evaluate import assess
from src.retrieval_balanced.model import SharedTower
from src.retrieval_balanced.train import load_model as load_retriever
from src.retrieval_longtail.run import setup,load_selection
from src.retrieval_longtail.train import read_json,write_json


class TeacherDIN(RichDIN):
    def teacher_spec(self):
        if isinstance(self.teacher,SharedTower):return {'kind':'shared','architecture':self.teacher.architecture}
        return {'kind':'v2','architecture':super().teacher_spec()}


def load_model(path,device='cuda'):
    c=torch.load(path,map_location=device,weights_only=True);a=c['architecture'];spec=a['teacher_spec']
    cls=SharedTower if spec['kind']=='shared' else TwoTowerV2
    teacher=cls(**spec['architecture']).to(device)
    model=TeacherDIN(a['num_users'],a['num_items'],a['config']|{'device':str(device)},teacher)
    model.load_state_dict(c['state']);model.eval()
    return model,c


def candidate_signature(cache,counts,teacher,checkpoint_sha,topk=100):
    return {'source':cache.metadata['signature'],'teacher_checkpoint':checkpoint_sha,
        'teacher_parameters':parameter_digest(teacher),'generator':digest_file(__file__),
        'teacher_code':digest_file(ROOT/'src/retrieval_balanced/model.py'),
        'topk':min(topk,len(counts)-1),'score_batch_size':512,'target_injection':False}


def candidate_cache(cache,counts,teacher,directory,checkpoint_sha,topk=100):
    directory=Path(directory);k=min(topk,len(counts)-1);sig=candidate_signature(cache,counts,teacher,checkpoint_sha,topk)
    def worker():
        directory.mkdir();idx=np.flatnonzero(cache['label']==1);rows=len(idx)
        scorer=Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),next(teacher.parameters()).device,model=teacher)
        arrays={};types={'ids':('int32',(rows,k)),'scores':('float32',(rows,k)),
            'indices':('int64',(rows,)),'event_ids':('int64',(rows,)),'user_ids':('int64',(rows,)),'v2_ranks':('int32',(rows,))}
        try:
            for name,(dtype,shape) in types.items():arrays[name]=np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
            for offset in range(0,rows,512):
                chosen=idx[offset:offset+512];scores=scorer.scores(cache,chosen)
                ids,values=stable_topk(scores,scorer.candidate_ids,k)
                # Targets are consulted only after independently selecting the candidate list.
                target=torch.tensor(np.array(cache['item_index'][chosen]),device=scorer.device,dtype=torch.long)
                sl=slice(offset,offset+len(chosen));arrays['ids'][sl]=ids.cpu().numpy();arrays['scores'][sl]=values.cpu().numpy()
                arrays['indices'][sl]=chosen;arrays['event_ids'][sl]=cache['event_id'][chosen];arrays['user_ids'][sl]=cache['user_id'][chosen]
                arrays['v2_ranks'][sl]=candidate_ranks(ids,values,target).cpu().numpy()
        finally:
            for v in arrays.values():v.flush();v._mmap.close()
        metadata={'rows':rows,'signature':sig,'array_hashes':{n:digest_file(directory/(n+'.npy')) for n in types},
            'v2_ranks_field_semantics':'legacy interface name; rank in CURRENT teacher candidates, not necessarily V2'}
        write_json(directory/'metadata.json',metadata)
    run_component(directory,sig,worker)
    return CandidateCache(directory,read_json(directory/'metadata.json'))


def rank_signature(train,val,train_c,val_c,teacher,cfg,seed):
    return {'config':cfg,'seed':seed,'teacher_parameters':parameter_digest(teacher),'source':digest_file(__file__),
        'upstream':{str(p.relative_to(ROOT)):digest_file(p) for p in (ROOT/'src/ranking_v2').glob('*.py')},
        'dependencies':{p:digest_file(ROOT/p) for p in ['src/retrieval_balanced/model.py','src/retrieval/data.py',
            'src/ranking/model.py','src/ranking/evaluate.py','src/ranking/metrics.py','src/ranking/adapter.py']},
        'versions':{'torch':str(torch.__version__),'numpy':str(np.__version__)},
        'train':train.metadata['signature'],'validation':val.metadata['signature'],
        'train_candidates':train_c.metadata['signature'],'val_candidates':val_c.metadata['signature']}


def train_ranker(train,val,train_c,val_c,teacher,cfg,directory,seed):
    directory=Path(directory);teacher_before=parameter_digest(teacher)
    sig=rank_signature(train,val,train_c,val_c,teacher,cfg,seed)
    def worker():
        directory.mkdir();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        device=torch.device(cfg['device'])
        if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
        model=TeacherDIN(teacher.user_embedding.num_embeddings-1,teacher.item_embedding.num_embeddings-1,cfg,teacher)
        architecture={'num_users':teacher.user_embedding.num_embeddings-1,'num_items':teacher.item_embedding.num_embeddings-1,
                      'config':cfg,'teacher_spec':model.teacher_spec()}
        trainable=[p for p in model.parameters() if p.requires_grad];repeat=list(model.repeat_linear.parameters());repeat_ids={id(p) for p in repeat}
        optimizer=torch.optim.Adam([{'params':[p for p in trainable if id(p) not in repeat_ids]},
            {'params':repeat,'lr':cfg['learning_rate']*cfg.get('repeat_lr_multiplier',10.)}],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        rng=np.random.default_rng(seed);path=directory/'best.pt';curve=[];start=time.perf_counter();updates=0;stale=0
        def save(epoch):
            probe=tensor_batch(val,np.arange(min(16,len(val))),device)
            with torch.no_grad():expected=model(probe['users'],probe['items'],probe['tabs'],probe['history'])['rank'].cpu()
            torch.save({'architecture':architecture,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'epoch':epoch,'seed':seed,'reload_probe':expected},path)
        base=assess(model,val,val_c,cfg,directory/'validation_baseline.npz')
        if base['max_abs_score_correction']!=0:raise ValueError('Initial residual not exactly zero')
        best=base['ranking']['ndcg@20'];best_epoch=-1;best_metric=base;save(-1);curve.append({'epoch':-1,'validation':base})
        eligible=np.flatnonzero(train_c['v2_ranks']>0)
        if not len(eligible):raise ValueError('No positive targets in independently generated pool')
        for epoch in range(cfg['epochs']):
            model.train();total=0.;steps=0
            order=rng.choice(eligible,min(cfg['queries_per_epoch'],len(eligible)),replace=False)
            for offset in range(0,len(order),cfg['query_batch_size']):
                optimizer.zero_grad(set_to_none=True)
                chosen=choice_batch(train,train_c,order[offset:offset+cfg['query_batch_size']],rng,cfg['competitors'],cfg['mask_past'])
                b=tensor_batch(train,chosen['indices'],device);k=chosen['items'].shape[1];q=len(chosen['indices'])
                aux=tensor_batch(train,rng.integers(0,len(train),size=cfg['aux_batch_size']),device)
                u=torch.cat([b['users'].repeat_interleave(k),aux['users']]);t=torch.cat([b['tabs'].repeat_interleave(k),aux['tabs']])
                h=torch.cat([b['history'].repeat_interleave(k,dim=0),aux['history']])
                i=torch.cat([torch.tensor(chosen['items'].ravel(),device=device,dtype=torch.long),aux['items']])
                with torch.no_grad():ab=(teacher.encode_user(aux['users'],aux['tabs'],aux['history'])*teacher.encode_item(aux['items'])).sum(1)/teacher.temperature
                canonical=torch.cat([torch.tensor(chosen['base_scores'].ravel(),device=device),ab])
                out=model(u,i,t,h,base_scores=canonical)
                choice=F.cross_entropy(out['rank'][:q*k].reshape(q,k),torch.zeros(q,device=device,dtype=torch.long))
                observed=F.binary_cross_entropy_with_logits(out['ctr'][q*k:],aux['labels'])
                loss=choice+cfg['aux_weight']*observed+cfg['delta_reg']*out['delta'][:q*k].square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DIN loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,5.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite DIN gradient')
                optimizer.step();total+=loss.item();steps+=1;updates+=1
            m=assess(model,val,val_c,cfg,directory/f'validation_epoch{epoch}.npz')
            curve.append({'epoch':epoch,'loss':total/steps,'validation':m})
            print(f'{directory.name} epoch={epoch} N20={m["ranking"]["ndcg@20"]:.6f} R20={m["ranking"]["recall@20"]:.6f}',flush=True)
            if m['ranking']['ndcg@20']>best:best=m['ranking']['ndcg@20'];best_epoch=epoch;best_metric=m;stale=0;save(epoch)
            else:stale+=1
            if stale>=cfg['patience']:break
        if teacher_before!=parameter_digest(teacher) or any(p.grad is not None for p in teacher.parameters()):raise ValueError('Frozen teacher changed')
        restored,saved=load_model(path,device);probe=tensor_batch(val,np.arange(min(16,len(val))),device)
        with torch.no_grad():a=restored(probe['users'],probe['items'],probe['tabs'],probe['history'])['rank']
        torch.testing.assert_close(a.cpu(),saved['reload_probe'].cpu(),rtol=0,atol=0)
        if parameter_digest(restored.teacher)!=teacher_before:raise ValueError('Reloaded graph teacher differs from frozen teacher')
        original_rank=directory/('validation_baseline.npz' if best_epoch==-1 else f'validation_epoch{best_epoch}.npz')
        replay_rank=directory/'selected_validation_replay.npz'
        assess(restored,val,val_c,cfg,replay_rank)
        with np.load(original_rank) as a,np.load(replay_rank) as b:
            if set(a.files)!=set(b.files) or not all(np.array_equal(a[k],b[k]) for k in a.files):
                raise ValueError('Selected checkpoint changed full validation rankings after reload')
        write_json(directory/'training.json',{'config':cfg,'seed':seed,'curve':curve,'best_epoch':best_epoch,
            'best_validation':best_metric,'checkpoint':str(path.resolve()),'checkpoint_sha256':digest_file(path),
            'teacher_parameters':teacher_before,'teacher_unchanged':True,'reload_identical':True,
            'full_validation_rank_replay_identical':True,
            'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-start,
            'trained_parameter_sha256':parameter_digest(restored),
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None})
    run_component(directory,sig,worker)
    return read_json(directory/'training.json')


def adapt(cfg,out):
    from src.retrieval_balanced.run import replicate
    frozen=load_selection(out);rep=replicate(cfg,out);selected=rep['selected_for_adaptation']
    if (out/'adaptation_manifest.json').exists():
        from src.retrieval_longtail.verify import validate_outputs
        validate_outputs(out,out/'adaptation_manifest.json')
        return read_json(out/'adaptation.json')
    if selected is None:
        result={'status':'SKIPPED','reason':'No candidate passed material three-seed validation gate'}
        write_json(out/'adaptation.json',result);return result
    counts,caches=setup(cfg,['standard/train','standard/validation'])
    try:
        rcfg=read_json(ROOT/'experiments/ranking_v2/selection.json')['config']
        runs={}
        for name in ['control',selected]:
            teacher_run=rep['groups'][name]['42'];teacher,_=load_retriever(teacher_run['checkpoint'],cfg['device'])
            candidates={p:candidate_cache(c,counts,teacher,out/'adaptation/candidates'/name/p.replace('/','_'),teacher_run['checkpoint_sha256']) for p,c in caches.items()}
            try:
                runs[name]={}
                for seed in [42,43,44]:
                    teacher,_=load_retriever(teacher_run['checkpoint'],cfg['device'])
                    runs[name][str(seed)]=train_ranker(caches['standard/train'],caches['standard/validation'],
                        candidates['standard/train'],candidates['standard/validation'],teacher,rcfg,
                        out/'adaptation/runs'/f'{name}_{seed}',seed)
            finally:
                for c in candidates.values():c.close()
        result={'status':'TRAINED','selected_retriever':selected,'runs':runs,
            'retrieval_seed':42,'ranking_seeds':[42,43,44],'test_used_for_selection':False,
            'replication_sha256':digest_file(out/'replication.json')}
        write_json(out/'adaptation.json',result)
    finally:
        for c in caches.values():c.close()
    # Only after all adaptation checkpoints are validation-selected, reassess test partitions.
    counts,caches=setup(cfg,['standard/validation','standard/test','random/validation','random/test'])
    try:
        results={}
        for name,group in runs.items():
            teacher_run=rep['groups'][name]['42'];teacher,_=load_retriever(teacher_run['checkpoint'],cfg['device'])
            candidates={p:candidate_cache(c,counts,teacher,out/'adaptation/candidates'/name/p.replace('/','_'),teacher_run['checkpoint_sha256']) for p,c in caches.items()}
            try:
                results[name]={}
                for seed,r in group.items():
                    model,_=load_model(r['checkpoint'],cfg['device']);results[name][seed]={}
                    for part,c in caches.items():
                        path=out/'adaptation/ranks'/f'{name}_{seed}'/(part.replace('/','_')+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                        m=assess(model,c,candidates[part],rcfg,path);m['rank_artifact']=str(path.resolve());results[name][seed][part]=m
                        print('ADAPTED '+name+' '+seed+' '+part+' R20='+str(m['ranking']['recall@20']),flush=True)
            finally:
                for c in candidates.values():c.close()
        result['evaluation']=results;write_json(out/'adaptation.json',result)
        write_json(out/'adaptation_manifest.json',{'output_hashes':{str(p.relative_to(out)):digest_file(p) for p in
            [out/'adaptation.json',*sorted((out/'adaptation/ranks').rglob('*.npz'))]}})
        return result
    finally:
        for c in caches.values():c.close()
