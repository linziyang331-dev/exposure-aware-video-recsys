"""Deterministic full-positive-sequence native SASRec training and early stopping."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from src.data.download import digest_file
from src.retrieval_v2.train import parameter_digest
from src.ranking_v2.inference import deterministic
from src.sequential.model import SequentialModel
from src.sequential.provenance import code_hashes
from src.sequential.evaluate import catalog_evaluate


def load_model(path,device='cuda'):
    cp=torch.load(path,map_location=device,weights_only=True)
    model=SequentialModel(cp['config']|{'device':str(device)},cp['counts'].cpu().numpy())
    model.load_state_dict(cp['state']);model.eval();return model,cp


def train_run(train,val,counts,cfg,directory,seed):
    signature={'config':cfg,'seed':seed,'train':train.metadata['signature'],'validation':val.metadata['signature'],
               'counts_sha256':hashlib.sha256(np.asarray(counts,dtype=np.int64).tobytes()).hexdigest(),'code':code_hashes()}
    fp=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/'best.pt';result_path=directory/'training.json'
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['fingerprint']!=fp or digest_file(path)!=old['checkpoint_sha256']:raise ValueError('Completed sequence run changed; choose a new directory')
        return old
    deterministic();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device(cfg['device'])
    if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    model=SequentialModel(cfg,counts);initial=parameter_digest(model)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    indices=np.flatnonzero((train['label']==1)&(train['history_length']>0)&(train['item_index']>0))
    if not len(indices):raise ValueError('No nonempty observed positive sequence targets')
    generator=np.random.default_rng(seed);curve=[];best=-float('inf');stale=0;updates=0;start=time.perf_counter()
    def save(destination,epoch):
        model.eval();probe=torch.as_tensor(np.array(val['history'][:min(16,len(val))],copy=True),device=device,dtype=torch.long)
        with torch.no_grad():expected=model.scores(probe).cpu()
        torch.save({'config':cfg,'counts':torch.as_tensor(counts),'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'epoch':epoch,'seed':seed,'fingerprint':fp,'reload_probe':expected},destination)
    initial_metrics=catalog_evaluate(model,val,counts,cfg['eval_batch_size'],directory/'validation_initial.npz')
    save(directory/'initial.pt',-1);curve.append({'epoch':-1,'validation':initial_metrics})
    for epoch in range(cfg['epochs']):
        model.train();epoch_start=time.perf_counter();total=0.;seen=0;order=generator.permutation(indices)
        for offset in range(0,len(order),cfg['batch_size']):
            chosen=order[offset:offset+cfg['batch_size']]
            h=torch.as_tensor(np.array(train['history'][chosen],copy=True),device=device,dtype=torch.long)
            targets=torch.as_tensor(np.array(train['item_index'][chosen],copy=True),device=device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=model.loss(h,targets)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sequence loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite sequence gradient')
            optimizer.step();updates+=1;total+=loss.item()*len(chosen);seen+=len(chosen)
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-epoch_start
        metrics=catalog_evaluate(model,val,counts,cfg['eval_batch_size'],directory/f'validation_epoch{epoch}.npz')
        curve.append({'epoch':epoch,'observed_positive_sequence_ce':total/seen,'train_seconds':seconds,'validation':metrics})
        print(f'{directory.name} epoch={epoch} CE={total/seen:.6f} valR20={metrics["recall@20"]:.6f} valN20={metrics["ndcg@20"]:.6f} seconds={seconds:.1f}',flush=True)
        if metrics['ndcg@20']>best:best=metrics['ndcg@20'];best_epoch=epoch;best_metrics=metrics;save(path,epoch);stale=0
        else:stale+=1
        (directory/'learning_curve.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
        if stale>=cfg['patience']:break
    final=parameter_digest(model);restored,cp=load_model(path,device)
    probe=torch.as_tensor(np.array(val['history'][:min(16,len(val))],copy=True),device=device,dtype=torch.long)
    with torch.no_grad():actual=restored.scores(probe).cpu()
    torch.testing.assert_close(actual,cp['reload_probe'].cpu(),rtol=0,atol=0)
    if not torch.equal(restored.core.item_embedding.weight[0],torch.zeros(cfg['hidden_size'],device=device)):
        raise ValueError('Padding embedding changed')
    result={'status':'TRAINED','config':cfg,'seed':seed,'signature':signature,'fingerprint':fp,'checkpoint':str(path.resolve()),
            'checkpoint_sha256':digest_file(path),'best_epoch':best_epoch,'best_validation':best_metrics,'curve':curve,
            'baseline_validation':initial_metrics,'eligible_training_queries':len(indices),'empty_positive_train_skipped':int(((train['label']==1)&(train['history_length']==0)).sum()),
            'parameters_changed':initial!=final,'trained_parameter_sha256':parameter_digest(restored),'reload_identical':True,
            'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-start,
            'peak_process_working_set_MiB':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)/2**20,
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
