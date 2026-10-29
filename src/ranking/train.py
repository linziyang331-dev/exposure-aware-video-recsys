"""Observed-exposure BCE, validation LogLoss early stopping, actual upstream DIN/LR."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch.nn import functional as F
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.data import tensor_batch
from src.ranking.model import RankingModel
from src.ranking.metrics import exposure_metrics
from src.ranking.provenance import implementation_hashes


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model(path,device='cuda'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    cfg=checkpoint['architecture']['config']|{'device':str(device)}
    model=RankingModel(checkpoint['architecture']['num_users'],checkpoint['architecture']['num_items'],cfg)
    model.load_state_dict(checkpoint['state']);model.eval();return model,checkpoint


@torch.no_grad()
def predict(model,cache,batch_size=2048):
    model.eval();device=next(model.parameters()).device;logits=np.empty(len(cache),dtype=np.float32)
    for start in range(0,len(cache),batch_size):
        indices=np.arange(start,min(start+batch_size,len(cache)));b=tensor_batch(cache,indices,device)
        logits[indices]=model(b['users'],b['items'],b['tabs'],b['history']).cpu().numpy()
    return logits


def train_run(train,validation,num_users,num_items,config,directory,seed):
    if not np.isin(train['label'],[0,1]).all():raise ValueError('Only observed binary exposure labels')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    signature={'config':config,'seed':seed,'train':train.metadata['signature'],
               'validation':validation.metadata['signature'],'implementation':implementation_hashes()}
    if config['pretrained_items']:
        from src.ranking.provenance import verify_v2
        signature['v2_checkpoint_sha256']=verify_v2()['checkpoint_sha256']
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    result_path=directory/'training.json';path=directory/'best.pt'
    if result_path.exists():
        old=json.loads(result_path.read_text(encoding='utf-8'))
        if old['fingerprint']!=fingerprint or digest_file(path)!=old['checkpoint_sha256']:
            raise ValueError('Training signature/checkpoint changed; preserve old output')
        return old
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device(config['device'])
    if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    model=RankingModel(num_users,num_items,config)
    if config['pretrained_items']:
        from src.retrieval_v2.train import load_model as load_v2
        from src.ranking.provenance import verify_v2
        base,_=load_v2(ROOT/verify_v2()['checkpoint'],device)
        model.initialize_item_embeddings(base.item_embedding.weight.detach());del base
    initial=parameter_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=np.random.default_rng(seed);best=float('inf');stale=0;curve=[];updates=0;start=time.perf_counter()
    for epoch in range(config['epochs']):
        epoch_start=time.perf_counter();model.train();order=generator.permutation(len(train));total=0.;seen=0
        for offset in range(0,len(order),config['batch_size']):
            indices=order[offset:offset+config['batch_size']];b=tensor_batch(train,indices,device)
            # Upstream Dice uses BatchNorm; a singleton tail is evaluated with running statistics.
            if len(indices)==1:model.eval()
            optimizer.zero_grad(set_to_none=True)
            logits=model(b['users'],b['items'],b['tabs'],b['history'])
            loss=F.binary_cross_entropy_with_logits(logits,b['labels'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite ranking loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite ranking gradients')
            optimizer.step();updates+=1;total+=loss.item()*len(indices);seen+=len(indices)
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-epoch_start
        metrics=exposure_metrics(validation['label'],predict(model,validation,config['eval_batch_size']))
        row={'epoch':epoch,'train_bce':total/seen,'train_seconds':seconds,'validation':metrics};curve.append(row)
        print(f'{directory.name} epoch={epoch} BCE={total/seen:.5f} valAUC={metrics["auc"]} valLL={metrics["logloss"]:.6f}',flush=True)
        if metrics['logloss']<best:
            best=metrics['logloss'];stale=0;model.eval();probe=tensor_batch(validation,np.arange(min(32,len(validation))),device)
            with torch.no_grad():expected=model(probe['users'],probe['items'],probe['tabs'],probe['history']).cpu()
            torch.save({'architecture':{'num_users':num_users,'num_items':num_items,'config':config},
                        'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'best_epoch':epoch,
                        'seed':seed,'fingerprint':fingerprint,'reload_probe_logits':expected},path)
        else:stale+=1
        (directory/'learning_curve.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
        if stale>=config['patience']:break
    restored,checkpoint=load_model(path,device);probe=tensor_batch(validation,np.arange(min(32,len(validation))),device)
    with torch.no_grad():actual=restored(probe['users'],probe['items'],probe['tabs'],probe['history']).cpu()
    torch.testing.assert_close(actual,checkpoint['reload_probe_logits'].cpu(),rtol=0,atol=0)
    trained=parameter_digest(restored)
    if trained==initial:raise AssertionError('No actual parameter update')
    result={'status':'TRAINED','config':config,'seed':seed,'signature':signature,'fingerprint':fingerprint,
            'best_epoch':checkpoint['best_epoch'],'best_validation':curve[checkpoint['best_epoch']]['validation'],
            'curve':curve,'checkpoint':str(path.resolve()),'checkpoint_sha256':digest_file(path),
            'initial_parameter_sha256':initial,'trained_parameter_sha256':trained,'parameters_changed':True,
            'reload_identical':True,'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-start,
            'peak_process_working_set_MiB':psutil.Process().memory_info().peak_wset/2**20,
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
