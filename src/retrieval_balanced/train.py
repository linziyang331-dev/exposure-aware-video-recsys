"""Fixed-budget full-catalog training with a material-tail checkpoint rule."""
import hashlib
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
from src.retrieval_v2.train import parameter_digest,implementation_hashes
from src.retrieval_longtail.train import read_json,write_json,evaluate_model
from src.retrieval_balanced.model import SharedTower
from src.retrieval_balanced.objectives import mixture_weights,choice_loss
from src.retrieval_balanced.selection import qualifies,utility


def sources():
    files=[ROOT/'src/retrieval_balanced'/n for n in ['graph.py','model.py','objectives.py','selection.py','train.py']]
    files += [ROOT/'src/retrieval_longtail/train.py',ROOT/'src/retrieval_longtail/contracts.py',ROOT/'src/exposure/run.py']
    return {'original':implementation_hashes(),'new':{p.relative_to(ROOT).as_posix():digest_file(p) for p in files}}


def array_digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def signature(train,val,counts,ids,weights,cfg,baseline,gate,seed):
    return {'config':cfg,'baseline':baseline,'gate':gate,'seed':int(seed),'sources':sources(),
        'train':train.metadata['signature'],'validation':val.metadata['signature'],'counts':array_digest(counts),
        'neighbor_ids':array_digest(ids),'neighbor_weights':array_digest(weights),
        'versions':{'torch':str(torch.__version__),'numpy':str(np.__version__)}}


def load_model(path,device='cuda'):
    c=torch.load(path,map_location=device,weights_only=True)
    model=SharedTower(**c['architecture']).to(device);model.load_state_dict(c['state']);model.eval()
    return model,c


def backbone_digest(model):
    sha=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):sha.update(name.encode());sha.update(p.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def train_run(train,val,counts,ids,neighbor_weights,cfg,baseline,gate,directory,seed):
    directory=Path(directory);sig=signature(train,val,counts,ids,neighbor_weights,cfg,baseline,gate,seed)
    def worker():
        directory.mkdir();device=torch.device(cfg['device'])
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
        architecture={k:cfg[k] for k in ['embedding_dim','temperature','history_decay','graph_alpha','neighbors']}
        architecture.update(num_users=int(train['user_index'].max()),num_items=len(counts)-1)
        model=SharedTower(**architecture).to(device);model.set_neighbors(ids,neighbor_weights)
        initial=parameter_digest(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        candidate_ids=torch.arange(1,len(counts),device=device)
        weights=mixture_weights(counts,cfg['rho']);wt=torch.tensor(weights,device=device)
        lp=torch.tensor(np.log((counts[1:]+1)/(counts.sum()+len(counts)-1)),dtype=torch.float32,device=device)
        indices=np.flatnonzero(train['label']==1);rng=np.random.default_rng(seed)
        curve=[];qualified_key=None;fallback_key=-1.;roles={};start=time.perf_counter();updates=0
        for epoch in range(cfg['epochs']):
            model.train();order=rng.permutation(indices);total=0.;samples=0;begin=time.perf_counter()
            for offset in range(0,len(order),cfg['batch_size']):
                chosen=order[offset:offset+cfg['batch_size']];b=tensor_batch(train,chosen,device)
                optimizer.zero_grad(set_to_none=True)
                scores=model.encode_user(b['users'],b['tabs'],b['history'])@model.encode_item(candidate_ids).T/model.temperature
                loss=F.cross_entropy(scores,b['items']-1) if cfg['rho']==0 and cfg['tau']==0 else choice_loss(scores,b['items'],wt,lp,cfg['tau'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();total+=loss.item()*len(chosen);samples+=len(chosen);updates+=1
            if device.type=='cuda':torch.cuda.synchronize()
            seconds=time.perf_counter()-begin
            current=directory/'current.npz';m=evaluate_model(model,val,counts,cfg['eval_batch_size'],current)
            ok=qualifies(m,baseline,gate);key=(utility(m,baseline),m['ndcg@20'])
            row={'epoch':epoch,'loss':total/samples,'train_seconds':seconds,'validation':m,'qualified':ok,'tail_utility':key[0]}
            curve.append(row);save=[]
            if m['ndcg@20']>fallback_key:fallback_key=m['ndcg@20'];save.append('fallback')
            if ok and (qualified_key is None or key>qualified_key):qualified_key=key;save.append('eligible')
            for role in save:
                torch.save({'architecture':architecture,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'epoch':epoch,'seed':int(seed)},directory/(role+'.pt'))
                (directory/(role+'.npz')).write_bytes(current.read_bytes());roles[role]=epoch
            current.unlink();write_json(directory/'curve.json',curve)
            print(f'{directory.name} epoch={epoch} N20={m["ndcg@20"]:.6f} R20={m["recall@20"]:.6f} tail20={m["groups"]["tail_target"].get("recall@20",0):.6f} tail100={m["groups"]["tail_target"].get("recall@100",0):.6f} qualifies={ok}',flush=True)
        role='eligible' if 'eligible' in roles else 'fallback';best_epoch=roles[role]
        path=directory/'best.pt';path.write_bytes((directory/(role+'.pt')).read_bytes())
        best_rank=directory/'best_validation.npz';best_rank.write_bytes((directory/(role+'.npz')).read_bytes())
        restored,_=load_model(path,device);reload_path=directory/'reload.npz'
        evaluate_model(restored,val,counts,cfg['eval_batch_size'],reload_path)
        with np.load(best_rank) as a,np.load(reload_path) as b:
            if not all(np.array_equal(a[k],b[k]) for k in a.files):raise AssertionError('Reload changed actual rankings')
        reload_path.unlink()
        result={'config':cfg,'seed':int(seed),'baseline':baseline,'gate':gate,'curve':curve,'best_epoch':best_epoch,
            'qualified':role=='eligible','best_validation':curve[best_epoch]['validation'],
            'tail_utility':curve[best_epoch]['tail_utility'],'checkpoint':str(path.resolve()),'checkpoint_sha256':digest_file(path),
            'initial_parameter_sha256':initial,'trained_parameter_sha256':parameter_digest(restored),
            'backbone_parameter_sha256':backbone_digest(restored),'reload_identical':True,
            'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-start,
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
            'weights':{'event_mean':float(np.dot(counts,weights)/counts.sum()),'max':float(weights.max()),
                'effective_sample_size':float(counts.sum()**2/np.dot(counts,weights**2))},
            'parameter_count':sum(p.numel() for p in model.parameters())}
        write_json(directory/'training.json',result)
    run_component(directory,sig,worker)
    return read_json(directory/'training.json')
