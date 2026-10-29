"""Deterministic full-catalog learning with immutable, reload-checked runs."""
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
from src.exposure.run import run_component
from src.retrieval.data import tensor_batch
from src.retrieval_v2.model import TwoTowerV2
from src.retrieval_v2.train import parameter_digest, implementation_hashes
from src.retrieval_v2.evaluate import Scorer, evaluate
from src.retrieval_longtail.model import GatedTwoTower
from src.retrieval_longtail.contracts import frequency_weights, weighted_choice_loss, summarize


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def source_hashes():
    files=[ROOT/'src/retrieval_longtail'/n for n in ['contracts.py','model.py','train.py']]
    return {'legacy':implementation_hashes(),'longtail':{p.name:digest_file(p) for p in files},
            'immutable_runner':digest_file(ROOT/'src/exposure/run.py')}


def load_model(path,device='cuda'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    cls=GatedTwoTower if checkpoint['model']=='gated' else TwoTowerV2
    model=cls(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    return model,checkpoint


def training_signature(train,validation,counts,config,seed):
    return {'config':config,'seed':int(seed),'sources':source_hashes(),
        'train':train.metadata['signature'],'validation':validation.metadata['signature'],
        'counts':np.asarray(counts).tolist(),'versions':{'torch':str(torch.__version__),'numpy':str(np.__version__)}}


def evaluate_model(model,cache,counts,batch_size,artifact):
    scorer=Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),next(model.parameters()).device,model=model)
    result=evaluate(scorer,cache,counts,batch_size=batch_size,artifact=artifact)
    idx=np.flatnonzero(cache['label']==1)
    with np.load(artifact) as saved:
        extra=summarize(saved['ranks'],saved['users'],cache['item_index'][idx],cache['history'][idx],counts)
    result.update(extra)
    return result


def train_run(train,validation,counts,config,directory,seed):
    if config['model'] not in {'recency','gated'}:raise ValueError('Unknown model')
    directory=Path(directory)
    weights=frequency_weights(counts,config['beta'],config['weight_cap'])
    signature=training_signature(train,validation,counts,config,seed)
    def worker():
        directory.mkdir()
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        device=torch.device(config['device'])
        if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
        architecture={'num_users':int(train['user_index'].max()),'num_items':len(counts)-1,
            'embedding_dim':config['embedding_dim'],'temperature':config['temperature'],
            'history_mode':'recency','history_decay':config['history_decay'],'item_bias':False}
        cls=GatedTwoTower if config['model']=='gated' else TwoTowerV2
        model=cls(**architecture).to(device);initial=parameter_digest(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        candidate_ids=torch.arange(1,len(counts),device=device)
        weight_tensor=torch.tensor(weights,device=device)
        indices=np.flatnonzero(train['label']==1)
        rng=np.random.default_rng(seed);curve=[];best=-1.;stale=0;updates=0;start=time.perf_counter()
        best_path=directory/'best.pt';val_path=directory/'best_validation.npz'
        for epoch in range(config['epochs']):
            model.train();total=0.;samples=0;begin=time.perf_counter()
            order=rng.permutation(indices)
            for offset in range(0,len(order),config['batch_size']):
                chosen=order[offset:offset+config['batch_size']];batch=tensor_batch(train,chosen,device)
                optimizer.zero_grad(set_to_none=True)
                scores=model.encode_user(batch['users'],batch['tabs'],batch['history'])@model.encode_item(candidate_ids).T/model.temperature
                loss=F.cross_entropy(scores,batch['items']-1) if config['beta']==0 else weighted_choice_loss(scores,batch['items'],weight_tensor)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();updates+=1;total+=loss.item()*len(chosen);samples+=len(chosen)
            if device.type=='cuda':torch.cuda.synchronize()
            train_seconds=time.perf_counter()-begin
            artifact=directory/'current_validation.npz'
            metric=evaluate_model(model,validation,counts,config['eval_batch_size'],artifact)
            row={'epoch':epoch,'training_loss':total/samples,'train_seconds':train_seconds,'validation':metric}
            curve.append(row);score=metric['ndcg@20']
            print(f'{directory.name} epoch={epoch} N20={score:.6f} R20={metric["recall@20"]:.6f} tailR100={metric["groups"]["tail_target"].get("recall@100",0):.6f}',flush=True)
            if score>best:
                best=score;stale=0;best_epoch=epoch
                torch.save({'model':config['model'],'architecture':architecture,'seed':int(seed),'best_epoch':epoch,
                    'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()}},best_path)
                val_path.write_bytes(artifact.read_bytes())
            else:stale+=1
            artifact.unlink()
            write_json(directory/'learning_curve.json',curve)
            if stale>=config['patience']:break
        restored,_=load_model(best_path,device)
        checked=evaluate_model(restored,validation,counts,config['eval_batch_size'],directory/'reload_validation.npz')
        with np.load(val_path) as a,np.load(directory/'reload_validation.npz') as b:
            if not all(np.array_equal(a[k],b[k]) for k in a.files):raise AssertionError('Reload changed validation ranks')
        (directory/'reload_validation.npz').unlink()
        changed=parameter_digest(restored)
        if changed==initial:raise AssertionError('No learning')
        result={'status':'TRAINED','seed':int(seed),'config':config,'architecture':architecture,
            'best_epoch':best_epoch,'best_validation':curve[best_epoch]['validation'],'curve':curve,
            'checkpoint':str(best_path.resolve()),'checkpoint_sha256':digest_file(best_path),
            'initial_parameter_sha256':initial,'trained_parameter_sha256':changed,
            'checkpoint_reload_identical':True,'optimizer_updates':updates,
            'weight_min':float(weights[1:].min()),'weight_max':float(weights[1:].max()),
            'train_event_mean_weight':float(np.dot(counts,weights)/np.sum(counts)),
            'parameter_count':sum(p.numel() for p in restored.parameters()),
            'elapsed_seconds':time.perf_counter()-start,
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
            'process_peak_working_set_MiB':psutil.Process().memory_info().peak_wset/2**20}
        write_json(directory/'training.json',result)
    run_component(directory,signature,worker)
    return read_json(directory/'training.json')
