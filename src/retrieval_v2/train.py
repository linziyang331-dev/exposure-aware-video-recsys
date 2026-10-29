"""Bounded validation-only standalone two-tower optimization."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch.nn import functional as F
from src.data.download import digest_file
from src.retrieval.data import tensor_batch
from src.retrieval.model import TwoTower,contrastive_loss
from src.retrieval.provenance import code_hashes
from src.retrieval_v2.model import TwoTowerV2
from src.retrieval_v2.objectives import full_catalog_loss
from src.retrieval_v2.evaluate import Scorer,evaluate


def implementation_hashes():
    directory=Path(__file__).parent
    return {'original':code_hashes(),'v2':{p.name:digest_file(p) for p in sorted(directory.glob('*.py'))
            if p.name in {'model.py','objectives.py','train.py','evaluate.py'}}}


def build_model(config,num_users,num_items):
    if config['model']=='legacy':
        architecture={k:config[k] for k in ['embedding_dim','output_dim','hidden_dim','temperature']}
        architecture.update(num_users=num_users,num_items=num_items,use_history=config['history_mode']!='none')
        return TwoTower(**architecture),architecture
    if config['model']!='residual':raise ValueError('Unknown model')
    architecture={k:config[k] for k in ['embedding_dim','temperature','history_mode','item_bias']}
    architecture.update(num_users=num_users,num_items=num_items,history_decay=config.get('history_decay',.9))
    return TwoTowerV2(**architecture),architecture


def load_model(path,device='cuda'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    cls=TwoTower if checkpoint['model']=='legacy' else TwoTowerV2
    model=cls(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    return model,checkpoint


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def batch_losses(model,batch,objective,config,candidate_ids):
    user=model.encode_user(batch['users'],batch['tabs'],batch['history'])
    if objective=='full':
        return {'full':full_catalog_loss(user@model.encode_item(candidate_ids).T/model.temperature,batch['items'])}
    item=model.encode_item(batch['items'])
    observed=F.binary_cross_entropy_with_logits((user*item).sum(1)/model.temperature,batch['labels'])
    losses={'bce':observed}
    if objective in {'mixed','contrastive'}:
        positives=torch.nonzero(batch['labels']==1).flatten()
        limit=config['contrastive_max_queries']
        if len(positives)>limit:positives=positives[torch.randperm(len(positives),device=user.device)[:limit]]
        contrast=user.sum()*0
        if len(positives)>1:
            targets=batch['items'][positives];pool=torch.unique(targets)
            contrast=contrastive_loss(user[positives],model.encode_item(pool),targets,pool,batch['history'][positives],model.temperature)
        losses['contrastive']=contrast
    return losses


def train_run(train,validation,counts,config,directory,seed):
    if config['objective'] not in {'mixed','bce','contrastive','full'}:raise ValueError('Unknown objective')
    if config['model'] not in {'legacy','residual'}:raise ValueError('Unknown model')
    signature={'config':config,'seed':seed,'train':train.metadata['signature'],
               'validation':validation.metadata['signature'],'code':implementation_hashes()}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/'training.json';path=directory/'best.pt'
    if result_path.exists():
        previous=json.loads(result_path.read_text(encoding='utf-8'))
        if previous['fingerprint']!=fingerprint or digest_file(path)!=previous['checkpoint_sha256']:
            raise ValueError('Completed experiment changed; preserve it and use a new directory')
        print('Reusing '+directory.name,flush=True);return previous
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device(config['device']);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    model,architecture=build_model(config,int(train['user_index'].max()),len(counts)-1)
    model=model.to(device);initial=parameter_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    indices=np.flatnonzero(train['label']==1) if config['objective'] in {'full','contrastive'} else np.arange(len(train))
    generator=np.random.default_rng(seed);curve=[];best=-1.;stale=0;updates=0;start=time.perf_counter()
    candidate_ids=torch.arange(1,len(counts),device=device)
    for epoch in range(config['epochs']):
        model.train();epoch_start=time.perf_counter();totals={};samples=0
        order=generator.permutation(indices)
        for offset in range(0,len(order),config['batch_size']):
            chosen=order[offset:offset+config['batch_size']];batch=tensor_batch(train,chosen,device)
            # Caches are left-aligned; history length clipping is done by cache construction.
            optimizer.zero_grad(set_to_none=True)
            losses=batch_losses(model,batch,config['objective'],config,candidate_ids)
            if config['objective']=='mixed':loss=losses['bce']+config['contrastive_weight']*losses['contrastive']
            else:loss=losses[config['objective']]
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();updates+=1;samples+=len(chosen)
            for name,value in losses.items():totals[name]=totals.get(name,0.)+value.item()*len(chosen)
        if device.type=='cuda':torch.cuda.synchronize()
        train_seconds=time.perf_counter()-epoch_start
        metrics=evaluate(Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),device,model=model),
                         validation,counts,batch_size=config['eval_batch_size'])
        row={'epoch':epoch,'losses':{k:v/samples for k,v in totals.items()},'train_seconds':train_seconds,'validation':metrics}
        curve.append(row);score=metrics['ndcg@20']
        print(f'{directory.name} epoch={epoch} loss={row["losses"]} valR20={metrics["recall@20"]:.5f} valN20={score:.5f}',flush=True)
        if score>best:
            best=score;stale=0;probe=tensor_batch(validation,np.arange(min(32,len(validation))),device)
            with torch.no_grad():expected=model(probe['users'],probe['items'],probe['tabs'],probe['history']).cpu()
            torch.save({'model':config['model'],'architecture':architecture,
                        'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'best_epoch':epoch,'seed':seed,'fingerprint':fingerprint,'reload_probe_scores':expected},path)
        else:stale+=1
        (directory/'learning_curve.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
        if stale>=config['patience']:break
    restored,checkpoint=load_model(path,device);probe=tensor_batch(validation,np.arange(min(32,len(validation))),device)
    with torch.no_grad():actual=restored(probe['users'],probe['items'],probe['tabs'],probe['history']).cpu()
    torch.testing.assert_close(actual,checkpoint['reload_probe_scores'].cpu(),rtol=0,atol=0)
    trained=parameter_digest(restored)
    if trained==initial:raise AssertionError('No parameter update')
    result={'status':'TRAINED','seed':seed,'config':config,'fingerprint':fingerprint,'signature':signature,
            'best_epoch':checkpoint['best_epoch'],'best_validation':curve[checkpoint['best_epoch']]['validation'],
            'curve':curve,'checkpoint':str(path.resolve()),'checkpoint_sha256':digest_file(path),
            'initial_parameter_sha256':initial,'trained_parameter_sha256':trained,'parameters_changed':True,
            'checkpoint_reload_identical':True,'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-start,
            'process_peak_working_set_MiB':psutil.Process().memory_info().peak_wset/2**20,
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
