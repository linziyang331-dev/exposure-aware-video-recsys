"""Controlled BCE or implicit-choice training; actual exposures alone label the CTR head."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.data import tensor_batch
from src.retrieval_v2.model import TwoTowerV2
from src.retrieval_v2.train import parameter_digest
from src.ranking.provenance import verify_v2,implementation_hashes as original_hashes
from src.ranking_v2.model import RichDIN
from src.ranking_v2.data import choice_batch
from src.ranking_v2.evaluate import assess


def code_hashes():
    return {'original':original_hashes(),'revision':{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')
           if p.name in {'model.py','data.py','train.py','evaluate.py'}}}


def load_model(path,device='cuda'):
    checkpoint=torch.load(path,map_location=device,weights_only=True);arch=checkpoint['architecture']
    teacher=TwoTowerV2(**arch['teacher_spec']).to(device) if arch['teacher_spec'] is not None else None
    cfg=arch['config']|{'device':str(device)}
    model=RichDIN(arch['num_users'],arch['num_items'],cfg,teacher)
    model.load_state_dict(checkpoint['state']);model.eval();return model,checkpoint


def train_run(train,validation,train_candidates,validation_candidates,num_users,num_items,config,directory,seed,teacher=None):
    if config['objective'] not in {'bce','choice'}:raise ValueError('Unknown revision objective')
    if not np.isin(train['label'],[0,1]).all():raise ValueError('Only actual observed binary labels in BCE')
    needs=config['residual'] or config['teacher_features'] or config['pretrained']
    if teacher is None and needs:
        from src.retrieval_v2.train import load_model as load_v2
        frozen=verify_v2();teacher,_=load_v2(ROOT/frozen['checkpoint'],config['device'])
    teacher_before=parameter_digest(teacher) if teacher is not None else None
    signature={'config':config,'seed':seed,'train':train.metadata['signature'],'validation':validation.metadata['signature'],
               'train_candidates':train_candidates.metadata['signature'] if train_candidates is not None else None,
               'validation_candidates':validation_candidates.metadata['signature'],'teacher_parameters':teacher_before,'code':code_hashes()}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/'best.pt';result_path=directory/'training.json'
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['fingerprint']!=fingerprint or digest_file(path)!=old['checkpoint_sha256']:raise ValueError('Revision run changed; use a new directory')
        return old
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device(config['device'])
    if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    model=RichDIN(num_users,num_items,config,teacher);initial=parameter_digest(model)
    architecture={'num_users':num_users,'num_items':num_items,'config':config,'teacher_spec':model.teacher_spec()}
    trainable=[p for p in model.parameters() if p.requires_grad]
    repeat_parameters=list(model.repeat_linear.parameters())
    repeated_ids={id(p) for p in repeat_parameters}
    remaining=[p for p in trainable if id(p) not in repeated_ids]
    optimizer=torch.optim.Adam([{'params':remaining},{'params':repeat_parameters,
          'lr':config['learning_rate']*config.get('repeat_lr_multiplier',10.)}],
          lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=np.random.default_rng(seed);curve=[];start=time.perf_counter();updates=0;stale=0
    def select(metrics):return metrics['ranking']['ndcg@20'] if config['select_metric']=='ndcg@20' else -metrics['pointwise']['logloss']
    def save(epoch,metrics):
        model.eval();probe=tensor_batch(validation,np.arange(min(16,len(validation))),device)
        with torch.no_grad():expected=model(probe['users'],probe['items'],probe['tabs'],probe['history'])['rank'].cpu()
        torch.save({'architecture':architecture,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'epoch':epoch,'seed':seed,'fingerprint':fingerprint,'reload_probe':expected},path)
        return metrics
    base=assess(model,validation,validation_candidates,config,directory/'baseline_validation_ranks.npz')
    best=select(base);best_metrics=save(-1,base);best_epoch=-1
    curve.append({'epoch':-1,'validation':base})
    if config['residual']:
        if base['max_abs_score_correction']!=0:raise ValueError('Zero residual did not preserve frozen baseline')
    eligible=np.flatnonzero(train_candidates['v2_ranks']>0) if config['objective']=='choice' else None
    for epoch in range(config['epochs']):
        model.train();epoch_start=time.perf_counter();totals={};steps=0
        if config['objective']=='choice':
            order=rng.choice(eligible,min(config['queries_per_epoch'],len(eligible)),replace=False)
            size=config['query_batch_size']
        else:order=rng.permutation(len(train));size=config['batch_size']
        for offset in range(0,len(order),size):
            optimizer.zero_grad(set_to_none=True)
            if config['objective']=='choice':
                choice=choice_batch(train,train_candidates,order[offset:offset+size],rng,config['competitors'],config['mask_past'])
                b=tensor_batch(train,choice['indices'],device);k=choice['items'].shape[1];q=len(choice['indices'])
                aux_idx=rng.integers(0,len(train),size=config['aux_batch_size']);aux=tensor_batch(train,aux_idx,device)
                users=torch.cat([b['users'].repeat_interleave(k),aux['users']]);tabs=torch.cat([b['tabs'].repeat_interleave(k),aux['tabs']])
                history=torch.cat([b['history'].repeat_interleave(k,dim=0),aux['history']])
                items=torch.cat([torch.as_tensor(choice['items'].ravel(),device=device,dtype=torch.long),aux['items']])
                with torch.no_grad():
                    tu=model.teacher.encode_user(aux['users'],aux['tabs'],aux['history']);ti=model.teacher.encode_item(aux['items'])
                    aux_base=(tu*ti).sum(1)/model.teacher.temperature
                canonical=torch.cat([torch.as_tensor(choice['base_scores'].ravel(),device=device),aux_base])
                out=model(users,items,tabs,history,base_scores=canonical)
                choice_loss=F.cross_entropy(out['rank'][:q*k].reshape(q,k),torch.zeros(q,device=device,dtype=torch.long))
                bce=F.binary_cross_entropy_with_logits(out['ctr'][q*k:],aux['labels'])
                reg=out['delta'][:q*k].square().mean()
                loss=choice_loss+config['aux_weight']*bce+config['delta_reg']*reg
                values={'choice':choice_loss.item(),'observed_bce':bce.item(),'delta_square':reg.item()}
            else:
                b=tensor_batch(train,order[offset:offset+size],device)
                if len(b['users'])==1:model.eval()
                out=model(b['users'],b['items'],b['tabs'],b['history'])
                loss=F.binary_cross_entropy_with_logits(out['ctr'],b['labels']);values={'observed_bce':loss.item()}
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite revision loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite revision gradients')
            optimizer.step();steps+=1;updates+=1
            for key,value in values.items():totals[key]=totals.get(key,0.)+value
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-epoch_start
        metrics=assess(model,validation,validation_candidates,config,directory/f'validation_epoch{epoch}.npz')
        row={'epoch':epoch,'losses':{k:v/steps for k,v in totals.items()},'train_seconds':seconds,'validation':metrics,
             'learned_repeat_weight':float(model.repeat_linear.weight.detach().item())};curve.append(row)
        print(f'{directory.name} epoch={epoch} loss={row["losses"]} valN20={metrics["ranking"]["ndcg@20"]:.6f} AUC={metrics["pointwise"]["auc"]:.6f}',flush=True)
        if select(metrics)>best:best=select(metrics);best_epoch=epoch;best_metrics=save(epoch,metrics);stale=0
        else:stale+=1
        (directory/'learning_curve.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
        if stale>=config['patience']:break
    final_digest=parameter_digest(model)
    restored,checkpoint=load_model(path,device);probe=tensor_batch(validation,np.arange(min(16,len(validation))),device)
    with torch.no_grad():actual=restored(probe['users'],probe['items'],probe['tabs'],probe['history'])['rank'].cpu()
    torch.testing.assert_close(actual,checkpoint['reload_probe'].cpu(),rtol=0,atol=0)
    teacher_unchanged=model.teacher is None or (teacher_before==parameter_digest(model.teacher) and all(p.grad is None for p in model.teacher.parameters()))
    if not teacher_unchanged:raise ValueError('Frozen retriever was modified')
    result={'status':'TRAINED','config':config,'seed':seed,'signature':signature,'fingerprint':fingerprint,
            'baseline_validation':base,'best_epoch':best_epoch,'best_validation':best_metrics,'curve':curve,
            'selected_baseline':best_epoch==-1,'optimization_changed_parameters':initial!=final_digest,
            'checkpoint':str(path.resolve()),'checkpoint_sha256':digest_file(path),'reload_identical':True,
            'teacher_unchanged':teacher_unchanged,'teacher_parameter_sha256':teacher_before,
            'trained_parameter_sha256':parameter_digest(restored),'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-start,
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
