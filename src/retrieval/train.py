"""Validation-selected GPU training; only the caller-supplied train/validation data."""
import argparse
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
from src.retrieval.evaluate import Scorer,evaluate_partition
from src.retrieval.model import TwoTower,contrastive_loss


def architecture(config,num_users,num_items,use_history):
    return {name:config[name] for name in ['embedding_dim','output_dim','hidden_dim','temperature']}|{
        'num_users':num_users,'num_items':num_items,'use_history':use_history}


def load_model(path,device='cuda'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model=TwoTower(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    return model,checkpoint


def train_run(train,validation,counts,config,directory,seed,use_history=True,*,smoke=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    signature={'config':config,'seed':seed,'use_history':use_history,'smoke':smoke,
               'train':train.metadata['signature'],'validation':validation.metadata['signature'],
               'model_code_sha256':digest_file(Path(__file__).parent/'model.py'),'train_code_sha256':digest_file(__file__)}
    signature['evaluator_code_sha256']=digest_file(Path(__file__).parent/'evaluate.py')
    signature['metric_code_sha256']=digest_file(Path(__file__).parent/'metrics.py')
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True,default=str).encode()).hexdigest()
    result_path=directory/'training.json';checkpoint_path=directory/'best.pt'
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous['fingerprint']!=fingerprint or digest_file(checkpoint_path)!=previous['checkpoint_sha256']:
            raise ValueError('Existing training run fingerprint/checkpoint changed; preserve it and use another directory')
        print(f'Reusing completed {directory.name}',flush=True)
        return previous
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if config['device']=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for actual retrieval run')
        torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(config['device'])
    arch=architecture(config,int(train['user_index'].max()),len(counts)-1,use_history)
    model=TwoTower(**arch).to(device)
    def parameter_digest(state):
        digest=hashlib.sha256()
        for name,value in sorted(state.items()):digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    initial_digest=parameter_digest(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    all_indices=np.arange(len(train))
    if smoke:all_indices=all_indices[:min(4096,len(all_indices))]
    generator=np.random.default_rng(seed)
    best=-1.;best_epoch=-1;stale=0;curve=[];start=time.perf_counter();updates=0
    for epoch in range(1 if smoke else config['epochs']):
        model.train();epoch_start=time.perf_counter();observed_total=contrastive_total=0.;samples=0
        order=generator.permutation(all_indices)
        for offset in range(0,len(order),config['batch_size']):
            indices=order[offset:offset+config['batch_size']];batch=tensor_batch(train,indices,device)
            optimizer.zero_grad(set_to_none=True)
            user=model.encode_user(batch['users'],batch['tabs'],batch['history'])
            item=model.encode_item(batch['items'])
            logits=(user*item).sum(dim=1)/model.temperature
            observed=F.binary_cross_entropy_with_logits(logits,batch['labels'])
            positives=torch.nonzero(batch['labels']==1).flatten()
            if len(positives)>config['contrastive_max_queries']:
                positives=positives[torch.randperm(len(positives),device=device)[:config['contrastive_max_queries']]]
            contrastive=torch.zeros((),device=device)
            if len(positives)>1 and config['contrastive_weight']:
                targets=batch['items'][positives];pool=torch.unique(targets)
                contrastive=contrastive_loss(user[positives],model.encode_item(pool),targets,pool,
                                             batch['history'][positives],model.temperature)
            loss=observed+config['contrastive_weight']*contrastive
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
            optimizer.step();updates+=1
            observed_total+=observed.item()*len(indices);contrastive_total+=contrastive.item()*len(indices);samples+=len(indices)
        if device.type=='cuda':torch.cuda.synchronize()
        train_seconds=time.perf_counter()-epoch_start
        scorer=Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),device,model=model)
        metrics=evaluate_partition(scorer,validation,counts,batch_size=config['eval_batch_size'],query_limit=128 if smoke else None)
        row={'epoch':epoch,'observed_bce':observed_total/samples,'contrastive':contrastive_total/samples,
             'train_seconds':train_seconds,'validation':metrics}
        curve.append(row);score=metrics['ndcg@20']
        print(f'{directory.name} epoch={epoch} BCE={row["observed_bce"]:.5f} contrast={row["contrastive"]:.5f} val_NDCG20={score:.6f} train_s={train_seconds:.2f}',flush=True)
        if score>best:
            best=score;best_epoch=epoch;stale=0
            probe=tensor_batch(validation,np.arange(min(32,len(validation))),device)
            with torch.no_grad():probe_scores=model(probe['users'],probe['items'],probe['tabs'],probe['history']).cpu()
            torch.save({'architecture':arch,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'seed':seed,'best_epoch':epoch,'training_fingerprint':fingerprint,'smoke_only':smoke,
                        'reload_probe_scores':probe_scores},checkpoint_path)
        else:stale+=1
        (directory/'learning_curve.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
        if stale>=config['patience']:break
    restored,checkpoint=load_model(checkpoint_path,device)
    probe=tensor_batch(validation,np.arange(min(32,len(validation))),device)
    with torch.no_grad():
        actual=restored(probe['users'],probe['items'],probe['tabs'],probe['history']).cpu()
    torch.testing.assert_close(actual,checkpoint['reload_probe_scores'].cpu(),rtol=0,atol=0)
    trained_digest=parameter_digest(restored.state_dict())
    if trained_digest==initial_digest:raise AssertionError('Training did not change parameters')
    result={'status':'TRAINED','seed':seed,'use_history':use_history,'smoke_only':smoke,'fingerprint':fingerprint,
            'signature':signature,'best_epoch':best_epoch,'best_validation_ndcg20':best,
            'checkpoint_sha256':digest_file(checkpoint_path),'checkpoint':str(checkpoint_path),
            'checkpoint_reload_identical':True,'optimizer_updates':updates,'curve':curve,
            'initial_parameter_sha256':initial_digest,'trained_parameter_sha256':trained_digest,'parameters_changed':True,
            'elapsed_seconds':time.perf_counter()-start,'process_peak_working_set_MiB':psutil.Process().memory_info().peak_wset/2**20,
            'peak_torch_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
    result_path.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--no-history',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    from src.retrieval.run import load_setup
    config,paths,vocabulary,counts,caches=load_setup(['standard/train','standard/validation'],smoke=args.smoke)
    name=('two_tower_no_history' if args.no_history else 'two_tower')+f'_seed{args.seed}'
    train_run(caches['standard/train'],caches['standard/validation'],counts,config,paths['artifacts']/name,args.seed,not args.no_history,smoke=args.smoke)


if __name__=='__main__':main()
