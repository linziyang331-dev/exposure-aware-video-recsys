"""Fixed-budget learning from static logged transitions; no oracle inputs."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from src.offline_rl.data import validate_log_structure
from src.offline_rl.loss import cql_loss
from src.offline_rl.model import QNetwork,all_q_values


def log_fingerprint(log):
    digest=hashlib.sha256()
    for key,array in sorted(log.items()):
        array=np.ascontiguousarray(array)
        digest.update(json.dumps([key,str(array.dtype),array.shape]).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def _config(config):
    expected={'states','actions','horizon','hidden','steps','batch_size','learning_rate','gamma',
              'target_update','trace_interval','gradient_clip'}
    if set(config)!=expected:raise ValueError('Training accepts dimensions and optimizer settings only')
    for name in ['states','actions','horizon','steps','batch_size','target_update','trace_interval']:
        if not isinstance(config[name],int) or isinstance(config[name],bool) or config[name]<1:raise ValueError('Invalid '+name)
    if not isinstance(config['hidden'],list) or not config['hidden'] or any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in config['hidden']):
        raise ValueError('Hidden widths must be positive integers')
    if (not np.isfinite([config['gamma'],config['learning_rate'],config['gradient_clip']]).all()
            or not 0<=config['gamma']<=1 or min(config['learning_rate'],config['gradient_clip'])<=0):
        raise ValueError('Invalid optimizer or discount configuration')


def _tensors(log):
    return {name:torch.as_tensor(log[name]) for name in ['state','time','action','reward','next_state','terminated']}


def _loss(model,target,batch,gamma,alpha):
    return cql_loss(model(batch['state'],batch['time']),batch['action'],batch['reward'].float(),
        target(batch['next_state'],batch['time']+1),batch['terminated'],gamma,alpha)


def fit_model(train_log,validation_log,config,alpha,seed,output):
    _config(config)
    if not np.isfinite(alpha) or alpha<0 or not isinstance(seed,int) or seed<0:raise ValueError('Invalid alpha/seed')
    for log in [train_log,validation_log]:validate_log_structure(log,config['states'],config['actions'],config['horizon'])
    if np.intersect1d(train_log['episode_id'],validation_log['episode_id']).size:raise ValueError('Training and validation episodes overlap')
    signature={'config':config,'alpha':float(alpha),'seed':seed,'train_log_sha256':log_fingerprint(train_log),
               'validation_log_sha256':log_fingerprint(validation_log),'implementation':'discrete_CQL_H_MSE_temperature1'}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(seed);model=QNetwork(config['states'],config['horizon'],config['actions'],config['hidden'])
        target=copy.deepcopy(model);target.requires_grad_(False);target.eval()
        initial=torch.cat([p.detach().ravel().clone() for p in model.parameters()])
        initial_hash=hashlib.sha256(initial.numpy().tobytes()).hexdigest()
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        train=_tensors(train_log);valid=_tensors(validation_log);rng=np.random.default_rng(seed+101)
        history=[];q_history=[];target_history=[];snapshot_steps=[];sample_digest=hashlib.sha256();gradient_max=0.
        diagnostic={key:value[:2048] for key,value in train.items()}
        def snapshot(step):
            model.eval()
            with torch.no_grad():
                a=_loss(model,target,diagnostic,config['gamma'],alpha)
                b=_loss(model,target,valid,config['gamma'],alpha)
            history.append({'step':step,'train_diagnostic_rows':len(diagnostic['state']),
                'train_loss':float(a['loss']),'train_bellman':float(a['bellman']),
                'train_conservative':float(a['conservative']),'validation_bellman':float(b['bellman']),
                'max_gradient_norm_so_far':gradient_max})
            q_history.append(all_q_values(model));target_history.append(all_q_values(target));snapshot_steps.append(step)
        snapshot(0)
        for step in range(1,config['steps']+1):
            indices=rng.integers(0,len(train_log['action']),size=config['batch_size'])
            sample_digest.update(indices.tobytes());batch={k:v[indices] for k,v in train.items()}
            model.train();loss=_loss(model,target,batch,config['gamma'],alpha)
            if not torch.isfinite(loss['loss']):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            gradient_max=max(gradient_max,float(norm));optimizer.step()
            if step%config['target_update']==0:target.load_state_dict(model.state_dict())
            if step%config['trace_interval']==0 or step==config['steps']:snapshot(step)
        final=torch.cat([p.detach().ravel() for p in model.parameters()])
        result={'signature':signature,'steps':config['steps'],'snapshot_steps':snapshot_steps,'history':history,
                'initial_parameter_sha256':initial_hash,'batch_sequence_sha256':sample_digest.hexdigest(),
                'parameter_change_norm':float(torch.linalg.vector_norm(final-initial)),
                'target_has_gradients':any(p.grad is not None for p in target.parameters()),
                'checkpoint_selection':'fixed final step; no reward-based selection',
                'oracle_inputs_used_for_training':False}
        model.eval();torch.save({'state_dict':model.state_dict(),'spec':model.spec,'signature':signature},output/'model.pt')
        np.save(output/'q_history.npy',np.stack(q_history))
        np.save(output/'target_q_history.npy',np.stack(target_history))
        (output/'training.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        reloaded=load_checkpoint(output/'model.pt',signature)
        np.testing.assert_array_equal(all_q_values(reloaded),q_history[-1])
        return result


def load_checkpoint(path,expected_signature):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['signature']!=expected_signature:raise ValueError('Checkpoint training input/config signature changed')
    model=QNetwork(**saved['spec']);model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite checkpoint parameters')
    return model
