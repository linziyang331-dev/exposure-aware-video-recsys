"""Whole synthetic episodes, with explicit terminal and transition validation."""
import numpy as np
from src.offline_rl.environment import policy_array,validate_world


def collect_log(world,mu,episodes,seed,episode_offset=0):
    validate_world(world);pi=policy_array(world,mu)
    if (not isinstance(episodes,(int,np.integer)) or episodes<1
            or not isinstance(episode_offset,(int,np.integer)) or episode_offset<0):
        raise ValueError('Positive episode count and nonnegative integer offset required')
    rng=np.random.default_rng(seed);h=world['horizon'];s,a=world['reward'].shape
    uniforms=rng.random((episodes,h,3))
    state=np.searchsorted(world['initial'].cumsum(),rng.random(episodes),side='right').clip(max=s-1)
    shape=(episodes,h)
    log={key:np.empty(shape,dtype=np.int64) for key in ['state','action','next_state']}
    log['reward']=np.empty(shape);log['pscore']=np.empty(shape)
    for t in range(h):
        action=(pi[t,state].cumsum(1)<uniforms[:,t,0,None]).sum(1).clip(max=a-1)
        next_state=(world['transition'][state,action].cumsum(1)<uniforms[:,t,1,None]).sum(1).clip(max=s-1)
        log['state'][:,t]=state;log['action'][:,t]=action;log['next_state'][:,t]=next_state
        log['reward'][:,t]=uniforms[:,t,2]<world['reward'][state,action]
        log['pscore'][:,t]=pi[t,state,action];state=next_state
    log={key:values.ravel() for key,values in log.items()}
    log['episode_id']=np.repeat(np.arange(episode_offset,episode_offset+episodes),h)
    log['time']=np.tile(np.arange(h),episodes);log['terminated']=log['time']==h-1
    validate_log(world,mu,log);return log


def validate_log(world,mu,log):
    validate_world(world);pi=policy_array(world,mu);h,s,a=pi.shape
    metadata=validate_log_structure(log,s,a,h)
    r=np.asarray(log['reward']);p=np.asarray(log['pscore']);st=log['state'];act=log['action'];ns=log['next_state']
    if not np.allclose(p,pi[log['time'],st,act],rtol=0,atol=1e-12):raise ValueError('Logged behavior probability mismatch')
    if (world['transition'][st,act,ns]<=0).any():raise ValueError('Impossible recorded transition')
    if ((world['reward'][st,act]==0)&(r==1)).any() or ((world['reward'][st,act]==1)&(r==0)).any():
        raise ValueError('Impossible recorded reward')
    if (world['initial'][log['state'].reshape(-1,h)[:,0]]<=0).any():raise ValueError('Impossible initial state')
    return metadata


def validate_log_structure(log,s,a,h):
    """Validation available to the learner without oracle MDP tables."""
    required={'episode_id','time','state','action','reward','next_state','terminated','pscore'}
    if set(log)!=required:raise ValueError('Logged transition fields differ')
    n=len(log['action'])
    if n<1 or n%h or any(np.asarray(v).shape!=(n,) for v in log.values()):raise ValueError('Complete aligned episodes required')
    for key,limit in [('state',s),('next_state',s),('action',a),('time',h)]:
        v=np.asarray(log[key])
        if v.dtype.kind not in 'iu' or ((v<0)|(v>=limit)).any():raise ValueError('Invalid integer '+key)
    ids=np.asarray(log['episode_id']);matrix=ids.reshape(-1,h)
    if ids.dtype.kind not in 'iu' or (ids<0).any() or (matrix!=matrix[:,:1]).any() or len(np.unique(ids))!=n//h:
        raise ValueError('Each episode must occur once as a complete contiguous block')
    if not np.array_equal(log['time'],np.tile(np.arange(h),n//h)):raise ValueError('Episode time order changed')
    done=np.asarray(log['terminated'])
    if done.dtype.kind!='b' or not np.array_equal(done,log['time']==h-1):raise ValueError('Finite horizon terminal flags changed')
    if not np.array_equal(log['next_state'].reshape(-1,h)[:,:-1],log['state'].reshape(-1,h)[:,1:]):
        raise ValueError('next_state does not match the next transition')
    r=np.asarray(log['reward']);p=np.asarray(log['pscore']);st=log['state'];act=log['action'];ns=log['next_state']
    if not np.isfinite(r).all() or not np.isin(r,[0.,1.]).all():raise ValueError('Binary logged rewards required')
    if not np.isfinite(p).all() or ((p<=0)|(p>1)).any():raise ValueError('Positive logged probabilities required')
    return {'episodes':n//h,'transitions':n,'terminal_rows':int(done.sum())}


def action_counts(log,states,actions,horizon):
    result=np.zeros((horizon,states,actions),dtype=np.int64)
    np.add.at(result,(log['time'],log['state'],log['action']),1)
    return result
