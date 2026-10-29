"""Exact simulator returns, episode uncertainty, and distinct support diagnostics."""
import numpy as np
from src.offline_rl.data import collect_log
from src.offline_rl.environment import evaluate_policy,policy_array


def rollout(world,policy,episodes,seed,gamma):
    if not np.isfinite(gamma) or not 0<=gamma<=1:raise ValueError('Invalid discount')
    log=collect_log(world,policy,episodes,seed)
    rewards=log['reward'].reshape(episodes,world['horizon'])
    return {'returns':rewards@(gamma**np.arange(world['horizon'])),
            'undiscounted':rewards.sum(1)}


def mean_interval(values,repeats,seed):
    values=np.asarray(values,dtype=np.float64)
    if (values.ndim!=1 or len(values)<2 or not np.isfinite(values).all()
            or not isinstance(repeats,int) or repeats<1):raise ValueError('Finite independent episode values and positive repeats required')
    rng=np.random.default_rng(seed);draws=np.empty(repeats)
    for i in range(repeats):draws[i]=values[rng.integers(0,len(values),size=len(values))].mean()
    lo,hi=np.quantile(draws,[.025,.975]);sd=float(values.std(ddof=1))
    return {'mean':float(values.mean()),'sample_std':sd,'standard_error':sd/np.sqrt(len(values)),
            'low':float(lo),'high':float(hi),'episodes':len(values),'bootstrap_repeats':repeats,
            'confidence':.95,'unit':'independent complete simulated episodes; fixed trained policies'}


def compare_returns(first,second,repeats,seed):
    first=np.asarray(first);second=np.asarray(second)
    if first.shape!=second.shape:raise ValueError('Paired evaluation episode shapes differ')
    return mean_interval(second-first,repeats,seed)


def empirical_bc(counts):
    counts=np.asarray(counts)
    if counts.ndim!=3 or not np.isfinite(counts).all() or (counts<0).any() or counts.sum()<=0:
        raise ValueError('Nonempty time/state/action counts required')
    global_action=counts.sum((0,1)).astype(float);global_action/=global_action.sum()
    total=counts.sum(-1,keepdims=True)
    return np.divide(counts,total,out=np.broadcast_to(global_action,counts.shape).copy(),where=total>0)


def policy_metrics(world,policy,mu,counts,gamma,q_prediction=None):
    pi=policy_array(world,policy);behavior=policy_array(world,mu);counts=np.asarray(counts)
    if counts.shape!=pi.shape or not np.isfinite(counts).all() or (counts<0).any():raise ValueError('Invalid action counts')
    exact=evaluate_policy(world,pi,gamma)
    occupancy=exact['occupancy'][:,:,None]*pi
    frequency=occupancy/world['horizon']
    action_mass=frequency.sum((0,1))
    result={key:exact[key] for key in ['discounted_return','undiscounted_return','forward_return']}
    result.update(zero_behavior_support_fraction=float(frequency[behavior==0].sum()),
        unobserved_training_action_fraction=float(frequency[counts==0].sum()),
        action_catalog_coverage=float((action_mass>1e-12).mean()),action_occupancy=action_mass.tolist())
    if q_prediction is not None:
        q=np.asarray(q_prediction)
        if q.shape!=pi.shape or not np.isfinite(q).all():raise ValueError('Invalid predicted Q table')
        predicted=float((world['initial'][:,None]*pi[0]*q[0]).sum())
        result.update(predicted_start_value=predicted,q_start_bias=predicted-exact['discounted_return'],
            occupancy_weighted_policy_q_mse=float((frequency*(q-exact['q_values'])**2).sum()))
    return result
