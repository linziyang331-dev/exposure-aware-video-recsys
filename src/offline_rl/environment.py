"""Known synthetic transitions; no inference of real KuaiRand user dynamics."""
import numpy as np
from scipy.special import softmax


def validate_world(world):
    r=np.asarray(world['reward']);p=np.asarray(world['transition']);initial=np.asarray(world['initial'])
    if (r.ndim!=2 or min(r.shape)<2 or p.shape!=(*r.shape,r.shape[0]) or initial.shape!=(r.shape[0],)
            or not np.isfinite(r).all() or ((r<0)|(r>1)).any()
            or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1.,rtol=0,atol=1e-12)
            or not np.isfinite(initial).all() or (initial<0).any() or not np.isclose(initial.sum(),1.)
            or not isinstance(world['horizon'],(int,np.integer)) or world['horizon']<1):
        raise ValueError('Invalid finite MDP probabilities, rewards or horizon')


def make_world(config):
    for name in ['topics','fatigue_levels','horizon','world_seed']:
        if not isinstance(config[name],int) or isinstance(config[name],bool) or config[name]<(0 if name=='world_seed' else 1):
            raise ValueError('Invalid integer world '+name)
    topics,fatigue,h=config['topics'],config['fatigue_levels'],config['horizon']
    if min(topics,fatigue)<2 or h<1:raise ValueError('At least two topics/fatigue levels and a positive horizon required')
    states,actions=topics*fatigue,topics*2
    reward=np.empty((states,actions));transition=np.zeros((states,actions,states))
    rng=np.random.default_rng(config['world_seed']);quality=rng.uniform(-.025,.025,actions)
    for state in range(states):
        interest,level=divmod(state,fatigue)
        for action in range(actions):
            topic,form=divmod(action,2);match=topic==interest
            reward[state,action]=np.clip(.12+.58*match+.12*(form==0)
                -match*level*(.23 if form==0 else .08)+quality[action],.02,.98)
            interests=np.full(topics,.03/topics)
            interests[interest]+=.97*(1. if match else .65)
            if not match:interests[topic]+=.97*.35
            levels=np.zeros(fatigue)
            target=min(level+1,fatigue-1) if match and form==0 else max(level-1,0)
            if not match:target=0
            levels[target]+=.85;levels[level]+=.15
            transition[state,action]=(interests[:,None]*levels[None,:]).ravel()
    initial=np.zeros(states);initial[::fatigue]=1/topics
    world={'reward':reward,'transition':transition,'initial':initial,'horizon':int(h),
           'topics':topics,'fatigue_levels':fatigue,'world_seed':config['world_seed']}
    validate_world(world);return world


def policy_array(world, policy):
    s,a=world['reward'].shape;h=world['horizon'];pi=np.asarray(policy,dtype=np.float64)
    if pi.shape==(s,a):pi=np.broadcast_to(pi,(h,s,a))
    if (pi.shape!=(h,s,a) or not np.isfinite(pi).all() or (pi<0).any()
            or not np.allclose(pi.sum(-1),1.,rtol=0,atol=1e-12)):
        raise ValueError('Finite normalized policy must align with time/state/action space')
    return pi


def behavior_policy(world, scenario):
    if scenario not in ['exploratory','narrow']:raise ValueError('Unknown logging scenario')
    s,a=world['reward'].shape;fatigue=world['fatigue_levels']
    match=np.arange(s)[:,None]//fatigue==np.arange(a)[None,:]//2
    score=1.5*match+.8*(np.arange(a)[None,:]%2==0)+np.linspace(.6,-.6,a)[None,:]
    if scenario=='exploratory':return .9*softmax(score,axis=1)+.1/a
    keep=np.argsort(-score,axis=1,kind='stable')[:,:2]
    masked=np.full_like(score,-np.inf);row=np.arange(s)[:,None];masked[row,keep]=score[row,keep]
    return softmax(masked,axis=1)


def evaluate_policy(world, policy, gamma):
    validate_world(world)
    if not np.isfinite(gamma) or not 0<=gamma<=1:raise ValueError('Discount must be in [0,1]')
    pi=policy_array(world,policy);h,s,_=pi.shape
    def backward(discount):
        values=np.zeros((h+1,s));q=np.zeros_like(pi)
        for t in range(h-1,-1,-1):
            q[t]=world['reward']+discount*np.einsum('sak,k->sa',world['transition'],values[t+1])
            values[t]=(pi[t]*q[t]).sum(1)
        return values,q
    values,q=backward(gamma);undiscounted,_=backward(1.)
    occupancy=np.zeros((h,s));occupancy[0]=world['initial']
    for t in range(1,h):occupancy[t]=np.einsum('s,sa,sak->k',occupancy[t-1],pi[t-1],world['transition'])
    forward=float(np.sum((occupancy[:,:,None]*pi*world['reward'][None,:,:]).sum((1,2))*gamma**np.arange(h)))
    value=float(world['initial']@values[0])
    if not np.isclose(value,forward,rtol=1e-12,atol=1e-12):raise ValueError('Forward/backward value disagreement')
    return {'discounted_return':value,'undiscounted_return':float(world['initial']@undiscounted[0]),
            'state_values':values,'q_values':q,'occupancy':occupancy,'forward_return':forward}


def optimal_policy(world,gamma):
    validate_world(world)
    if not np.isfinite(gamma) or not 0<=gamma<=1:raise ValueError('Discount must be in [0,1]')
    s,a=world['reward'].shape;h=world['horizon'];value=np.zeros(s);policy=np.zeros((h,s,a))
    for t in range(h-1,-1,-1):
        q=world['reward']+gamma*np.einsum('sak,k->sa',world['transition'],value)
        chosen=q.argmax(1);policy[t,np.arange(s),chosen]=1.;value=q[np.arange(s),chosen]
    return policy
