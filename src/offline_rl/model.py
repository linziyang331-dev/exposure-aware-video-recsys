"""Time-aware multihead Q network for a small, fully observed simulation."""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot


class QNetwork(nn.Module):
    def __init__(self,states,horizon,actions,hidden):
        super().__init__()
        if min(states,actions)<2 or horizon<1 or not hidden or min(hidden)<1:raise ValueError('Invalid network dimensions')
        self.spec={'states':states,'horizon':horizon,'actions':actions,'hidden':list(hidden)}
        widths=[states+horizon+1,*hidden,actions];layers=[]
        for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(widths)-2:layers.append(nn.ReLU())
        self.network=nn.Sequential(*layers)

    def encode(self,state,time):
        if (state.ndim!=1 or time.shape!=state.shape or state.dtype!=torch.int64 or time.dtype!=torch.int64
                or ((state<0)|(state>=self.spec['states'])).any()
                or ((time<0)|(time>self.spec['horizon'])).any()):raise ValueError('Invalid time/state indices')
        return torch.cat([one_hot(state,self.spec['states']),one_hot(time,self.spec['horizon']+1)],dim=1).float()

    def forward(self,state,time):return self.network(self.encode(state,time))


def all_q_values(model):
    model.eval();s=model.spec['states'];h=model.spec['horizon']
    with torch.no_grad():q=model(torch.arange(s).repeat(h),torch.arange(h).repeat_interleave(s))
    return q.numpy().reshape(h,s,model.spec['actions']).copy()


def greedy_policy(q):
    q=np.asarray(q)
    if q.ndim!=3 or not np.isfinite(q).all():raise ValueError('Finite time/state/action Q table required')
    result=np.zeros_like(q,dtype=np.float64);t,s=np.indices(q.shape[:2])
    result[t,s,q.argmax(-1)]=1.;return result
