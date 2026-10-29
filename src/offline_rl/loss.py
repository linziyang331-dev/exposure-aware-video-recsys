"""Discrete CQL(H), temperature one, with a detached DQN Bellman target."""
import math
import torch


def cql_loss(q,action,reward,next_q,terminated,gamma,alpha):
    if (q.ndim!=2 or not len(q) or next_q.shape!=q.shape or action.shape!=(len(q),)
            or action.dtype!=torch.int64 or reward.shape!=(len(q),) or terminated.shape!=(len(q),)
            or terminated.dtype!=torch.bool or not torch.isfinite(q).all() or not torch.isfinite(next_q).all()
            or not torch.isfinite(reward).all() or ((reward<0)|(reward>1)).any()
            or ((action<0)|(action>=q.shape[1])).any()
            or not math.isfinite(gamma) or not 0<=gamma<=1 or not math.isfinite(alpha) or alpha<0):
        raise ValueError('Invalid aligned Q values, binary terminal flags, rewards, discount or alpha')
    selected=q.gather(1,action[:,None]).squeeze(1)
    with torch.no_grad():target=reward+gamma*(~terminated).to(q.dtype)*next_q.detach().max(1).values
    bellman=(selected-target).square().mean()
    conservative=(torch.logsumexp(q,dim=1)-selected).mean()
    return {'loss':bellman+alpha*conservative,'bellman':bellman,'conservative':conservative}
