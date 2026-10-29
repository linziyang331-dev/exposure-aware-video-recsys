"""Shared item neighborhood encoder; independent catalog vectors and context query."""
import math
import torch
from torch.nn import functional as F
from src.retrieval_v2.model import TwoTowerV2


class SharedTower(TwoTowerV2):
    def __init__(self,num_users,num_items,embedding_dim=32,temperature=.1,
                 graph_alpha=0.,neighbors=32,history_decay=.9,history_mode='recency',item_bias=False):
        super().__init__(num_users,num_items,embedding_dim,temperature,history_mode,item_bias,history_decay)
        if graph_alpha<0 or neighbors<1 or history_mode!='recency' or item_bias:
            raise ValueError('Nonnegative graph residual and bias-free recency model required')
        self.graph_alpha=float(graph_alpha);self.neighbors=int(neighbors);self._eval_bank=None
        self.register_buffer('neighbor_ids',torch.zeros(num_items+1,neighbors,dtype=torch.long))
        self.register_buffer('neighbor_weights',torch.zeros(num_items+1,neighbors))
        self.architecture=dict(num_users=num_users,num_items=num_items,embedding_dim=embedding_dim,
            temperature=temperature,graph_alpha=graph_alpha,neighbors=neighbors,history_decay=history_decay,
            history_mode=history_mode,item_bias=item_bias)

    def set_neighbors(self,ids,weights):
        a=torch.as_tensor(ids,device=self.neighbor_ids.device,dtype=torch.long)
        w=torch.as_tensor(weights,device=self.neighbor_weights.device,dtype=torch.float32)
        if (a.shape!=self.neighbor_ids.shape or w.shape!=a.shape or (a<0).any()
                or (a>=len(self.item_embedding.weight)).any() or not torch.isfinite(w).all() or (w<0).any()
                or ((a==0)&(w!=0)).any() or (w[0]!=0).any()):raise ValueError('Invalid neighbor graph buffers')
        sums=w.sum(1);nonempty=sums>0
        if not torch.allclose(sums[nonempty],torch.ones_like(sums[nonempty]),atol=1e-5):raise ValueError('Normalized graph rows required')
        self.neighbor_ids.copy_(a);self.neighbor_weights.copy_(w);self._eval_bank=None

    def train(self,mode=True):
        self._eval_bank=None
        return super().train(mode)

    def bank(self):
        if not self.training and not torch.is_grad_enabled() and self._eval_bank is not None:return self._eval_bank
        raw=F.normalize(self.item_embedding.weight,dim=-1,eps=1e-8)
        if self.graph_alpha:
            shared=(raw[self.neighbor_ids]*self.neighbor_weights[:,:,None]).sum(1)
            value=F.normalize(raw+self.graph_alpha*F.normalize(shared,dim=-1,eps=1e-8),dim=-1,eps=1e-8)
        else:value=raw
        if not self.training and not torch.is_grad_enabled():self._eval_bank=value
        return value

    def encode_item(self,items):
        if self.graph_alpha==0:return super().encode_item(items)
        return self.bank()[items]

    def encode_user(self,users,tabs,history):
        if self.graph_alpha==0:return super().encode_user(users,tabs,history)
        query=self.user_embedding(users)+self.tab_embedding(tabs)
        mask=history>0;vectors=self.bank()[history];age=mask.sum(1,keepdim=True)-mask.cumsum(1)
        w=mask.to(vectors.dtype)*torch.exp(age.to(vectors.dtype)*math.log(self.history_decay))
        pooled=(vectors*w[:,:,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1e-8)
        return F.normalize(query+self.history_scale.exp().clamp(max=10)*pooled,dim=-1,eps=1e-8)
