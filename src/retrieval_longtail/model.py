"""Candidate-independent context attention and adaptive historical contribution."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.retrieval_v2.model import TwoTowerV2


class GatedTwoTower(TwoTowerV2):
    def __init__(self,num_users,num_items,embedding_dim=32,temperature=.1,
                 history_mode='recency',item_bias=False,history_decay=.9):
        super().__init__(num_users,num_items,embedding_dim,temperature,history_mode,item_bias,history_decay)
        if history_mode!='recency' or item_bias:raise ValueError('Gated experiment uses recency and no item bias')
        self.attention_query=nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.history_gate=nn.Linear(embedding_dim*2+1,1)
        nn.init.zeros_(self.attention_query.weight)
        nn.init.zeros_(self.history_gate.weight);nn.init.zeros_(self.history_gate.bias)

    def encode_user(self,users,tabs,history):
        context=self.user_embedding(users)+self.tab_embedding(tabs)
        mask=history>0;length=mask.sum(1,keepdim=True)
        vectors=F.normalize(self.item_embedding(history),dim=-1,eps=1e-8)
        query=self.attention_query(F.normalize(context,dim=-1,eps=1e-8))
        age=length-mask.cumsum(1)
        logits=(vectors*query[:,None,:]).sum(-1)+age.to(vectors.dtype)*math.log(self.history_decay)
        # Finite mask avoids softmax(-inf,...,-inf) for empty histories.
        weights=torch.softmax(logits.masked_fill(~mask,-1e4),dim=1)*mask
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        pooled=(vectors*weights.unsqueeze(-1)).sum(1)
        features=torch.cat([F.normalize(context,dim=-1,eps=1e-8),
                            F.normalize(pooled,dim=-1,eps=1e-8),length.to(context.dtype)/50],dim=1)
        gate=2*torch.sigmoid(self.history_gate(features))
        return F.normalize(context+self.history_scale.exp().clamp(max=10)*gate*pooled,dim=-1,eps=1e-8)
