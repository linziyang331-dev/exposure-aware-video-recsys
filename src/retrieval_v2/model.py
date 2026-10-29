"""Shared item-space history residuals, independent of the candidate target."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerV2(nn.Module):
    def __init__(self,num_users,num_items,embedding_dim=32,temperature=.1,
                 history_mode='mean',item_bias=False,history_decay=.9):
        super().__init__()
        if history_mode not in {'none','mean','recency'}:raise ValueError('Unknown history mode')
        if temperature<=0 or not 0<history_decay<=1:raise ValueError('Invalid temperature/decay')
        self.temperature=float(temperature);self.history_mode=history_mode
        self.history_decay=float(history_decay);self.has_bias=bool(item_bias)
        self.user_embedding=nn.Embedding(num_users+1,embedding_dim,padding_idx=0)
        self.item_embedding=nn.Embedding(num_items+1,embedding_dim,padding_idx=0)
        self.tab_embedding=nn.Embedding(15,embedding_dim)
        self.history_scale=nn.Parameter(torch.tensor(0.))
        if item_bias:self.item_bias=nn.Embedding(num_items+1,1,padding_idx=0)
        for layer in [self.user_embedding,self.item_embedding,self.tab_embedding]:
            nn.init.normal_(layer.weight,std=.02)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_();self.item_embedding.weight[0].zero_()
            if item_bias:self.item_bias.weight.zero_()

    def encode_item(self,items):
        vector=F.normalize(self.item_embedding(items),dim=-1,eps=1e-8)
        if self.has_bias:
            vector=torch.cat([vector,self.temperature*self.item_bias(items)],dim=-1)
        return vector

    def encode_user(self,users,tabs,history):
        query=self.user_embedding(users)+self.tab_embedding(tabs)
        if self.history_mode!='none':
            mask=history>0
            vectors=F.normalize(self.item_embedding(history),dim=-1,eps=1e-8)
            weights=mask.to(vectors.dtype)
            if self.history_mode=='recency':
                age=mask.sum(1,keepdim=True)-mask.cumsum(1)
                weights=weights*torch.exp(age.to(vectors.dtype)*math.log(self.history_decay))
            pooled=(vectors*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-8)
            query=query+self.history_scale.exp().clamp(max=10)*pooled
        vector=F.normalize(query,dim=-1,eps=1e-8)
        if self.has_bias:vector=torch.cat([vector,torch.ones_like(vector[:,:1])],dim=-1)
        return vector

    def forward(self,users,items,tabs,history):
        return (self.encode_user(users,tabs,history)*self.encode_item(items)).sum(-1)/self.temperature
