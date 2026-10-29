"""A decomposable, history/context two-tower implemented with standard PyTorch."""
import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    def __init__(self,num_users,num_items,embedding_dim=32,output_dim=32,hidden_dim=64,temperature=.1,use_history=True):
        super().__init__()
        if temperature<=0:raise ValueError('Temperature must be positive')
        self.temperature=float(temperature);self.use_history=use_history
        self.user_embedding=nn.Embedding(num_users+1,embedding_dim,padding_idx=0)
        self.item_embedding=nn.Embedding(num_items+1,embedding_dim,padding_idx=0)
        self.tab_embedding=nn.Embedding(15,8)
        self.user_mlp=nn.Sequential(nn.Linear(2*embedding_dim+8,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
        self.item_mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))

    def encode_user(self,users,tabs,history):
        embedded=self.item_embedding(history)
        mask=(history>0).unsqueeze(-1)
        mean=(embedded*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
        if not self.use_history:mean=torch.zeros_like(mean)
        vector=self.user_mlp(torch.cat([self.user_embedding(users),mean,self.tab_embedding(tabs)],dim=1))
        return F.normalize(vector,p=2,dim=1,eps=1e-8)

    def encode_item(self,items):
        return F.normalize(self.item_mlp(self.item_embedding(items)),p=2,dim=1,eps=1e-8)

    def forward(self,users,items,tabs,history):
        return (self.encode_user(users,tabs,history)*self.encode_item(items)).sum(dim=1)/self.temperature


def contrastive_loss(user_vectors,item_vectors,targets,pool_ids,history,temperature):
    if len(torch.unique(pool_ids))!=len(pool_ids):raise ValueError('Contrastive pool must deduplicate item IDs')
    matches=targets[:,None]==pool_ids[None,:]
    if not matches.any(dim=1).all():raise ValueError('Each positive target must exist in contrastive pool')
    past=(history[:,:,None]==pool_ids[None,None,:]).any(dim=1)
    logits=user_vectors@item_vectors.T/temperature
    logits=logits.masked_fill(past&~matches,-torch.inf)
    return F.cross_entropy(logits,matches.to(torch.int64).argmax(dim=1))
