"""Thin wrappers around complete, unmodified RecBole DIN and LR models."""
import torch
from torch import nn
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender.din import DIN
from recbole.model.context_aware_recommender.lr import LR
from src.ranking.adapter import DatasetAdapter,upstream_config


def recent_history(history,length):
    if history.ndim!=2 or length<=0 or history.shape[1]==0:raise ValueError('Nonempty padded history width required')
    lengths=(history>0).sum(1);kept=lengths.clamp(max=length)
    offsets=(lengths-kept)[:,None]+torch.arange(length,device=history.device)[None,:]
    gathered=history.gather(1,offsets.clamp(max=history.shape[1]-1))
    return gathered.masked_fill(torch.arange(length,device=history.device)[None,:]>=kept[:,None],0)


class RankingModel(nn.Module):
    def __init__(self,num_users,num_items,config):
        super().__init__()
        self.config=dict(config);self.kind=config['kind'];self.history_length=config['history_length']
        config_upstream=upstream_config(config);dataset=DatasetAdapter(num_users,num_items,config_upstream)
        if self.kind=='din':self.core=DIN(config_upstream,dataset)
        elif self.kind=='lr':self.core=LR(config_upstream,dataset)
        else:raise ValueError('Unknown ranking model')
        if self.kind=='din' and config['context_bias']:
            self.user_bias=nn.Embedding(num_users+1,1,padding_idx=0);self.tab_bias=nn.Embedding(15,1)
            nn.init.zeros_(self.user_bias.weight);nn.init.zeros_(self.tab_bias.weight)
        self.to(config['device'])

    def forward(self,users,items,tabs,history):
        if self.kind=='lr':return self.core(Interaction({'user_id':users,'item_id':items,'tab':tabs}))
        history=recent_history(history,self.history_length)
        logits=self.core.forward(users,history,(history>0).sum(1),items)
        if hasattr(self,'user_bias'):logits=logits+self.user_bias(users).squeeze(1)+self.tab_bias(tabs).squeeze(1)
        return logits

    def initialize_item_embeddings(self,weight):
        if self.kind!='din':raise ValueError('Pretrained item embeddings apply to DIN only')
        table=self.core.embedding_layer.token_embedding_table['item'].embedding.weight
        if tuple(table.shape)!=tuple(weight.shape):raise ValueError('Pretrained item shape mismatch')
        with torch.no_grad():table.copy_(weight.to(table.device))
