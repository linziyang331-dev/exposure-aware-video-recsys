"""Domain adapter around unmodified RecBole SASRec, including its native loss flow."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender.sasrec import SASRec
from src.ranking.adapter import DatasetAdapter
from src.ranking.model import recent_history


def prepare_history(history,length):
    seq=recent_history(history,length)
    return seq,(seq>0).sum(1)


class KnownItemCrossEntropy(nn.Module):
    """Class0 is padding, not an alternative video or a feedback observation."""
    def forward(self,logits,targets):
        if (targets<=0).any() or (targets>=logits.shape[1]).any():raise ValueError('Known positive target required')
        return F.cross_entropy(logits[:,1:],targets-1)


def upstream_config(c):
    return {'USER_ID_FIELD':'user_id','ITEM_ID_FIELD':'item_id','LIST_SUFFIX':'_list',
            'ITEM_LIST_LENGTH_FIELD':'item_length','NEG_PREFIX':'neg_',
            'MAX_ITEM_LIST_LENGTH':c['history_length'],'device':c['device'],
            'n_layers':c['n_layers'],'n_heads':c['n_heads'],'hidden_size':c['hidden_size'],
            'inner_size':c['inner_size'],'hidden_dropout_prob':c['dropout'],'attn_dropout_prob':c['dropout'],
            'hidden_act':'gelu','layer_norm_eps':1e-12,'initializer_range':.02,'loss_type':'CE'}


class SequentialModel(nn.Module):
    def __init__(self,config,counts):
        super().__init__();self.config=dict(config)
        counts=torch.as_tensor(np.array(counts,copy=True),dtype=torch.long)
        if len(counts)<2 or (counts<0).any():raise ValueError('Nonnegative training catalog counts required')
        self.register_buffer('counts',counts)
        self.register_buffer('popularity',torch.log1p(counts[1:].float()))
        cfg=upstream_config(config);self.core=SASRec(cfg,DatasetAdapter(0,len(counts)-1,cfg))
        self.core.loss_fct=KnownItemCrossEntropy()
        with torch.no_grad():
            self.core.item_embedding.weight[0].zero_()
            if not config['position_embeddings']:self.core.position_embedding.weight.zero_()
        if not config['position_embeddings']:self.core.position_embedding.requires_grad_(False)
        self.to(config['device'])

    def loss(self,history,targets):
        seq,length=prepare_history(history,self.config['history_length'])
        if (length==0).any():raise ValueError('Only nonempty training sequences enter sequence loss')
        return self.core.calculate_loss(Interaction({'item_id_list':seq,'item_length':length,'item_id':targets}))

    def scores(self,history):
        seq,length=prepare_history(history,self.config['history_length']);present=length>0
        result=self.popularity[None,:].expand(len(seq),-1).clone()
        if present.any():
            result[present]=self.core.full_sort_predict(Interaction({'item_id_list':seq[present],
                         'item_length':length[present]}))[:,1:]
        return result
