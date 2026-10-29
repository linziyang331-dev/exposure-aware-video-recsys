"""Extend feature routing; reuse upstream DIN embeddings, attention and Dice MLP.

Feature routing follows the DIN paper / DeepCTR-Torch full sparse-input path.
The RecBole package files and the frozen V2 model are never edited.
"""
import torch
import math
from torch import nn
from recbole.model.sequential_recommender.din import DIN
from recbole.model.layers import MLPLayers
from src.ranking.adapter import DatasetAdapter,upstream_config
from src.ranking.model import recent_history


class RichDIN(nn.Module):
    def __init__(self,num_users,num_items,config,retriever=None):
        super().__init__();self.config=dict(config)
        self.teacher=retriever
        if (config['residual'] or config['teacher_features'] or config['pretrained']) and retriever is None:
            raise ValueError('Declared retrieval features require the frozen retriever')
        if retriever is not None:
            if retriever.has_bias:raise ValueError('Use the declared bias-free frozen V2')
            retriever.requires_grad_(False);retriever.eval()
        cfg=upstream_config(config);dataset=DatasetAdapter(num_users,num_items,cfg)
        self.core=DIN(cfg,dataset)
        dim=config['embedding_dim'];self.tab_embedding=nn.Embedding(15,8)
        nn.init.normal_(self.tab_embedding.weight,std=.02)
        width=5*dim+10
        if config['teacher_features']:width+=3*retriever.item_embedding.embedding_dim+1
        self.core.dnn_mlp_layers=MLPLayers([width]+config['hidden_sizes'],activation='Dice',dropout=config['dropout'],bn=True)
        self.core.dnn_predict_layers=nn.Linear(config['hidden_sizes'][-1],1)
        self.rank_delta=nn.Linear(config['hidden_sizes'][-1],1)
        nn.init.zeros_(self.rank_delta.weight);nn.init.zeros_(self.rank_delta.bias)
        self.repeat_linear=nn.Linear(1,1,bias=False);nn.init.zeros_(self.repeat_linear.weight)
        self.to(config['device'])
        if config['pretrained']:
            with torch.no_grad():
                self.core.embedding_layer.token_embedding_table['item'].embedding.weight.copy_(retriever.item_embedding.weight)
                self.core.embedding_layer.token_embedding_table['user'].embedding.weight.copy_(retriever.user_embedding.weight)

    def train(self,mode=True):
        super().train(mode)
        if self.teacher is not None:self.teacher.eval()
        return self

    def teacher_spec(self):
        t=self.teacher
        if t is None:return None
        return {'num_users':t.user_embedding.num_embeddings-1,'num_items':t.item_embedding.num_embeddings-1,
                'embedding_dim':t.item_embedding.embedding_dim,'temperature':t.temperature,
                'history_mode':t.history_mode,'item_bias':t.has_bias,'history_decay':t.history_decay}

    def forward(self,users,items,tabs,history,base_scores=None):
        short=recent_history(history,self.config['history_length'])
        sparse,dense=self.core.embedding_layer(users,torch.cat([short,items[:,None]],dim=1))
        if dense['user'] is not None or dense['item'] is not None:raise ValueError('This revision supports audited ID-only metadata')
        item=sparse['item'].flatten(-2);user=sparse['user'].flatten(-2)
        keys=item[:,:-1];target=item[:,-1];lengths=(short>0).sum(1)
        if self.config['attention']:interest=self.core.attention(target,keys,lengths).squeeze(1)
        else:
            mask=(short>0).unsqueeze(-1);interest=(keys*mask).sum(1)/lengths[:,None].clamp_min(1)
        if not self.config['user_features']:user=torch.zeros_like(user)
        seen=((history==items[:,None])&(history>0)).any(1).float()
        if not self.config.get('repeat_feature',True):seen=torch.zeros_like(seen)
        length_feature=torch.log1p((history>0).sum(1).float())/math.log(51)
        features=[interest,target,interest*target,user,user*target,self.tab_embedding(tabs),seen[:,None],length_feature[:,None]]
        base=torch.zeros(len(users),device=users.device)
        if self.teacher is not None:
            with torch.no_grad():
                tu=self.teacher.encode_user(users,tabs,history);ti=self.teacher.encode_item(items)
                base=(tu*ti).sum(1)/self.teacher.temperature
            if base_scores is not None:
                if base_scores.shape!=(len(users),) or not torch.isfinite(base_scores).all():raise ValueError('Finite aligned canonical scores required')
                base=base_scores
            if self.config['teacher_features']:features.extend([tu,ti,tu*ti,base[:,None]])
        elif base_scores is not None:raise ValueError('Base scores require the declared frozen teacher')
        hidden=self.core.dnn_mlp_layers(torch.cat(features,dim=1))
        ctr=self.core.dnn_predict_layers(hidden).squeeze(1)
        neural=self.rank_delta(hidden).squeeze(1)
        if not self.config.get('neural_rank',True):neural=torch.zeros_like(neural)
        repeat=self.repeat_linear(seen[:,None]).squeeze(1)
        delta=neural+repeat
        rank=base+delta if self.config['residual'] else ctr
        return {'rank':rank,'ctr':ctr,'delta':delta,'base':base,'neural_delta':neural,'repeat_delta':repeat}
