"""Minimal metadata adapter; never invokes upstream split or negative sampling."""
import torch
from recbole.data.interaction import Interaction
from recbole.utils import FeatureType


class DatasetAdapter:
    def __init__(self,num_users,num_items,config):
        self.config=config;self.sizes={'user_id':num_users+1,'item_id':num_items+1,'tab':15,'label':1}
        self.field2type={name:FeatureType.TOKEN for name in ['user_id','item_id','tab']}
        self.field2type['label']=FeatureType.FLOAT

    def num(self,field):return self.sizes[field]
    def fields(self,**kwargs):return ['user_id','item_id','tab','label']
    def get_user_feature(self):return Interaction({'user_id':torch.arange(self.sizes['user_id'])})
    def get_item_feature(self):return Interaction({'item_id':torch.arange(self.sizes['item_id'])})


def upstream_config(config):
    return {'USER_ID_FIELD':'user_id','ITEM_ID_FIELD':'item_id','LABEL_FIELD':'label',
            'LIST_SUFFIX':'_list','ITEM_LIST_LENGTH_FIELD':'item_length','NEG_PREFIX':'neg_',
            'MAX_ITEM_LIST_LENGTH':config['history_length'],'embedding_size':config['embedding_dim'],
            'mlp_hidden_size':config['hidden_sizes'],'device':config['device'],'pooling_mode':'mean',
            'dropout_prob':config['dropout'],'double_tower':False,'numerical_features':[]}
