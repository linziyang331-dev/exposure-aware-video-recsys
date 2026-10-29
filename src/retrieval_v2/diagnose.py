"""Actual gradients, tiny memorization and train/validation diagnostic evidence."""
import json
import numpy as np
import torch
from src.retrieval.data import tensor_batch
from src.retrieval.train import load_model as load_original
from src.retrieval_v2.train import build_model,batch_losses
from src.retrieval_v2.run import setup,write_json
from src.retrieval_v2.objectives import full_catalog_loss
from src.retrieval_v2.evaluate import Scorer,evaluate
from src.retrieval.metrics import target_ranks
from src.data.common import ROOT


class View:
    def __init__(self,cache,indices):self.cache=cache;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,name):return self.cache[name][self.indices]


def main():
    cfg,output,prepared,vocabulary,counts,parts=setup();train=parts['standard/train'];validation=parts['standard/validation']
    torch.set_num_threads(4);torch.manual_seed(42);device=cfg['device']
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    original=json.loads((ROOT/'experiments/retrieval_selected/selection.json').read_text())
    result={'test_partitions_loaded':False,'diagnostic_only':True,'gradient_batches':[],'models':{}}
    random=np.random.default_rng(20261028)
    training_sample=random.choice(len(train),min(8192,len(train)),replace=False)
    for at in ['initial','trained']:
        model=build_model(cfg,len(vocabulary['user_raw_to_index']),len(counts)-1)[0].to(device) if at=='initial' else load_original(original['training_runs']['two_tower_seed42']['checkpoint'],device)[0]
        for _ in range(3):
            chosen=random.choice(len(train),1024,replace=False);batch=tensor_batch(train,chosen,device)
            loss=batch_losses(model,batch,'mixed',cfg,torch.arange(1,len(counts),device=device))
            params=[p for p in model.parameters() if p.requires_grad];grads=[]
            for key in ['bce','contrastive']:
                g=torch.autograd.grad(loss[key],params,retain_graph=True,allow_unused=True)
                grads.append(torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(params,g)]))
            result['gradient_batches'].append({'stage':at,'bce':loss['bce'].item(),'contrastive':loss['contrastive'].item(),
                'bce_gradient_norm':grads[0].norm().item(),'contrastive_gradient_norm':grads[1].norm().item(),
                'cosine':torch.nn.functional.cosine_similarity(grads[0][None],grads[1][None]).item()})
        sample=View(train,training_sample)
        result['models'][at]={'train_sample':evaluate(Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),device,model=model),sample,counts),
                             'validation':evaluate(Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),device,model=model),validation,counts)}
    positives=np.flatnonzero(train['label']==1);_,first=np.unique(train['user_index'][positives],return_index=True)
    chosen=positives[first[:64]];batch=tensor_batch(train,chosen,device);ids=torch.arange(1,len(counts),device=device)
    torch.manual_seed(42);model=build_model(cfg,len(vocabulary['user_raw_to_index']),len(counts)-1)[0].to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    def scores():return model.encode_user(batch['users'],batch['tabs'],batch['history'])@model.encode_item(ids).T/model.temperature
    before=scores();before_loss=full_catalog_loss(before,batch['items']).item()
    for _ in range(100):
        opt.zero_grad();loss=full_catalog_loss(scores(),batch['items']);loss.backward();opt.step()
    after=scores();ranks=target_ranks(after,ids,batch['items'])
    result['memorization']={'queries':64,'updates':100,'before_loss':before_loss,'after_loss':full_catalog_loss(after,batch['items']).item(),
                           'after_hit1':(ranks==1).float().mean().item(),'not_quality_benchmark':True}
    write_json(ROOT/'reports/retrieval_optimization_diagnostics.json',result)
    print(json.dumps({'gradient_batches':result['gradient_batches'],'memorization':result['memorization']},indent=2),flush=True)
    for part in parts.values():part.close()


if __name__=='__main__':main()
