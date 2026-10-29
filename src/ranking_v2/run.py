"""Controlled revision trials, using only standard train and validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from src.data.common import ROOT
from src.retrieval.data import load_partition,training_counts
from src.retrieval.provenance import read_verified_vocabulary
from src.retrieval_v2.provenance import verify_evidence
from src.ranking.provenance import verify_v2
from src.ranking.data import write_json
from src.ranking.candidates import load_candidates
from src.ranking_v2.train import train_run


def setup(parts=('standard/train','standard/validation')):
    frozen=verify_v2();old=json.loads((ROOT/'experiments/ranking/selection.json').read_text())
    verify_evidence(ROOT,old['evidence'],old['config'])
    cfg=yaml.safe_load((ROOT/'configs/ranking_v2.yaml').read_text(encoding='utf-8'))
    paths={k:(ROOT/cfg[v]).resolve() for k,v in [('prepared','prepared_directory'),('cache','cache_directory'),('output','output_directory')]}
    for path in paths.values():
        if path.drive.upper()!='D:' or not path.is_relative_to(ROOT.resolve()):raise ValueError('D: project paths required')
    vocabulary=read_verified_vocabulary(paths['prepared']);counts=training_counts(vocabulary)
    caches={p:load_partition(paths['prepared'],p,paths['cache'],50) for p in parts}
    candidates={p:load_candidates(p,caches[p],counts,paths['output'] if p=='standard/train' else ROOT/'experiments/ranking',frozen,cfg['device'])
                for p in parts}
    return cfg,paths,vocabulary,counts,caches,candidates,frozen


def trial(name,overrides):
    cfg,paths,vocabulary,counts,caches,candidates,frozen=setup();cfg.update(overrides)
    if '/' in name or '\\' in name or name in {'.','..'}:raise ValueError('Simple trial name required')
    output=paths['output'];path=output/'tuning.json'
    if (output/'selection.json').exists():raise ValueError('Revision selection is frozen')
    ledger=json.loads(path.read_text()) if path.exists() else {'status':'VALIDATION_ONLY','test_partitions_loaded':False,'trials':{}}
    if name not in ledger['trials'] and len(ledger['trials'])>=cfg['trial_limit']:raise ValueError('Revision training configuration budget exhausted')
    run=train_run(caches['standard/train'],caches['standard/validation'],candidates['standard/train'],candidates['standard/validation'],
                  len(vocabulary['user_raw_to_index']),len(counts)-1,cfg,output/'artifacts/trials'/name,42)
    ledger['trials'][name]={'config':cfg,'training_result_path':str((output/'artifacts/trials'/name/'training.json').resolve()),
                           'best_validation':run['best_validation'],'best_epoch':run['best_epoch'],'checkpoint_sha256':run['checkpoint_sha256']}
    write_json(path,ledger)
    print(name+' BEST N20='+str(run['best_validation']['ranking']['ndcg@20'])+' epoch='+str(run['best_epoch']),flush=True)
    for c in candidates.values():c.close()
    for c in caches.values():c.close()
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True)
    p.add_argument('--control-bce',action='store_true');p.add_argument('--learning-rate',type=float);p.add_argument('--delta-reg',type=float)
    p.add_argument('--no-attention',action='store_true');p.add_argument('--no-user',action='store_true');p.add_argument('--linear-only',action='store_true')
    p.add_argument('--direct-choice',action='store_true');p.add_argument('--no-repeat',action='store_true')
    a=p.parse_args();overrides={k:v for k,v in [('learning_rate',a.learning_rate),('delta_reg',a.delta_reg)] if v is not None}
    if a.control_bce:overrides.update(objective='bce',residual=False,teacher_features=False,pretrained=False,
                                     select_metric='logloss',learning_rate=.003,dropout=.3)
    if a.no_attention:overrides['attention']=False
    if a.no_user:overrides['user_features']=False
    if a.linear_only:overrides['neural_rank']=False
    if a.direct_choice:overrides.update(residual=False,teacher_features=False,pretrained=True)
    if a.no_repeat:overrides['repeat_feature']=False
    trial(a.name,overrides)


if __name__=='__main__':main()
