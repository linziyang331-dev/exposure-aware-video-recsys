"""Bounded train/validation-only upstream DIN/LR experiments."""
import argparse
import json
from pathlib import Path
from src.ranking.data import setup,write_json
from src.ranking.train import train_run,load_model
from src.ranking.candidates import load_candidates
from src.ranking.evaluate import evaluate_candidates
from src.ranking.finalize import validation_signature


def trial(name,overrides):
    cfg,paths,vocabulary,counts,caches,frozen=setup();cfg.update(overrides)
    if '/' in name or '\\' in name or name in {'.','..'}:raise ValueError('Simple name required')
    ledger_path=paths['output']/'tuning.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {
        'status':'VALIDATION_ONLY','test_partitions_loaded':False,'trials':{},'din_limit':cfg['din_trial_limit']}
    if (paths['output']/'selection.json').exists():raise ValueError('Ranking already frozen; no further tuning')
    count=sum(t['config']['kind']=='din' for t in ledger['trials'].values())
    if cfg['kind']=='din' and name not in ledger['trials'] and count>=ledger['din_limit']:raise ValueError('DIN tuning budget exhausted')
    directory=paths['output']/'artifacts/tuning'/name
    run=train_run(caches['standard/train'],caches['standard/validation'],len(vocabulary['user_raw_to_index']),len(counts)-1,cfg,directory,42)
    candidates=load_candidates('standard/validation',caches['standard/validation'],counts,paths['output'],frozen,cfg['device'])
    model,_=load_model(run['checkpoint'],cfg['device'])
    ranks,logits=evaluate_candidates(model,caches['standard/validation'],candidates,cfg['rerank_alphas'],
                                      cfg['candidate_query_batch_size'],directory/'validation_ranks')
    import numpy as np
    np.save(directory/'validation_candidate_logits.npy',logits)
    result={'config':cfg,'training_result_path':str((directory/'training.json').resolve()),
            'pointwise_validation':run['best_validation'],'candidate_validation':ranks,
            'checkpoint_sha256':run['checkpoint_sha256'],'candidate_signature':candidates.metadata['signature']}
    result['validation_evidence']=validation_signature(result,candidates,directory/'validation_candidate_logits.npy')
    ledger['trials'][name]=result;write_json(ledger_path,ledger)
    pure=ranks['results']['pure'];best=max(ranks['results'].items(),key=lambda p:p[1]['ndcg@20'])
    print(f'{name} pointwise AUC={run["best_validation"]["auc"]:.6f}; pure N20={pure["ndcg@20"]:.6f}; best={best[0]} N20={best[1]["ndcg@20"]:.6f}',flush=True)
    candidates.close()
    for cache in caches.values():cache.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True)
    parser.add_argument('--kind',choices=['lr','din']);parser.add_argument('--history-length',type=int)
    parser.add_argument('--learning-rate',type=float);parser.add_argument('--dropout',type=float)
    parser.add_argument('--pretrained-items',action='store_true',default=None)
    args=vars(parser.parse_args());name=args.pop('name');trial(name,{k:v for k,v in args.items() if v is not None})


if __name__=='__main__':main()
