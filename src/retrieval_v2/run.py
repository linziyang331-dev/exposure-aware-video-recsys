"""Independent train/validation-only optimization ledger; no test loading here."""
import argparse
import json
from pathlib import Path
import yaml
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.data import load_partition,training_counts
from src.retrieval.provenance import read_verified_vocabulary,verify_frozen_inputs
from src.retrieval_v2.train import train_run,implementation_hashes


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def setup(history_length=None):
    cfg=yaml.safe_load((ROOT/'configs/retrieval_v2.yaml').read_text(encoding='utf-8'))
    prepared=(ROOT/cfg['prepared_directory']).resolve();output=(ROOT/cfg['output_directory']).resolve()
    cache=(ROOT/cfg['cache_directory']).resolve()
    for path in [prepared,output,cache]:
        if not path.is_relative_to(ROOT.resolve()) or path.drive.upper()!='D:':raise ValueError('D: project paths required')
    original=json.loads((ROOT/'experiments/retrieval_selected/selection.json').read_text(encoding='utf-8'))
    verify_frozen_inputs(original,prepared)
    vocabulary=read_verified_vocabulary(prepared);counts=training_counts(vocabulary)
    length=history_length or cfg['history_length']
    parts={p:load_partition(prepared,p,cache,length) for p in ['standard/train','standard/validation']}
    return cfg,output,prepared,vocabulary,counts,parts


def trial(name,overrides):
    cfg,output,prepared,vocabulary,counts,parts=setup(overrides.get('history_length'))
    cfg.update(overrides)
    path=output/'tuning.json'
    ledger=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'status':'VALIDATION_ONLY','test_partitions_loaded':False,'selection_metric':'standard validation NDCG@20',
        'primary_seed':42,'limit':cfg['tuning_config_limit'],'trials':{}}
    if '/' in name or '\\' in name or name in {'.','..'}:raise ValueError('Simple trial name required')
    if name not in ledger['trials'] and len(ledger['trials'])>=ledger['limit']:raise ValueError('Tuning budget exhausted')
    frozen=output/'selection.json'
    if frozen.exists():raise ValueError('Selection already frozen; no further tuning')
    result=train_run(parts['standard/train'],parts['standard/validation'],counts,cfg,output/'artifacts/tuning'/name,42)
    ledger['trials'][name]={'config':cfg,'result_path':str((output/'artifacts/tuning'/name/'training.json').resolve()),
                          'best_validation':result['best_validation'],'checkpoint_sha256':result['checkpoint_sha256'],
                          'trained_parameter_sha256':result['trained_parameter_sha256']}
    write_json(path,ledger)
    for part in parts.values():part.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--objective',choices=['mixed','bce','contrastive','full'])
    parser.add_argument('--model',choices=['legacy','residual']);parser.add_argument('--history-mode',choices=['none','mean','recency'])
    parser.add_argument('--item-bias',action='store_true',default=None)
    parser.add_argument('--history-length',type=int);parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--embedding-dim',type=int);parser.add_argument('--epochs',type=int);parser.add_argument('--patience',type=int)
    args=vars(parser.parse_args());name=args.pop('name');trial(name,{k:v for k,v in args.items() if v is not None})


if __name__=='__main__':main()
