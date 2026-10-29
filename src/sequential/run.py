"""Bounded SASRec standard train/validation experiments, without test tuning."""
import argparse
from pathlib import Path
import yaml
from src.data.common import ROOT
from src.ranking.data import write_json
from src.retrieval.data import load_partition,training_counts
from src.retrieval.provenance import read_verified_vocabulary
from src.sequential.provenance import read,verify_previous
from src.sequential.train import train_run


def output_path(value):
    output=(ROOT/value).resolve();expected=(ROOT/'experiments/sequential').resolve()
    if output!=expected or output.drive.upper()!='D:':
        raise ValueError('Formal SASRec trials require experiments/sequential; use standalone entry for reproduction')
    return output


def setup(parts=('standard/train','standard/validation')):
    frozen,din=verify_previous();cfg=yaml.safe_load((ROOT/'configs/sequential.yaml').read_text(encoding='utf-8'))
    paths={k:(ROOT/cfg[v]).resolve() for k,v in [('prepared','prepared_directory'),('cache','cache_directory'),('output','output_directory')]}
    output_path(cfg['output_directory'])
    for p in paths.values():
        if p.drive.upper()!='D:' or not p.is_relative_to(ROOT.resolve()):raise ValueError('Use D: project paths')
    vocab=read_verified_vocabulary(paths['prepared']);counts=training_counts(vocab)
    caches={p:load_partition(paths['prepared'],p,paths['cache'],50) for p in parts}
    return cfg,paths,counts,caches,frozen,din


def trials():
    cfg,paths,counts,caches,_,_=setup();out=paths['output'];path=out/'tuning.json'
    if (out/'selection.json').exists():raise ValueError('SASRec selection already frozen')
    ledger=read(path) if path.exists() else {'status':'VALIDATION_ONLY','test_partitions_loaded':False,'trials':{}}
    if set(ledger['trials'])-{'history20','history50','no_position'}:raise ValueError('Unexpected trial outside fixed budget')
    def fit(name,current):
        run=train_run(caches['standard/train'],caches['standard/validation'],counts,current,out/'artifacts/trials'/name,42)
        ledger['trials'][name]={'config':current,'training_result_path':str((out/'artifacts/trials'/name/'training.json').resolve()),
                               'checkpoint_sha256':run['checkpoint_sha256'],'best_epoch':run['best_epoch'],'best_validation':run['best_validation']}
        write_json(path,ledger);return run
    for h in [20,50]:fit(f'history{h}',cfg|{'history_length':h})
    name=max(['history20','history50'],key=lambda n:ledger['trials'][n]['best_validation']['ndcg@20'])
    fit('no_position',ledger['trials'][name]['config']|{'position_embeddings':False})
    for c in caches.values():c.close()
    print('Three SASRec validation configurations complete; primary='+name,flush=True)
    return ledger


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();trials()
