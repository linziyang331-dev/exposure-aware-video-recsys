"""Independently train a declared RecBole DIN/LR config, preserving benchmarks."""
import argparse
from pathlib import Path
import yaml
from scripts.train_retrieval_v2 import reproduction_config
from src.data.common import ROOT
from src.retrieval.data import load_partition
from src.retrieval.provenance import read_verified_vocabulary
from src.ranking.provenance import verify_v2
from src.ranking.train import train_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/ranking_selected.yaml')
    parser.add_argument('--output-directory',required=True);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
    cfg,output=reproduction_config(cfg,args.output_directory)
    benchmark=(ROOT/'experiments/ranking').resolve()
    if output==benchmark or output.is_relative_to(benchmark):raise ValueError('Preserve frozen ranking experiment')
    verify_v2()
    prepared=(ROOT/cfg['prepared_directory']).resolve();cache=(ROOT/cfg['cache_directory']).resolve()
    for path in [prepared,cache]:
        if path.drive.upper()!='D:' or not path.is_relative_to(ROOT.resolve()):raise ValueError('D: project paths required')
    vocabulary=read_verified_vocabulary(prepared)
    parts={p:load_partition(prepared,p,cache,50) for p in ['standard/train','standard/validation']}
    try:
        train_run(parts['standard/train'],parts['standard/validation'],len(vocabulary['user_raw_to_index']),
                  len(vocabulary['item_raw_to_index']),cfg,output,args.seed)
    finally:
        for part in parts.values():part.close()


if __name__=='__main__':main()
