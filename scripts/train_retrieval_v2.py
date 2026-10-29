"""Train a declared standalone two-tower config in a separate D: output directory."""
import argparse
from pathlib import Path
import yaml
from src.data.common import ROOT
from src.retrieval.data import load_partition,training_counts
from src.retrieval.provenance import read_verified_vocabulary
from src.retrieval_v2.train import train_run


def reproduction_config(config,output_directory):
    output=(ROOT/output_directory).resolve();experiments=(ROOT/'experiments').resolve()
    if output.drive.upper()!='D:' or output==experiments or not output.is_relative_to(experiments):
        raise ValueError('Choose a D: project directory below experiments/')
    protected=[experiments/name for name in ['retrieval_v2','retrieval_selected','retrieval']]
    if any(output==path or output.is_relative_to(path) for path in protected):
        raise ValueError('Choose a separate reproduction output directory')
    return config|{'output_directory':output.relative_to(ROOT.resolve()).as_posix()},output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/retrieval_v2_selected.yaml')
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
    cfg,output=reproduction_config(cfg,args.output_directory)
    prepared=(ROOT/cfg['prepared_directory']).resolve();cache=(ROOT/cfg['cache_directory']).resolve()
    for path in [prepared,cache,output]:
        if not path.is_relative_to(ROOT.resolve()) or path.drive.upper()!='D:':raise ValueError('Use D: project paths')
    vocab=read_verified_vocabulary(prepared);counts=training_counts(vocab)
    parts={p:load_partition(prepared,p,cache,cfg['history_length']) for p in ['standard/train','standard/validation']}
    try:train_run(parts['standard/train'],parts['standard/validation'],counts,cfg,output,args.seed)
    finally:
        for part in parts.values():part.close()


if __name__=='__main__':main()
