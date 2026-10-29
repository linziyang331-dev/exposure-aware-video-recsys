"""Train the declared SASRec configuration in a protected independent D: directory."""
import argparse
from scripts.train_retrieval_v2 import reproduction_config as base_output
from src.data.common import ROOT
from src.retrieval_v2.provenance import verify_evidence
from src.sequential.provenance import read
from src.sequential.run import setup
from src.sequential.train import train_run


def reproduction_config(config,output_directory):
    cfg,out=base_output(config,output_directory)
    for name in ['ranking','ranking_v2','sequential']:
        p=(ROOT/'experiments'/name).resolve()
        if out==p or out.is_relative_to(p):raise ValueError('Preserve benchmark experiments')
    return cfg,out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',default='experiments/sequential/selection.json')
    p.add_argument('--output-directory',required=True);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();sel=read(ROOT/args.selection);verify_evidence(ROOT,sel['evidence'],sel['config'])
    cfg,out=reproduction_config(sel['config'],args.output_directory);_,_,counts,caches,_,_=setup()
    try:train_run(caches['standard/train'],caches['standard/validation'],counts,cfg,out,args.seed)
    finally:
        for c in caches.values():c.close()


if __name__=='__main__':main()
