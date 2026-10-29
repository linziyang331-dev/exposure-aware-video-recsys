"""Reproduce the selected revised DIN in a separate D: project directory."""
import argparse
import json
from scripts.train_retrieval_v2 import reproduction_config as base_output
from src.data.common import ROOT
from src.ranking_v2.run import setup
from src.ranking_v2.train import train_run
from src.retrieval_v2.provenance import verify_evidence


def reproduction_config(config,output_directory):
    cfg,output=base_output(config,output_directory)
    for name in ['ranking','ranking_v2']:
        benchmark=(ROOT/'experiments'/name).resolve()
        if output==benchmark or output.is_relative_to(benchmark):raise ValueError('Preserve frozen ranking experiments')
    return cfg,output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',default='experiments/ranking_v2/selection.json')
    p.add_argument('--output-directory',required=True);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();selected=json.loads((ROOT/args.selection).read_text(encoding='utf-8'))
    verify_evidence(ROOT,selected['evidence'],selected['config'])
    cfg,output=reproduction_config(selected['config'],args.output_directory)
    _,_,vocab,counts,caches,candidates,_=setup()
    try:
        run=train_run(caches['standard/train'],caches['standard/validation'],candidates['standard/train'],candidates['standard/validation'],
                      len(vocab['user_raw_to_index']),len(counts)-1,cfg,output,args.seed)
        print('TRAINED '+str(run['checkpoint'])+' N20='+str(run['best_validation']['ranking']['ndcg@20']))
    finally:
        for c in candidates.values():c.close()
        for c in caches.values():c.close()


if __name__=='__main__':main()
