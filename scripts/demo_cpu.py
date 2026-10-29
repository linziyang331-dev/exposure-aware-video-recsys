"""Small self-contained CPU/OPE demo; not the frozen recommendation benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.offline_rl.run import get_config
from src.offline_rl.experiment import run_experiment
from src.offline_rl.verify import verify_experiment
from src.exposure.ope import ope_terms,point_estimates


def run_demo(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Demo outputs are preserved; use a new directory')
    config,_=get_config(smoke=True)
    config.update(training_episodes=[16],validation_episodes=8,seeds=[42],alphas=[0.,.2],
                  rollout_episodes=100,bootstrap_repeats=20)
    config['world']['horizon']=3
    config['model'].update(hidden=[16],steps=20,batch_size=32,target_update=10,trace_interval=10)
    # fit_model in this independent prototype explicitly creates CPU tensors.
    output.mkdir(parents=True)
    experiment=output/'experiment'
    run_experiment(config,experiment)
    verification=verify_experiment(config,experiment)
    # Uniform logging/target: IPS = SNIPS = empirical reward = 0.5.
    terms=ope_terms(np.array([0,1,0,1]),np.array([1.,0.,0.,1.]),np.full(4,.5),
                    np.full((4,2),.5),behavior_policy=np.full((4,2),.5))
    estimates=point_estimates(terms)
    if verification['status']!='PASS' or estimates['ips']!=.5 or estimates['snips']!=.5:
        raise ValueError('Public demo verification failed')
    result={'status':'PASS','scope':'small synthetic CPU workflow, not formal benchmark reproduction',
            'real_top20_improvement_claimed':False,'frozen_assets_required':False,
            'training_device':'cpu','verification':verification,'ope':estimates}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',default='.runtime/demos/first-run')
    args=parser.parse_args();output=(ROOT/args.output_directory).resolve()
    if output.drive.upper()!='D:' or not output.is_relative_to(ROOT/'.runtime/demos') or output==ROOT/'.runtime/demos':
        raise ValueError('Use an explicit D: project .runtime/demos/<new-name> directory')
    result=run_demo(output)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
