"""D-drive-only reproducible CQL CLI. Completed experiments are immutable."""
import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import yaml

from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.run import run_component,validate_component


def output_path(value):
    path=(ROOT/value).resolve()
    if path.drive.upper()!='D:' or path.parent!=(ROOT/'experiments').resolve() or not path.name.startswith('offline_rl_'):
        raise ValueError('Use a separate D: project experiments/offline_rl_* directory')
    return path


def get_config(override=None,smoke=False):
    cfg=yaml.safe_load((ROOT/'configs/offline_rl.yaml').read_text(encoding='utf-8'))
    if smoke:
        cfg.update(training_episodes=[30],validation_episodes=15,seeds=[42],rollout_episodes=300,bootstrap_repeats=50)
        cfg['world']['horizon']=4;cfg['model'].update(steps=80,trace_interval=40,target_update=20,hidden=[16,16],batch_size=32)
    target=override or cfg['output_directory']+('_smoke' if smoke else '')
    output=output_path(target);cfg['output_directory']=output.relative_to(ROOT).as_posix();cfg['smoke']=bool(smoke)
    for key in ['validation_episodes','rollout_episodes','bootstrap_repeats','bootstrap_seed']:
        if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or cfg[key]<1:raise ValueError('Invalid '+key)
    for key in ['training_episodes','seeds']:
        values=cfg[key]
        if not values or len(values)!=len(set(values)) or any(not isinstance(v,int) or isinstance(v,bool) or v<(1 if key=='training_episodes' else 0) for v in values):
            raise ValueError('Invalid matrix '+key)
    alphas=cfg['alphas']
    if (not alphas or len(alphas)!=len(set(alphas)) or 0. not in alphas
            or not np.isfinite(alphas).all() or min(alphas)<0 or cfg['main_alpha'] not in alphas or cfg['main_alpha']<=0):
        raise ValueError('DQN and a positive predeclared main CQL alpha required')
    if set(cfg['scenarios'])!={'exploratory','narrow'} or len(cfg['scenarios'])!=2:raise ValueError('Both unique logging scenarios required')
    from src.offline_rl.environment import make_world
    from src.offline_rl.experiment import training_config
    from src.offline_rl.train import _config
    _config(training_config(cfg,make_world(cfg['world'])))
    if cfg['rollout_episodes']<2:raise ValueError('At least two evaluation episodes required')
    return cfg,output


def signature(config):
    files=list((ROOT/'src/offline_rl').glob('*.py'))+list((ROOT/'tests').glob('test_offline_rl_*.py'))
    files += [ROOT/p for p in ['configs/offline_rl.yaml','src/data/common.py','src/data/download.py',
        'src/exposure/run.py','src/exposure/verify.py','src/exposure/synthetic.py']]
    inputs=['reports/data_audit.md','reports/data_audit.json','reports/retrieval_v2_freeze.json',
            'experiments/ranking_v2/selection.json','reports/exposure_delivery_validation.json']
    return {'config':config,'sources':{p.relative_to(ROOT).as_posix():digest_file(p) for p in sorted(files)},
        'existing_evidence':{name:digest_file(ROOT/name) for name in inputs},
        'environment':{'python':sys.version.split()[0],'pearl_module_present':importlib.util.find_spec('pearl') is not None,
            **{n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','matplotlib']}}}


def fit(config,output):
    from src.offline_rl.experiment import run_experiment
    return run_component(output/'experiment',signature(config),lambda:run_experiment(config,output/'experiment'))


def verify(config,output):
    from src.offline_rl.verify import verify_experiment
    component=validate_component(output/'experiment',signature(config))
    result=verify_experiment(config,output/'experiment')
    result['experiment_fingerprint']=component['fingerprint']
    result['experiment_manifest_sha256']=digest_file(output/'experiment/component.json')
    (output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('VERIFIED offline RL',flush=True);return result


def report(config,output):
    from src.offline_rl.report import create_report
    validate_component(output/'experiment',signature(config));return create_report(config,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','fit','verify','report'],default='all')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--output-directory')
    args=parser.parse_args();config,output=get_config(args.output_directory,args.smoke)
    if args.phase=='all':fit(config,output);verify(config,output);report(config,output)
    else:globals()[args.phase](config,output)


if __name__=='__main__':main()
