"""Reproducible exposure module CLI with immutable completed components."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime,timezone,timedelta
from pathlib import Path

import numpy as np
import yaml

from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.verify import assert_result_equal


def output_path(value):
    path=(ROOT/value).resolve()
    if path.drive.upper()!='D:' or path.parent!=(ROOT/'experiments').resolve() or not path.name.startswith('exposure_'):
        raise ValueError('Use a separate D: project experiments/exposure_* directory')
    return path


def get_config(override=None,smoke=False):
    cfg=yaml.safe_load((ROOT/'configs/exposure.yaml').read_text(encoding='utf-8'))
    if smoke:
        cfg['bootstrap_repeats']=20
        cfg['synthetic'].update(n_contexts=128,n_actions=4,context_dim=3,strengths=[0.,3.],
            sample_sizes=[100,500],repeats=3,clips=[None,5.],reward_training_rows=300,
            coverage_rows=200,coverage_repeats=3,coverage_bootstrap_repeats=20,
            learning_rows=300,learning_seeds=[42])
    target=override or (cfg['output_directory']+'_smoke' if smoke else cfg['output_directory'])
    output=output_path(target);cfg['output_directory']=output.relative_to(ROOT).as_posix();cfg['smoke']=bool(smoke)
    for key in ['prepared_directory','cache_directory']:
        path=(ROOT/cfg[key]).resolve()
        if path.drive.upper()!='D:' or not path.is_relative_to(ROOT):raise ValueError('D: project input/cache paths required')
    s=cfg['synthetic']
    for key in ['bootstrap_repeats','calibration_bins']:
        if not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError('Invalid '+key)
    for key in ['n_contexts','n_actions','context_dim','repeats','reward_training_rows','coverage_rows','coverage_repeats','coverage_bootstrap_repeats','learning_rows']:
        if not isinstance(s[key],int) or s[key]<1:raise ValueError('Invalid synthetic '+key)
    if min(s['n_contexts'],s['n_actions'],s['context_dim'])<2 or s['repeats']<2 or not 0<s['exploration']<=1:
        raise ValueError('Invalid synthetic population, repetitions or exploration')
    if (not s['strengths'] or any(not np.isfinite(x) or x<0 for x in s['strengths'])
            or not s['sample_sizes'] or any(not isinstance(x,int) or x<1 for x in s['sample_sizes'])
            or not s['clips'] or any(x is not None and (not np.isfinite(x) or x<=0) for x in s['clips'])
            or not s['learning_seeds'] or len(s['target_policies'])!=2
            or set(s['target_policies'])!={'uniform','reward_softmax'}
            or any(len(s[key])!=len(set(s[key])) for key in ['strengths','sample_sizes','clips','learning_seeds'])):
        raise ValueError('Invalid predeclared synthetic matrix')
    return cfg,output


def _fingerprint(signature):
    return hashlib.sha256(json.dumps(signature,sort_keys=True,allow_nan=False).encode()).hexdigest()


def validate_component(directory,signature):
    directory=Path(directory)
    manifest=json.loads((directory/'component.json').read_text(encoding='utf-8'))
    if manifest['fingerprint']!=_fingerprint(signature) or manifest['signature']!=signature:
        raise ValueError('Completed component configuration/source/input signature changed; use a new output directory')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='component.json'}
    if actual!=set(manifest['output_hashes']):raise ValueError('Component output file set changed')
    for name,expected in manifest['output_hashes'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest_file(path)!=expected:
            raise ValueError('Completed component output changed: '+name)
    return manifest


def run_component(directory,signature,worker):
    directory=Path(directory);directory.parent.mkdir(parents=True,exist_ok=True)
    if (directory/'component.json').exists():return validate_component(directory,signature)
    if directory.exists():raise FileExistsError('Incomplete component preserved; choose a new output directory')
    lock=directory.parent/(directory.name+'.lock')
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(directory))
    try:
        worker()
        files={p.relative_to(directory).as_posix():digest_file(p) for p in sorted(directory.rglob('*')) if p.is_file()}
        if not files:raise ValueError('Component produced no results')
        manifest={'status':'COMPLETE','fingerprint':_fingerprint(signature),'signature':signature,
                  'output_hashes':files,'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat()}
        (directory/'component.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def _source_hashes():
    files=list((ROOT/'src/exposure').glob('*.py'))+list((ROOT/'tests').glob('test_exposure_*.py'))
    files+=[ROOT/'configs/exposure.yaml',ROOT/'src/retrieval/data.py',ROOT/'src/retrieval/metrics.py',
            ROOT/'src/ranking/metrics.py',ROOT/'src/data/common.py']
    return {p.relative_to(ROOT).as_posix():digest_file(p) for p in sorted(files)}


def _real_inputs(config):
    prepared=ROOT/config['prepared_directory']
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    paths=[prepared/'manifest.json',prepared/'vocabulary.json',ROOT/'experiments/ranking_v2/selection.json',
           ROOT/'experiments/ranking_v2/evaluation.json',ROOT/'reports/ranking_v2_artifact_validation.json',
           ROOT/'reports/retrieval_v2_freeze.json']
    paths += [prepared/r['path'] for r in manifest['partitions'].values()]
    evaluation=json.loads((ROOT/'experiments/ranking_v2/evaluation.json').read_text(encoding='utf-8'))
    for part in evaluation['baseline']:
        paths += [Path(evaluation['baseline'][part]['artifact']),
                  Path(evaluation['results']['din_seed42'][part]['rank_artifact']),
                  Path(evaluation['results']['din_seed42'][part]['prediction_artifact'])]
    return {p.resolve().relative_to(ROOT).as_posix():digest_file(p) for p in sorted(set(paths))}


def signature(config,phase):
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','torch','pyarrow']}
    return {'config':config,'phase':phase,'sources':_source_hashes(),
            'inputs':_real_inputs(config) if phase=='real' else {},
            'environment':{'python':sys.version.split()[0],**versions}}


def synthetic(config,output):
    from src.exposure.synthetic import run_synthetic
    return run_component(output/'synthetic',signature(config,'synthetic'),lambda:run_synthetic(config,output/'synthetic'))


def real(config,output):
    from src.exposure.real import run_real
    return run_component(output/'real',signature(config,'real'),lambda:run_real(config,output/'real'))


def verify(config,output):
    from src.exposure.verify import verify_synthetic,verify_real
    manifests={phase:validate_component(output/phase,signature(config,phase)) for phase in ['synthetic','real']}
    result={'status':'PASS','synthetic':verify_synthetic(config,output/'synthetic'),
            'real':verify_real(config,output/'real'),
            'component_fingerprints':{k:v['fingerprint'] for k,v in manifests.items()}}
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('VERIFIED exposure module',flush=True)
    return result


def report(config,output):
    from src.exposure.report import create_report
    for phase in ['synthetic','real']:validate_component(output/phase,signature(config,phase))
    return create_report(config,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','synthetic','real','verify','report'],default='all')
    parser.add_argument('--output-directory');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();config,output=get_config(args.output_directory,args.smoke)
    if args.phase=='all':
        synthetic(config,output);real(config,output);verify(config,output);report(config,output)
    else:globals()[args.phase](config,output)


if __name__=='__main__':main()
