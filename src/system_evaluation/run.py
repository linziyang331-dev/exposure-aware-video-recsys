"""Unified read-only acceptance and separate-task system evaluation CLI."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.run import run_component,validate_component
from src.exposure.verify import assert_result_equal
from src.exposure.synthetic import write_csv


def output_path(value):
    path=(ROOT/value).resolve()
    if path.drive.upper()!='D:' or path.parent!=(ROOT/'experiments').resolve() or not path.name.startswith('system_evaluation_'):
        raise ValueError('Use a separate D: project experiments/system_evaluation_* directory')
    return path


def get_config(override=None,smoke=False):
    cfg=yaml.safe_load((ROOT/'configs/system_evaluation.yaml').read_text(encoding='utf-8'))
    if smoke:cfg.update(deep_audit=False,bootstrap_repeats=20,warmup_batches=2,measurement_batches=8)
    out=output_path(override or cfg['output_directory']+('_smoke' if smoke else ''))
    cfg['output_directory']=out.relative_to(ROOT).as_posix();cfg['smoke']=bool(smoke)
    for name in ['bootstrap_repeats','bootstrap_seed','probe_queries_per_partition','warmup_batches','measurement_batches']:
        if not isinstance(cfg[name],int) or isinstance(cfg[name],bool) or cfg[name]<1:raise ValueError('Positive integer '+name+' required')
    if (not isinstance(cfg['deep_audit'],bool) or cfg['probe_queries_per_partition']%512
            or not cfg['benchmark_batches'] or len(set(cfg['benchmark_batches']))!=len(cfg['benchmark_batches'])
            or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in cfg['benchmark_batches'])):
        raise ValueError('Invalid audit, probe or benchmark configuration')
    for name in ['prepared_directory','cache_directory']:
        p=(ROOT/cfg[name]).resolve()
        if p.drive.upper()!='D:' or not p.is_relative_to(ROOT):raise ValueError('D: project inputs/cache required')
    if cfg['device'] not in ['cuda','cpu'] or cfg['device']=='cuda' and not torch.cuda.is_available():raise ValueError('Configured device unavailable')
    return cfg,out


def signature(config):
    from src.system_evaluation.registry import load_registry
    from src.system_evaluation.analysis import audit_modules
    reg=load_registry();audit_modules(reg['assets'],False)
    prepared=ROOT/config['prepared_directory'];manifest=reg['assets'].read(prepared/'manifest.json')
    reg['assets'].check(prepared/'vocabulary.json')
    for p in manifest['partitions'].values():reg['assets'].check(prepared/p['path'],p['sha256'])
    files=list((ROOT/'src/system_evaluation').glob('*.py'))+list((ROOT/'tests').glob('test_system_*.py'))
    files += [ROOT/'configs/system_evaluation.yaml',ROOT/'src/exposure/run.py',ROOT/'src/exposure/verify.py']
    return {'config':config,'sources':{p.relative_to(ROOT).as_posix():digest_file(p) for p in sorted(files)},
        'inputs':dict(sorted(reg['assets'].hashes.items())),
        'environment':{'python':sys.version.split()[0],**{n:importlib.metadata.version(n) for n in ['numpy','torch','scikit-learn','pyarrow','matplotlib']},
            'hardware':torch.cuda.get_device_name() if config['device']=='cuda' else 'CPU'}}


def csv_rows(result):
    return [{**{k:r[k] for k in ['task','partition','space','model','primary','experimental']},
        **{k:r['metrics'][k] for k in ['queries','users','recall@20','ndcg@20','recall@100','ndcg@100','mrr','mrr_cutoff']}}
        for r in result['rank_results']]


def table_rows(result):
    return {'ranking_metrics.csv':csv_rows(result),
        'subgroups.csv':[{**{k:r[k] for k in ['partition','space','model','cohort']},
            **{k:r['metrics'].get(k) for k in ['queries','recall@20','ndcg@20','recall@100']}} for r in result['subgroups']],
        'pointwise_metrics.csv':[{**{k:r[k] for k in ['task','partition','model']},
            **{k:r['metrics'][k] for k in ['rows','positives','auc','logloss','brier','ece']}} for r in result['pointwise_results']],
        'pipeline_funnel.csv':[{'partition':part,'cohort':name,**f} for part,groups in result['candidate_funnel'].items() for name,f in groups.items()]}


def check_csv(path,rows):
    import csv
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);actual=list(reader)
        if reader.fieldnames!=list(rows[0]):raise ValueError('CSV column contract differs')
    expected=[{k:'' if v is None else str(v) for k,v in row.items()} for row in rows]
    if actual!=expected:raise ValueError('CSV values differ: '+str(path))


def check_latency(pipeline,config):
    from src.system_evaluation.pipeline import latency_summary,LATENCY_SCOPE
    if pipeline['latency_scope']!=LATENCY_SCOPE:raise ValueError('Offline timing scope changed')
    if set(pipeline['latency'])!={str(n) for n in config['benchmark_batches']}:raise ValueError('Benchmark batch set changed')
    for key,value in pipeline['latency'].items():
        if value['measured_batches']!=config['measurement_batches']:raise ValueError('Timing count changed')
        compare_analysis(latency_summary(value['raw_seconds'],int(key)),value)


def verification_record(config,component,folder,analysis,pipeline):
    return {'status':'PASS','rank_artifacts_recomputed':len(analysis['rank_results']),
        'exposure_predictions_recomputed':len(analysis['pointwise_results']),'subgroup_rows_recomputed':len(analysis['subgroups']),
        'pipeline_queries_replayed':sum(v['queries'] for v in pipeline['checks'].values()),
        'deep_module_recomputation':config['deep_audit'],'latency_observations_replayed':False,
        'latency_validity_and_summary_checked':True,'component_fingerprint':component['fingerprint'],
        'manifest_sha256':digest_file(Path(folder)/'component.json')}


def build(config,output):
    from src.system_evaluation.analysis import analyze
    from src.system_evaluation.pipeline import probe_pipeline
    def worker():
        folder=output/'evaluation';folder.mkdir(parents=True)
        analysis=analyze(config)
        (folder/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        pipeline=probe_pipeline(config,folder)
        (folder/'pipeline.json').write_text(json.dumps(pipeline,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        for name,rows in table_rows(analysis).items():write_csv(folder/name,rows)
    return run_component(output/'evaluation',signature(config),worker)


def compare_analysis(expected,actual):assert_result_equal(expected,actual)


def verify(config,output):
    from src.system_evaluation.analysis import analyze
    from src.system_evaluation.pipeline import probe_pipeline
    from src.system_evaluation.report import validate_report_delivery
    component=validate_component(output/'evaluation',signature(config));folder=output/'evaluation'
    if (output/'report_delivery.json').exists():validate_report_delivery(config,output)
    saved=json.loads((folder/'analysis.json').read_text(encoding='utf-8'));actual=analyze(config);compare_analysis(saved,actual)
    pipeline=json.loads((folder/'pipeline.json').read_text(encoding='utf-8'))
    replay=probe_pipeline(config,folder,verify=True)
    compare_analysis(replay,{k:v for k,v in pipeline.items() if k not in ['latency','latency_scope']})
    check_latency(pipeline,config)
    for name,rows in table_rows(actual).items():check_csv(folder/name,rows)
    result=verification_record(config,component,folder,actual,replay)
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('VERIFIED unified system evaluation',flush=True);return result


def report(config,output):
    from src.system_evaluation.report import create_report
    return create_report(config,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all','build','verify','report'],default='all')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--output-directory')
    args=parser.parse_args();config,out=get_config(args.output_directory,args.smoke)
    if args.phase=='all':build(config,out);verify(config,out);report(config,out)
    else:globals()[args.phase](config,out)


if __name__=='__main__':main()
