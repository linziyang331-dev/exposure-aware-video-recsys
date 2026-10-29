"""Bounded validation-only selection; test access follows an immutable freeze."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.data import load_partition,training_counts
from src.retrieval.provenance import read_verified_vocabulary
from src.retrieval_v2.provenance import verify_evidence
from src.retrieval_longtail.train import read_json,write_json,train_run,load_model,evaluate_model
from src.retrieval_longtail.contracts import choose_trial,selection_passes,mean_metrics


def get_config(override=None):
    cfg=yaml.safe_load((ROOT/'configs/retrieval_longtail.yaml').read_text(encoding='utf-8'))
    output=(ROOT/(override or cfg['output_directory'])).resolve()
    if (output.drive.upper()!='D:' or output.parent!=(ROOT/'experiments').resolve()
            or not output.name.startswith('retrieval_longtail_')):
        raise ValueError('Separate D: experiments/retrieval_longtail_* output required')
    for key in ['prepared_directory','cache_directory']:
        path=(ROOT/cfg[key]).resolve()
        if path.drive.upper()!='D:' or not path.is_relative_to(ROOT):raise ValueError('D: project paths required')
    if cfg['seeds']!=[42,43,44]:raise ValueError('Predeclared independent seeds42/43/44 required')
    if cfg.get('reuse_trials_from'):
        prior=(ROOT/cfg['reuse_trials_from']).resolve()
        if (prior.parent!=(ROOT/'experiments').resolve() or not prior.name.startswith('retrieval_longtail_')
                or prior==output):raise ValueError('Separate prior longtail directory required')
    return cfg,output


def old_integrity():
    frozen=read_json(ROOT/'reports/retrieval_v2_freeze.json')
    verify_evidence(ROOT,frozen['evidence'],frozen['evidence']['config'])
    if digest_file(ROOT/frozen['checkpoint'])!=frozen['checkpoint_sha256']:raise ValueError('Frozen V2 changed')
    return frozen


def setup(cfg,partitions):
    old_integrity()
    vocabulary=read_verified_vocabulary(ROOT/cfg['prepared_directory']);counts=training_counts(vocabulary)
    caches={p:load_partition(ROOT/cfg['prepared_directory'],p,ROOT/cfg['cache_directory'],cfg['history_length']) for p in partitions}
    if 'standard/train' in caches:
        c=caches['standard/train'];actual=np.bincount(c['item_index'][c['label']==1],minlength=len(counts))
        if not np.array_equal(actual,counts):raise ValueError('Training count/positive events disagree')
    return counts,caches


def trial_matrix(cfg,output):
    if (Path(output)/'selection.json').exists():raise ValueError('Selection frozen; no more tuning')
    trials=cfg['trials'];names=[t['name'] for t in trials]
    if (not 1<=len(trials)<=8 or len(names)!=len(set(names))
            or any(not n or '/' in n or '\\' in n or n in {'.','..'} for n in names)):
        raise ValueError('One to eight unique simple trial names required')
    return trials


def freeze_selection(output,payload,files):
    output=Path(output);path=output/'selection.json'
    if path.exists():raise ValueError('Selection already frozen')
    write_json(path,payload|{'evidence':{str(Path(p).resolve()):digest_file(p) for p in sorted(set(map(Path,files)))}})
    return load_selection(output)


def load_selection(output):
    result=read_json(Path(output)/'selection.json')
    for name,sha in result['evidence'].items():
        path=Path(name).resolve()
        if not path.is_relative_to(ROOT) or digest_file(path)!=sha:raise ValueError('Frozen selection evidence changed: '+name)
    canonical=Path(output)/'trials.json'
    if str(canonical.resolve()) not in result['evidence']:
        raise ValueError('Frozen canonical trial ledger required')
    if {k:v for k,v in result.items() if k!='evidence'}!=read_json(canonical):
        raise ValueError('Selection payload differs from frozen trial ledger')
    for name,run in result.get('runs',{}).items():
        path=Path(run['checkpoint']).parent/'training.json'
        if str(path.resolve()) not in result['evidence'] or read_json(path)!=run:
            raise ValueError('Embedded training run differs from frozen file: '+name)
    return result


def train_config(cfg,trial):
    keys=['device','embedding_dim','temperature','history_decay','weight_cap','weight_decay',
          'batch_size','eval_batch_size','epochs','patience']
    return {k:cfg[k] for k in keys}|{k:trial[k] for k in ['model','learning_rate','beta']}


def tune(cfg,output):
    trials=trial_matrix(cfg,output)
    counts,caches=setup(cfg,['standard/train','standard/validation'])
    try:
        runs={}
        for trial in trials:
            prior=ROOT/cfg['reuse_trials_from'] if cfg.get('reuse_trials_from') else output
            name=trial['name'];runs[name]=train_run(caches['standard/train'],caches['standard/validation'],counts,
                train_config(cfg,trial),prior/'tuning'/name,42)
        # Exact original training control guards accidental protocol/implementation changes.
        original=read_json(ROOT/'experiments/retrieval_v2/artifacts/tuning/residual_recency/training.json')
        if runs['control']['trained_parameter_sha256']!=original['trained_parameter_sha256']:
            raise ValueError('Matched control did not reproduce frozen V2 parameters')
        metrics={n:r['best_validation'] for n,r in runs.items()}
        selected=choose_trial({n:m for n,m in metrics.items() if n!='control'},metrics['control'],cfg['gate'])
        payload={'status':'VALIDATION_SELECTION_FROZEN','config':cfg,'selected_trial':selected,
            'selection_uses':'standard validation only; NDCG20 among tail-gain and retention-qualified trials',
            'test_loaded_for_tuning':False,'test_is_iterative_reassessment':True,
            'gate':cfg['gate'],'runs':runs,'metrics':metrics,'matched_control_parameters_identical':True}
        write_json(output/'trials.json',payload)
        files=[output/'trials.json',ROOT/'configs/retrieval_longtail.yaml',ROOT/'reports/retrieval_v2_freeze.json',
               ROOT/cfg['prepared_directory']/'manifest.json',ROOT/cfg['prepared_directory']/'vocabulary.json']
        files+=list((ROOT/'src/retrieval_longtail').glob('*.py'))
        for run in runs.values():files+=list(Path(run['checkpoint']).parent.glob('*'))
        selection=freeze_selection(output,payload,files)
        print('FROZEN selected_trial='+str(selected),flush=True)
        return selection
    finally:
        for c in caches.values():c.close()


def replicate(cfg,output):
    selection=load_selection(output)
    if selection['config']!=cfg:raise ValueError('Config changed after selection')
    counts,caches=setup(cfg,['standard/train','standard/validation'])
    try:
        groups={'control':{}}
        if selection['selected_trial'] is not None:groups['candidate']={}
        for group in groups:
            name='control' if group=='control' else selection['selected_trial']
            original=selection['runs'][name]
            for seed in cfg['seeds']:
                groups[group][str(seed)]=original if seed==42 else train_run(caches['standard/train'],caches['standard/validation'],
                    counts,original['config'],output/'replication'/f'{group}_seed{seed}',seed)
        means={g:mean_metrics([r['best_validation'] for r in runs.values()]) for g,runs in groups.items()}
        passed='candidate' in means and selection_passes(means['candidate'],means['control'],cfg['gate'])
        payload={'groups':groups,'validation_mean':means,'three_seed_validation_gate_passed':bool(passed),
            'selected_trial':selection['selected_trial'],'selection_sha256':digest_file(output/'selection.json')}
        path=output/'replication.json'
        if path.exists():
            if read_json(path)!=payload:raise ValueError('Replication changed')
        else:write_json(path,payload)
        return payload
    finally:
        for c in caches.values():c.close()


def evaluate(cfg,output):
    selection=load_selection(output)
    if selection['config']!=cfg:raise ValueError('Config changed after freeze')
    replication=replicate(cfg,output)
    if (output/'evaluation.json').exists():raise ValueError('Reassessment already complete; use verify, not retuning')
    counts,caches=setup(cfg,['standard/validation','standard/test','random/validation','random/test'])
    try:
        results={}
        for group,runs in replication['groups'].items():
            results[group]={}
            for seed,run in runs.items():
                model,_=load_model(run['checkpoint'],cfg['device']);results[group][seed]={}
                for part,c in caches.items():
                    artifact=output/'ranks'/f'{group}_seed{seed}'/(part.replace('/','_')+'.npz')
                    metric=evaluate_model(model,c,counts,cfg['eval_batch_size'],artifact)
                    metric['rank_artifact']=str(artifact.resolve());results[group][seed][part]=metric
                    print(group+' seed'+seed+' '+part+' R20='+str(metric['recall@20'])+' tailR100='+str(metric['groups']['tail_target']['recall@100']),flush=True)
        data={'results':results,'three_seed_validation_gate_passed':replication['three_seed_validation_gate_passed'],
            'selection_sha256':digest_file(output/'selection.json'),'replication_sha256':digest_file(output/'replication.json'),
            'end_to_end_din_adapted':False,'test_is_iterative_reassessment':True}
        write_json(output/'evaluation.json',data)
        hashes={str(p.relative_to(output)):digest_file(p) for p in output.rglob('*.npz')}
        hashes.update({n:digest_file(output/n) for n in ['evaluation.json','replication.json','selection.json','diagnosis.json']})
        write_json(output/'evaluation_manifest.json',{'output_hashes':hashes})
        return data
    finally:
        for c in caches.values():c.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['diagnose','tune','replicate','evaluate','verify','report','reproduce','all'],default='all')
    p.add_argument('--output-directory');args=p.parse_args();cfg,output=get_config(args.output_directory)
    if args.phase in ['diagnose','all']:
        from src.retrieval_longtail.diagnose import diagnose
        diagnose(cfg,output)
    if args.phase in ['tune','all']:tune(cfg,output)
    if args.phase in ['replicate','all']:replicate(cfg,output)
    if args.phase in ['evaluate','all']:evaluate(cfg,output)
    if args.phase in ['reproduce','all']:
        from src.retrieval_longtail.verify import reproduce
        reproduce(cfg,output)
    if args.phase in ['verify','all']:
        from src.retrieval_longtail.verify import verify
        verify(cfg,output)
    if args.phase in ['report','all']:
        from src.retrieval_longtail.report import report
        report(cfg,output)


if __name__=='__main__':main()
