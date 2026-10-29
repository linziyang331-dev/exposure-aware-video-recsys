"""Recompute saved metrics, align events and independently replay inference."""
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.run import validate_component
from src.exposure.verify import assert_result_equal
from src.retrieval.data import tensor_batch
from src.retrieval.metrics import target_ranks
from src.retrieval.statistics import paired_user_bootstrap
from src.retrieval_longtail.contracts import summarize,group_masks,choose_trial,mean_metrics,selection_passes
from src.retrieval_longtail.train import (read_json,write_json,load_model,train_run,
    training_signature,parameter_digest)


def validate_outputs(directory,manifest_path):
    directory=Path(directory).resolve();manifest=read_json(manifest_path)
    if not manifest['output_hashes']:raise ValueError('Empty output manifest')
    for name,sha in manifest['output_hashes'].items():
        p=(directory/name).resolve()
        if not p.is_relative_to(directory) or digest_file(p)!=sha:raise ValueError('Output changed: '+name)
    return len(manifest['output_hashes'])


def verify_rank_file(path,cache,counts):
    idx=np.flatnonzero(cache['label']==1)
    with np.load(path) as a:
        if (set(a.files)!={'ranks','users','event_ids'} or len(np.unique(a['event_ids']))!=len(idx)
                or not np.array_equal(a['event_ids'],cache['event_id'][idx])
                or not np.array_equal(a['users'],cache['user_id'][idx])):
            raise ValueError('Rank artifact event/user alignment changed')
        if (a['ranks']>len(counts)-1).any():raise ValueError('Rank exceeds full catalog')
        return summarize(a['ranks'],a['users'],cache['item_index'][idx],cache['history'][idx],counts)


def compare_metrics(actual,saved):
    for key in ['queries','users','recall@20','ndcg@20','recall@100','ndcg@100','mrr','user_macro','groups']:
        assert_result_equal(actual[key],saved[key])


def _reproduction_record(output,selection):
    names=['control']+([selection['selected_trial']] if selection['selected_trial'] else [])
    checks={}
    for name in names:
        if name not in selection['runs']:raise ValueError('Missing original reproduction control/candidate')
        old=selection['runs'][name];old_dir=Path(old['checkpoint']).parent
        new_dir=Path(output)/'reproduction'/name
        if not (new_dir/'training.json').is_file():raise ValueError('Missing independent reproduction model: '+name)
        new=read_json(new_dir/'training.json')
        if new['config']!=old['config'] or new['seed']!=42 or Path(new['checkpoint']).resolve()!=(new_dir/'best.pt').resolve():
            raise ValueError('Reproduction config/seed/checkpoint mismatch')
        signature=read_json(old_dir/'component.json')['signature']
        validate_component(old_dir,signature);validate_component(new_dir,signature)
        for run in [old,new]:
            model,_=load_model(run['checkpoint'],'cpu')
            if parameter_digest(model)!=run['trained_parameter_sha256']:raise ValueError('Reproduction parameter evidence mismatch')
        if old['trained_parameter_sha256']!=new['trained_parameter_sha256']:raise ValueError('Independent retraining parameter mismatch')
        with np.load(old_dir/'best_validation.npz') as a,np.load(new_dir/'best_validation.npz') as b:
            if set(a.files)!=set(b.files) or not all(np.array_equal(a[k],b[k]) for k in a.files):
                raise ValueError('Independent retraining rank mismatch')
        if len(old['curve'])!=len(new['curve']):raise ValueError('Curve length changed')
        for a,b in zip(old['curve'],new['curve']):
            if a['epoch']!=b['epoch'] or a['training_loss']!=b['training_loss']:raise ValueError('Reproduction curve mismatch')
            compare_metrics(a['validation'],b['validation'])
        checks[name]={'parameters_identical':True,'all_validation_ranks_identical':True,
            'all_epoch_losses_and_rank_metrics_identical':True,
            'parameter_sha256':new['trained_parameter_sha256'],'reproduction_run':str(new_dir.resolve())}
    return {'status':'PASS','scope':'same-machine independent directory from-scratch seed42 retraining; timing excluded',
            'selection_sha256':digest_file(Path(output)/'selection.json'),'models':checks}


def verify_reproduction(output,selection):
    result=read_json(Path(output)/'reproduction.json')
    expected={'control'}|({selection['selected_trial']} if selection['selected_trial'] else set())
    if set(result.get('models',{}))!=expected:raise ValueError('Reproduction must include exact required control/candidate')
    actual=_reproduction_record(output,selection)
    if actual!=result:raise ValueError('Reproduction summary differs from actual retraining evidence')
    return result


def reproduce(cfg,output):
    from src.retrieval_longtail.run import load_selection,setup
    selection=load_selection(output)
    if selection['config']!=cfg:raise ValueError('Config differs from frozen selection')
    if (output/'report_manifest.json').exists():validate_outputs(output,output/'report_manifest.json')
    if (output/'reproduction.json').exists():return verify_reproduction(output,selection)
    counts,caches=setup(cfg,['standard/train','standard/validation'])
    try:
        names=['control']+([selection['selected_trial']] if selection['selected_trial'] else [])
        for name in names:
            old=selection['runs'][name]
            train_run(caches['standard/train'],caches['standard/validation'],counts,old['config'],
                      output/'reproduction'/name,42)
        result=_reproduction_record(output,selection)
        write_json(output/'reproduction.json',result)
        return result
    finally:
        for c in caches.values():c.close()


def verify(cfg,output):
    from src.retrieval_longtail.run import load_selection,setup,old_integrity
    selection=load_selection(output)
    if selection['config']!=cfg:raise ValueError('Config differs from frozen selection')
    if (output/'report_manifest.json').exists():validate_outputs(output,output/'report_manifest.json')
    count=validate_outputs(output,output/'evaluation_manifest.json')
    replication=read_json(output/'replication.json');evaluation=read_json(output/'evaluation.json')
    if (evaluation['selection_sha256']!=digest_file(output/'selection.json')
            or evaluation['replication_sha256']!=digest_file(output/'replication.json')
            or replication['selection_sha256']!=digest_file(output/'selection.json')):
        raise ValueError('Freeze/replication/evaluation chain changed')
    counts,caches=setup(cfg,['standard/train','standard/validation','standard/test','random/validation','random/test'])
    try:
        run_checks=0
        runs=list(selection['runs'].values())
        runs+= [r for group in replication['groups'].values() for r in group.values()]
        repro=verify_reproduction(output,selection)
        runs += [read_json(Path(r['reproduction_run'])/'training.json') for r in repro['models'].values()]
        for directory in sorted({str(Path(r['checkpoint']).parent) for r in runs}):
            r=read_json(Path(directory)/'training.json')
            signature=training_signature(caches['standard/train'],caches['standard/validation'],counts,r['config'],r['seed'])
            validate_component(directory,signature)
            model,_=load_model(r['checkpoint'],cfg['device'])
            if parameter_digest(model)!=r['trained_parameter_sha256']:raise ValueError('Parameter digest differs')
            actual=verify_rank_file(Path(directory)/'best_validation.npz',caches['standard/validation'],counts)
            compare_metrics(actual,r['best_validation']);run_checks+=1
        metrics={n:r['best_validation'] for n,r in selection['runs'].items()}
        chosen=choose_trial({n:m for n,m in metrics.items() if n!='control'},metrics['control'],cfg['gate'])
        if chosen!=selection['selected_trial']:raise ValueError('Frozen selection not derived from validation gate')
        means={g:mean_metrics([r['best_validation'] for r in rs.values()]) for g,rs in replication['groups'].items()}
        assert_result_equal(means,replication['validation_mean'])
        gate=bool('candidate' in means and selection_passes(means['candidate'],means['control'],cfg['gate']))
        if gate!=replication['three_seed_validation_gate_passed'] or gate!=evaluation['three_seed_validation_gate_passed']:
            raise ValueError('Validation gate evidence changed')
        replayed=0;rank_files=0
        for group,seeds in evaluation['results'].items():
            for seed,parts in seeds.items():
                model,_=load_model(replication['groups'][group][seed]['checkpoint'],cfg['device'])
                with torch.no_grad():item=model.encode_item(torch.arange(1,len(counts),device=cfg['device']))
                for part,saved in parts.items():
                    cache=caches[part];path=Path(saved['rank_artifact'])
                    compare_metrics(verify_rank_file(path,cache,counts),saved);rank_files+=1
                    idx=np.flatnonzero(cache['label']==1)[:cfg['eval_batch_size']]
                    batch=tensor_batch(cache,idx,cfg['device'])
                    with torch.no_grad():scores=model.encode_user(batch['users'],batch['tabs'],batch['history'])@item.T/model.temperature
                    ranks=target_ranks(scores,torch.arange(1,len(counts),device=cfg['device']),batch['items']).cpu().numpy()
                    with np.load(path) as a:
                        if not np.array_equal(ranks,a['ranks'][:len(idx)]):raise ValueError('Actual checkpoint rank replay changed')
                    # Scoring only receives context; labels/targets are not inputs to either encoder.
                    replayed+=len(idx)
        paired={}
        if 'candidate' in evaluation['results']:
            for part,cache in caches.items():
                if part=='standard/train':continue
                idx=np.flatnonzero(cache['label']==1);paired[part]={}
                with (np.load(evaluation['results']['candidate']['42'][part]['rank_artifact']) as a,
                      np.load(evaluation['results']['control']['42'][part]['rank_artifact']) as b):
                    masks={'all':np.ones(len(idx),dtype=bool)}|group_masks(cache['item_index'][idx],cache['history'][idx],counts)
                    for g in ['all','head_target','tail_target']:
                        m=masks[g]
                        ci=paired_user_bootstrap(a['ranks'][m],b['ranks'][m],a['users'][m],cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                        ci100=paired_user_bootstrap(np.where((a['ranks'][m]>0)&(a['ranks'][m]<=100),1,0),
                            np.where((b['ranks'][m]>0)&(b['ranks'][m]<=100),1,0),a['users'][m],cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                        paired[part][g]={k:ci[k] for k in ['recall@20','ndcg@20']}|{'recall@100':ci100['recall@20']}
        frozen=old_integrity()
        result={'status':'PASS','rank_files_recomputed':rank_files,'checkpoint_context_rank_replays':replayed,
            'training_runs_validated':run_checks,'evaluation_output_hashes_validated':count,
            'old_v2_frozen_files_unchanged':len(frozen['evidence']['files']),
            'selection_and_three_seed_gate_recomputed':True,'paired_seed42_user_cluster_intervals':paired,
            'scope':'full saved-rank metric recomputation; first512 positives/partition/model checkpoint replay; fixed batch512',
            'three_seed_validation_gate_passed':gate,'end_to_end_din_adapted':False,
            'evaluation_manifest_sha256':digest_file(output/'evaluation_manifest.json'),
            'reproduction_sha256':digest_file(output/'reproduction.json')}
        write_json(output/'verification.json',result)
        print('VERIFIED '+str(rank_files)+' rank files, '+str(replayed)+' actual inference replays',flush=True)
        return result
    finally:
        for c in caches.values():c.close()
