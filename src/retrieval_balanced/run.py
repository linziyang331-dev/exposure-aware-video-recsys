"""Train-only graph and validation-only, predeclared balanced retrieval experiments."""
import argparse
import hashlib
import inspect
from pathlib import Path
import numpy as np
import yaml
from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.run import run_component,validate_component
from src.retrieval_longtail.run import setup as old_setup,freeze_selection,load_selection
from src.retrieval_longtail.train import read_json,write_json,evaluate_model
from src.retrieval_longtail.contracts import mean_metrics
from src.retrieval_balanced.graph import build_neighbors
from src.retrieval_balanced.train import train_run,load_model,backbone_digest
from src.retrieval_balanced.selection import qualifies,utility


def config(override=None):
    cfg=yaml.safe_load((ROOT/'configs/retrieval_balanced.yaml').read_text('utf-8'))
    out=(ROOT/(override or cfg['output_directory'])).resolve()
    if out.drive.upper()!='D:' or out.parent!=(ROOT/'experiments').resolve() or not out.name.startswith('retrieval_balanced_'):
        raise ValueError('Independent D: retrieval_balanced_* output required')
    for key in ['prepared_directory','cache_directory']:
        path=(ROOT/cfg[key]).resolve()
        if not path.is_relative_to(ROOT) or path.drive.upper()!='D:':raise ValueError('D: project path required')
    for key in ['reuse_trials_from','reuse_replication_from']:
        if cfg.get(key):
            path=(ROOT/cfg[key]).resolve()
            if path.drive.upper()!='D:' or path.parent!=(ROOT/'experiments').resolve() or not path.name.startswith('retrieval_balanced_') or path==out:
                raise ValueError('Separate D: project balanced experiment origin required')
    names=[r['name'] for r in cfg['trials']]
    if len(names)!=20 or len(set(names))!=20 or any('/' in n or '\\' in n or n in ['.','..'] for n in names):raise ValueError('Predeclared twenty unique trials required')
    if cfg['seeds']!=[42,43,44]:raise ValueError('Independent seeds42/43/44 required')
    return cfg,out


def baseline_metrics():
    load_selection(ROOT/'experiments/retrieval_longtail_main_v2')
    prior=read_json(ROOT/'experiments/retrieval_longtail_main_v2/replication.json')
    return {seed:r['best_validation'] for seed,r in prior['groups']['control'].items()}


def graph(cfg,out,counts,train):
    sig={'train':train.metadata['signature'],'num_users':int(train['user_index'].max()),'counts':counts.tolist(),
         'k':32,'code':digest_file(ROOT/'src/retrieval_balanced/graph.py'),
         'wrapper_code':hashlib.sha256(inspect.getsource(graph).encode()).hexdigest()}
    directory=out/'graph'
    def worker():
        directory.mkdir()
        idx=np.flatnonzero(train['label']==1)
        ids,w=build_neighbors(train['user_index'][idx],train['item_index'][idx],sig['num_users'],len(counts)-1,32)
        np.savez_compressed(directory/'neighbors.npz',ids=ids,weights=w)
        write_json(directory/'summary.json',{'training_positive_events':len(idx),'catalog_size':len(counts)-1,
            'directed_nonzero_edges':int((w>0).sum()),'items_without_neighbors':int((w[1:].sum(1)==0).sum()),
            'source':'standard/train positive binary user/item pairs only'})
    run_component(directory,sig,worker)
    with np.load(directory/'neighbors.npz') as a:return np.array(a['ids']),np.array(a['weights'])


def subset_graph(ids,weights,k):
    n=np.array(ids[:,:k],copy=True);w=np.array(weights[:,:k],copy=True);s=w.sum(1,keepdims=True)
    np.divide(w,s,out=w,where=s>0)
    return n,w


def trial_config(cfg,trial):
    keys=['device','embedding_dim','temperature','history_decay','graph_alpha','neighbors','rho','tau',
          'learning_rate','weight_decay','batch_size','eval_batch_size','epochs']
    return {k:trial.get(k,cfg[k]) for k in keys}


def select_replicated(groups,gate):
    means={n:mean_metrics([r['best_validation'] for r in group.values()]) for n,group in groups.items()}
    baseline=means['control'];passed=[]
    for name in groups:
        if name=='control':continue
        stable=all(all(groups[name][s]['best_validation']['groups']['tail_target'][k]>=
            groups['control'][s]['best_validation']['groups']['tail_target'][k] for k in ['recall@20','recall@100'])
            for s in ['42','43','44'])
        if stable and qualifies(means[name],baseline,gate):passed.append(name)
    chosen=max(passed,key=lambda n:(utility(means[n],baseline),means[n]['ndcg@20'])) if passed else None
    return means,passed,chosen


def tune(cfg,out):
    if (out/'selection.json').exists():raise ValueError('Selection frozen; cannot tune again')
    counts,caches=old_setup(cfg,['standard/train','standard/validation']);baselines=baseline_metrics()
    try:
        ids,w=graph(cfg,out,counts,caches['standard/train']);runs={}
        for t in cfg['trials']:
            c=trial_config(cfg,t);n,v=subset_graph(ids,w,c['neighbors'])
            origin=ROOT/cfg['reuse_trials_from'] if cfg.get('reuse_trials_from') else out
            runs[t['name']]=train_run(caches['standard/train'],caches['standard/validation'],counts,n,v,c,
                baselines['42'],cfg['gate'],origin/'trials'/t['name'],42)
        original=read_json(ROOT/'experiments/retrieval_v2/artifacts/tuning/residual_recency/training.json')
        if runs['control']['backbone_parameter_sha256']!=original['trained_parameter_sha256']:raise ValueError('Original control weights not reproduced')
        eligible=[n for n,r in runs.items() if r['qualified']]
        ordered=sorted(eligible,key=lambda n:(-runs[n]['tail_utility'],-runs[n]['best_validation']['ndcg@20'],n))
        candidates=ordered[:1]
        if ordered:
            accuracy=max(ordered,key=lambda n:(runs[n]['best_validation']['ndcg@20'],n))
            if accuracy not in candidates:candidates.append(accuracy)
        payload={'config':cfg,'runs':runs,'baselines':baselines,'candidate_names':candidates,
            'selected_trial':candidates[0] if candidates else None,'test_used_for_selection':False,
            'control_backbone_reproduced':True,'selection_rule':'material tail utility under overall/head98% retention'}
        write_json(out/'trials.json',payload)
        files=list((ROOT/'src/retrieval_balanced').glob('*.py'))+[ROOT/'configs/retrieval_balanced.yaml',out/'trials.json',out/'graph/component.json',out/'graph/neighbors.npz']
        for r in runs.values():files+=list(Path(r['checkpoint']).parent.glob('*'))
        files += [ROOT/'experiments/retrieval_longtail_main_v2/selection.json',ROOT/'experiments/retrieval_longtail_main_v2/replication.json']
        result=freeze_selection(out,payload,files)
        print('FROZEN candidates '+str(candidates),flush=True);return result
    finally:
        for c in caches.values():c.close()


def replicate(cfg,out):
    frozen=load_selection(out)
    if frozen['config']!=cfg:raise ValueError('Config changed after freeze')
    counts,caches=old_setup(cfg,['standard/train','standard/validation'])
    try:
        ids,w=graph(cfg,out,counts,caches['standard/train']);names=['control']+frozen['candidate_names'];runs={}
        for name in names:
            base=frozen['runs'][name];runs[name]={'42':base}
            n,v=subset_graph(ids,w,base['config']['neighbors'])
            origin=ROOT/cfg['reuse_replication_from'] if cfg.get('reuse_replication_from') else out
            for seed in [43,44]:runs[name][str(seed)]=train_run(caches['standard/train'],caches['standard/validation'],counts,n,v,
                base['config'],frozen['baselines'][str(seed)],cfg['gate'],origin/'replication'/f'{name}_seed{seed}',seed)
        means,passed,chosen=select_replicated(runs,cfg['gate'])
        result={'groups':runs,'validation_means':means,'baseline_mean':means['control'],'eligible_candidates':passed,
            'selected_for_adaptation':chosen,'selection_sha256':digest_file(out/'selection.json')}
        path=out/'replication.json'
        if path.exists() and read_json(path)!=result:raise ValueError('Replication evidence changed')
        if not path.exists():write_json(path,result)
        return result
    finally:
        for c in caches.values():c.close()


def evaluate(cfg,out):
    frozen=load_selection(out);rep=replicate(cfg,out)
    if (out/'evaluation.json').exists():raise ValueError('Evaluation completed; verify instead')
    counts,caches=old_setup(cfg,['standard/validation','standard/test','random/validation','random/test'])
    try:
        results={}
        for name,group in rep['groups'].items():
            results[name]={}
            for seed,r in group.items():
                model,_=load_model(r['checkpoint'],cfg['device']);results[name][seed]={}
                for part,c in caches.items():
                    path=out/'ranks'/f'{name}_{seed}'/(part.replace('/','_')+'.npz')
                    m=evaluate_model(model,c,counts,cfg['eval_batch_size'],path);m['rank_artifact']=str(path.resolve())
                    results[name][seed][part]=m
                    print(name+' '+seed+' '+part+' R20='+str(m['recall@20'])+' tailR100='+str(m['groups']['tail_target']['recall@100']),flush=True)
        result={'results':results,'selected_for_adaptation':rep['selected_for_adaptation'],
            'selection_sha256':digest_file(out/'selection.json'),'replication_sha256':digest_file(out/'replication.json'),
            'test_is_iterative_reassessment':True}
        write_json(out/'evaluation.json',result)
        write_json(out/'evaluation_manifest.json',{'output_hashes':{str(p.relative_to(out)):digest_file(p) for p in
            [out/'evaluation.json',out/'replication.json',out/'selection.json',*sorted((out/'ranks').rglob('*.npz'))]}})
        return result
    finally:
        for c in caches.values():c.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['tune','replicate','evaluate','adapt','reproduce','verify','report'],required=True)
    p.add_argument('--output-directory');a=p.parse_args();cfg,out=config(a.output_directory)
    if a.phase in ['tune','replicate','evaluate']:globals()[a.phase](cfg,out)
    elif a.phase=='reproduce':
        from src.retrieval_balanced.verify import reproduce
        reproduce(cfg,out)
    else:
        module=__import__('src.retrieval_balanced.'+a.phase,fromlist=[a.phase]);getattr(module,a.phase)(cfg,out)


if __name__=='__main__':main()
