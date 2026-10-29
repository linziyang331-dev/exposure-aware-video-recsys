"""Actual-rank verification, source-bound runs and independent retraining evidence."""
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.run import validate_component
from src.exposure.verify import assert_result_equal
from src.retrieval.data import tensor_batch
from src.retrieval.metrics import target_ranks,stable_topk
from src.retrieval.statistics import paired_user_bootstrap
from src.retrieval_v2.train import parameter_digest
from src.retrieval_longtail.run import load_selection,setup
from src.retrieval_longtail.train import read_json,write_json
from src.retrieval_longtail.verify import verify_rank_file,compare_metrics,validate_outputs
from src.retrieval_longtail.contracts import group_masks
from src.retrieval_balanced.selection import qualifies,utility
from src.retrieval_balanced.train import train_run,load_model,signature


def selected_epoch(curve,baseline,gate):
    good=[r for r in curve if qualifies(r['validation'],baseline,gate)]
    if good:return max(good,key=lambda r:(utility(r['validation'],baseline),r['validation']['ndcg@20']))['epoch'],True
    return max(curve,key=lambda r:r['validation']['ndcg@20'])['epoch'],False


def repro_names(rep):
    return ['control']+([rep['selected_for_adaptation']] if rep['selected_for_adaptation'] else [])


def reproduction_record(cfg,out,frozen,rep,train,val,counts,ids,w):
    from src.retrieval_balanced.run import subset_graph
    records={}
    for name in repro_names(rep):
        original=rep['groups'][name]['42'];directory=out/'reproduction'/name
        if not (directory/'training.json').exists():raise ValueError('Required independent reproduction missing')
        new=read_json(directory/'training.json');n,v=subset_graph(ids,w,original['config']['neighbors'])
        expected=signature(train,val,counts,n,v,original['config'],frozen['baselines']['42'],cfg['gate'],42)
        validate_component(directory,expected)
        if new['config']!=original['config'] or new['seed']!=42:raise ValueError('Reproduction config/seed differs')
        a,_=load_model(original['checkpoint'],'cpu');b,_=load_model(new['checkpoint'],'cpu')
        if parameter_digest(a)!=parameter_digest(b) or parameter_digest(a)!=original['trained_parameter_sha256']:
            raise ValueError('Independent retraining parameters differ')
        with np.load(Path(original['checkpoint']).parent/'best_validation.npz') as x,np.load(directory/'best_validation.npz') as y:
            if not all(np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Independent retraining validation ranks differ')
        if len(new['curve'])!=len(original['curve']):raise ValueError('Independent retraining curve length differs')
        for x,y in zip(original['curve'],new['curve']):
            if x['loss']!=y['loss'] or x['qualified']!=y['qualified']:raise ValueError('Independent retraining loss/selection differs')
            compare_metrics(x['validation'],y['validation'])
        records[name]={'parameter_sha256':parameter_digest(b),'all_losses_and_validation_rank_metrics_identical':True,
                       'all_selected_validation_ranks_identical':True,'directory':str(directory.resolve())}
    return {'status':'PASS','scope':'same-machine seed42 from-scratch independent directories; timing excluded',
            'selection_sha256':digest_file(out/'selection.json'),'models':records}


def reproduce(cfg,out):
    from src.retrieval_balanced.run import graph,subset_graph
    frozen=load_selection(out);rep=read_json(out/'replication.json')
    if frozen['config']!=cfg:raise ValueError('Frozen config changed')
    if (out/'report_manifest.json').exists():validate_outputs(out,out/'report_manifest.json')
    counts,caches=setup(cfg,['standard/train','standard/validation'])
    try:
        ids,w=graph(cfg,out,counts,caches['standard/train']);path=out/'reproduction.json'
        if not path.exists():
            for name in repro_names(rep):
                r=rep['groups'][name]['42'];n,v=subset_graph(ids,w,r['config']['neighbors'])
                train_run(caches['standard/train'],caches['standard/validation'],counts,n,v,r['config'],
                    frozen['baselines']['42'],cfg['gate'],out/'reproduction'/name,42)
            record=reproduction_record(cfg,out,frozen,rep,caches['standard/train'],caches['standard/validation'],counts,ids,w)
            write_json(path,record)
        else:
            record=reproduction_record(cfg,out,frozen,rep,caches['standard/train'],caches['standard/validation'],counts,ids,w)
            if read_json(path)!=record:raise ValueError('Completed reproduction evidence changed')
        return record
    finally:
        for c in caches.values():c.close()


def verify(cfg,out):
    from src.retrieval_balanced.run import graph,subset_graph,select_replicated
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    frozen=load_selection(out)
    if frozen['config']!=cfg:raise ValueError('Frozen config changed')
    if (out/'report_manifest.json').exists():validate_delivery(out)
    validate_outputs(out,out/'evaluation_manifest.json');rep=read_json(out/'replication.json');evaluation=read_json(out/'evaluation.json')
    if (rep['selection_sha256']!=digest_file(out/'selection.json')
            or evaluation['selection_sha256']!=digest_file(out/'selection.json')
            or evaluation['replication_sha256']!=digest_file(out/'replication.json')):raise ValueError('Frozen lineage differs')
    counts,caches=setup(cfg,['standard/train','standard/validation','standard/test','random/validation','random/test'])
    try:
        ids,w=graph(cfg,out,counts,caches['standard/train'])
        runs=list(frozen['runs'].values())+[r for g in rep['groups'].values() for r in g.values()]
        checked=set()
        for r in runs:
            directory=Path(r['checkpoint']).parent
            if directory in checked:continue
            if not directory.resolve().is_relative_to(ROOT):raise ValueError('Run outside workspace')
            if read_json(directory/'training.json')!=r:raise ValueError('Embedded run differs from actual component')
            n,v=subset_graph(ids,w,r['config']['neighbors'])
            expected=signature(caches['standard/train'],caches['standard/validation'],counts,n,v,r['config'],
                frozen['baselines'][str(r['seed'])],cfg['gate'],r['seed'])
            validate_component(directory,expected)
            if selected_epoch(r['curve'],r['baseline'],r['gate'])!=(r['best_epoch'],r['qualified']):raise ValueError('Selected epoch violates material-tail rule')
            compare_metrics(verify_rank_file(directory/'best_validation.npz',caches['standard/validation'],counts),r['best_validation'])
            model,_=load_model(r['checkpoint'],cfg['device'])
            if parameter_digest(model)!=r['trained_parameter_sha256']:raise ValueError('Checkpoint parameters differ')
            np.testing.assert_array_equal(model.neighbor_ids.cpu().numpy(),n)
            np.testing.assert_allclose(model.neighbor_weights.cpu().numpy(),v,rtol=0,atol=0)
            checked.add(directory)
        means,eligible,chosen=select_replicated(rep['groups'],cfg['gate'])
        assert_result_equal(means,rep['validation_means'])
        if chosen!=rep['selected_for_adaptation'] or eligible!=rep['eligible_candidates']:raise ValueError('Three-seed material gate differs')
        record=reproduction_record(cfg,out,frozen,rep,caches['standard/train'],caches['standard/validation'],counts,ids,w)
        if record!=read_json(out/'reproduction.json'):raise ValueError('Reproduction record differs from actual evidence')
        rank_files=0;replays=0;paired={}
        for name,group in evaluation['results'].items():
            for seed,parts in group.items():
                model,_=load_model(rep['groups'][name][seed]['checkpoint'],cfg['device'])
                with torch.no_grad():bank=model.encode_item(torch.arange(1,len(counts),device=cfg['device']))
                for part,m in parts.items():
                    c=caches[part];p=Path(m['rank_artifact']);actual=verify_rank_file(p,c,counts);compare_metrics(actual,m);rank_files+=1
                    rows=np.flatnonzero(c['label']==1)[:512];b=tensor_batch(c,rows,cfg['device'])
                    with torch.no_grad():score=model.encode_user(b['users'],b['tabs'],b['history'])@bank.T/model.temperature
                    rank=target_ranks(score,torch.arange(1,len(counts),device=cfg['device']),b['items']).cpu().numpy()
                    with np.load(p) as a:np.testing.assert_array_equal(rank,a['ranks'][:len(rows)])
                    replays+=len(rows)
            if name!='control':
                paired[name]={}
                for part,c in caches.items():
                    if part=='standard/train':continue
                    idx=np.flatnonzero(c['label']==1);masks={'all':np.ones(len(idx),bool)}|group_masks(c['item_index'][idx],c['history'][idx],counts)
                    with (np.load(group['42'][part]['rank_artifact']) as a,
                          np.load(evaluation['results']['control']['42'][part]['rank_artifact']) as b):
                        paired[name][part]={}
                        for g in ['all','head_target','tail_target']:
                            m=masks[g];ci=paired_user_bootstrap(a['ranks'][m],b['ranks'][m],a['users'][m],cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                            c100=paired_user_bootstrap(np.where((a['ranks'][m]>0)&(a['ranks'][m]<=100),1,0),
                                np.where((b['ranks'][m]>0)&(b['ranks'][m]<=100),1,0),a['users'][m],cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                            paired[name][part][g]={k:ci[k] for k in ['recall@20','ndcg@20']}|{'recall@100':c100['recall@20']}
        adaptation=None
        if chosen is not None and not (out/'adaptation.json').exists():raise ValueError('Eligible retriever requires completed DIN adaptation before final verification')
        if (out/'adaptation.json').exists():adaptation=verify_adaptation(cfg,out,frozen,rep,counts,caches)
        result={'status':'PASS','training_components_checked':len(checked),'rank_files_recomputed':rank_files,
            'actual_retriever_context_replays':replays,'selected_for_adaptation':chosen,'paired_seed42':paired,
            'adaptation':adaptation,'reproduction_sha256':digest_file(out/'reproduction.json'),
            'evaluation_manifest_sha256':digest_file(out/'evaluation_manifest.json'),
            'model_changes_to_original':False,'test_is_iterative_reassessment':True}
        write_json(out/'verification.json',result)
        print('VERIFIED '+str(rank_files)+' retrieval ranks, '+str(replays)+' actual contexts; adapted='+str(adaptation is not None),flush=True)
        return result
    finally:
        for c in caches.values():c.close()


def verify_adaptation(cfg,out,frozen,rep,counts,caches):
    from src.retrieval_balanced.adapt import candidate_signature,rank_signature,load_model as load_din
    from src.ranking.candidates import CandidateCache
    from src.ranking.metrics import candidate_ranks
    adaptation=read_json(out/'adaptation.json')
    if adaptation['status']=='SKIPPED':
        if rep['selected_for_adaptation'] is not None:raise ValueError('Adaptation incorrectly skipped')
        return {'status':'SKIPPED'}
    validate_outputs(out,out/'adaptation_manifest.json')
    if adaptation['selected_retriever']!=rep['selected_for_adaptation'] or adaptation['replication_sha256']!=digest_file(out/'replication.json'):
        raise ValueError('Adaptation teacher/selection lineage differs')
    result={'status':'PASS','rank_files_recomputed':0,'actual_ranker_context_replays':0,'candidate_rows_checked':0,'metrics':{}}
    for name,group in adaptation['runs'].items():
        retriever=rep['groups'][name]['42'];teacher,_=load_model(retriever['checkpoint'],cfg['device']);candidate_sets={}
        try:
            for part,c in caches.items():
                path=out/'adaptation/candidates'/name/part.replace('/','_')
                validate_component(path,candidate_signature(c,counts,teacher,retriever['checkpoint_sha256']))
                ca=CandidateCache(path,read_json(path/'metadata.json'));candidate_sets[part]=ca
                idx=np.flatnonzero(c['label']==1)
                np.testing.assert_array_equal(ca['indices'],idx);np.testing.assert_array_equal(ca['event_ids'],c['event_id'][idx])
                for offset in range(0,len(ca),512):
                    s=slice(offset,offset+512);n=np.array(ca['ids'][s]);v=np.array(ca['scores'][s])
                    if (n<=0).any() or (n>=len(counts)).any() or not np.isfinite(v).all() or (np.diff(np.sort(n,axis=1),axis=1)==0).any():raise ValueError('Invalid candidates')
                    t=torch.tensor(np.array(c['item_index'][idx[s]]),dtype=torch.long)
                    ranks=candidate_ranks(torch.tensor(n),torch.tensor(v),t).numpy()
                    np.testing.assert_array_equal(ranks,ca['v2_ranks'][s])
                rows=idx[:512];b=tensor_batch(c,rows,cfg['device'])
                with torch.no_grad():
                    scores=teacher.encode_user(b['users'],b['tabs'],b['history'])@teacher.encode_item(torch.arange(1,len(counts),device=cfg['device'])).T/teacher.temperature
                    top,values=stable_topk(scores,torch.arange(1,len(counts),device=cfg['device']),100)
                np.testing.assert_array_equal(top.cpu().numpy(),ca['ids'][:len(rows)])
                np.testing.assert_allclose(values.cpu().numpy(),ca['scores'][:len(rows)],rtol=0,atol=0)
                result['candidate_rows_checked']+=len(ca)
            result['metrics'][name]={}
            for seed,r in group.items():
                directory=Path(r['checkpoint']).parent
                if read_json(directory/'training.json')!=r:raise ValueError('Adapted run metadata differs')
                validate_component(directory,rank_signature(caches['standard/train'],caches['standard/validation'],
                    candidate_sets['standard/train'],candidate_sets['standard/validation'],teacher,r['config'],int(seed)))
                model,saved=load_din(r['checkpoint'],cfg['device'])
                if parameter_digest(model.teacher)!=retriever['trained_parameter_sha256']:raise ValueError('Restored frozen teacher changed')
                if parameter_digest(model)!=r['trained_parameter_sha256']:raise ValueError('Adapted model changed')
                result['metrics'][name][seed]={}
                for part,m in adaptation['evaluation'][name][seed].items():
                    c=caches[part];ca=candidate_sets[part];p=Path(m['rank_artifact']);actual=verify_rank_file(p,c,counts)
                    for key in ['queries','users','recall@20','ndcg@20','recall@100','ndcg@100']:
                        assert_result_equal(actual[key],m['ranking'][key])
                    with np.load(p) as a:
                        if (a['ranks']>100).any():raise ValueError('Adapted ranks exceed their frozen pool')
                        np.testing.assert_array_equal(a['ranks']>0,ca['v2_ranks']>0)
                        size=min(256,len(ca));idx=ca['indices'][:size];b=tensor_batch(c,idx,cfg['device']);ids=np.array(ca['ids'][:size]);k=ids.shape[1]
                        with torch.no_grad():v=model(b['users'].repeat_interleave(k),torch.tensor(ids.ravel(),device=cfg['device']),
                            b['tabs'].repeat_interleave(k),b['history'].repeat_interleave(k,dim=0),
                            base_scores=torch.tensor(np.array(ca['scores'][:size]).ravel(),device=cfg['device']))['rank'].reshape(size,k)
                        ranks=candidate_ranks(torch.tensor(ids,device=cfg['device']),v,b['items']).cpu().numpy()
                        np.testing.assert_array_equal(ranks,a['ranks'][:size])
                    result['metrics'][name][seed][part]=actual
                    result['rank_files_recomputed']+=1;result['actual_ranker_context_replays']+=size
        finally:
            for ca in candidate_sets.values():ca.close()
    return result


def validate_delivery(out):
    count=validate_outputs(out,out/'report_manifest.json');m=read_json(out/'report_manifest.json')
    allowed={'reports/balanced_retrieval_report.md','reports/balanced_retrieval_results.json'}
    frozen=read_json(out/'selection.json');canonical=(ROOT/frozen['config']['output_directory']).resolve()
    expected=allowed if out.resolve()==canonical else set()
    if set(m.get('global_mirrors',{}))!=expected:raise ValueError('Required global report mirror hashes missing or unexpected')
    for name,sha in m.get('global_mirrors',{}).items():
        if name not in allowed or digest_file(ROOT/name)!=sha:raise ValueError('Published report mirror changed')
    return count
