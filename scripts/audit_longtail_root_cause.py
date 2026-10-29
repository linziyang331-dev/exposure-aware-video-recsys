"""Read-only long-tail mechanism audit. No fitting, model updates or deployment."""
import argparse
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.exposure.run import run_component,validate_component
from src.retrieval.data import tensor_batch
from src.retrieval.metrics import target_ranks,stable_topk,summarize_ranks
from src.retrieval_v2.model import TwoTowerV2
from src.retrieval_v2.train import load_model,parameter_digest
from src.retrieval_longtail.run import get_config,setup,load_selection
from src.retrieval_longtail.train import read_json,write_json
from src.retrieval_longtail.contracts import group_masks,frequency_weights,summarize


def competition_breakdown(full_ranks,tail_ranks,k):
    a=np.asarray(full_ranks);b=np.asarray(tail_ranks)
    if a.ndim!=1 or not len(a) or a.shape!=b.shape or (b<=0).any() or (a<b).any():
        raise ValueError('Aligned known tail ranks with full>=tail>0 required')
    hit=a<=k;crowded=(a>k)&(b<=k);failed=b>k
    return {'queries':len(a),'global_hits':int(hit.sum()),'head_crowding_losses':int(crowded.sum()),
            'within_tail_failures':int(failed.sum())}


def paired_outcomes(tower,itemcf,k):
    a=np.asarray(tower);b=np.asarray(itemcf)
    if a.shape!=b.shape or a.ndim!=1:raise ValueError('Aligned ranks required')
    ah=(a>0)&(a<=k);bh=(b>0)&(b<=k)
    return {'queries':len(a),'both_hit':int((ah&bh).sum()),'tower_only':int((ah&~bh).sum()),
            'itemcf_only':int((~ah&bh).sum()),'both_miss':int((~ah&~bh).sum())}


def weighting_audit(counts):
    c=np.asarray(counts);masks=group_masks(np.arange(len(c)),np.zeros((len(c),1),dtype=int),c)
    result={}
    for beta in [0.,.25,.5]:
        w=frequency_weights(c,beta,3.);raw=np.minimum((c.max()/np.maximum(c[1:],1))**beta,3.)
        result[str(beta)]={'global_event_mean_weight':float(np.dot(c,w)/c.sum()),
            'catalog_fraction_at_raw_cap':float(np.mean(raw==3.)),
            'groups':{g:{'weighted_event_share':float(np.dot(c[m],w[m])/np.dot(c,w)),
                         'weight_min':float(w[m].min()) if m.any() else None,
                         'weight_max':float(w[m].max()) if m.any() else None}
                      for g,m in masks.items() if g in ['head_target','tail_target','freq_zero','freq_1_5','freq_6_20','freq_21_100']}}
    return result


def choice_mass(scores,targets):
    scores=np.asarray(scores,dtype=np.float64);targets=np.asarray(targets)
    if (scores.ndim!=2 or targets.shape!=(len(scores),) or not len(scores)
            or (targets<=0).any() or (targets>scores.shape[1]).any()):
        raise ValueError('Full catalog logits and known one-based targets required')
    values=np.exp(scores-scores.max(1,keepdims=True));values/=values.sum(1,keepdims=True)
    positive=np.bincount(targets,minlength=scores.shape[1]+1)[1:]/len(scores)
    competition=values.mean(0)
    return {'attraction_mass':positive,'competition_mass':competition,
            'logit_shift_derivative':competition-positive}


def aligned(path,cache):
    idx=np.flatnonzero(cache['label']==1)
    with np.load(path) as a:
        if not np.array_equal(a['event_ids'],cache['event_id'][idx]) or not np.array_equal(a['users'],cache['user_id'][idx]):
            raise ValueError('Positive events/users differ: '+str(path))
        return np.array(a['ranks'])


def quantiles(a):
    return np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else None


def inspect_data(counts,caches,train_end_ms):
    train=caches['standard/train'];exposure=np.bincount(train['item_index'],minlength=len(counts))
    positive=train['label']==1
    unique=np.unique(np.column_stack([train['item_index'][positive],train['user_index'][positive]]),axis=0)
    users=np.bincount(unique[:,0],minlength=len(counts))
    first=np.full(len(counts),np.iinfo(np.int64).max,dtype=np.int64)
    np.minimum.at(first,train['item_index'],train['time_ms'])
    masks=group_masks(np.arange(len(counts)),np.zeros((len(counts),1),dtype=int),counts)
    groups={}
    for name in ['head_target','tail_target','freq_zero','freq_1_5','freq_6_20','freq_21_100','freq_101_plus']:
        m=masks[name];n=int(m.sum());e=int(exposure[m].sum());p=int(counts[m].sum())
        groups[name]={'catalog_items':n,'training_exposures':e,'training_positive_events':p,
            'observed_train_click_rate':p/e if e else None,
            'exposures_per_item_quantiles':quantiles(exposure[m]),'positive_events_per_item_quantiles':quantiles(counts[m]),
            'distinct_positive_users_quantiles':quantiles(users[m]),
            'items_with_no_more_than_5_positive_users':int((users[m]<=5).sum()),
            'first_observed_in_last3_train_days':int((first[m]>=train_end_ms-3*86400000).sum())}
    partitions={}
    for part,c in caches.items():
        m=group_masks(c['item_index'],c['history'],counts);pos=c['label']==1
        partitions[part]={'rows':len(c),'positive_events':int(pos.sum()),'groups':{g:{
            'exposures':int(x.sum()),'positive_events':int((pos&x).sum()),
            'positive_event_fraction':float((pos&x).sum()/pos.sum()),
            'observed_click_rate':float(c['label'][x].mean()) if x.any() else None}
            for g,x in m.items() if g in ['head_target','tail_target','cold_target','freq_zero','freq_1_5']}}
    return {'catalog_support':groups,'partitions':partitions}


def audit(config,output):
    counts,caches=setup(config,['standard/train','standard/validation','standard/test','random/validation','random/test'])
    try:
        selection=load_selection(ROOT/config['output_directory'])
        frozen=read_json(ROOT/'reports/retrieval_v2_freeze.json')
        model,checkpoint=load_model(ROOT/frozen['checkpoint'],config['device'])
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        before=parameter_digest(model)
        manifest=read_json(ROOT/config['prepared_directory']/'manifest.json')
        data=inspect_data(counts,caches,manifest['signature']['protocol']['train_end_ms'])
        ids=torch.arange(1,len(counts),device=config['device'])
        with torch.no_grad():vectors=model.encode_item(ids)
        item_masks=group_masks(np.arange(len(counts)),np.zeros((len(counts),1),dtype=int),counts)
        head=item_masks['head_target'];tail_ids=ids[torch.tensor(~head[1:],device=ids.device)]
        train=caches['standard/train'];rng=np.random.default_rng(20261029)
        prior_rows=np.sort(rng.choice(len(train),4096,replace=False));sumq=np.zeros(vectors.shape[1],dtype=np.float64)
        for offset in range(0,len(prior_rows),512):
            b=tensor_batch(train,prior_rows[offset:offset+512],ids.device)
            with torch.no_grad():q=model.encode_user(b['users'],b['tabs'],b['history'])
            sumq+=q.cpu().numpy().astype(np.float64).sum(0)
        meanq=torch.tensor(sumq/len(prior_rows),device=ids.device,dtype=vectors.dtype)
        prior=(meanq@vectors.T/model.temperature).cpu().numpy()
        val=caches['standard/validation'];idx=np.flatnonzero(val['label']==1);targets=val['item_index'][idx]
        arrays={k:np.zeros(len(idx),dtype=np.int32) for k in ['full_ranks','tail_only_ranks','center05_ranks','center1_ranks','top100_head_slots']}
        arrays.update(event_ids=np.array(val['event_id'][idx]),users=np.array(val['user_id'][idx]),items=np.array(targets))
        p=torch.tensor(prior,device=ids.device)
        for offset in range(0,len(idx),512):
            rows=idx[offset:offset+512];b=tensor_batch(val,rows,ids.device)
            with torch.no_grad():scores=model.encode_user(b['users'],b['tabs'],b['history'])@vectors.T/model.temperature
            sl=slice(offset,offset+len(rows))
            arrays['full_ranks'][sl]=target_ranks(scores,ids,b['items']).cpu().numpy()
            arrays['tail_only_ranks'][sl]=target_ranks(scores[:,~torch.tensor(head[1:],device=ids.device)],tail_ids,b['items']).cpu().numpy()
            arrays['center05_ranks'][sl]=target_ranks(scores-.5*p[None,:],ids,b['items']).cpu().numpy()
            arrays['center1_ranks'][sl]=target_ranks(scores-p[None,:],ids,b['items']).cpu().numpy()
            top,_=stable_topk(scores,ids,100)
            arrays['top100_head_slots'][sl]=head[top.cpu().numpy()].sum(1)
        original_path=ROOT/'experiments/retrieval_v2/artifacts/ranks/two_tower_v2_seed42/standard_validation.npz'
        if not np.array_equal(arrays['full_ranks'],aligned(original_path,val)):raise ValueError('Frozen baseline rank replay differs')
        vmasks=group_masks(targets,val['history'][idx],counts);tm=vmasks['tail_target']
        competition={str(k):competition_breakdown(arrays['full_ranks'][tm],arrays['tail_only_ranks'][tm],k) for k in [20,100]}
        probes={name:summarize(arrays[key],arrays['users'],targets,val['history'][idx],counts)
                for name,key in [('original','full_ranks'),('center05','center05_ranks'),('center1','center1_ranks')]}
        slot_summary={g:{'queries':int(m.sum()),'mean_head_slots_in_top100':float(arrays['top100_head_slots'][m].mean())}
                      for g,m in vmasks.items() if m.any()}
        comparators={};pipeline={};ranking=read_json(ROOT/'experiments/ranking_v2/evaluation.json')
        for part,c in caches.items():
            if part=='standard/train':continue
            positive=np.flatnonzero(c['label']==1);masks=group_masks(c['item_index'][positive],c['history'][positive],counts)
            tower=aligned(ROOT/'experiments/retrieval_v2/artifacts/ranks/two_tower_v2_seed42'/(part.replace('/','_')+'.npz'),c)
            cf=aligned(ROOT/'experiments/retrieval_selected/artifacts/ranks/itemcf'/(part.replace('/','_')+'.npz'),c)
            din=aligned(ranking['results']['din_seed42'][part]['rank_artifact'],c)
            comparators[part]={};pipeline[part]={}
            for g in ['head_target','tail_target','freq_zero','freq_1_5','freq_6_20','freq_21_100','freq_101_plus']:
                m=masks[g]
                if not m.any():continue
                comparators[part][g]={'tower':summarize_ranks(tower[m],c['user_id'][positive][m],ks=(20,100)),
                    'itemcf':summarize_ranks(cf[m],c['user_id'][positive][m],ks=(20,100)),
                    'paired20':paired_outcomes(tower[m],cf[m],20),'paired100':paired_outcomes(tower[m],cf[m],100)}
                inpool=(tower[m]>0)&(tower[m]<=100);hit=(din[m]>0)&(din[m]<=20)
                if not np.array_equal(inpool,din[m]>0):raise ValueError('DIN candidate membership differs')
                pipeline[part][g]={'queries':int(m.sum()),'outside_pool':int((~inpool).sum()),
                    'pool_not_top20':int((inpool&~hit).sum()),'top20_hits':int(hit.sum())}
        # Exact score-shift derivative accounting, sampled train queries; not parameter-gradient causality.
        pos=np.flatnonzero(train['label']==1);grad_rows=np.sort(rng.choice(pos,8192,replace=False))
        mass={name:np.zeros(len(counts)-1) for name in ['attraction_mass','competition_mass','logit_shift_derivative']}
        for offset in range(0,len(grad_rows),512):
            b=tensor_batch(train,grad_rows[offset:offset+512],ids.device)
            with torch.no_grad():scores=model.encode_user(b['users'],b['tabs'],b['history'])@vectors.T/model.temperature
            values=choice_mass(scores.cpu().numpy(),b['items'].cpu().numpy())
            for name in mass:mass[name]+=values[name]/(len(grad_rows)/512)
        grouped_mass={g:{name:float(value[m[1:]].sum()) for name,value in mass.items()} for g,m in item_masks.items()
                      if g in ['head_target','tail_target','freq_zero','freq_1_5','freq_6_20','freq_21_100']}
        torch.manual_seed(42);initial=TwoTowerV2(**checkpoint['architecture'])
        old_training=read_json(ROOT/'experiments/retrieval_v2/artifacts/tuning/residual_recency/training.json')
        if parameter_digest(initial)!=old_training['initial_parameter_sha256']:raise ValueError('Initial embedding reconstruction differs')
        a=initial.item_embedding.weight.detach().numpy();b=model.item_embedding.weight.detach().cpu().numpy()
        cos=(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)).clip(1e-12)
        movement={g:{'initial_to_fitted_cosine_quantiles':quantiles(cos[m]),
                     'relative_parameter_change_quantiles':quantiles(np.linalg.norm(b[m]-a[m],axis=1)/np.linalg.norm(a[m],axis=1).clip(1e-12))}
                  for g,m in item_masks.items() if g in ['head_target','tail_target','freq_zero','freq_1_5']}
        from scipy.stats import spearmanr
        curves={n:{'best_epoch':r['best_epoch'],'rows':[{'epoch':x['epoch'],'loss':x['training_loss'],
                    'metrics':{k:x['validation'][k] for k in ['recall@20','ndcg@20','recall@100']},
                    'head':x['validation']['groups']['head_target'],'tail':x['validation']['groups']['tail_target']}
                    for x in r['curve']]} for n,r in selection['runs'].items()}
        result={'scope':'read-only root-cause diagnostics; no training, new selection, deployment or causal exposure claim',
            'data':data,'effective_training_weights':weighting_audit(counts),'tail_rank_competition':competition,
            'validation_fixed_score_probes':probes,'validation_top100_head_slots':slot_summary,
            'training_context_mean_score_spearman_with_log_click_count':float(spearmanr(prior,np.log1p(counts[1:])).statistic),
            'train_sample_score_shift_derivative_groups':grouped_mass,'train_derivative_sample_size':8192,
            'initial_to_fitted_embedding_movement':movement,'itemcf_comparison':comparators,
            'frozen_pipeline_failure_breakdown':pipeline,'existing_training_curves':curves,
            'model_parameters_unchanged':before==parameter_digest(model),
            'probe_design':{'mean_score_train_exposure_contexts':4096,'centering_alphas':[0,.5,1.],
                            'probe_partition':'standard/validation','tail_only_is_comparable_full_catalog_metric':False}}
        if not result['model_parameters_unchanged']:raise ValueError('Diagnostic modified model')
        output.mkdir();write_json(output/'analysis.json',result)
        np.savez_compressed(output/'validation_probes.npz',**arrays)
        np.savez_compressed(output/'training_probes.npz',prior_rows=prior_rows,derivative_rows=grad_rows,prior_scores=prior,
            **mass)
        print('ROOT AUDIT COMPLETE; tail competition '+str(competition),flush=True)
        return result
    finally:
        for c in caches.values():c.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-directory',default='experiments/retrieval_longtail_root_cause')
    args=p.parse_args();cfg,_=get_config();out=(ROOT/args.output_directory).resolve()
    if out.parent!=(ROOT/'experiments').resolve() or not out.name.startswith('retrieval_longtail_root_cause'):
        raise ValueError('Independent D: project root-cause directory required')
    selection=load_selection(ROOT/cfg['output_directory'])
    files=[Path(__file__),ROOT/'tests/test_longtail_root_cause.py',ROOT/'docs/longtail_root_cause_audit.md',
           ROOT/'reports/retrieval_longtail_delivery_validation.json',ROOT/'experiments/ranking_v2/evaluation.json']
    files += [Path(x) for x in selection['evidence']]
    signature={'config':cfg,'inputs':{p.relative_to(ROOT).as_posix():digest_file(p) for p in sorted(set(files))},
               'audit_seeds':20261029,'probe_alphas':[0,.5,1.],'model_updates':False}
    run_component(out/'audit',signature,lambda:audit(cfg,out/'audit'))
    validate_component(out/'audit',signature)


if __name__=='__main__':main()
