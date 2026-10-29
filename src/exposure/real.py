"""Observed source-shift and calibration diagnostics; never real-log OPE."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.data import load_partition, training_counts
from src.retrieval.provenance import read_verified_vocabulary
from src.retrieval.metrics import summarize_ranks
from src.retrieval_v2.provenance import verify_evidence
from src.ranking.provenance import verify_v2
from src.exposure.metrics import binary_metrics,binary_terms,fit_platt,apply_platt,shared_stratum_weights,js_divergence_counts
from src.exposure.uncertainty import paired_cluster_means
from src.exposure.synthetic import write_csv


def check_temporal_fit(fit_times, evaluation_times):
    fit=np.asarray(fit_times);evaluation=np.asarray(evaluation_times)
    if not len(fit) or not len(evaluation) or not np.isfinite(fit).all() or not np.isfinite(evaluation).all() or fit.max()>=evaluation.min():
        raise ValueError('Calibration fit timestamps must be strictly earlier than every evaluation row')


def _rank_metrics(ranks, users):
    if not len(ranks):return {'queries':0,'status':'EMPTY_COHORT'}
    result=summarize_ranks(ranks,users,ks=(20,100))
    result['mrr@100']=result.pop('mrr');result['user_macro']['mrr@100']=result['user_macro'].pop('mrr')
    return result


def _rank_terms(ranks):
    hit=(ranks>0)&(ranks<=20);gain=np.zeros(len(ranks));gain[hit]=1/np.log2(ranks[hit]+1)
    return np.column_stack([hit.astype(float),gain])


def _flat_binary(metrics):
    return {k:metrics.get(k) for k in ['rows','positives','positive_rate','auc','logloss','brier','ece']}


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def run_real(config, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    selected_path=ROOT/'experiments/ranking_v2/selection.json'
    selected=_read(selected_path);verify_evidence(ROOT,selected['evidence'],selected['config']);verify_v2()
    verification_path=ROOT/'reports/ranking_v2_artifact_validation.json';verification=_read(verification_path)
    if verification['status']!='PASS':raise ValueError('Frozen DIN artifacts did not pass original verification')
    evaluation_path=ROOT/'experiments/ranking_v2/evaluation.json'
    if digest_file(evaluation_path)!=verification['artifact_hashes'].get(evaluation_path.relative_to(ROOT).as_posix()):
        raise ValueError('Original evaluation artifact changed')
    previous=_read(evaluation_path)
    prepared=ROOT/config['prepared_directory'];cache_root=ROOT/config['cache_directory']
    vocabulary=read_verified_vocabulary(prepared);counts=training_counts(vocabulary)
    manifest=_read(prepared/'manifest.json')
    paths=[selected_path,verification_path,evaluation_path,prepared/'manifest.json',prepared/'vocabulary.json']
    caches={};logits={};ranks={};positive_rows={};schemas={}
    try:
        for part in ['standard/train','standard/validation','random/validation','standard/test','random/test']:
            cache=load_partition(prepared,part,cache_root,50);caches[part]=cache
            paths.extend([cache.root/'metadata.json',prepared/manifest['partitions'][part]['path']])
            if part=='standard/train':continue
            pred_path=Path(previous['results']['din_seed42'][part]['prediction_artifact'])
            rank_paths={'v2':Path(previous['baseline'][part]['artifact']),
                        'din':Path(previous['results']['din_seed42'][part]['rank_artifact'])}
            for path in [pred_path,*rank_paths.values()]:
                expected=verification['artifact_hashes'].get(path.relative_to(ROOT).as_posix())
                if digest_file(path)!=expected:raise ValueError('Frozen prediction/rank artifact changed: '+str(path))
                paths.append(path)
            with np.load(pred_path) as p:
                np.testing.assert_array_equal(p['event_ids'],cache['event_id']);np.testing.assert_array_equal(p['labels'],cache['label'])
                logits[part]=np.array(p['logits'],dtype=np.float64)
            idx=np.flatnonzero(cache['label']==1);positive_rows[part]=idx;ranks[part]={}
            for name,path in rank_paths.items():
                with np.load(path) as r:
                    np.testing.assert_array_equal(r['event_ids'],cache['event_id'][idx]);np.testing.assert_array_equal(r['users'],cache['user_id'][idx])
                    rank=np.array(r['ranks'])
                    if ((rank<0)|(rank>100)).any():raise ValueError('Invalid frozen candidate rank')
                    ranks[part][name]=rank
            np.testing.assert_array_equal(ranks[part]['v2']>0,ranks[part]['din']>0)
            schemas[part]=pq.read_schema(prepared/manifest['partitions'][part]['path']).names

        fits={}
        for name,part in [('platt_standard_validation','standard/validation'),('platt_random_validation','random/validation')]:
            fits[name]=fit_platt(logits[part],caches[part]['label'])|{'fit_partition':part,
                'fit_time_min_ms':int(caches[part]['time_ms'].min()),'fit_time_max_ms':int(caches[part]['time_ms'].max())}
            for test in ['standard/test','random/test']:check_temporal_fit(caches[part]['time_ms'],caches[test]['time_ms'])

        first,second=caches['standard/test'],caches['random/test']
        common_users=np.intersect1d(first['user_id'][first['user_index']>0],second['user_id'][second['user_index']>0])
        common_items=np.intersect1d(first['item_index'][first['item_index']>0],second['item_index'][second['item_index']>0])
        train=caches['standard/train'];user_positive=np.bincount(train['user_index'],weights=train['label'],minlength=len(vocabulary['user_raw_to_index'])+1)
        head=np.zeros(len(counts),dtype=bool);order=np.argsort(-counts[1:],kind='stable')+1
        head[order[:int(np.ceil(.2*(len(counts)-1)))]]=True
        cohorts={};groups={};cohort_table=[];calibration_table=[];calibration_results={};all_rank_metrics={}
        partitions={p:{'rows':len(c),'positives':int(c['label'].sum()),'users':len(np.unique(c['user_id'])),
                       'min_time_ms':int(c['time_ms'].min()),'max_time_ms':int(c['time_ms'].max())} for p,c in caches.items()}
        for part in ['standard/test','random/test']:
            c=caches[part];known_user=c['user_index']>0;known_item=c['item_index']>0
            common=np.isin(c['user_id'],common_users)&np.isin(c['item_index'],common_items)
            history_amount=user_positive[c['user_index']]
            masks={'all':np.ones(len(c),dtype=bool),'tab_1':c['tab']==1,
                   'common_warm_user_item_tab1':common&(c['tab']==1),
                   'warm_user_and_item':known_user&known_item,'cold_user':~known_user,'cold_item':~known_item,
                   'head_item':known_item&head[c['item_index']],'tail_item':known_item&~head[c['item_index']],
                   'known_user_zero_positive':known_user&(history_amount==0),
                   'user_positive_1_5':known_user&(history_amount>=1)&(history_amount<=5),
                   'user_positive_6_20':known_user&(history_amount>=6)&(history_amount<=20),
                   'user_positive_gt20':known_user&(history_amount>20)}
            masks.update({'tab_'+str(int(t)):c['tab']==t for t in np.unique(c['tab'])})
            cohorts[part]=masks;groups[part]={};all_rank_metrics[part]={}
            idx=positive_rows[part]
            for name,mask in masks.items():
                if not mask.any():
                    groups[part][name]={'exposure':{'rows':0,'status':'EMPTY_COHORT'},'ranking':{}};continue
                exposure=binary_metrics(c['label'][mask],logits[part][mask],bins=config['calibration_bins'])
                query_mask=mask[idx];ranking={model:_rank_metrics(r[query_mask],c['user_id'][idx][query_mask]) for model,r in ranks[part].items()}
                groups[part][name]={'exposure':exposure,'ranking':ranking,
                                    'fraction_of_source_exposures':float(mask.mean())}
                for model,rm in ranking.items():
                    cohort_table.append({'partition':part,'cohort':name,'model':model,**_flat_binary(exposure),
                        'positive_queries':rm['queries'],'recall@20':rm.get('recall@20'),'ndcg@20':rm.get('ndcg@20'),
                        'recall@100':rm.get('recall@100')})
            all_rank_metrics[part]=groups[part]['all']['ranking']
            calibration_results[part]={}
            for name,fit in [('uncalibrated',None),*fits.items()]:
                adjusted=logits[part] if fit is None else apply_platt(logits[part],fit)
                calibration_results[part][name]={}
                for cohort in ['all','tab_1','common_warm_user_item_tab1']:
                    mask=masks[cohort]
                    if not mask.any():continue
                    m=binary_metrics(c['label'][mask],adjusted[mask],bins=config['calibration_bins'])
                    calibration_results[part][name][cohort]=m
                    for cell in m['calibration']:calibration_table.append({'partition':part,'method':name,'cohort':cohort,**cell})
            print('real diagnostic '+part+' exposures='+str(len(c))+' positive_queries='+str(len(idx)),flush=True)

        # Balance the shared USER/TAB distribution; do not claim action/time matching.
        common_masks={p:np.isin(caches[p]['user_id'],common_users)&np.isin(caches[p]['item_index'],common_items) for p in ['standard/test','random/test']}
        p1,p2='standard/test','random/test';m1,m2=common_masks[p1],common_masks[p2]
        w1,w2,strata=shared_stratum_weights(caches[p1]['user_id'][m1],caches[p1]['tab'][m1],caches[p2]['user_id'][m2],caches[p2]['tab'][m2])
        standardized={}
        for part,mask,w in [(p1,m1,w1),(p2,m2,w2)]:
            use=w>0
            standardized[part]=binary_metrics(caches[part]['label'][mask][use],logits[part][mask][use],weights=w[use],bins=config['calibration_bins'])

        uncertainty={'source_gap':{},'calibration_delta':{}}
        for cohort in ['all','common_warm_user_item_tab1']:
            a,b=cohorts[p1][cohort],cohorts[p2][cohort]
            if not a.any() or not b.any():continue
            uncertainty['source_gap'][cohort]={'direction':'random minus standard; observational source contrast',
                'exposure_metrics':['positive_rate','logloss','brier'],
                'exposure':paired_cluster_means(binary_terms(caches[p1]['label'][a],logits[p1][a]),caches[p1]['user_id'][a],
                    binary_terms(caches[p2]['label'][b],logits[p2][b]),caches[p2]['user_id'][b],
                    repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'])}
            ia,ib=positive_rows[p1],positive_rows[p2];qa,qb=a[ia],b[ib]
            uncertainty['source_gap'][cohort]['ranking_metrics']=['recall@20','ndcg@20']
            uncertainty['source_gap'][cohort]['ranking']={}
            for model in ['v2','din']:
                if qa.any() and qb.any():
                    uncertainty['source_gap'][cohort]['ranking'][model]=paired_cluster_means(
                        _rank_terms(ranks[p1][model][qa]),caches[p1]['user_id'][ia][qa],
                        _rank_terms(ranks[p2][model][qb]),caches[p2]['user_id'][ib][qb],
                        repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'])
        for part in [p1,p2]:
            c=caches[part];base=binary_terms(c['label'],logits[part])[:,1:]
            uncertainty['calibration_delta'][part]={}
            for name,fit in fits.items():
                after=binary_terms(c['label'],apply_platt(logits[part],fit))[:,1:]
                uncertainty['calibration_delta'][part][name]=paired_cluster_means(base,c['user_id'],after,c['user_id'],
                    repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'])

        distributions={}
        for name,field,size in [('tab','tab',15),('item_exposure','item_index',len(counts))]:
            a=np.bincount(first[field],minlength=size);b=np.bincount(second[field],minlength=size)
            distributions[name]={'js_divergence_bits':js_divergence_counts(a,b),'standard_counts':a.tolist(),'random_counts':b.tolist()}
        day_tables={}
        for part,c in caches.items():
            day=(np.asarray(c['time_ms'])+8*3600000)//86400000
            days,inv=np.unique(day,return_inverse=True);n=np.bincount(inv);pos=np.bincount(inv,weights=c['label'])
            day_tables[part]=[{'beijing_epoch_day':int(d),'rows':int(count),'positive_rate':float(p/count)} for d,count,p in zip(days,n,pos)]
        result={'status':'DIAGNOSED','real_ope_status':'NOT_IDENTIFIABLE_FROM_AVAILABLE_METADATA',
                'real_propensity_fabricated':False,'real_policy_value_reported':False,
                'scope':'observational source-shift and probability calibration; frozen original Top20 rankings',
                'partitions':partitions,'common_cohort':{'warm_users':len(common_users),'known_items':len(common_items),
                    'definition':'shared warm raw users and shared known item IDs; tab1 subset is separate',
                    'limits':'Cohort membership uses evaluation identifiers, not labels; no matching of requests or action frequencies'},
                'groups':groups,'frozen_top20':all_rank_metrics,'calibration_fits':fits,
                'calibration_results':calibration_results,'standardization':{'definition':strata,'metrics':standardized},
                'uncertainty':uncertainty,'distribution_diagnostics':distributions,'daily_label_rates':day_tables,
                'schema_propensity_fields':{p:[f for f in fields if f in {'pscore','propensity','action_probability','behavior_probability'}] for p,fields in schemas.items()},
                'test_is_iterative_reassessment':True,
                'calibration_protocol':'both calibrators predefined; fit only earlier validation; no test selection; random-validation fit uses early random feedback',
                'ranking_changed_by_calibration':False,
                'uncertainty_scope':'conditional on fixed model/calibrators; user-cluster resampling; subgroup analyses descriptive, no multiplicity-adjusted claims',
                'input_file_hashes':{p.resolve().relative_to(ROOT).as_posix():digest_file(p) for p in sorted(set(paths))}}
        write_csv(output/'real_cohorts.csv',cohort_table);write_csv(output/'real_calibration_bins.csv',calibration_table)
        (output/'real_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        return result
    finally:
        for cache in caches.values():cache.close()
