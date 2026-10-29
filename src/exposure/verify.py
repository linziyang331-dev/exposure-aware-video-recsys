"""Recompute reported quantities, beyond artifact existence/hash checks."""
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from src.data.common import ROOT
from src.exposure.ope import ope_terms,summarize_ope
from src.exposure.uncertainty import bootstrap_ope


def assert_result_equal(expected, actual, path='result'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(expected)!=set(actual):raise ValueError('Result keys differ at '+path)
        for key in expected:assert_result_equal(expected[key],actual[key],path+'.'+str(key))
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):raise ValueError('Result rows differ at '+path)
        for i,(a,b) in enumerate(zip(expected,actual)):assert_result_equal(a,b,path+f'[{i}]')
    elif isinstance(expected,float):
        if not isinstance(actual,(int,float)) or not np.isfinite(expected) or not np.isfinite(actual) or not np.isclose(expected,actual,rtol=1e-8,atol=1e-10):
            raise ValueError('Result numeric value differs at '+path)
    elif expected!=actual:
        raise ValueError('Result value differs at '+path)


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def verify_coverage_rows(cfg, truth, intervals, aggregates):
    """Validate the entire coverage grid and every reported summary field."""
    from src.exposure.synthetic import wilson_interval
    cases=['correct_pscore_oracle_q','correct_pscore_wrong_q','wrong_pscore_oracle_q','wrong_pscore_wrong_q']
    estimators=['naive','ips','snips','dm','dr']
    expected={(case,estimator,repeat) for case in cases for estimator in estimators
              for repeat in range(cfg['coverage_repeats'])}
    counts=defaultdict(lambda:[0,0,0]);seen=set()
    for row in intervals:
        repeat=int(row['repeat']);key=(row['case'],row['estimator']);identity=key+(repeat,)
        if identity not in expected or identity in seen:raise ValueError('Coverage condition/repeat grid differs')
        seen.add(identity)
        if int(row['seed'])!=cfg['world_seed']+900000+repeat:raise ValueError('Coverage seed differs')
        assert_result_equal(truth,float(row['truth']))
        if row['defined'] not in ['True','False'] or row['covered'] not in ['True','False']:
            raise ValueError('Invalid coverage boolean')
        defined=row['defined']=='True'
        if defined:
            low,high=float(row['low']),float(row['high'])
            if not np.isfinite([low,high]).all() or low>high:raise ValueError('Invalid coverage bounds')
            covered=low<=truth<=high
        else:
            if row['low']!='' or row['high']!='':raise ValueError('Undefined interval has bounds')
            covered=False
        if (row['covered']=='True')!=covered:raise ValueError('Coverage indicator differs from its interval')
        counts[key][0]+=int(covered);counts[key][1]+=int(defined);counts[key][2]+=1
    if seen!=expected:raise ValueError('Incomplete coverage condition/repeat grid')
    rebuilt={}
    for (case,estimator),(covered,valid,total) in counts.items():
        rebuilt[(case,estimator)]={'case':case,'estimator':estimator,'target':'uniform',
            'rows':cfg['coverage_rows'],'covered':covered,'valid_intervals':valid,'repeats':total,
            'coverage_over_all_repeats':covered/total,'coverage_given_defined':covered/valid if valid else None,
            'coverage_wilson95':wilson_interval(covered,total),'nominal_confidence':.95,
            'bootstrap_repeats':cfg['coverage_bootstrap_repeats']}
    seen_aggregates=set()
    for row in aggregates:
        key=(row['case'],row['estimator'])
        if key not in rebuilt or key in seen_aggregates:raise ValueError('Coverage aggregate groups differ')
        seen_aggregates.add(key);assert_result_equal(rebuilt[key],row)
    if seen_aggregates!=set(rebuilt):raise ValueError('Incomplete coverage aggregates')


def verify_synthetic(config, directory):
    directory=Path(directory);result=_read(directory/'synthetic_results.json');cfg=config['synthetic']
    from src.exposure.synthetic import policy_value,reward_features,_case_inputs,_terms,sample_log,wilson_interval
    with np.load(directory/'artifacts/world.npz') as z:
        world={key:np.array(z[key]) for key in ['context','reward_logits','expected_reward','logging_popularity']}
        world['targets']={'uniform':np.array(z['target_uniform']),'reward_softmax':np.array(z['target_reward_softmax'])}
    np.testing.assert_array_equal(world['expected_reward'],expit(world['reward_logits']))
    truth={name:policy_value(world,world['targets'][name]) for name in cfg['target_policies']}
    assert_result_equal(result['known_values'],truth)
    learned=np.load(directory/'artifacts/independent_reward_prediction.npy')
    with np.load(directory/'artifacts/representative_log.npz') as z:
        rep={name:np.array(z[name]) for name in z.files}
    for cap in cfg['clips']:
        terms=ope_terms(rep['action'],rep['reward'],rep['pscore'],rep['target_policy'],
            reward_prediction=rep['reward_prediction'],behavior_policy=rep['behavior_policy'],clip=cap)
        rebuilt=summarize_ope(terms)|{'bootstrap':bootstrap_ope(terms,repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'])}
        assert_result_equal(result['representative'][str(cap)],rebuilt)

    rows=_csv(directory/'synthetic_replicates.csv')
    expected_count=len(cfg['strengths'])*len(cfg['sample_sizes'])*cfg['repeats']*len(cfg['target_policies'])*5*len(cfg['clips'])
    if len(rows)!=expected_count:raise ValueError('Missing Monte Carlo result rows')
    grouped=defaultdict(list);unique=set()
    cases=['correct_pscore_oracle_q','correct_pscore_wrong_q','wrong_pscore_oracle_q',
           'wrong_pscore_wrong_q','correct_pscore_learned_q']
    expected={(strength,n,target,case,cap,repeat) for strength in cfg['strengths'] for n in cfg['sample_sizes']
              for target in cfg['target_policies'] for case in cases for cap in cfg['clips']
              for repeat in range(cfg['repeats'])}
    for row in rows:
        cap=None if row['cap']=='' else float(row['cap'])
        key=(float(row['strength']),int(row['rows']),row['target'],row['case'],cap)
        identity=key+(int(row['repeat']),)
        if identity not in expected or identity in unique:raise ValueError('Monte Carlo condition/repeat grid differs')
        unique.add(identity)
        expected_seed=(cfg['world_seed']+100000+cfg['strengths'].index(key[0])*10000
                       +cfg['sample_sizes'].index(key[1])*100+identity[-1])
        if int(row['seed'])!=expected_seed:raise ValueError('Monte Carlo seed differs')
        if float(row['truth'])!=truth[row['target']]:raise ValueError('Monte Carlo truth changed')
        for estimator in ['naive','ips','snips','dm','dr']:
            if row[estimator]!='':grouped[key+(estimator,)].append(float(row[estimator]))
    if unique!=expected:raise ValueError('Incomplete Monte Carlo grid')
    rebuilt={}
    for key,values in grouped.items():
        strength,n,policy,case,cap,estimator=key;v=np.asarray(values);error=v-truth[policy]
        rebuilt[key]={'strength':strength,'rows':n,'target':policy,'case':case,'cap':cap,'estimator':estimator,
                      'truth':truth[policy],'repeats':len(v),'mean':float(v.mean()),'bias':float(error.mean()),
                      'rmse':float(np.sqrt((error**2).mean())),
                      'sample_std':float(v.std(ddof=1)) if len(v)>1 else None}
    if len(rebuilt)!=len(result['metrics']):raise ValueError('Monte Carlo aggregate groups differ')
    for row in result['metrics']:
        key=tuple(row[k] for k in ['strength','rows','target','case','cap','estimator'])
        assert_result_equal(row,rebuilt[key])

    coverage=_csv(directory/'synthetic_coverage_intervals.csv')
    verify_coverage_rows(cfg,truth['uniform'],coverage,result['coverage'])
    # Independently replay one full coverage log and all four misspecification cases.
    seed=cfg['world_seed']+900000
    log=sample_log(world,max(cfg['strengths']),cfg['coverage_rows'],seed,cfg['exploration'])
    ids=log['context_ids'];target=world['targets']['uniform'][ids]
    for case,(mu,prediction) in _case_inputs(log,world['expected_reward'][ids],learned[ids]).items():
        if case=='correct_pscore_learned_q':continue
        intervals=bootstrap_ope(_terms(log,target,mu,prediction,None),
            repeats=cfg['coverage_bootstrap_repeats'],seed=seed+1000)['intervals']
        saved={row['estimator']:row for row in coverage if row['case']==case and row['repeat']=='0'}
        for estimator,interval in intervals.items():
            assert_result_equal(interval['low'],float(saved[estimator]['low']))
            assert_result_equal(interval['high'],float(saved[estimator]['high']))

    n,a=world['expected_reward'].shape
    features=reward_features(world,np.repeat(np.arange(n),a),np.tile(np.arange(a),n))
    q=world['expected_reward'].ravel()
    for run in result['weighted_learning']:
        for model in run['models'].values():
            z=features@np.asarray(model['coefficients'])+model['intercept']
            risk=float((np.logaddexp(0,z)-q*z).mean())
            assert_result_equal(model['true_uniform_logloss'],risk)
            chosen=z.reshape(n,a).argmax(1)
            value=float(world['expected_reward'][np.arange(n),chosen].mean())
            assert_result_equal(model['true_greedy_policy_value'],value)
    return {'status':'PASS','monte_carlo_rows_checked':len(rows),'aggregate_rows_recomputed':len(rebuilt),
            'representative_estimators_and_bootstrap_recomputed':True,
            'coverage_intervals_checked':len(coverage),'coverage_logs_replayed':1,
            'weighted_learning_true_risk_recomputed':True}


def verify_real(config, directory):
    from src.exposure.real import run_real
    original=_read(Path(directory)/'real_results.json')
    # Regenerate all cohorts, calibrators, metrics and user-cluster intervals.
    temporary_root=ROOT/'.runtime/tmp';temporary_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='exposure_verify_',dir=temporary_root) as temp:
        actual=run_real(config,Path(temp))
        assert_result_equal(original,actual)
    if original['real_policy_value_reported'] or original['real_propensity_fabricated']:
        raise ValueError('Real-log identifiability contract was violated')
    return {'status':'PASS','all_real_cohorts_calibration_and_uncertainty_recomputed':True,
            'numeric_tolerance':{'relative':1e-8,'absolute':1e-10},
            'original_ranking_and_prediction_inputs_verified':True,'no_real_propensity_fabricated':True}
