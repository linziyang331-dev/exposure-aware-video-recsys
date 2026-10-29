"""A finite, known world for identifiable OPE; no probabilities from real logs."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression

from src.exposure.ope import ope_terms, point_estimates, summarize_ope
from src.exposure.uncertainty import bootstrap_ope


def make_world(config, seed=None):
    n, a, d = config['n_contexts'], config['n_actions'], config['context_dim']
    if n < 2 or a < 2 or d < 2:
        raise ValueError('At least two contexts, actions and context dimensions required')
    rng = np.random.default_rng(config['world_seed'] if seed is None else seed)
    context = rng.normal(size=(n, d))
    beta = rng.normal(0, .55, size=(d, a))
    action_effect = np.linspace(-.6, .6, a)
    nonlinear = (.65 * np.sin(1.5 * context[:, :1]) * np.cos(np.arange(a))[None, :]
                 + .4 * (context[:, :1] * context[:, 1:2]) * np.sin(np.arange(a))[None, :])
    logits = -.7 + context @ beta + action_effect + nonlinear
    return {'context': context, 'reward_logits': logits, 'expected_reward': expit(logits),
            'logging_popularity': np.linspace(1.5, -1.5, a),
            'targets': {'uniform': np.full((n, a), 1 / a),
                        'reward_softmax': softmax(.7 * logits, axis=1)}}


def policy_value(world, policy):
    policy = np.asarray(policy)
    if policy.shape != world['expected_reward'].shape or not np.allclose(policy.sum(1), 1):
        raise ValueError('Policy must align with the finite population')
    return float((policy * world['expected_reward']).sum(1).mean())


def sample_log(world, strength, rows, seed, exploration=.05):
    if not isinstance(rows, (int, np.integer)) or rows < 1 or not 0 <= exploration <= 1 or strength < 0:
        raise ValueError('Positive row count and valid exploration/logging strength required')
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(world['context']), size=rows)
    score = strength * (.6 * world['reward_logits'][ids] + world['logging_popularity'][None, :])
    a = score.shape[1]
    mu = (1 - exploration) * softmax(score, axis=1) + exploration / a
    action = (mu.cumsum(1) < rng.random((rows, 1))).sum(1).clip(max=a-1)
    reward = rng.binomial(1, world['expected_reward'][ids, action]).astype(np.float64)
    return {'context_ids': ids, 'action': action, 'reward': reward,
            'pscore': mu[np.arange(rows), action], 'behavior_policy': mu}


def reward_features(world, context_ids, actions):
    a = world['expected_reward'].shape[1]
    return np.concatenate([world['context'][context_ids], np.eye(a)[actions]], axis=1)


def fit_reward_prediction(world, training_log):
    """Fit on separate logged outcomes; never regress on oracle reward tables."""
    features = reward_features(world, training_log['context_ids'], training_log['action'])
    if len(np.unique(training_log['reward'])) != 2:
        raise ValueError('Reward training log needs both binary outcomes')
    model = LogisticRegression(C=1., max_iter=300, random_state=0)
    model.fit(features, training_log['reward'])
    n, a = world['expected_reward'].shape
    features = reward_features(world, np.repeat(np.arange(n), a), np.tile(np.arange(a), n))
    return model.predict_proba(features)[:, 1].reshape(n, a)


def _case_inputs(log, oracle, learned):
    mu = log['behavior_policy']
    wrong = .5 * mu + .5 / mu.shape[1]
    constant = np.full_like(oracle, .1)
    return {'correct_pscore_oracle_q': (mu, oracle),
            'correct_pscore_wrong_q': (mu, constant),
            'wrong_pscore_oracle_q': (wrong, oracle),
            'wrong_pscore_wrong_q': (wrong, constant),
            'correct_pscore_learned_q': (mu, learned)}


def _terms(log, target, mu, prediction, clip):
    action = log['action']
    return ope_terms(action, log['reward'], mu[np.arange(len(action)), action], target,
                     reward_prediction=prediction, behavior_policy=mu, clip=clip)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise ValueError('Cannot export an empty result table')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def wilson_interval(successes, total):
    if total < 1: return [None, None]
    z = 1.959963984540054; p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [float(center-half), float(center+half)]


def run_synthetic(config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cfg = config['synthetic']; world = make_world(cfg)
    artifact = output / 'artifacts'; artifact.mkdir(exist_ok=True)
    np.savez_compressed(artifact/'world.npz', **{k: world[k] for k in ['context','reward_logits','expected_reward','logging_popularity']},
                        target_uniform=world['targets']['uniform'], target_reward_softmax=world['targets']['reward_softmax'])
    reward_train_seed = cfg['world_seed'] + 501
    train = sample_log(world, 1.5, cfg['reward_training_rows'], reward_train_seed, cfg['exploration'])
    learned = fit_reward_prediction(world, train)
    np.save(artifact/'independent_reward_prediction.npy', learned)
    truth = {name: policy_value(world, world['targets'][name]) for name in cfg['target_policies']}
    raw, grouped = [], defaultdict(list)
    for si, strength in enumerate(cfg['strengths']):
        for ni, n in enumerate(cfg['sample_sizes']):
            for repeat in range(cfg['repeats']):
                seed = cfg['world_seed'] + 100000 + si*10000 + ni*100 + repeat
                log = sample_log(world, strength, n, seed, cfg['exploration'])
                ids = log['context_ids']; oracle = world['expected_reward'][ids]
                cases = _case_inputs(log, oracle, learned[ids])
                for policy in cfg['target_policies']:
                    target = world['targets'][policy][ids]
                    for case, (mu, prediction) in cases.items():
                        for cap in cfg['clips']:
                            t = _terms(log, target, mu, prediction, cap)
                            values = point_estimates(t)
                            row = {'strength': strength, 'rows': n, 'repeat': repeat, 'seed': seed,
                                   'target': policy, 'case': case, 'cap': cap, 'truth': truth[policy], **values}
                            raw.append(row)
                            for estimator, value in values.items():
                                if value is not None:
                                    grouped[(strength,n,policy,case,cap,estimator)].append(value)
            print(f'synthetic strength={strength} rows={n} repeats={cfg["repeats"]} complete', flush=True)
    aggregates = []
    for (strength,n,policy,case,cap,estimator), values in grouped.items():
        values = np.asarray(values); error = values - truth[policy]
        aggregates.append({'strength':strength,'rows':n,'target':policy,'case':case,'cap':cap,
                           'estimator':estimator,'truth':truth[policy],'repeats':len(values),
                           'mean':float(values.mean()),'bias':float(error.mean()),
                           'rmse':float(np.sqrt((error**2).mean())),
                           'sample_std':float(values.std(ddof=1)) if len(values)>1 else None})
    write_csv(output/'synthetic_replicates.csv', raw)
    write_csv(output/'synthetic_metrics.csv', aggregates)

    # Coverage is an empirical experiment, with its own Monte Carlo uncertainty.
    coverage, coverage_counts, coverage_raw = [], defaultdict(lambda: [0,0,0]), []
    for repeat in range(cfg['coverage_repeats']):
        seed = cfg['world_seed'] + 900000 + repeat
        log = sample_log(world, max(cfg['strengths']), cfg['coverage_rows'], seed, cfg['exploration'])
        ids = log['context_ids']; target = world['targets']['uniform'][ids]
        for case, (mu, prediction) in _case_inputs(log, world['expected_reward'][ids], learned[ids]).items():
            if case == 'correct_pscore_learned_q': continue
            t = _terms(log, target, mu, prediction, None)
            intervals = bootstrap_ope(t, repeats=cfg['coverage_bootstrap_repeats'], seed=seed+1000)
            for estimator, interval in intervals['intervals'].items():
                low, high = interval['low'], interval['high']; key = (case, estimator)
                coverage_counts[key][2] += 1
                if low is not None:
                    coverage_counts[key][1] += 1
                    coverage_counts[key][0] += int(low <= truth['uniform'] <= high)
                coverage_raw.append({'repeat':repeat,'seed':seed,'case':case,'estimator':estimator,
                    'truth':truth['uniform'],'low':low,'high':high,
                    'defined':low is not None,'covered':bool(low is not None and low<=truth['uniform']<=high)})
    for (case, estimator), (covered, valid, total) in coverage_counts.items():
        coverage.append({'case':case,'estimator':estimator,'target':'uniform',
                         'rows':cfg['coverage_rows'],'covered':covered,'valid_intervals':valid,'repeats':total,
                         'coverage_over_all_repeats':covered/total,
                         'coverage_given_defined':covered/valid if valid else None,
                         'coverage_wilson95':wilson_interval(covered,total),
                         'nominal_confidence':.95,'bootstrap_repeats':cfg['coverage_bootstrap_repeats']})
    write_csv(output/'synthetic_coverage.csv', [{**r,'coverage_wilson95':json.dumps(r['coverage_wilson95'])} for r in coverage])
    write_csv(output/'synthetic_coverage_intervals.csv',coverage_raw)
    representative = sample_log(world, max(cfg['strengths']), max(cfg['sample_sizes']), cfg['world_seed']+999999, cfg['exploration'])
    ids = representative['context_ids']; target = world['targets']['uniform'][ids]
    representative_results = {}
    for cap in cfg['clips']:
        t = _terms(representative, target, representative['behavior_policy'], learned[ids], cap)
        representative_results[str(cap)] = summarize_ope(t) | {
            'bootstrap': bootstrap_ope(t, repeats=config['bootstrap_repeats'], seed=config['bootstrap_seed'])}
    np.savez_compressed(artifact/'representative_log.npz', **representative,
                        target_policy=target, reward_prediction=learned[ids])

    # A demonstrably unsupported target is rejected rather than assigned a value.
    unsupported_rejected = False
    try:
        ope_terms(np.array([0]), np.array([1.]), np.array([1.]), np.full((1,2),.5),
                  behavior_policy=np.array([[1.,0.]]))
    except ValueError: unsupported_rejected = True
    if not unsupported_rejected: raise AssertionError('Unsupported target was accepted')
    from src.exposure.learning import fit_weighted_logistic
    learning = [fit_weighted_logistic(world, cfg['learning_rows'], seed,
                max(cfg['strengths']), cfg['exploration']) for seed in cfg['learning_seeds']]
    result = {'status':'SIMULATED', 'estimand':'single-action policy reward on a known finite population',
              'config':cfg,'known_values':truth,'reward_model_training_seed':reward_train_seed,
              'reward_model_training_rows':cfg['reward_training_rows'],
              'reward_model_independent_of_evaluation_logs':True,
              'reward_model_scope':'linear context plus action indicators; intentionally lacks simulator nonlinear interactions',
              'metrics':aggregates,'coverage':coverage,'representative':representative_results,
              'unsupported_target_rejected':unsupported_rejected,'weighted_learning':learning,
              'limits':['Not real KuaiRand policy value','Not a Top20 slate or long-term RL value',
                        'Reward-model fit held fixed across Monte Carlo logs; coverage is conditional on it',
                        'Capped and misspecified estimators are not guaranteed unbiased']}
    (output/'synthetic_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result
