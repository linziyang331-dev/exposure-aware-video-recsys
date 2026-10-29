"""Independent NumPy implementations of IPS, SNIPS, DM and DR.

Single-action contextual bandits; not full-slate or sequential-policy value.
References are recorded in the module design. No upstream code is vendored.
"""
import numpy as np


def _probability_matrix(value, n, name):
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim != 2 or array.shape[0] != n or array.shape[1] < 1
            or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any()
            or not np.allclose(array.sum(1), 1., rtol=0, atol=1e-10)):
        raise ValueError(name + ' must be finite, normalized (rounds, actions) probabilities')
    return array


def ope_terms(action, reward, pscore, target_policy, reward_prediction=None,
              behavior_policy=None, clip=None):
    action = np.asarray(action)
    reward = np.asarray(reward, dtype=np.float64)
    pscore = np.asarray(pscore, dtype=np.float64)
    if action.ndim != 1 or action.dtype.kind not in 'iu' or not len(action):
        raise ValueError('Nonempty integer action vector required')
    n = len(action)
    if reward.shape != (n,) or not np.isfinite(reward).all() or ((reward < 0) | (reward > 1)).any():
        raise ValueError('Aligned finite rewards in [0,1] required')
    if pscore.shape != (n,) or not np.isfinite(pscore).all() or ((pscore <= 0) | (pscore > 1)).any():
        raise ValueError('Aligned positive logged-action probabilities in (0,1] required')
    target = _probability_matrix(target_policy, n, 'Target policy')
    if ((action < 0) | (action >= target.shape[1])).any():
        raise ValueError('Action outside target action space')
    row = np.arange(n)
    support = 'NOT_VERIFIABLE_FROM_LOGGED_PSCORE_ONLY'
    if behavior_policy is not None:
        behavior = _probability_matrix(behavior_policy, n, 'Behavior policy')
        if behavior.shape != target.shape:
            raise ValueError('Behavior and target action spaces differ')
        if not np.allclose(behavior[row, action], pscore, rtol=1e-10, atol=0):
            raise ValueError('Logged pscore disagrees with supplied behavior policy')
        if ((behavior == 0) & (target > 0)).any():
            raise ValueError('Target policy has mass outside behavior support; value is not identifiable')
        support = 'VERIFIED_GIVEN_SUPPLIED_BEHAVIOR_DISTRIBUTION'
    if clip is not None and (not np.isscalar(clip) or not np.isfinite(clip) or clip <= 0):
        raise ValueError('Weight cap must be finite and positive, or None')
    weights = target[row, action] / pscore
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite importance weights')
    used = weights.copy() if clip is None else np.minimum(weights, float(clip))
    dm = dr = None
    if reward_prediction is not None:
        prediction = np.asarray(reward_prediction, dtype=np.float64)
        if (prediction.shape != target.shape or not np.isfinite(prediction).all()
                or ((prediction < 0) | (prediction > 1)).any()):
            raise ValueError('Finite reward probabilities matching target action space required')
        dm = (target * prediction).sum(1)
        dr = dm + used * (reward - prediction[row, action])
    return {'naive': reward.copy(), 'weights': used, 'raw_weights': weights,
            'ips': used * reward, 'dm': dm, 'dr': dr, 'clip': clip,
            'support_status': support}


def weight_diagnostics(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Nonnegative finite weight vector required')
    with np.errstate(over='ignore'):
        total = float(weights.sum())
    if not np.isfinite(total):raise ValueError('Weight total exceeds floating-point range')
    maximum = float(weights.max())
    scaled = weights/maximum if maximum else weights
    square_sum = float(scaled @ scaled)
    scaled_total = float(scaled.sum())
    ess = scaled_total * scaled_total / square_sum if square_sum else 0.
    top_count = max(1, int(np.ceil(.01 * len(weights))))
    largest = np.partition(weights, len(weights) - top_count)[-top_count:]
    return {'sum': total, 'mean': total/len(weights),
            'ess': ess, 'ess_fraction': ess/len(weights),
            'zero_fraction': float((weights == 0).mean()),
            'top1_percent_weight_share': float(largest.sum() / total) if total else None,
            'quantiles': {str(q): float(np.quantile(weights, q)) for q in [0, .5, .9, .95, .99, .999, 1]}}


def point_estimates(terms):
    denominator = float(terms['weights'].sum())
    estimates = {name: float(terms[name].mean()) if terms[name] is not None else None
                 for name in ['naive', 'ips', 'dm', 'dr']}
    estimates['snips'] = float(terms['ips'].sum() / denominator) if denominator else None
    return estimates


def summarize_ope(terms):
    denominator = float(terms['weights'].sum())
    estimates = point_estimates(terms)
    return {'rounds': len(terms['ips']), 'estimates': estimates,
            'snips_status': 'DEFINED' if denominator else 'UNDEFINED_ZERO_WEIGHT',
            'support_status': terms['support_status'],
            'weight_diagnostics': {'cap': terms['clip'],
                'clipped_fraction': float((terms['weights'] < terms['raw_weights']).mean()),
                'raw': weight_diagnostics(terms['raw_weights']),
                'used': weight_diagnostics(terms['weights'])},
            'estimand': 'single-action expected reward under supplied target policy',
            'clipping_note': 'Clipping may introduce bias and invalidate the usual double-robust guarantee' if terms['clip'] is not None else None}
