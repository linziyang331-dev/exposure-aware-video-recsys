"""Percentile resampling of complete units, with correct event/ratio denominators."""
import numpy as np


def _groups(n, clusters):
    if clusters is None:
        return np.arange(n), np.ones(n, dtype=np.int64), 'independent_logged_round'
    clusters = np.asarray(clusters)
    if clusters.shape != (n,) or (clusters.dtype.kind in 'fiu' and not np.isfinite(clusters).all()):
        raise ValueError('Aligned nonmissing cluster identifiers required')
    _, inverse = np.unique(clusters, return_inverse=True)
    return inverse, np.bincount(inverse), 'complete_cluster_event_weighted'


def _check_bootstrap(repeats, confidence):
    if not isinstance(repeats, (int, np.integer)) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError('Positive integer bootstrap repeats required')
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('Confidence must lie strictly inside (0,1)')


def bootstrap_ope(terms, clusters=None, repeats=500, seed=42, confidence=.95):
    _check_bootstrap(repeats, confidence)
    n = len(terms['ips'])
    inverse, counts, unit = _groups(n, clusters)
    names = [name for name in ['naive', 'ips', 'dm', 'dr'] if terms[name] is not None]
    totals = np.column_stack([np.bincount(inverse, weights=terms[name], minlength=len(counts)) for name in names]
                            + [np.bincount(inverse, weights=terms['weights'], minlength=len(counts))])
    draws = {name: np.empty(repeats) for name in names + ['snips']}
    rng = np.random.default_rng(seed)
    for b in range(repeats):
        sample = rng.integers(0, len(counts), size=len(counts))
        summed = totals[sample].sum(0)
        denom = counts[sample].sum()
        for j, name in enumerate(names): draws[name][b] = summed[j] / denom
        weight_sum = summed[-1]
        draws['snips'][b] = summed[names.index('ips')] / weight_sum if weight_sum else np.nan
    result = {}
    alpha = (1 - confidence) / 2
    for name, values in draws.items():
        valid = values[np.isfinite(values)]
        interval = np.quantile(valid, [alpha, 1 - alpha]) if len(valid) else [None, None]
        result[name] = {'low': float(interval[0]) if len(valid) else None,
                        'high': float(interval[1]) if len(valid) else None,
                        'valid_replicates': len(valid), 'undefined_replicates': repeats - len(valid)}
    return {'intervals': result, 'repeats': repeats, 'confidence': confidence,
            'resampling_unit': unit, 'units': len(counts), 'rows': n,
            'method': 'percentile; SNIPS numerator and denominator recomputed in every draw',
            'conditioning': 'supplied policies and reward predictions held fixed'}


def paired_cluster_means(first, first_users, second, second_users, repeats=500, seed=42, confidence=.95):
    """second minus first; same user-cluster draw jointly resamples both sources.

    Different source rows need not be paired events. Missing users contribute
    zero events in that source, and each source retains its own event denominator.
    """
    _check_bootstrap(repeats, confidence)
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    u1 = np.asarray(first_users); u2 = np.asarray(second_users)
    if first.ndim == 1: first = first[:, None]
    if second.ndim == 1: second = second[:, None]
    if (not len(first) or not len(second) or first.shape[1] != second.shape[1]
            or u1.shape != (len(first),) or u2.shape != (len(second),)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError('Finite aligned additive metrics and user IDs required')
    _, inverse = np.unique(np.concatenate([u1, u2]), return_inverse=True)
    i1, i2 = inverse[:len(u1)], inverse[len(u1):]
    n = inverse.max() + 1
    c1, c2 = np.bincount(i1, minlength=n), np.bincount(i2, minlength=n)
    g1 = np.column_stack([np.bincount(i1, weights=first[:, k], minlength=n) for k in range(first.shape[1])])
    g2 = np.column_stack([np.bincount(i2, weights=second[:, k], minlength=n) for k in range(second.shape[1])])
    rng = np.random.default_rng(seed); draws = []
    for _ in range(repeats):
        sample = rng.integers(0, n, size=n)
        d1, d2 = c1[sample].sum(), c2[sample].sum()
        if d1 and d2: draws.append(g2[sample].sum(0) / d2 - g1[sample].sum(0) / d1)
    alpha = (1 - confidence) / 2
    intervals = np.quantile(np.asarray(draws), [alpha, 1-alpha], axis=0) if draws else None
    diff = second.mean(0) - first.mean(0)
    return {'difference': diff.tolist(), 'low': intervals[0].tolist() if draws else None,
            'high': intervals[1].tolist() if draws else None, 'repeats': repeats,
            'valid_replicates': len(draws), 'users_in_union': int(n),
            'first_rows': len(first), 'second_rows': len(second), 'confidence': confidence,
            'unit': 'jointly resampled full user clusters; separate event denominators'}
