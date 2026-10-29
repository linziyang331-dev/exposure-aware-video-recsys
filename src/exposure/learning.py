"""Known-propensity weighted feedback loss; real logs are not assigned weights."""
import torch
from torch.nn import functional as F


def weighted_bce(logits, reward, weights, normalization='ips'):
    if (logits.ndim != 1 or not len(logits) or reward.shape != logits.shape or weights.shape != logits.shape
            or not torch.isfinite(logits).all() or not torch.isfinite(reward).all()
            or not torch.isfinite(weights).all() or (weights < 0).any()
            or ((reward < 0) | (reward > 1)).any()):
        raise ValueError('Finite aligned logits, bounded rewards and nonnegative weights required')
    losses = F.binary_cross_entropy_with_logits(logits, reward, reduction='none')
    if normalization == 'ips': return (losses * weights).mean()
    if normalization == 'snips':
        total = weights.sum()
        if total.item() <= 0: raise ValueError('Self-normalized loss has zero total weight')
        return (losses * weights).sum() / total
    raise ValueError('Unknown weighted loss normalization')


def fit_weighted_logistic(world, rows, seed, strength=3., exploration=.05):
    """Controlled, misspecified feedback model trained only on logged outcomes.

    Evaluation risk integrates exact expected rewards on a different finite
    population objective, uniformly across actions. No holdout labels tune it.
    """
    import numpy as np
    from src.exposure.synthetic import sample_log, reward_features
    torch.set_num_threads(4)
    log = sample_log(world, strength, rows, seed+2000000, exploration)
    features = torch.as_tensor(reward_features(world, log['context_ids'], log['action']), dtype=torch.float64)
    reward = torch.as_tensor(log['reward'], dtype=torch.float64)
    n, a = world['expected_reward'].shape
    evaluation = torch.as_tensor(reward_features(world, np.repeat(np.arange(n),a), np.tile(np.arange(a),n)), dtype=torch.float64)
    expected = torch.as_tensor(world['expected_reward'].ravel(), dtype=torch.float64)
    raw = (1/a) / log['pscore']
    results = {}
    for name, weights in [('unweighted',np.ones(rows)),('ips',raw),('ips_clip5',np.minimum(raw,5.))]:
        model = torch.nn.Linear(features.shape[1],1,dtype=torch.float64)
        with torch.no_grad(): model.weight.zero_(); model.bias.zero_()
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=80, tolerance_grad=1e-8,
                                      tolerance_change=1e-10, line_search_fn='strong_wolfe')
        weight = torch.as_tensor(weights,dtype=torch.float64); evaluations = 0
        def closure():
            nonlocal evaluations
            optimizer.zero_grad()
            loss = weighted_bce(model(features).squeeze(1), reward, weight) + .001*model.weight.square().sum()
            loss.backward(); evaluations += 1
            return loss
        optimizer.step(closure)
        final_objective=closure()
        gradient_max=max(float(p.grad.detach().abs().max()) for p in model.parameters())
        with torch.no_grad():
            logits = model(evaluation).squeeze(1)
            risk = F.binary_cross_entropy_with_logits(logits, expected).item()
            probabilities = logits.sigmoid().reshape(n,a).numpy()
            chosen = probabilities.argmax(1)
            value = world['expected_reward'][np.arange(n),chosen].mean()
            oracle_value = world['expected_reward'].max(1).mean()
        results[name] = {'true_uniform_logloss':risk,'true_greedy_policy_value':float(value),
                         'greedy_policy_regret':float(oracle_value-value),
                         'parameter_norm':float(model.weight.detach().norm()),
                         'coefficients':model.weight.detach().reshape(-1).tolist(),
                         'intercept':float(model.bias.detach().item()),
                         'training_objective':float(final_objective.detach()),
                         'objective_gradient_max':gradient_max,
                         'closure_evaluations':evaluations,'training_rows':rows,
                         'target_training_distribution':'uniform action, same context distribution'}
    return {'seed':seed,'logging_strength':strength,'models':results,
            'weights_from_known_synthetic_pscore':True,
            'evaluation_uses_finite_population_expected_rewards':True,
            'scope':'synthetic misspecified reward learning; no real recommender changed'}
