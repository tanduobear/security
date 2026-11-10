"""P4 recording adapter for P3's paired, class-stratified percentile bootstrap.

P3 has no replicate-return interface. This isolated adapter uses the same RNG
draw order and sklearn metric definitions; equivalence is tested against P3.
No thresholds, models, fitting, data loaders, or filesystem access here.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_arrays(y, a, b, ids_a, ids_b):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    ia, ib = np.asarray(ids_a), np.asarray(ids_b)
    require(all(x.ndim == 1 for x in (y, a, b, ia, ib)), 'Inputs must be one dimensional')
    require(len(y) > 0 and len({len(x) for x in (y, a, b, ia, ib)}) == 1, 'Input lengths differ')
    require(np.isin(y, [0, 1]).all() and set(np.unique(y)) == {0, 1}, 'Both binary classes required')
    for scores in (a, b):
        require(scores.dtype.kind == 'f' and np.isfinite(scores).all() and
                ((scores >= 0) & (scores <= 1)).all(), 'Invalid attack probabilities')
    for ids in (ia, ib):
        require(all(isinstance(x, str) and x.strip() for x in ids) and len(set(ids)) == len(ids), 'Invalid row IDs')
    require(np.array_equal(ia, ib), 'Paired row IDs/order differ')
    return y, a, b


def metric_value(y, scores, metric, target=None):
    if metric == 'ap':
        require(target is None, 'AP has no FPR target')
        return float(average_precision_score(y, scores))
    require(metric == 'tpr_at_fpr' and target in (.001, .01, .05), 'Unapproved metric/FPR target')
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
    return float(np.max(tpr[fpr <= target]))


def sample_indices(rng, negative, positive):
    # Exact P3 order, including RNG consumption: benign first, then attack.
    return np.concatenate([rng.choice(negative, size=len(negative), replace=True),
                           rng.choice(positive, size=len(positive), replace=True)])


def replicates(y, a, b, *, metric, target, seed, count, state=None, on_indices=None):
    require(type(seed) is int and seed >= 0 and type(count) is int and count >= 1, 'Invalid seed/count')
    rng = np.random.default_rng(seed)
    if state is not None:
        rng.bit_generator.state = state
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    require(len(negative) and len(positive), 'Both classes required')
    for _ in range(count):
        indices = sample_indices(rng, negative, positive)
        if on_indices is not None:
            on_indices(indices)
        # One shared index vector for A and B, no weights or family balancing.
        va = metric_value(y[indices], a[indices], metric, target)
        vb = metric_value(y[indices], b[indices], metric, target)
        require(np.isfinite([va, vb]).all(), 'Invalid replicate; never silently drop a sample')
        yield va, vb, va - vb, rng.bit_generator.state


def summarize(deltas, *, value_a, value_b, metric, target, y, seed=2026, confidence=.95):
    values = np.asarray(deltas, dtype=float)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), 'Invalid replicate distribution')
    require(0 < confidence < 1, 'Invalid confidence')
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail], method='linear')
    return {'status': 'ok', 'metric': metric, 'target_fpr': target,
            'difference_definition': 'A_minus_B', 'value_a': float(value_a), 'value_b': float(value_b),
            'difference': float(value_a - value_b), 'ci_lower': float(low), 'ci_upper': float(high),
            'confidence': confidence, 'n_bootstrap': len(values), 'seed': seed,
            'resampling': 'paired_class_stratified_rows', 'interval_method': 'percentile',
            'n_samples': len(y), 'n_negative': int((y == 0).sum()), 'n_positive': int((y == 1).sum()),
            'quantile_method': 'linear', 'bootstrap_standard_error': float(values.std(ddof=1)),
            'valid_replicates': len(values), 'standard_error_note': 'Bootstrap distribution sample SD, not SD/sqrt(B).'}
