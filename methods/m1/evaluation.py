"""Adapters around P3, with explicit family lists and timing row identities."""
import numpy as np

from eval.metrics import classification_metrics, compute_curves, per_class_recall, tpr_at_fpr
from .data import integer, require, row_digest


def score_metrics(y, scores, metadata, threshold, families=None):
    require(isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and np.isfinite(threshold), 'Invalid threshold')
    require(len(metadata) == len(y), 'Metadata/sample count mismatch')
    flags = metadata['is_attempted'].to_numpy()
    require(all(isinstance(x, (bool, np.bool_)) for x in flags), 'Attempted flags must be booleans')
    predicted = (np.asarray(scores) >= threshold).astype(np.uint8)
    return {
        'curves': compute_curves(y, scores),
        'classification': classification_metrics(y, predicted, is_attempted=flags.astype(bool)),
        'operating_points': tpr_at_fpr(y, scores),
        'per_class': per_class_recall(y, predicted, metadata['attack_family'].to_numpy(), families=families),
    }


def family_coverage(y, families_observed, expected):
    expected = list(expected)
    require(bool(expected) and len(set(expected)) == len(expected) and 'benign' not in expected, 'Invalid expected attack families')
    counts = {name: int(((np.asarray(y) == 1) & (np.asarray(families_observed) == name)).sum()) for name in expected}
    require(set(np.asarray(families_observed)[np.asarray(y) == 1]).issubset(expected), 'Unknown attack family')
    return {'supports': counts, 'missing': [name for name, count in counts.items() if count == 0]}


def timing_indices(row_ids, *, batch_size, seed=None, selected_row_ids=None):
    """Use a shared explicit ID list, or a fixed engineering sampling seed.

    No class balancing is done. Callers record composition separately.
    Formal protocols must decide which common ID list to use.
    """
    integer(batch_size, 'timing batch size', 1)
    ids = np.asarray(row_ids, dtype=str)
    row_digest(ids)
    require(len(set(ids.tolist())) == len(ids) and len(ids) >= batch_size, 'Invalid timing population')
    if selected_row_ids is not None:
        selected = list(selected_row_ids)
        require(len(selected) == batch_size and len(set(selected)) == batch_size, 'Invalid selected timing IDs')
        lookup = {value: index for index, value in enumerate(ids)}
        require(set(selected).issubset(lookup), 'Timing IDs not present in evaluation population')
        return np.array([lookup[value] for value in selected], dtype=np.int64)
    integer(seed, 'timing seed')
    # Hash ranking is stable even if the caller's row ordering changes.
    keys = [row_digest([str(seed), value]) for value in ids]
    return np.array(sorted(range(len(ids)), key=keys.__getitem__)[:batch_size], dtype=np.int64)
