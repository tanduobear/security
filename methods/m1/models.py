"""Tree fitting and scoring adapters; no search or threshold policy defaults."""
from time import perf_counter
from contextlib import contextmanager

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from threadpoolctl import threadpool_limits

from .data import integer, require


def fit_tree(name, params, training, *, model_seed, fit_threads):
    require(training.split == 'train', 'Only training-fold input may be fitted')
    training.validate()
    integer(model_seed, 'model seed')
    integer(fit_threads, 'fit threads', 1)
    constructors = {'random_forest': RandomForestClassifier,
                    'hist_gradient_boosting': HistGradientBoostingClassifier}
    require(name in constructors, 'Unknown tree implementation')
    require('random_state' not in params, 'Pass model_seed separately')
    require(params.get('class_weight') is None, 'Do not double-apply P2 class weights')
    if name == 'random_forest':
        integer(params.get('n_jobs', 1), 'RF workers', 1)
        require(params.get('n_jobs', 1) <= fit_threads, 'RF workers exceed fit thread budget')
    # Pin worker count even when omitted; sklearn n_jobs=None is otherwise implicit.
    effective = dict(params)
    if name == 'random_forest':
        effective.setdefault('n_jobs', 1)
    model = constructors[name](**effective, random_state=model_seed)
    weights = np.empty(len(training.y), dtype=np.float64)
    for label in (0, 1):
        weights[training.y == label] = training.class_weights[label]
    require(np.unique(training.y).size == 2, 'Training requires both binary classes')
    with threadpool_limits(limits=fit_threads):
        started = perf_counter()
        model.fit(training.X, training.y, sample_weight=weights)
        seconds = perf_counter() - started
    return model, seconds


def attack_scores(model, X):
    classes = np.asarray(model.classes_)
    require(classes.shape == (2,) and set(classes.tolist()) == {0, 1}, 'Expected binary classes 0/1')
    probabilities = np.asarray(model.predict_proba(X))
    require(probabilities.shape == (len(X), 2), 'Expected two probability columns')
    require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Invalid probability output')
    require(np.allclose(probabilities.sum(axis=1), 1), 'Class probabilities must sum to one')
    positive_column = int(np.flatnonzero(classes == 1)[0])
    return probabilities[:, positive_column]


@contextmanager
def inference_workers(model, workers):
    """Limit RF's own workers in addition to P3's BLAS/OpenMP limit."""
    integer(workers, 'inference workers', 1)
    has_workers = hasattr(model, 'n_jobs')
    previous = model.n_jobs if has_workers else None
    if has_workers:
        model.n_jobs = workers
    try:
        yield
    finally:
        if has_workers:
            model.n_jobs = previous
