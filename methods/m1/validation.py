"""Strict smoke validation, without disguising validation data as test data."""
import json
import math
import re
import statistics

from jsonschema import Draft202012Validator, ValidationError
import numpy as np
import pandas as pd

from .data import assert_isolated, check_features, require, row_digest
from .evaluation import family_coverage, score_metrics


def equal(actual, expected, name):
    if expected is None:
        require(actual is None, f'{name}: expected null')
    else:
        require(actual is not None and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10), f'{name}: inconsistent value')


def sub_schema(value, definition, shared):
    try:
        Draft202012Validator({'$defs': shared['$defs'], **definition}).validate(value)
    except ValidationError as error:
        raise ValueError('Schema validation failed: ' + error.message) from error


def validate_metrics(metrics, samples, shared):
    sub_schema(metrics, shared['properties']['metrics'], shared)
    c = metrics['classification']
    tn, fp, fn, tp = (c['confusion_matrix'][k] for k in ('tn', 'fp', 'fn', 'tp'))
    n = tn + fp + fn + tp
    require(n == samples == c['n_samples'] and n > 0, 'Sample counts disagree')
    require(c['n_negative'] == tn + fp and c['n_positive'] == tp + fn, 'Class counts disagree')
    ratio = lambda a, b: a / b if b else None
    for key, value in [('accuracy', (tn + tp) / n), ('tpr', ratio(tp, tp + fn)),
                       ('fpr', ratio(fp, fp + tn)), ('precision', ratio(tp, tp + fp))]:
        equal(c[key], value, key)
    equal(c['macro_f1'], ((ratio(2 * tn, 2 * tn + fp + fn) or 0) + (ratio(2 * tp, 2 * tp + fp + fn) or 0)) / 2, 'macro F1')
    both = bool((tn + fp) and (tp + fn))
    require(c['both_true_classes_present'] == both, 'Class presence disagrees')
    curves, operating = metrics['curves'], metrics['operating_points']
    status = 'ok' if both else 'single_class'
    require(curves['status'] == operating['status'] == status, 'Curve status disagrees')
    if both:
        require(all(curves[k] is not None for k in ('roc_auc', 'average_precision', 'roc', 'pr')), 'Missing curves')
        roc, pr = curves['roc'], curves['pr']
        x, y, thresholds = roc['fpr'], roc['tpr'], roc['thresholds']
        require(len(x) == len(y) == len(thresholds) >= 2, 'ROC lengths disagree')
        require(x[0] == y[0] == 0 and x[-1] == y[-1] == 1, 'Invalid ROC endpoints')
        require(all(a <= b for a, b in zip(x, x[1:])) and all(a <= b for a, b in zip(y, y[1:])), 'Invalid ROC order')
        require(thresholds[0] is None and all(t is not None for t in thresholds[1:]), 'Invalid ROC threshold sentinel')
        require(all(a > b for a, b in zip(thresholds[1:], thresholds[2:])), 'Invalid ROC thresholds')
        equal(curves['roc_auc'], sum((b - a) * (u + v) / 2 for a, b, u, v in zip(x, x[1:], y, y[1:])), 'ROC area')
        precision, recall = pr['precision'], pr['recall']
        require(len(precision) == len(recall) == len(pr['thresholds']) + 1, 'PR lengths disagree')
        require(recall[0] == 1 and recall[-1] == 0 and all(a >= b for a, b in zip(recall, recall[1:])), 'Invalid PR order')
        require(pr['thresholds'] == list(reversed(thresholds[1:])), 'ROC/PR thresholds disagree')
        equal(curves['average_precision'], sum((a - b) * p for a, b, p in zip(recall, recall[1:], precision)), 'AP from curve')
        for point in operating['operating_points']:
            eligible = [i for i, value in enumerate(x) if value <= point['target_fpr']]
            index = max(eligible, key=lambda i: y[i])
            for key, value in [('actual_fpr', x[index]), ('tpr', y[index]), ('threshold', thresholds[index])]:
                equal(point[key], value, 'Operating ' + key)
            require(point['threshold_kind'] == ('above_max_score' if thresholds[index] is None else 'finite'), 'Operating threshold kind disagrees')
    else:
        require(all(curves[k] is None for k in ('roc_auc', 'average_precision', 'roc', 'pr')), 'Single class needs null curves')
        for p in operating['operating_points']:
            require(all(p[k] is None for k in ('actual_fpr', 'tpr', 'threshold')) and p['threshold_kind'] == 'undefined', 'Invalid single-class operating point')
    classes = metrics['per_class']['classes']
    names = [item['attack_family'] for item in classes]
    require(len(names) == len(set(names)) and 'benign' not in names, 'Invalid family entries')
    require(sum(item['support'] for item in classes) == tp + fn, 'Family supports disagree')
    require(sum(item['detected'] for item in classes) == tp, 'Family detections disagree')
    for item in classes:
        require(item['detected'] <= item['support'], 'Family detections exceed support')
        equal(item['recall'], ratio(item['detected'], item['support']), 'Family recall')
    attempted = c['attempted']
    if attempted['status'] == 'not_provided':
        require(all(attempted[k] is None for k in ('support', 'false_positives', 'fpr')), 'Missing attempted metadata must be null')
    else:
        support, false = attempted['support'], attempted['false_positives']
        require(support is not None and false is not None and 0 <= false <= min(support, fp) and support <= tn + fp, 'Invalid attempted counts')
        require(attempted['status'] == ('ok' if support else 'no_samples'), 'Attempted status disagrees')
        equal(attempted['fpr'], ratio(false, support), 'Attempted FPR')


def validate_inference(timing, shared):
    sub_schema(timing, shared['properties']['cost']['properties']['inference'], shared)
    require(len(timing['batch_times_ms']) == timing['repeats'], 'Timing repeat count disagrees')
    median = statistics.median(timing['batch_times_ms'])
    equal(timing['median_batch_ms'], median, 'Median latency')
    equal(timing['amortized_ms_per_flow'], median / timing['batch_size'], 'Amortized latency')
    equal(timing['throughput_flows_per_second'], 1000 * timing['batch_size'] / median, 'Throughput')


def validate_record(record, shared):
    json.dumps(record, allow_nan=False)
    required = {'purpose', 'evaluation_split', 'config', 'data', 'decision_rule', 'models', 'provenance', 'comparison'}
    require(required.issubset(record) and set(record).issubset(required | {'format_version'}), 'Missing or unexpected smoke sections')
    require(record['purpose'] == 'development_smoke_only' and record['evaluation_split'] == 'validation_subset', 'Not a smoke validation record')
    cfg, data, rule, models = (record[k] for k in ('config', 'data', 'decision_rule', 'models'))
    require(cfg['purpose'] == record['purpose'], 'Configuration purpose disagrees')
    check_features(data['features'])
    require(data['directory'] == cfg['dataset_directory'] and bool(data['sampling']), 'Data identity disagrees')
    for name in ('train_sha256', 'validation_sha256', 'preprocessor_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', data[name]) is not None, 'Invalid data SHA256')
    for kind, key in [('train', 'train'), ('validation', 'validation')]:
        used, total = data[kind + '_used_rows'], data[kind + '_source_rows']
        require(isinstance(used, int) and not isinstance(used, bool) and 0 < used <= total, 'Invalid subset count')
        require(used == cfg[key + '_sample_rows'], 'Configuration subset count disagrees')
    weights = data['training_class_weights']
    require(set(map(str, weights)) == {'0', '1'} and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in weights.values()), 'Invalid training class weights')
    require(rule == {'threshold': cfg['threshold'], 'operator': '>=', 'source': 'predefined_for_smoke'}, 'Smoke decision rule disagrees')
    require(isinstance(rule['threshold'], (int, float)) and not isinstance(rule['threshold'], bool), 'Invalid threshold')
    require(set(models) == set(cfg['models']) == {'random_forest', 'hist_gradient_boosting'}, 'Model names disagree')
    for name, item in models.items():
        require(item['parameters'] == cfg['models'][name], 'Model configuration disagrees')
        require(isinstance(item['training_seconds'], (int, float)) and not isinstance(item['training_seconds'], bool) and item['training_seconds'] >= 0, 'Invalid training time')
        require(item['model_reload_prediction_equal'] is True, 'Model reload did not pass')
        validate_metrics(item['metrics'], data['validation_used_rows'], shared)
        validate_inference(item['inference'], shared)
        for key, expected in cfg['inference'].items():
            equal(item['inference']['threadpool_limit' if key == 'threads' else key], expected, 'Inference setting ' + key)
    families = data.get('expected_attack_families')
    if families is not None:
        require(len(set(families)) == len(families) and bool(families) and 'benign' not in families, 'Invalid full family list')
        for item in models.values():
            require([x['attack_family'] for x in item['metrics']['per_class']['classes']] == families, 'Incomplete family report')
        for split in ('train', 'validation'):
            coverage = data[split + '_coverage']
            require(set(coverage['supports']) == set(families), 'Incomplete coverage report')
            require(all(isinstance(v, int) and v >= 0 for v in coverage['supports'].values()), 'Invalid family support')
            require(coverage['missing'] == [f for f in families if coverage['supports'][f] == 0], 'Missing family list disagrees')
        require(data['validation_coverage']['supports'] == {x['attack_family']: x['support'] for x in models['random_forest']['metrics']['per_class']['classes']}, 'Coverage/metric supports disagree')
    comparison = record['comparison']
    require(comparison['scope'] == 'development validation subset only', 'Invalid comparison scope')
    a, b = comparison['method_a'], comparison['method_b']
    require(a in models and b in models and a != b, 'Invalid comparison models')
    result = comparison['result']
    sub_schema(result, shared['properties']['comparisons']['items']['properties']['result'], shared)
    require(result['metric'] == 'ap' and result['target_fpr'] is None, 'Unexpected smoke comparison metric')
    for key, expected in [('value_a', models[a]['metrics']['curves']['average_precision']), ('value_b', models[b]['metrics']['curves']['average_precision']),
                          ('difference', result['value_a'] - result['value_b']), ('n_samples', data['validation_used_rows']),
                          ('n_positive', models[a]['metrics']['classification']['n_positive']), ('n_negative', models[a]['metrics']['classification']['n_negative']),
                          ('n_bootstrap', cfg['bootstrap_repetitions']), ('seed', cfg.get('bootstrap_seed', cfg['seed']))]:
        equal(result[key], expected, 'Comparison ' + key)
    require(result['ci_lower'] <= result['ci_upper'], 'Reversed bootstrap interval')
    provenance = record['provenance']
    require(all(isinstance(provenance[k], str) and provenance[k] for k in ('git_commit', 'python', 'platform', 'note')), 'Missing provenance')
    require(bool(provenance['software']) and bool(provenance['files_sha256']), 'Missing software/source provenance')
    require(all(re.fullmatch('[0-9a-f]{64}', x) for x in provenance['files_sha256'].values()), 'Invalid source hash')
    if record.get('format_version') == '2':
        require(families is not None, 'New records require complete family reporting')
        require(data['sampling_seeds'] == cfg['sampling_seeds'], 'Sampling seeds disagree')
        for item in models.values():
            require(item['effective_parameters']['random_state'] == cfg['seed'], 'Effective model seed disagrees')
            require(item['effective_parameters']['class_weight'] is None, 'P2 weights were applied twice')
            require(item['inference']['model_worker_limit'] == cfg['inference']['threads'], 'Inference model workers disagree')
        for key in ('train_row_ids_sha256', 'validation_row_ids_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', data[key]) is not None, 'Missing row identity hash')
        ids = data['timing_row_ids']
        require(len(ids) == cfg['inference']['batch_size'] and len(set(ids)) == len(ids), 'Invalid timing identity')
        require(data['timing_row_ids_sha256'] == row_digest(ids), 'Timing row hash disagrees')
        require(data['timing_composition']['benign'] + data['timing_composition']['attack'] == len(ids), 'Timing composition disagrees')
    return True


def validate_saved_predictions(record, saved):
    """Independently recompute from persisted labels, families, flags and scores."""
    names = set(record['models'])
    required = {'y_true', 'row_ids', 'attack_family', 'is_attempted', 'train_row_ids'} | names
    require(set(saved) == required, 'Missing or unexpected prediction arrays')
    y = np.asarray(saved['y_true'])
    require(y.ndim == 1 and len(y) == record['data']['validation_used_rows'], 'Prediction sample count disagrees')
    require(np.isin(y, [0, 1]).all(), 'Invalid saved binary labels')
    for key in ('row_ids', 'attack_family', 'is_attempted'):
        require(np.asarray(saved[key]).shape == y.shape, 'Saved metadata lengths disagree')
    require(np.asarray(saved['is_attempted']).dtype.kind == 'b', 'Invalid saved attempted flags')
    row_digest(saved['row_ids'])
    row_digest(saved['train_row_ids'])
    require(len(saved['train_row_ids']) == record['data']['train_used_rows'], 'Training row identity count disagrees')
    assert_isolated(saved['train_row_ids'].tolist(), saved['row_ids'].tolist())
    for key, ids in [('train_row_ids_sha256', saved['train_row_ids']), ('validation_row_ids_sha256', saved['row_ids'])]:
        if key in record['data']:
            require(row_digest(ids) == record['data'][key], 'Saved row order/identity hash disagrees')
    metadata = pd.DataFrame({'attack_family': saved['attack_family'], 'is_attempted': saved['is_attempted']})
    families = record['data'].get('expected_attack_families')
    for name in names:
        scores = np.asarray(saved[name])
        require(scores.shape == y.shape and np.isfinite(scores).all(), 'Invalid saved score array')
        require(((scores >= 0) & (scores <= 1)).all(), 'Invalid tree probabilities')
        metrics = score_metrics(y, scores, metadata, record['decision_rule']['threshold'], families)
        require(metrics == record['models'][name]['metrics'], 'Saved predictions do not reproduce metrics')
    if families is not None:
        require(family_coverage(y, saved['attack_family'], families) == record['data']['validation_coverage'], 'Saved coverage disagrees')
    if 'timing_row_ids' in record['data']:
        mapping = dict(zip(saved['row_ids'].tolist(), y.tolist()))
        timing = record['data']['timing_row_ids']
        require(set(timing).issubset(mapping), 'Timing rows not in saved validation population')
        attack = sum(mapping[x] for x in timing)
        require(record['data']['timing_composition'] == {'benign': len(timing) - attack, 'attack': attack}, 'Timing labels disagree')
    return True
