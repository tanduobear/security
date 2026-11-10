"""Audit saved P6 artifacts only. No training, model loading or inference."""
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    import numpy as np
    import pandas as pd
    from eval.validate import validate_result
    from eval.metrics import classification_metrics, compute_curves, per_class_recall
    from methods.m3.run import signature

    def read(relative):
        return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))

    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Saved-artifact audit; no fit, model loading or inference.',
              'checks': [], 'failures': []}

    def check(name, operation):
        try:
            value = operation()
            if value is not None and not bool(value):
                raise ValueError('check returned False')
            report['checks'].append({'name': name, 'passed': True})
            print('OK:', name, flush=True)
        except Exception as exc:
            report['checks'].append({'name': name, 'passed': False, 'error': str(exc)})
            report['failures'].append(name)
            print('FAIL:', name, str(exc), flush=True)

    cfg = read('methods/m3/config.json')
    search = read('results/m3_runs/search.json')
    plan = read('results/m3_runs/plan.json')
    config_hash = signature(cfg)
    expected_pairs = {(c['id'], seed) for c in cfg['candidates'] for seed in cfg['seeds']}
    actual_pairs = [(r['candidate']['id'], r['seed']) for r in search['runs']]
    check('40 unique candidate/seed runs', lambda: len(actual_pairs) == 40
          and set(actual_pairs) == expected_pairs)
    check('formal search and frozen config', lambda: search['status'] == 'validation_search_complete'
          and not search['is_smoke'] and not search['test_accessed']
          and search['config_sha256'] == config_hash and plan['config'] == cfg)
    for split in ['train', 'val']:
        check(split + ' dataset hash', lambda s=split:
              sha(ROOT / f'data/split/original/stratified/{s}.parquet')
              == search['data'][s + '_sha256'] == plan['data'][s + '_sha256'])
    for run in search['runs']:
        name = f'{run["candidate"]["id"]}/{run["seed"]}'
        folder = Path('results/m3_runs') / run['candidate']['id'] / str(run['seed'])
        check(name + ' saved metadata', lambda f=folder, r=run: read(f / 'run.json') == r)
        check(name + ' checkpoint hash', lambda f=folder, r=run:
              sha(ROOT / f / 'model.pt') == r['model_sha256'])
    calculated = []
    for candidate in cfg['candidates']:
        values = [r['best_benign_validation_mse'] for r in search['runs']
                  if r['candidate']['id'] == candidate['id']]
        calculated.append((statistics.mean(values), candidate['id']))
    selected = min(calculated)[1]
    check('winner recomputed from benign validation only', lambda:
          selected == search['selected_candidate'])

    main_index = read('results/m3.json')
    aux_index = read('results/m3_runs/auxiliary.json')
    check('published main index matches local index', lambda:
          main_index == read('results/m3_runs/m3.json'))
    entries = main_index['records'] + aux_index['records']
    check('10 AE + 4 auxiliary record references', lambda:
          len(main_index['records']) == 10 and len(aux_index['records']) == 4
          and len({x['record'] for x in entries}) == 14)
    test_path = ROOT / 'data/split/original/stratified/test.parquet'
    test_hash = sha(test_path)
    truth = pd.read_parquet(test_path, columns=['row_id', 'row_index', 'y_binary',
                                               'attack_family', 'is_attempted'])
    truth = truth.sort_values('row_index').reset_index(drop=True)
    aggregate = []
    by_prediction = {}
    for entry in entries:
        key = (entry['method'], entry['seed'])
        by_prediction.setdefault(key, []).append(entry)
    for (method, seed), grouped in by_prediction.items():
        prediction_path = ROOT / 'results/m3_runs/predictions' / f'{method}_seed{seed}.parquet'
        prediction = pd.read_parquet(prediction_path)
        values = prediction.score.to_numpy()
        identity = ['row_id', 'y_binary', 'attack_family', 'is_attempted']
        check(f'{method}/{seed} real test row identity and labels', lambda:
              prediction.row_id.is_unique and len(prediction) == len(truth)
              and all(prediction[c].astype(str).tolist() == truth[c].astype(str).tolist()
                      for c in identity)
              and np.isfinite(values).all())
        curves = compute_curves(prediction.y_binary.to_numpy(), values)
        for entry in grouped:
            name = Path(entry['record']).name
            record = read(entry['record'])
            check(name + ' P3 schema and numerical consistency', lambda r=record:
                  validate_result(r))
            notes = json.loads(record['provenance']['notes'])
            check(name + ' recorded file identities', lambda r=record, n=notes:
                  r['dataset']['sha256'] == test_hash
                  and n['prediction_sha256'] == sha(prediction_path)
                  and n['code_sha256'] == sha(ROOT / 'methods/m3/run.py')
                  and n['config_sha256'] == config_hash)
            check(name + ' AP/AUC recomputed from saved scores', lambda r=record:
                  all(np.isclose(curves[k], r['metrics']['curves'][k], rtol=1e-10, atol=1e-12)
                      for k in ['average_precision', 'roc_auc']))
            pred = (values >= record['decision_rule']['threshold']).astype(int)
            actual = classification_metrics(prediction.y_binary.to_numpy(), pred,
                          is_attempted=prediction.is_attempted.to_numpy(dtype=bool))
            check(name + ' confusion matrix recomputed from saved scores', lambda r=record:
                  actual['confusion_matrix'] == r['metrics']['classification']['confusion_matrix'])
            family = per_class_recall(prediction.y_binary.to_numpy(), pred,
                                      prediction.attack_family.tolist())
            check(name + ' family recall recomputed from saved scores', lambda r=record:
                  family == r['metrics']['per_class'])
            check(name + ' index values match detailed record', lambda r=record, e=entry:
                  e['ap'] == r['metrics']['curves']['average_precision']
                  and e['roc_auc'] == r['metrics']['curves']['roc_auc']
                  and e['classification'] == r['metrics']['classification'])
            if method == 'Autoencoder':
                run = next(r for r in search['runs']
                           if r['candidate']['id'] == selected and r['seed'] == seed)
                check(name + ' frozen validation threshold', lambda r=record, e=entry, s=run:
                      r['decision_rule']['source'] == 'validation_benign_quantile'
                      and r['decision_rule']['threshold'] == s['thresholds'][str(e['quantile'])])
            aggregate.append(record)
    check('published threshold record copy', lambda:
          sha(ROOT / 'docs/m3_threshold.md') == sha(ROOT / 'results/m3_runs/m3_threshold.md'))
    for relative in ['docs/p6_text.md', 'methods/m3/README.md', 'figures/fig_m3_error_dist.png']:
        check('deliverable exists: ' + relative, lambda p=relative:
              (ROOT / p).is_file() and (ROOT / p).stat().st_size > 0)
    software = {}
    for package in ['torch', 'numpy', 'pandas', 'scikit-learn', 'duckdb', 'pyarrow',
                    'jsonschema', 'threadpoolctl', 'matplotlib', 'joblib']:
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = 'not installed'
    report['audit_environment'] = {'python': platform.python_version(),
                                    'interpreter': sys.executable, 'packages': software}
    report['selected_candidate'] = selected
    report['statistical_scope'] = 'Cross-method paired bootstrap remains for P7 integration.'
    report['calibration_limit'] = ('Thresholds agree with saved calibration records; independent '
                                   'reconstruction of validation quantiles requires model replay.')
    report['status'] = 'passed' if not report['failures'] else 'failed'
    out = ROOT / 'docs/p6_acceptance/final_audit.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Saved:', out)
    print('FINAL:', report['status'], '/', len(report['checks']), 'checks')
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
