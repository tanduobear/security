"""Collect saved formal results. Never trains or treats smoke results as final."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 43, 44, 45, 46]
APPROVAL_HASH = '8cea962ee3c6a221bb7307799d53dcf016779440b4dbebdeeea281e399917718'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def five(records, description):
    require(len(records) == 5 and sorted(r['seed'] for r in records) == SEEDS,
            description + ': require exactly one result for each seed 42–46')


def mean_std(values):
    values = list(values)
    require(len(values) == 5, 'Five observations are required; no silently dropped seeds')
    if all(v is None for v in values):
        return {'mean': None, 'sample_std': None, 'n': 5}
    require(all(v is not None and math.isfinite(float(v)) for v in values),
            'Missing/nonfinite observations must not be silently averaged')
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values), 'n': 5}


def metrics(value):
    curves = value.get('curves', value)
    cls = value['classification']
    cm = cls['confusion_matrix']
    n = sum(cm[k] for k in ['tn', 'fp', 'fn', 'tp'])
    require(n > 0 and n == cls['n_samples'], 'Confusion counts disagree')
    family = value.get('per_class', value.get('per_class_recall'))
    require(family is not None, 'Family recall is missing')
    points = value.get('operating_points', {}).get('operating_points', [])
    result = {'ap': curves['average_precision'], 'roc_auc': curves['roc_auc'],
              **{key: cls[key] for key in ['accuracy', 'macro_f1', 'fpr', 'tpr', 'precision']},
              'n_samples': n,
              'family': {x['attack_family']: {'support': x['support'], 'recall': x['recall']}
                         for x in family['classes']}}
    for target, suffix in [(0.01, '1pct'), (0.05, '5pct')]:
        point = next((p for p in points if p['target_fpr'] == target), None)
        result['tpr_at_fpr_' + suffix] = (point['tpr'] if point else value.get('tpr_at_fpr_' + suffix))
    for key in ['ap', 'roc_auc', 'accuracy', 'macro_f1', 'fpr', 'tpr', 'precision',
                'tpr_at_fpr_1pct', 'tpr_at_fpr_5pct']:
        require(result[key] is not None and 0 <= result[key] <= 1, 'Invalid/missing rate: ' + key)
    for key, numerator, denominator in [('fpr', cm['fp'], cm['tn'] + cm['fp']),
                                       ('tpr', cm['tp'], cm['tp'] + cm['fn'])]:
        require(denominator > 0 and math.isclose(result[key], numerator / denominator,
                                                abs_tol=1e-10), 'Inconsistent ' + key)
    for group in result['family'].values():
        require((group['support'] == 0 and group['recall'] is None) or
                (group['support'] > 0 and group['recall'] is not None and 0 <= group['recall'] <= 1),
                'Absent attack families must have recall=null, not zero')
    return result


def row(method, version, record, value, configuration, source):
    cost = record.get('cost', {})
    inference = record.get('latency_cpu', cost.get('inference', {}))
    return {'method': method, 'version': version, 'seed': int(record['seed']),
            'configuration': configuration, 'source': str(source),
            'training_seconds': record.get('training_seconds', cost.get('training_seconds')),
            'cpu_latency_ms_per_flow': inference.get('amortized_ms_per_flow'), **metrics(value)}


def original(root):
    all_rows = []
    m1_path = root / 'results/m1_test_summary/summary.json'
    m1 = load(m1_path)
    records = [r for r in m1['per_seed_threshold_records']
               if r['method'] == 'M1_GBDT' and r['config']['target_validation_fpr'] == 0.01]
    five(records, 'original M1')
    all_rows += [row('M1', 'original', r, r['metrics'], 'G05', m1_path.relative_to(root)) for r in records]
    rf = [r for r in m1['per_seed_threshold_records']
          if r['method'] == 'M1_RF_auxiliary' and r['config']['target_validation_fpr'] == 0.01]
    five(rf, 'original RF auxiliary')
    all_rows += [row('RF_auxiliary', 'original', r, r['metrics'], 'RF_FIXED', m1_path.relative_to(root)) for r in rf]
    m2_path = root / 'results/m2.json'
    m2 = load(m2_path)
    require(m2['dataset']['version'] == 'original', 'Wrong original M2 population')
    five(m2['runs'], 'original M2')
    all_rows += [row('M2', 'original', r, r['test'], 'CNN1D_fixed', m2_path.relative_to(root)) for r in m2['runs']]
    m3_path = root / 'results/m3_summary/summary.json'
    m3 = load(m3_path)
    records = [r for r in m3['records']
               if r['method'] == 'Autoencoder' and r['config']['benign_quantile'] == 0.99]
    five(records, 'original M3')
    all_rows += [row('M3', 'original', r, r['metrics'], r['config']['id'], m3_path.relative_to(root)) for r in records]
    return all_rows


def validate_m3_search(search, cfg):
    require(search['status'] == 'validation_search_complete' and search['is_smoke'] is False
            and search['test_accessed'] is False, 'M3 formal search/freeze evidence missing')
    expected = {(c['id'], seed) for c in cfg['candidates'] for seed in SEEDS}
    require(len(search['runs']) == 40 and {(r['candidate']['id'], r['seed']) for r in search['runs']} == expected,
            'M3 requires 40 unique configuration/seed runs')
    candidates = {c['id']: c for c in cfg['candidates']}
    require(all(r['candidate'] == candidates[r['candidate']['id']] for r in search['runs']),
            'M3 candidate parameters changed')
    ranking = [(statistics.mean(r['best_benign_validation_mse'] for r in search['runs']
                                if r['candidate']['id'] == name), name) for name in candidates]
    require(search['selected_candidate'] == min(ranking)[1], 'M3 selected winner does not match benign-validation rule')


def corrected(run_root, m1_run=None):
    candidates = sorted((run_root / 'results/p7_corrected/m1/formal').glob('*/evaluation_index.json'))
    if m1_run is None:
        require(len(candidates) == 1, 'Select exactly one completed M1 run with --m1-run; found ' + str(len(candidates)))
        directory = candidates[0].parent
    else:
        directory = Path(m1_run).resolve()
    expected_commit = subprocess.check_output(['git', '-c', 'safe.directory=' + str(run_root), 'rev-parse', '3632afa'],
                                               cwd=run_root, text=True).strip()
    require(directory.is_relative_to((run_root / 'results/p7_corrected/m1/formal').resolve()),
            'M1 results must be inside the isolated formal output root')
    require(not (directory / 'FAILED.json').exists(), 'M1 run has FAILED.json')
    index = load(directory / 'evaluation_index.json')
    require(index['status'] == 'complete' and index['git_commit'] == expected_commit,
            'M1 run incomplete or not run from 3632afa')
    approval = load(directory / 'execution_approval.json')
    payload = {k: v for k, v in approval.items() if k != 'approval'}
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(payload_hash == APPROVAL_HASH == approval['approval']['payload_sha256']
            and approval['approval']['status'] == 'approved', 'M1 v2 approval mismatch')
    require(approval['runner_sha256'] == sha(run_root / 'methods/p7/run_m1_full.py'),
            'M1 runner differs from approved source')
    require(approval['config_sha256'] == sha(run_root / 'methods/p7/m1_corrected.json')
            and load(directory / 'config.json') == load(run_root / 'methods/p7/m1_corrected.json'),
            'M1 saved configuration mismatch')
    frozen = load(directory / 'validation_frozen.json')
    require(frozen['test_accessed'] is False and len(frozen['jobs']) == 10,
            'M1 freeze evidence missing/incomplete')
    require({(j['model_id'], j['seed']) for j in frozen['jobs']} ==
            {(model, seed) for model in ['G05', 'RF_FIXED'] for seed in SEEDS},
            'M1 requires G05 and RF_FIXED with all five seeds')
    manifest = load(directory / 'MANIFEST.json')
    require(manifest['status'] == 'p7_m1_corrected_full_rerun_complete', 'M1 manifest is not a formal completion manifest')
    for name, info in manifest['files'].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory), 'Archive path escapes run root')
        require(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
                'M1 manifest mismatch: ' + name)
    all_rows = []
    keys = set()
    for entry in index['records']:
        path = (directory / entry['path']).resolve()
        require(path.is_relative_to(directory), 'Record path escapes run root')
        value = load(path)
        require(value['dataset']['version'] == 'corrected' and value['dataset']['split'] == 'test'
                and value['dataset']['protocol'] == 'stratified' and value['dataset']['variant'] == 'main',
                'M1 record population mismatch')
        require(value['test_row_ids_sha256'] == index['test_row_ids_sha256'], 'M1 Test row identity mismatch')
        key = (value['model_id'], value['seed'], value['threshold_calibration']['target_fpr'])
        require(key not in keys, 'Duplicate M1 model/seed/threshold record')
        keys.add(key)
        method = 'M1' if value['model_id'] == 'G05' else 'RF_auxiliary'
        normalized = row(method, 'corrected', value, value['metrics'], value['model_id'],
                         'run-root/' + path.relative_to(run_root).as_posix())
        if value['threshold_calibration']['target_fpr'] == 0.01:
            all_rows.append(normalized)
    require(len(index['records']) == 20 and keys ==
            {(model, seed, q) for model in ['G05', 'RF_FIXED'] for seed in SEEDS for q in [.01, .05]},
            'M1 record count/uniqueness mismatch')
    for method in ['M1', 'RF_auxiliary']:
        five([r for r in all_rows if r['method'] == method], 'corrected ' + method)

    m2_path = run_root / 'results/p7_corrected/m2_corrected.json'
    m2 = load(m2_path)
    require(m2['dataset']['version'] == 'corrected' and m2['dataset']['protocol'] == 'stratified'
            and m2['dataset']['variant'] == 'main', 'M2 dataset mismatch')
    require(m2['training_config']['device'] == 'cuda' and m2['evaluation_config']['evaluation_device'] == 'cuda',
            'M2 was expected to use CUDA; do not silently accept a CPU fallback')
    five(m2['runs'], 'corrected M2')
    all_rows += [row('M2', 'corrected', r, r['test'], 'CNN1D_fixed',
                     'run-root/' + m2_path.relative_to(run_root).as_posix()) for r in m2['runs']]
    for stage in ['M2Train', 'M2Evaluate', 'M3Search', 'M3Evaluate']:
        receipt = load(run_root / 'results/p7_corrected/receipts' / (stage + '.json'))
        require(receipt['status'] == 'complete' and receipt['git_commit'] == expected_commit,
                stage + ' receipt missing or version mismatch')
        config = 'methods/p7/m2_corrected.yaml' if stage.startswith('M2') else 'methods/p7/m3_corrected.json'
        require(receipt['config_hashes'][config] == sha(run_root / config), stage + ' receipt config mismatch')

    m3_dir = run_root / 'results/p7_corrected/m3'
    search = load(m3_dir / 'search.json')
    cfg = load(run_root / 'methods/p7/m3_corrected.json')
    validate_m3_search(search, cfg)
    require(load(m3_dir / 'plan.json')['config'] == cfg, 'M3 frozen plan differs from prepared config')
    manifest3 = load(run_root / 'data/split/corrected/stratified/preprocess_manifest.json')
    for split in ['train', 'val']:
        require(search['data'][split + '_sha256'] == manifest3['output'][split]['main']['sha256'],
                'M3 search data identity mismatch')
    for run in search['runs']:
        path = m3_dir / run['candidate']['id'] / str(run['seed']) / 'model.pt'
        require(sha(path) == run['model_sha256'], 'M3 model hash mismatch')
    index3 = load(m3_dir / 'm3.json')
    require(index3['status'] == 'test_evaluation_complete' and index3['dataset'] ==
            {'version': 'corrected', 'protocol': 'stratified', 'variant': 'main'}, 'M3 evaluation identity mismatch')
    require(index3['selected_candidate'] == search['selected_candidate'], 'M3 selected candidate mismatch')
    expected_test_hash = load(run_root / 'data/split/corrected/stratified/preprocess_manifest.json')['output']['test']['main']['sha256']
    chosen = []
    require(len(index3['records']) == 10, 'M3 requires five seeds at both quantiles')
    require({(r['seed'], r['quantile']) for r in index3['records']} ==
            {(s, q) for s in SEEDS for q in [0.95, 0.99]}, 'M3 duplicate or missing threshold/seed record')
    for item in index3['records']:
        path = (run_root / item['record']).resolve()
        require(path.is_relative_to(m3_dir), 'M3 record path must remain isolated')
        record = load(path)
        require(record['dataset']['sha256'] == expected_test_hash and record['dataset']['split'] == 'test'
                and record['dataset']['version'] == 'corrected', 'M3 Test identity mismatch')
        require(record['provenance']['code_commit'] == expected_commit, 'M3 execution version mismatch')
        notes = json.loads(record['provenance']['notes'])
        require(notes['code_sha256'] == sha(run_root / 'methods/m3/run.py'), 'M3 evaluated with different implementation')
        run = next(r for r in search['runs'] if r['candidate']['id'] == search['selected_candidate']
                   and r['seed'] == record['seed'])
        q = item['quantile']
        require(record['decision_rule']['source'] == 'validation_benign_quantile'
                and record['decision_rule']['threshold'] == run['thresholds'][str(q)], 'M3 threshold changed')
        prediction = m3_dir / 'predictions' / f'Autoencoder_seed{record["seed"]}.parquet'
        require(sha(prediction) == notes['prediction_sha256'], 'M3 prediction hash mismatch')
        if q == 0.99:
            chosen.append(row('M3', 'corrected', record, record['metrics'], search['selected_candidate'],
                              'run-root/' + path.relative_to(run_root).as_posix()))
    five(chosen, 'corrected M3 q99')
    all_rows += chosen
    counts = {r['n_samples'] for r in all_rows}
    require(len(counts) == 1, 'Corrected methods used different Test sample counts')
    audit = {'status': 'passed', 'required_training_commit': expected_commit,
             'm1_run': directory.relative_to(run_root).as_posix(), 'm1_approval_hash': APPROVAL_HASH,
             'corrected_test_rows': next(iter(counts)),
             'limitations': ['M2 base evaluator does not export per-row scores; row-order identity is not independently checked here.',
                            'This is file/provenance verification, not an independent replay of models.',
                            'CPU timings from different machines are not a controlled speed comparison.']}
    return all_rows, audit


def aggregate(rows):
    result = []
    for method, version in sorted({(r['method'], r['version']) for r in rows}):
        selected = [r for r in rows if r['method'] == method and r['version'] == version]
        five(selected, method + '/' + version)
        group = {'method': method, 'dataset_version': version, 'configuration': selected[0]['configuration'],
                 'evaluation_split': 'test', 'seed_count': 5, 'family': {}}
        for key in ['ap', 'roc_auc', 'tpr_at_fpr_1pct', 'tpr_at_fpr_5pct', 'accuracy', 'macro_f1',
                    'fpr', 'tpr', 'precision', 'training_seconds', 'cpu_latency_ms_per_flow']:
            group[key] = mean_std(r.get(key) for r in selected)
        families = sorted(set().union(*(r['family'] for r in selected)))
        for family in families:
            require(all(family in r['family'] for r in selected), 'Family lists differ across seeds')
            counts = {r['family'][family]['support'] for r in selected}
            require(len(counts) == 1, 'Family support differs across seeds')
            group['family'][family] = {'support': next(iter(counts)),
                                       **mean_std(r['family'][family]['recall'] for r in selected)}
        result.append(group)
    return result


def write(root, rows, audit, preview=False):
    # Preview can never overwrite a final deliverable or masquerade as corrected results.
    out = root / ('results/p7_preview' if preview else 'results')
    out.mkdir(parents=True, exist_ok=True)
    groups = aggregate(rows)
    payload = {'status': 'preview_original_only' if preview else 'formal_corrected_integration',
               'audit': audit, 'groups': groups, 'per_seed': rows,
               'version_delta_note': 'Difference of means across independent populations; not a paired confidence interval.'}
    (out / 'p7_figure_data.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    csv_rows = []
    for group in groups:
        entry = {k: group[k] for k in ['method', 'dataset_version', 'configuration', 'evaluation_split', 'seed_count']}
        for name, values in group.items():
            if isinstance(values, dict) and 'mean' in values:
                entry[name + '_mean'] = values['mean']
                entry[name + '_sample_std'] = values['sample_std']
        csv_rows.append(entry)
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (out / 'p7_acceptance.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print('Saved:', out / 'summary.csv', 'and p7_figure_data.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT.parent / 'p7-run-3632afa')
    parser.add_argument('--m1-run', type=Path)
    parser.add_argument('--preview-original', action='store_true')
    args = parser.parse_args()
    rows = original(ROOT)
    if args.preview_original:
        return write(ROOT, rows, {'status': 'preview_only_no_corrected_results_checked'}, True)
    formal, audit = corrected(args.run_root.resolve(), args.m1_run)
    rows.extend(formal)
    write(ROOT, rows, audit)
    from export_corrected_summary import export
    export(ROOT, args.run_root.resolve(), audit)


if __name__ == '__main__':
    main()
