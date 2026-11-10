"""Serial, guarded P4 Bootstrap pipeline: trial -> 20 tasks -> verify -> publish.

Only a user-authorized --launch starts workers. All prediction and numerical
work happens after Windows Job/affinity verification and GO. No training or
Parquet/model loader is imported. No automatic retry.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import uuid

from .search_gbdt import ROOT, GIB, require, read, sha, digest, resource_slot, run_lock
from .seal_gbdt import artifact_paths, file_inventory, freeze_files

CONFIG = ROOT / 'methods/m1/execution.bootstrap.v1.json'
COLUMNS = ['seed', 'metric', 'fpr_target', 'bootstrap_iteration', 'gbdt_value', 'rf_value', 'delta']
METRICS = [('ap', None), ('tpr_at_fpr', .001), ('tpr_at_fpr', .01), ('tpr_at_fpr', .05)]
RESOURCES = {'outer_parallel_jobs': 1, 'metric_threads': 1, 'job_commit_limit_bytes': 8 * GIB,
             'logical_cpu_limit': 4, 'min_start_free_bytes': 14 * GIB, 'min_running_free_bytes': 6 * GIB,
             'rss_abort_bytes': 8 * GIB, 'min_disk_free_bytes': 20 * GIB}
TEST_RUN = '.tmp/m1_final_runs/evaluate-20261109T122704-2bf90cbb95e04ceea7c40993a6ec5012'
TEST_MANIFEST = '34a1503cd829290714d87dc6c84165295d92323dd1d44839ecbf5cbb49430c4f'
TEST_SUMMARY = 'cb2a95db154792d5dc0d3fb5f2710aba3ed6cc04f12a964e958cf26f371c23e3'
P3_SHA = 'aa593a22c51ad4d657c2c09716b103fe921c3c7f5c51cb597bd866094fefcd97'


def now():
    return datetime.now(timezone.utc).isoformat()


def publish_bytes(path, data):
    """Flush then exclusive hard-link publication. Never replace any file."""
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def write(path, value):
    publish_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def code_identity():
    files = ['methods/m1/bootstrap.py', 'methods/m1/bootstrap_stats.py', 'methods/m1/bootstrap_delivery.py',
             'tests/test_m1_bootstrap.py', 'methods/m1/resource_trial.py', 'methods/m1/search_gbdt.py',
             'methods/m1/seal_gbdt.py', 'eval/metrics.py', 'eval/schema.json', 'methods/m1/requirements-lock.txt']
    return {name: sha(ROOT / name) for name in files}


def make_config():
    return {'schema_version': '1', 'stage': 'p4_formal_paired_bootstrap',
        'statistics': {'models': ['G05', 'RF_FIXED'], 'model_seeds': [42, 43, 44, 45, 46],
            'repetitions': 2000, 'confidence': .95, 'bootstrap_seed': 2026,
            'difference': 'GBDT_minus_RF', 'resampling': 'paired_class_stratified_rows',
            'interval': 'percentile', 'quantile_method': 'linear',
            'metrics': [{'metric': m, 'fpr_target': t} for m, t in METRICS]},
        'input': {'archive': TEST_RUN, 'manifest_sha256': TEST_MANIFEST,
            'public_summary': 'results/m1_test_summary/summary.json', 'public_summary_sha256': TEST_SUMMARY,
            'rows': 346544, 'negative': 282506, 'positive': 64038,
            'row_ids_sha256': '33455428ecd6ebe255c60819f77958321e643a2f9104a8fcc2cc2ae9685422c3',
            'labels_sha256': '7096835d7bca2b55d6dc5743f68046466cea0e6153a36eb20d8836b9bf0286c8'},
        'resources': dict(RESOURCES), 'trial': {'model_seed': 42, 'metric': 'ap', 'repetitions': 20, 'counts_as_formal': False},
        'checkpoint_replicates': 100, 'automatic_retries': 0,
        'independent_verification': {'recompute_all_CIs_and_SE': True, 'regenerate_all_rng_checkpoints': True,
                                     'recheck_metric_iterations': [1, 1000, 2000]},
        'implementation': 'P4 replicate-recording adapter, exact P3 RNG and sklearn definitions; no P3 edits',
        'p3_sha256': P3_SHA, 'unified_file_sha256': '245618b00d9290d765f8cd185da1c1d3281e4310727c6bdd205297b386aae997',
        'code': code_identity(), 'delivery': {'directory': 'results/m1_bootstrap',
            'report': 'docs/m1_bootstrap_report.md',
            'figures': ['figures/fig_m1_bootstrap_ap.png', 'figures/fig_m1_bootstrap_tpr.png', 'figures/fig_m1_bootstrap_distribution.png'],
            'csv_columns': COLUMNS, 'csv_rows': 40000, 'overwrite': False},
        'approval': {'status': 'pending', 'source': None, 'approved_at': None, 'payload_sha256': None}}


def payload(config):
    return {k: v for k, v in config.items() if k != 'approval'}


def validate(config):
    require(config['stage'] == 'p4_formal_paired_bootstrap', 'Wrong Bootstrap scope')
    expected = make_config()
    require(payload(config) == payload(expected), 'Frozen Bootstrap parameters, source or code identity changed')
    a = config['approval']
    require(a['status'] == 'approved' and a['source'] and a['approved_at'] and
            a['payload_sha256'] == digest(payload(config)), 'Bootstrap approval/digest invalid')
    unified = read(ROOT / 'methods/m1/protocol.unified.v1.json')
    require(sha(ROOT / 'methods/m1/protocol.unified.v1.json') == config['unified_file_sha256'] and
            unified['approval']['status'] == 'approved' and unified['statistics']['paired_bootstrap'] and
            unified['model_seeds'] == config['statistics']['model_seeds'], 'Approved unified protocol conflict')
    require(sha(ROOT / 'eval/metrics.py') == P3_SHA, 'P3 changed')
    return config


def freeze():
    # Existing stage-scoped approved files remain byte-identical; old pending drafts stay pending.
    config = make_config()
    old = read(ROOT / 'methods/m1/execution.proposed.json')['bootstrap']
    require((old['repetitions'], old['confidence'], old['seed']) == (2000, .95, 2026), 'Original statistical proposal conflict')
    config['approval'] = {'status': 'approved', 'source': 'Explicit user authorization: P4 Formal Paired Bootstrap End-to-End Execution, sections A-J; engineering, trial, formal, verification and publication.',
        'approved_at': '2026-11-09', 'payload_sha256': digest(payload(config))}
    validate(config)
    write(CONFIG, config)
    print('FROZEN_CONFIG_SHA256', sha(CONFIG), flush=True)


def tasks(config, trial=False):
    if trial:
        return [{'id': 'trial_seed42_ap', 'seed': 42, 'metric': 'ap', 'target': None, 'count': 20, 'trial': True}]
    return [{'id': f'seed{s}_' + ('ap' if t is None else 'tpr_' + str(t)),
             'seed': s, 'metric': m, 'target': t, 'count': 2000, 'trial': False}
            for s in config['statistics']['model_seeds'] for m, t in METRICS]


@contextmanager
def source_guard(config):
    run = ROOT / config['input']['archive']
    public = ROOT / config['input']['public_summary']
    paths = [*artifact_paths(run).values(), run / 'MANIFEST.json', public]
    paths += [ROOT / name for name in config['code']]
    paths += [ROOT / 'methods/m1/protocol.unified.v1.json']
    with freeze_files(paths):
        validate(config)
        require(sha(public) == TEST_SUMMARY and sha(run / 'MANIFEST.json') == TEST_MANIFEST, 'Test source pin mismatch')
        manifest = read(run / 'MANIFEST.json')
        require(manifest['status'] == 'p4_test_evaluation_complete' and file_inventory(run) == manifest['files'], 'Test source integrity mismatch')
        summary = read(public)
        require(summary['dataset']['n_samples'] == 346544 and summary['selected_gbdt'] == 'G05', 'Test dataset/winner differs')
        yield run, summary
        require(file_inventory(run) == manifest['files'], 'Original Test archive changed')


def load_pair(config, source, summary, seed):
    import numpy as np
    from .bootstrap_stats import validate_arrays, metric_value
    from .pr_curve_data import row_digest
    arrays = []
    for model in ('G05', 'RF_FIXED'):
        with np.load(source / 'predictions' / f'{model}_seed_{seed}.npz', allow_pickle=False) as z:
            arrays.append((z['row_ids'], z['y_true'], z['y_score']))
    ia, y, a = arrays[0];ib, yb, b = arrays[1]
    require(np.array_equal(y, yb), 'Paired labels differ')
    validate_arrays(y, a, b, ia, ib)
    require(len(y) == 346544 and int((y == 0).sum()) == 282506 and int((y == 1).sum()) == 64038, 'Class population differs')
    require(row_digest(ia) == config['input']['row_ids_sha256'] and
            hashlib.sha256(np.asarray(y, dtype=np.uint8).tobytes()).hexdigest() == config['input']['labels_sha256'], 'Test row/label identity differs')
    expected = {}
    for method, scores in [('M1_GBDT', a), ('M1_RF_auxiliary', b)]:
        record = next(r for r in summary['per_seed_threshold_records'] if r['method'] == method and r['seed'] == seed and r['config']['target_validation_fpr'] == .01)
        expected[(method, 'ap', None)] = record['metrics']['average_precision']
        for op in record['metrics']['operating_points']['operating_points']:
            expected[(method, 'tpr_at_fpr', op['target_fpr'])] = op['tpr']
        for metric, target in METRICS:
            require(metric_value(y, scores, metric, target) == expected[(method, metric, target)], 'Original Test point metric mismatch')
    return y, a, b, expected


def csv_bytes(rows):
    import io
    stream = io.StringIO(newline='');writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(COLUMNS);writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def read_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream);require(reader.fieldnames == COLUMNS, 'CSV columns differ')
        rows = list(reader)
        require(all(set(row) == set(COLUMNS) and all(v is not None for v in row.values()) for row in rows), 'Malformed CSV row width')
        return rows


def task_binding(config, task):
    return digest({'config': payload(config), 'task': task})


def completed_chunks(directory, config, task):
    """Reject every orphan or changed committed segment; do not recompute it."""
    directory = Path(directory)
    checkpoints = sorted((directory / 'checkpoints').glob('*.json'))
    rows, state, previous, seconds = [], None, '0' * 64, 0.
    expected_csv = set()
    for number, path in enumerate(checkpoints):
        checkpoint = read(path)
        require(path.stem == f'{number:04}' and checkpoint['binding'] == task_binding(config, task) and
                checkpoint['previous'] == previous and checkpoint['start'] == len(rows), 'Resume identity/chain mismatch')
        relative = f'chunks/{number:04}.csv';expected_csv.add(relative)
        data = directory / relative
        require(checkpoint['csv_sha256'] == sha(data), 'Resume CSV hash mismatch')
        part = read_rows(data)
        require(len(part) == checkpoint['end'] - checkpoint['start'], 'Checkpoint count mismatch')
        for i, row in enumerate(part, len(rows) + 1):
            require(int(row['seed']) == task['seed'] and row['metric'] == task['metric'] and
                    row['fpr_target'] == ('' if task['target'] is None else str(task['target'])) and
                    int(row['bootstrap_iteration']) == i, 'Checkpoint row identity mismatch')
        rows.extend(part);state = checkpoint['rng_state'];seconds += checkpoint['compute_seconds'];previous = sha(path)
    actual = {p.relative_to(directory).as_posix() for p in (directory / 'chunks').glob('*.csv')}
    require(actual == expected_csv and len(rows) <= task['count'], 'Uncommitted/missing chunk: manual inspection required')
    return rows, state, previous, seconds


def execute_task(run, config, task, source, summary, phase):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .bootstrap_stats import replicates, summarize
    directory = run / 'tasks' / task['id'];directory.mkdir(parents=True, exist_ok=True)
    rows, state, previous, previous_seconds = completed_chunks(directory, config, task)
    resumed_count = len(rows)
    y, a, b, expected = load_pair(config, source, summary, task['seed'])
    started = now();seconds = previous_seconds
    while len(rows) < task['count']:
        offset = len(rows);count = min(config['checkpoint_replicates'], task['count'] - offset)
        batch = [];timer = perf_counter()
        with threadpool_limits(limits=1):
            for i, (va, vb, delta, state) in enumerate(replicates(y, a, b, metric=task['metric'], target=task['target'],
                    seed=2026, count=count, state=state), offset + 1):
                batch.append([task['seed'], task['metric'], '' if task['target'] is None else task['target'], i, va, vb, delta])
                if i % 20 == 0:phase(f"{task['id']} {i}/{task['count']}")
        elapsed = perf_counter() - timer;seconds += elapsed
        number = offset // config['checkpoint_replicates']
        csv_path = directory / f'chunks/{number:04}.csv';publish_bytes(csv_path, csv_bytes(batch))
        cp = {'binding': task_binding(config, task), 'start': offset, 'end': offset + count,
              'csv_sha256': sha(csv_path), 'rng_state': state, 'previous': previous, 'compute_seconds': elapsed}
        cp_path = directory / f'checkpoints/{number:04}.json';write(cp_path, cp);previous = sha(cp_path)
        rows.extend(read_rows(csv_path))
    result = summarize([float(r['delta']) for r in rows], value_a=expected[('M1_GBDT', task['metric'], task['target'])],
        value_b=expected[('M1_RF_auxiliary', task['metric'], task['target'])], metric=task['metric'], target=task['target'], y=y)
    if not (directory / 'result.json').exists():
        write(directory / 'result.json', {'task': task, 'binding': task_binding(config, task), 'result': result,
              'started_at': started, 'ended_at': now(), 'compute_seconds': seconds,
              'checkpoint_chain_sha256': previous, 'resumed_replicates': resumed_count})
    else:
        require(read(directory / 'result.json')['result'] == result, 'Existing result disagrees with CSV')


def record_completion(run, config, task, outcome, monitor):
    directory = run / 'tasks' / task['id']
    result = read(directory / 'result.json')
    require(result['binding'] == task_binding(config, task), 'Task completion binding mismatch')
    write(run / 'completed' / (task['id'] + '.json'), {'task': task, 'binding': task_binding(config, task),
          'files': file_inventory(directory), 'result_sha256': sha(directory / 'result.json'),
          'monitor': monitor.relative_to(run).as_posix(), 'outcome': outcome})


def verify_completed(run, config, task):
    path = run / 'completed' / (task['id'] + '.json')
    if not path.exists():return False
    record = read(path);directory = run / 'tasks' / task['id']
    require(record['task'] == task and record['binding'] == task_binding(config, task) and
            file_inventory(directory) == record['files'] and sha(directory / 'result.json') == record['result_sha256'], 'Completed task identity/hash mismatch')
    rows, _, chain, _ = completed_chunks(directory, config, task)
    require(len(rows) == task['count'] and read(directory / 'result.json')['checkpoint_chain_sha256'] == chain, 'Incomplete task checkpoint chain')
    require(record['outcome']['exit_code'] == 0 and record['outcome']['abort_reason'] is None and
            record['outcome']['guard_verified'], 'Task guard failed')
    return True


def abort_reason(sample, config):
    if sample['available_physical_bytes'] < config['min_running_free_bytes']:return 'free_memory_below_6_GiB'
    if sample['rss_bytes'] > config['rss_abort_bytes']:return 'tree_RSS_above_8_GiB'
    if sample['disk_free_bytes'] < config['min_disk_free_bytes']:return 'disk_free_below_20_GiB'


def protected(run, config, stage, task_id=None):
    from .resource_trial import WindowsAPI, monitored_child
    require(WindowsAPI().memory()['available_physical_bytes'] >= 14 * GIB, 'Start requires 14 GiB free')
    require(shutil.disk_usage(ROOT).free >= 20 * GIB, 'Start requires 20 GiB free disk')
    monitor = run / 'monitor' / (stage + '-' + (task_id or 'all') + '-' + uuid.uuid4().hex)
    monitor.mkdir(parents=True, exist_ok=False)
    outcome = monitored_child(monitor, {**RESOURCES, 'sample_seconds': 1, 'disk_sample_seconds': 5},
        worker_module='methods.m1.bootstrap', worker_args=('--worker', run, monitor, stage, task_id or '-'), reason_function=abort_reason)
    write(monitor / 'outcome.json', outcome)
    require(outcome['exit_code'] == 0 and outcome['abort_reason'] is None, 'Worker failed/aborted; no automatic retry. Evidence: ' + str(monitor))
    require(outcome['guard_verified'] and outcome['job_commit_limit_bytes'] == 8 * GIB and
            outcome['logical_cpu_limit'] == 4, 'Protected worker resource contract failed')
    require(read(monitor / 'worker_result.json')['status'] == stage + '_complete', 'Missing worker completion')
    return monitor, outcome


def worker(run, monitor, stage, task_id):
    require(sys.stdin.readline().strip() == 'GO', 'Protected GO missing')
    run, monitor = Path(run), Path(monitor)
    config = validate(read(run / 'config.json'))
    check_environment()
    def phase(value):
        (monitor / 'state.json').write_text(json.dumps({'phase': value}), encoding='utf-8')
        print(now(), value, flush=True)
    if stage == 'publish':
        from .bootstrap_delivery import publish
        publish(run, config, phase)
    else:
        with source_guard(config) as (source, summary):
            if stage in ('trial', 'task'):
                task = next(t for t in tasks(config, trial=stage == 'trial') if t['id'] == task_id)
                execute_task(run, config, task, source, summary, phase)
            elif stage == 'verify':
                from .bootstrap_delivery import independent_verify
                independent_verify(run, config, source, summary, phase)
            else:raise ValueError('Unknown worker stage')
    write(monitor / 'worker_result.json', {'status': stage + '_complete', 'training_fits': 0, 'parquet_accessed': False})


def initialize(config, kind):
    from .resource_trial import WindowsAPI
    require(WindowsAPI().memory()['available_physical_bytes'] >= 14 * GIB, 'Start requires 14 GiB free')
    run = ROOT / '.tmp' / ('m1_bootstrap_trials' if kind == 'trial' else 'm1_bootstrap_runs') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False);write(run / 'config.json', config)
    write(run / 'kind.json', {'kind': kind, 'created_at': now()})
    write(run / 'environment.json', check_environment())
    for name in config['code']:
        publish_bytes(run / 'sources' / name, (ROOT / name).read_bytes())
    return run


def check_environment():
    import platform
    from importlib.metadata import version
    require(Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve(), 'Original .venv required')
    versions = {p: version(p) for p in ('numpy', 'scikit-learn', 'matplotlib')}
    require(platform.python_version() == '3.13.0' and versions['numpy'] == '2.3.5' and
            versions['scikit-learn'] == '1.8.0', 'Original numerical environment changed')
    return {'python': platform.python_version(), 'packages': versions, 'platform': platform.platform(),
            'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'python_path': '.venv/Scripts/python.exe'}


def run_pipeline(config, resume=None):
    validate(config)
    # Runtime-only sleep inhibition, automatically released; no power plan edits.
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint
    require(kernel.SetThreadExecutionState(0x80000001), 'Temporary keep-awake request failed')
    try:
        with resource_slot():
            if resume:
                run = Path(resume).resolve()
                require(read(run / 'config.json') == config and read(run / 'kind.json')['kind'] == 'formal', 'Resume scope/config differs')
                require(not (run / 'PUBLISHED.json').exists(), 'Already completed; never rerun')
            else:
                trial = initialize(config, 'trial');print('TRIAL_RUN', trial, flush=True)
                task = tasks(config, trial=True)[0]
                monitor, outcome = protected(trial, config, 'trial', task['id'])
                record_completion(trial, config, task, outcome, monitor)
                require(verify_completed(trial, config, task), 'Trial acceptance failed')
                write(trial / 'TRIAL_COMPLETE.json', {'task': task, 'outcome': outcome,
                      'result_sha256': sha(trial / 'tasks' / task['id'] / 'result.json'), 'counts_as_formal': False})
                print('TRIAL PASSED; formal tasks start automatically', flush=True)
                run = initialize(config, 'formal')
                write(run / 'trial_reference.json', {'directory': str(trial), 'receipt_sha256': sha(trial / 'TRIAL_COMPLETE.json')})
            print('FORMAL_RUN', run, flush=True)
            with run_lock(run):
                reference = read(run / 'trial_reference.json');trial = Path(reference['directory'])
                require(sha(trial / 'TRIAL_COMPLETE.json') == reference['receipt_sha256'] and
                        verify_completed(trial, config, tasks(config, trial=True)[0]), 'Trial reference changed')
                allowed = {t['id'] for t in tasks(config)}
                require({p.stem for p in (run / 'completed').glob('*.json')}.issubset(allowed), 'Unknown completed task')
                for number, task in enumerate(tasks(config), 1):
                    if verify_completed(run, config, task):
                        print('SKIP HASH-VERIFIED', task['id'], flush=True);continue
                    print(f"TASK {number}/20 {task['id']} repetitions=2000", flush=True)
                    monitor, outcome = protected(run, config, 'task', task['id'])
                    record_completion(run, config, task, outcome, monitor)
                    print('COMPLETED', task['id'], 'peak Job GiB=', round(outcome['job_peak_commit_bytes'] / GIB, 3), flush=True)
                if not (run / 'independent_verification.json').exists():protected(run, config, 'verify')
                protected(run, config, 'publish')
                with freeze_files(artifact_paths(run).values()):
                    files = file_inventory(run)
                    write(run / 'MANIFEST.json', {'status': 'formal_bootstrap_complete', 'files': files,
                          'excluded_runtime_files': {'RUN.lock': 'OS runtime lock, no experimental content.'},
                          'public_manifest_sha256': read(run / 'PUBLISHED.json')['manifest_sha256']})
                    require(file_inventory(run) == files, 'Local archive changed during sealing')
                require(file_inventory(run) == read(run / 'MANIFEST.json')['files'], 'Local archive independent hash verification failed')
                print('LOCAL_ARCHIVE_MANIFEST_SHA256', sha(run / 'MANIFEST.json'), flush=True)
            print('BOOTSTRAP COMPLETE', run, flush=True)
            return run
    except BaseException as error:
        evidence = locals().get('run') or locals().get('trial')
        if evidence:
            write(evidence / ('FAILED_ATTEMPT-' + uuid.uuid4().hex + '.json'),
                  {'error_type': type(error).__name__, 'message': str(error), 'created_at': now(), 'automatic_retry': False})
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='plan', choices=['freeze', 'plan', 'run'])
    parser.add_argument('--launch', action='store_true');parser.add_argument('--resume', type=Path)
    parser.add_argument('--worker', nargs=4, metavar=('RUN', 'MONITOR', 'STAGE', 'TASK'))
    args = parser.parse_args(argv)
    if args.worker:return worker(*args.worker)
    if args.stage == 'freeze':return freeze()
    config = read(CONFIG) if CONFIG.exists() else make_config()
    if args.stage == 'plan':
        print(json.dumps({'config': config, 'tasks': tasks(config), 'task_count': 20, 'csv_rows': 40000}, indent=2));return
    require(args.launch, 'Explicit --launch required')
    run_pipeline(config, args.resume)


if __name__ == '__main__':main()
