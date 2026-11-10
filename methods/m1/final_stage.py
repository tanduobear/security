"""Independent, approval-gated calibration/evaluation of sealed G05 and RF.

The default plan is metadata-only. Parent imports are stdlib-only. All archive,
data and model work runs after the existing Windows Job monitor grants GO.
Neither this module nor final_stage_core contains any fitting or bootstrap call.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import uuid

from .search_gbdt import ROOT, GIB, digest, read, write, sha, require, resource_slot, abort_reason

CONFIG = ROOT / 'methods/m1/execution.final-test.proposed.json'
RESOURCES = {'outer_parallel_jobs': 1, 'inference_threads': 1,
             'prediction_batch_rows': 8192, 'job_commit_limit_bytes': 8 * GIB,
             'logical_cpu_limit': 4, 'min_start_free_bytes': 14 * GIB,
             'min_running_free_bytes': 6 * GIB, 'rss_abort_bytes': 8 * GIB,
             'duckdb_memory_limit_mb': 512, 'min_disk_free_bytes': 20 * GIB}


def payload(config):
    return {k: v for k, v in config.items() if k != 'approvals'}


def code_identity(project=ROOT):
    """Pin implementation bytes, independently of historical training sources."""
    project = Path(project)
    paths = [p for area in ('methods/m1', 'src', 'eval')
             for p in (project / area).glob('*.py')]
    paths += [project / p for p in ('eval/schema.json', 'docs/label_split_policy.json',
              'methods/m1/requirements-lock.txt', 'tools/aggregate_m1.py', 'schemas/m1.schema.json')]
    return {p.relative_to(project).as_posix(): sha(p) for p in sorted(paths)}


def validate(config):
    require(config['schema_version'] == '1' and config['stage'] == 'p4_final_evaluation', 'Wrong final-stage scope')
    require(config['seeds'] == [42, 43, 44, 45, 46], 'Five frozen seeds required')
    require(config['models'] == ['G05', 'RF_FIXED'], 'Only selected G05 and fixed RF allowed')
    require(config['dataset'] == {'version': 'original', 'protocol': 'stratified', 'variant': 'main', 'features': 140}, 'Data contract changed')
    require(config['reuse_trained_models'] is True and config['training_fits'] == 0, 'Retraining forbidden')
    require(config['bootstrap'] == 'deferred_separate_stage', 'No implicit bootstrap permitted')
    require(config['resources'] == RESOURCES and config['automatic_retries'] == 0, 'Resource/retry contract changed')
    require(config['thresholds'] == {'source_split': 'val', 'fpr_targets': [0.01, 0.05],
            'boundary_rule': 'strict_nextafter_benign_boundary', 'per_model_per_seed': True}, 'Invalid threshold proposal')
    timing = config['timing']
    require({k: timing[k] for k in ('source_split', 'device', 'batch_size', 'warmup', 'repeats', 'threads')} ==
            {'source_split': 'val', 'device': 'cpu', 'batch_size': 512, 'warmup': 5, 'repeats': 30, 'threads': 1}, 'Invalid timing protocol')
    require(timing['selection'] == 'shared_explicit_row_ids', 'Explicit common IDs required')
    require(config['attack_families_source'] == 'docs/label_split_policy.json', 'Family source changed')
    for key in ('unified_protocol_sha256', 'm1_summary_sha256'):
        require(isinstance(config[key], str) and len(config[key]) == 64, 'Missing public identity pin')
    require(set(config['archives']) == {'GBDT', 'RF'}, 'Two independent archives required')
    for spec in config['archives'].values():
        safe_path(ROOT, spec['path'])
        require(isinstance(spec['manifest_sha256'], str) and len(spec['manifest_sha256']) == 64, 'Missing manifest pin')
    return config


def safe_path(root, relative):
    root, value = Path(root).resolve(), Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Unsafe relative path')
    result = root / value
    require(result.resolve().is_relative_to(root), 'Path escaped root')
    return result


def require_approval(config, stage, *, launch=False, permit_test=False, project=ROOT):
    validate(config)
    require(stage in ('calibrate', 'evaluate') and launch is True, 'Explicit stage launch required')
    if stage == 'evaluate':
        require(permit_test is True, 'Explicit final Test access required')
    approval = config['approvals'][stage]
    require(approval['status'] == 'approved' and approval['source'] and approval['approved_at'], stage + ' approval pending')
    require(approval['payload_sha256'] == digest(payload(config)), 'Approved execution payload changed')
    require(approval['code_sha256'] == digest(code_identity(project)), 'Approved evaluation code changed')
    require(config['timing']['row_ids_path'] and config['timing']['row_ids_sha256'], 'Shared timing IDs not frozen')
    safe_path(project, config['timing']['row_ids_path'])


def plan(config, project=ROOT):
    validate(config)
    return {'stage': 'plan_only', 'task_count': 10, 'threshold_record_count': 20,
            'tasks': [{'task_id': f'{model}_seed_{seed}', 'model': model, 'seed': seed}
                      for model in config['models'] for seed in config['seeds']],
            'payload_sha256': digest(payload(config)), 'code_sha256': digest(code_identity(project)),
            'approvals': config['approvals'], 'test_accessed': False, 'training_fits': 0,
            'pending': [key for key, value in config['approvals'].items() if value['status'] != 'approved'] +
                       ([] if config['timing']['row_ids_path'] else ['shared_timing_row_ids'])}


def verify_output(run, expected_sha):
    """Hash verification only; never invokes a data loader or deserializes a model."""
    from .seal_gbdt import artifact_paths, freeze_files, file_inventory
    run = Path(run).resolve()
    require(sha(run / 'MANIFEST.json') == expected_sha, 'Output manifest pin mismatch')
    with freeze_files([*artifact_paths(run).values(), run / 'MANIFEST.json']):
        manifest = read(run / 'MANIFEST.json')
        require(manifest['status'] in ('validation_thresholds_frozen', 'p4_test_evaluation_complete'), 'Incomplete final stage')
        require(file_inventory(run) == manifest['files'], 'Output integrity mismatch')
        require(not (run / 'FAILED.json').exists(), 'Failed output cannot be reused')
        return manifest


def seal_output(run, stage):
    from .seal_gbdt import artifact_paths, freeze_files, file_inventory
    require(not (run / 'MANIFEST.json').exists(), 'Manifest already exists; never overwrite')
    with freeze_files(artifact_paths(run).values()):
        files = file_inventory(run)
        require('worker_result.json' in files and 'monitor/outcome.json' in files, 'Missing protected completion')
        require(read(run / 'worker_result.json') == {'status': stage + '_complete',
                'test_accessed': stage == 'evaluate', 'training_fits': 0, 'bootstrap_run': False}, 'Wrong worker result')
        require(not (run / 'FAILED.json').exists(), 'Failed stage cannot be sealed')
        outcome = read(run / 'monitor/outcome.json')
        require(outcome['exit_code'] == 0 and outcome['abort_reason'] is None and
                outcome['guard_verified'] and outcome['job_commit_limit_bytes'] == 8 * GIB and
                outcome['logical_cpu_limit'] == 4, 'Resource guard completion missing')
        require(file_inventory(run) == files, 'Output changed during sealing')
        write(run / 'MANIFEST.json', {'status': 'validation_thresholds_frozen' if stage == 'calibrate'
              else 'p4_test_evaluation_complete', 'files': files, 'test_accessed': stage == 'evaluate',
              'excluded_runtime_files': {'RUN.lock': 'Runtime coordination only, no experimental content.'},
              'self_hash_policy': 'Manifest is excluded from itself; pin SHA256 externally.'})
    result = sha(run / 'MANIFEST.json')
    verify_output(run, result)
    return result


def launch(config, stage, *, enabled=False, permit_test=False, calibration=None,
           calibration_sha=None, project=ROOT):
    require_approval(config, stage, launch=enabled, permit_test=permit_test, project=project)
    if stage == 'evaluate':
        require(calibration and calibration_sha, 'Frozen calibration directory and MANIFEST SHA256 required')
    from .resource_trial import WindowsAPI, monitored_child
    resources = config['resources']
    require(WindowsAPI().memory()['available_physical_bytes'] >= resources['min_start_free_bytes'], 'Start requires 14 GiB free')
    require(shutil.disk_usage(project).free >= resources['min_disk_free_bytes'], 'Start requires 20 GiB free disk')
    # Full raw final outputs stay inside the already ignored local workspace.
    run = Path(project) / '.tmp/m1_final_runs' / (stage + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex)
    with resource_slot():
        run.mkdir(parents=True, exist_ok=False)
        write(run / 'config.json', config)
        write(run / 'launch.json', {'stage': stage, 'launch': enabled, 'permit_test': permit_test,
              'calibration': str(Path(calibration).resolve()) if calibration else None, 'calibration_sha256': calibration_sha})
        (run / 'monitor').mkdir()
        try:
            outcome = monitored_child(run / 'monitor', {**resources, 'sample_seconds': 1, 'disk_sample_seconds': 5},
                worker_module='methods.m1.final_stage', worker_args=('--worker', run), reason_function=abort_reason)
            write(run / 'monitor/outcome.json', outcome)
            require(outcome['exit_code'] == 0 and outcome['abort_reason'] is None, 'Stage aborted; no automatic retry')
            manifest_sha = seal_output(run, stage)
        except Exception as error:
            write(run / 'FAILED.json', {'error_type': type(error).__name__, 'message': str(error), 'automatic_retry': False})
            raise
    print(json.dumps({'run': str(run), 'manifest_sha256': manifest_sha}), flush=True)
    return run


def worker(run):
    require(sys.stdin.readline().strip() == 'GO', 'Protected GO missing')
    run = Path(run)
    config, request = read(run / 'config.json'), read(run / 'launch.json')
    require_approval(config, request['stage'], launch=request['launch'], permit_test=request['permit_test'])
    # Heavy imports, archive hashes, P2 loading and model loading all follow GO.
    from .final_stage_core import execute
    def phase(name):
        (run / 'monitor/state.json').write_text(json.dumps({'phase': name}), encoding='utf-8')
        print(name, flush=True)
    execute(config, request, run, phase=phase)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='plan', choices=['plan', 'calibrate', 'evaluate'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--permit-final-test', action='store_true')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--calibration-sha256')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    config = read(args.config)
    if args.stage == 'plan':
        print(json.dumps(plan(config), indent=2))
        return 0
    launch(config, args.stage, enabled=args.launch, permit_test=args.permit_final_test,
           calibration=args.calibration, calibration_sha=args.calibration_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
