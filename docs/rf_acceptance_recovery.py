"""One authorized RF acceptance recovery; never trains or alters bound source/config.

Use the original run_rf accept worker unchanged. This separate sealer preserves the
failed system-Python attempt, requires exactly one new successful acceptance, and
retains all original seal checks. Both helper source and provenance are archived.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.m1 import run_rf as rf
from methods.m1.seal_gbdt import artifact_paths, file_inventory, freeze_files

RUN = ROOT / 'results/m1_rf_runs/20261102T103826-3bdbabff97db4e2f99ff8a8a6b0b7767'
RECOVERY = RUN / 'recovery/venv-acceptance-02'
FAILED = 'ACCEPT-b65ce02fb5374972808862184947ce7c'
CONFIG_SHA = '65fa1b47ad866b1d6d6db53105c47a3bb901bcb8ca005e5e86179b4de16d9043'


def assert_environment():
    from methods.m1.storage import environment
    import duckdb
    import sklearn
    import numpy
    expected = ROOT / '.venv/Scripts/python.exe'
    rf.require(Path(sys.executable).resolve() == expected.resolve(), 'Original .venv Python required')
    current = environment()
    rf.require(current == rf.read(RUN / 'binding.json')['environment'], 'Training environment differs')
    return {'executable': str(Path(sys.executable).resolve()), 'executable_sha256': rf.sha(expected),
            'python': sys.version, 'environment': current,
            'imported_versions': {'duckdb': duckdb.__version__, 'sklearn': sklearn.__version__, 'numpy': numpy.__version__}}


def baseline_unchanged(run, baseline):
    actual = file_inventory(run)
    for name, expected in baseline.items():
        rf.require(actual.get(name) == expected, 'Original archive file changed: ' + name)


def accepted_attempt(run, failed_files, summary_sha):
    failed = run / 'monitor' / FAILED
    rf.require(rf.inventory(failed) == failed_files, 'Previous failed acceptance changed')
    bad = rf.read(failed / 'outcome.json')
    rf.require(bad['exit_code'] == 1 and bad['abort_reason'] is None and
               "ModuleNotFoundError: No module named 'duckdb'" in (failed / 'worker.log').read_text(encoding='utf-8'),
               'Unexpected previous acceptance failure')
    rf.require(not (failed / 'worker_result.json').exists(), 'Failed attempt has success marker')
    others = [p for p in (run / 'monitor').glob('ACCEPT-*') if p.name != FAILED]
    rf.require(len(others) == 1, 'Exactly one explicitly authorized new acceptance required')
    accepted = others[0]
    outcome = rf.read(accepted / 'outcome.json')
    guard = rf.read(accepted / 'guard_verified.json')
    result = rf.read(accepted / 'worker_result.json')
    rf.require(outcome['exit_code'] == 0 and outcome['abort_reason'] is None and outcome['guard_verified'] and
               outcome['job_commit_limit_bytes'] == 8 * rf.GIB and outcome['logical_cpu_limit'] == 4 and
               guard['verified_before_GO'] and guard['job_commit_limit_bytes'] == 8 * rf.GIB and guard['logical_cpu_limit'] == 4,
               'New acceptance did not complete under required guards')
    rf.require(result == {'status': 'rf_accept_complete', 'test_accessed': False, 'summary_sha256': summary_sha},
               'New acceptance result/summary mismatch')
    return accepted


def prepare():
    environment = assert_environment()
    config = rf.validate(rf.read(rf.CONFIG))
    rf.require_execution(config, 'accept')
    rf.require(rf.sha(rf.CONFIG) == CONFIG_SHA and rf.read(RUN / 'config.json') == config, 'Config differs')
    with rf.resource_slot(), rf.run_lock(RUN), freeze_files(artifact_paths(RUN).values()):
        rf.require(not (RUN / 'summary.json').exists() and not (RUN / 'MANIFEST.json').exists(), 'Existing acceptance/seal')
        rf.require([p.name for p in (RUN / 'monitor').glob('ACCEPT-*')] == [FAILED], 'Unexpected acceptance attempts')
        rf.require(rf.binding(config) == rf.read(RUN / 'binding.json'), 'Current data/code/environment identity differs')
        baseline = file_inventory(RUN)
        failed_files = rf.inventory(RUN / 'monitor' / FAILED)
        RECOVERY.mkdir(parents=True, exist_ok=False)
        (RECOVERY / 'rf_acceptance_recovery.py').write_bytes(Path(__file__).read_bytes())
        rf.write(RECOVERY / 'environment_check.json', environment)
        rf.write(RECOVERY / 'preflight.json', {
            'scope': 'explicitly authorized second acceptance with original .venv; no retraining',
            'original_files': baseline, 'failed_attempt': FAILED, 'failed_files': failed_files,
            'config_sha256': CONFIG_SHA, 'recovery_script_sha256': rf.sha(Path(__file__)),
            'previous_failure_preserved': True, 'test_accessed': False})
    print(json.dumps({'preflight': 'PASS', 'original_files_pinned': len(baseline), 'environment': environment['imported_versions']}))


def seal():
    assert_environment()
    config = rf.validate(rf.read(rf.CONFIG))
    rf.require_execution(config, 'run')
    rf.require(rf.sha(rf.CONFIG) == CONFIG_SHA and rf.read(RUN / 'config.json') == config, 'Config differs')
    with rf.resource_slot(), rf.run_lock(RUN):
        preflight = rf.read(RECOVERY / 'preflight.json')
        rf.require(rf.sha(Path(__file__)) == preflight['recovery_script_sha256'] == rf.sha(RECOVERY / 'rf_acceptance_recovery.py'), 'Recovery code differs')
        baseline_unchanged(RUN, preflight['original_files'])
        summary_sha = rf.sha(RUN / 'summary.json')
        accepted = accepted_attempt(RUN, preflight['failed_files'], summary_sha)
        summary = rf.read(RUN / 'summary.json')
        resources = []
        for entry in summary['tasks']:
            job = rf.read(RUN / 'completed' / f"RF_FIXED_seed_{entry['seed']}.json")
            outcome = rf.read(RUN / job['monitor_directory'] / 'outcome.json')
            rf.require(job['validation_ap'] == entry['validation_ap'], 'AP mismatch')
            rf.require(outcome['exit_code'] == 0 and outcome['abort_reason'] is None and outcome['guard_verified'] and
                       outcome['job_commit_limit_bytes'] == 8 * rf.GIB and outcome['logical_cpu_limit'] == 4,
                       'Training guard/outcome invalid')
            counters = outcome['last_process_counters']
            resources.append({'seed': entry['seed'], 'original_ap': job['validation_ap'],
                              'independently_recomputed_ap': entry['validation_ap'], 'ap_exact_match': True,
                              'training_seconds': job['training_seconds'], 'monitor_outcome': outcome,
                              'mean_cpu_core_equivalent': (counters['user_cpu_seconds'] + counters['kernel_cpu_seconds']) / outcome['elapsed_seconds']})
        rf.write(RECOVERY / 'acceptance_receipt.json', {
            'successful_attempt': accepted.relative_to(RUN).as_posix(), 'failed_attempt_retained': FAILED,
            'summary_sha256': summary_sha, 'original_files_verified': len(preflight['original_files']),
            'per_seed': resources, 'AP_exact_matches': 5, 'test_accessed': False,
            'seal_compatibility': 'All original seal checks retained; exactly one preserved failed attempt plus one guarded successful attempt allowed.'})

        def checks(files):
            baseline_unchanged(RUN, preflight['original_files'])
            identity = rf.read(RUN / 'binding.json')
            rf.require(identity == rf.binding(config), 'Current identity differs')
            rf.require(summary['status'] == 'rf_validation_acceptance_complete' and summary['binding_sha256'] == rf.digest(identity), 'Acceptance missing')
            rf.require(summary['paired_reference'] == rf.read(RUN / 'paired_reference.json'), 'Pairing changed')
            rf.require(rf.read(RUN / 'checkpoint.json') == {'action': 'CONTINUE', 'binding_sha256': rf.digest(identity)}, 'Final checkpoint missing')
            rf.require_seed_checkpoints(RUN, config, identity)
            rf.require({p.stem for p in (RUN / 'completed').glob('*.json')} == {t['task_id'] for t in rf.tasks(config)}, 'Exactly five tasks required')
            for task in rf.tasks(config):
                path = RUN / 'completed' / (task['task_id'] + '.json')
                job = rf.read(path)
                rf.require(job['task'] == task and job['binding_sha256'] == rf.digest(identity), 'Task identity differs')
                rf.require(rf.inventory(rf.safe_path(RUN, job['artifact_directory'])) == job['files'], 'Task artifacts differ')
                entry = next(r for r in summary['tasks'] if r['seed'] == task['seed'])
                rf.require(entry['model_prediction_hashes'] == job['files'] and entry['validation_ap'] == job['validation_ap'] and
                           entry['completed_record_sha256'] == rf.sha(path), 'Accepted task record differs')
            rf.require(rf.sha(RUN / 'summary.json') == summary_sha, 'Summary changed')
            accepted_attempt(RUN, preflight['failed_files'], summary_sha)
            cache = rf.read(RUN / 'cache.json')
            rf.require(rf.inventory(rf.safe_path(RUN, cache['directory'])) == cache['files'], 'Cache differs')
            for name, expected in identity['code'].items():
                rf.require(rf.sha(RUN / 'sources' / name) == expected, 'Archived source differs')
            rf.require(rf.sha(rf.safe_path(ROOT, config['reference']['run']) / 'MANIFEST.json') == config['reference']['manifest_sha256'], 'GBDT reference changed')
        manifest_sha = rf.atomic_manifest(RUN, 'rf_auxiliary_validation_complete', checks)
    # Independent second pass with new locks/handles and fresh full inventory.
    verification = rf.verify_run(RUN, manifest_sha)
    rf.write(ROOT / '.tmp/rf_acceptance_final_verification.json', verification)
    print(json.dumps({'summary_sha256': summary_sha, **verification}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'seal'])
    args = parser.parse_args()
    {'prepare': prepare, 'seal': seal}[args.stage]()
