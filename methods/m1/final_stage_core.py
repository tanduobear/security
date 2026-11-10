"""Worker-only implementation. Reuses sealed train-fold models, never fits."""
from contextlib import contextmanager, ExitStack
import gc
import hashlib
import platform
from pathlib import Path
import sys

from .final_stage import ROOT, digest, read, write, sha, require, safe_path, code_identity, require_approval


@contextmanager
def verified_sources(config, project=ROOT):
    """Freeze both independent archives; validate historical and current identities."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    from tools.aggregate_m1 import validate_document
    from .seal_gbdt import artifact_paths, freeze_files, file_inventory
    project = Path(project)
    require(sha(project / 'results/m1.json') == config['m1_summary_sha256'], 'Public M1 pin mismatch')
    document = read(project / 'results/m1.json')
    validate_document(document, root=project)
    unified = read(project / 'methods/m1/protocol.unified.v1.json')
    require(unified['approval']['status'] == 'approved' and unified['sha256'] == config['unified_protocol_sha256'] and
            digest({k: v for k, v in unified.items() if k != 'sha256'}) == unified['sha256'], 'Unified protocol mismatch')
    require(document['gbdt']['selected_candidate_id'] == 'G05', 'Frozen winner differs')
    require(document['scope']['validation_only'], 'Source must be Validation-only')
    expected_inputs = document['dataset']['input_file_sha256']
    data_dir = project / 'data/split/original/stratified'
    runs = {name: safe_path(project, spec['path']) for name, spec in config['archives'].items()}
    pins = {name: sha(path) for name, path in [(name, project / name) for name in code_identity(project)]}
    external = [project / name for name in pins] + [data_dir / name for name in expected_inputs]
    external += [project / 'results/m1.json', project / 'methods/m1/protocol.unified.v1.json']
    external += [safe_path(project, x['path']) for x in document['provenance']['sources'].values()]
    external += [safe_path(project, config['timing']['row_ids_path'])]
    paths = {name: artifact_paths(run) for name, run in runs.items()}
    with freeze_files([*external, *[p for name, run in runs.items()
                       for p in [*paths[name].values(), run / 'MANIFEST.json']]]):
        require(all(sha(project / name) == value for name, value in pins.items()), 'Evaluation source changed before freeze')
        validate_document(document, root=project)
        manifests, jobs = {}, []
        reference = None
        for name, run in runs.items():
            spec = config['archives'][name]
            require(sha(run / 'MANIFEST.json') == spec['manifest_sha256'], 'Archive MANIFEST pin differs')
            public_pin = document['gbdt']['archive']['manifest_sha256'] if name == 'GBDT' else document['rf']['historical_verification']['RF_manifest_sha256']
            require(spec['manifest_sha256'] == public_pin, 'Archive differs from public evidence')
            manifest = read(run / 'MANIFEST.json')
            expected_status = 'gbdt_search_and_validation_selection_complete' if name == 'GBDT' else 'rf_auxiliary_validation_complete'
            require(manifest['status'] == expected_status and file_inventory(run) == manifest['files'], 'Archive status/hash mismatch')
            manifests[name] = manifest
            binding, archived_config = read(run / 'binding.json'), read(run / 'config.json')
            require(binding['inputs'] == expected_inputs and binding['config_sha256'] == digest(archived_config), 'Training data/config identity differs')
            expected_config = document['gbdt' if name == 'GBDT' else 'rf']['execution_config']
            require(archived_config == expected_config, 'Published training configuration differs')
            require(binding['unified_file_sha256'] == sha(project / 'methods/m1/protocol.unified.v1.json'), 'Unified identity differs')
            # Historical source snapshots remain untouched; core P2/P3/model adapters must match.
            for relative in ('src/model_loader.py', 'src/preprocess.py', 'eval/metrics.py',
                             'eval/schema.json', 'eval/validate.py', 'methods/m1/models.py',
                             'methods/m1/data.py', 'methods/m1/evaluation.py'):
                require(sha(project / relative) == binding['code'][relative], 'Core training dependency changed: ' + relative)
            check_environment(binding['environment'])
            model_name = 'G05' if name == 'GBDT' else 'RF_FIXED'
            if name == 'GBDT':
                require(sha(run / 'selection.json') == spec['selection_sha256'], 'Original selection changed')
                selection = read(run / 'selection.json')
                require(selection['selected_candidate_id'] == 'G05' and selection['binding_sha256'] == digest(binding), 'Winner/binding mismatch')
                result = document['gbdt']['selected_candidate']
                expected_ap = {x['seed']: x['validation_ap'] for x in result['tasks']}
                expected_params = result['parameters']
            else:
                result = document['rf']['validation_results']
                expected_ap = {x['seed']: x['validation_ap'] for x in result['tasks']}
                expected_params = expected_config['parameters']
            for seed in config['seeds']:
                task_id = f'{model_name}_seed_{seed}'
                job = read(run / 'completed' / (task_id + '.json'))
                task = job['task']
                require(task['task_id'] == task_id and task['seed'] == seed and task['params'] == expected_params and
                        job['binding_sha256'] == digest(binding) and not job['test_accessed'], 'Task identity mismatch')
                require(task['algorithm'] == ('hist_gradient_boosting' if name == 'GBDT' else 'random_forest'), 'Algorithm mismatch')
                directory = safe_path(run, job['artifact_directory'])
                require(directory.relative_to(run).parts[0] == 'attempts' and read(directory / 'task.json') == task, 'Task directory mismatch')
                require(set(job['files']) == {'model.joblib', 'task.json', 'validation.npz'}, 'Unexpected task artifacts')
                for relative, item in job['files'].items():
                    require(manifest['files'][str((directory / relative).relative_to(run).as_posix())] == item, 'Task manifest binding mismatch')
                with np.load(directory / 'validation.npz', allow_pickle=False) as saved:
                    ids, y, scores = saved['row_ids'], saved['y_true'], saved['y_score']
                    from .data import row_digest
                    require(ids.ndim == y.ndim == scores.ndim == 1 and len(y) == document['dataset']['validation_rows'], 'Validation shape mismatch')
                    require(len(set(ids)) == len(ids) and np.isin(y, [0, 1]).all() and set(np.unique(y)) == {0, 1}, 'Invalid Validation labels/IDs')
                    require(scores.dtype.kind == 'f' and np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), 'Invalid Validation scores')
                    require(float(average_precision_score(y, scores)) == job['validation_ap'] == expected_ap[seed], 'Independent Validation AP mismatch')
                    if reference is None:
                        reference = (ids.copy(), y.copy())
                    require(np.array_equal(ids, reference[0]) and np.array_equal(y, reference[1]), 'Models used different Validation populations')
                    if name == 'GBDT':
                        require(row_digest(ids) == job['validation_row_ids_sha256'], 'Recorded row order differs')
                jobs.append({**task, 'role': 'main' if name == 'GBDT' else 'auxiliary', 'candidate_id': model_name,
                             'model_path': directory / 'model.joblib', 'prediction_path': directory / 'validation.npz',
                             'model_sha256': job['files']['model.joblib']['sha256'],
                             'prediction_sha256': job['files']['validation.npz']['sha256'],
                             'training_seconds': job['training_seconds'], 'validation_ap': job['validation_ap']})
        for relative, expected in expected_inputs.items():
            require(sha(data_dir / relative) == expected, 'Current P2 input differs: ' + relative)
        manifest = read(data_dir / 'preprocess_manifest.json')
        require(manifest['status'] == 'verified_train_only_and_model_data_complete' and
                manifest['precision_guard']['checks']['no_transformed_group_overlap'], 'P2 verification missing')
        require(len(manifest['features']) == 140, 'P2 feature count differs')
        families = [x for x in read(project / config['attack_families_source'])['family_names'] if x != 'benign']
        require(len(families) == len(set(families)) == 14, 'Full family contract differs')
        cache = safe_path(runs['GBDT'], read(runs['GBDT'] / 'cache.json')['directory'])
        source = {'jobs': jobs, 'inputs': expected_inputs, 'features': manifest['features'], 'families': families,
                  'val_ids': reference[0], 'val_y': reference[1], 'cache': cache, 'data_dir': data_dir,
                  'p2_manifest': manifest, 'root': project, 'code': pins,
                  'm1_sha256': sha(project / 'results/m1.json'), 'archive_pins': config['archives']}
        yield source
        for name, run in runs.items():
            require(file_inventory(run) == manifests[name]['files'], 'Original archive changed during evaluation')
        require(all(sha(project / name) == value for name, value in pins.items()), 'Evaluation source changed')


def check_environment(original):
    from importlib.metadata import version
    require(Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve(), 'Use original .venv executable')
    require(platform.python_version() == original['python'], 'Use original training Python')
    for package in ('numpy', 'scikit-learn', 'duckdb', 'joblib'):
        require(version(package) == original['packages'][package], 'Training dependency version differs: ' + package)


def timing_rows(config, source):
    from .data import row_digest
    from .evaluation import timing_indices
    path = safe_path(source['root'], config['timing']['row_ids_path'])
    require(sha(path) == config['timing']['row_ids_sha256'], 'Common timing file hash differs')
    rows = read(path)
    require(rows['source_split'] == 'val' and rows['population_row_ids_sha256'] == row_digest(source['val_ids']), 'Timing population differs')
    chosen = timing_indices(source['val_ids'], batch_size=512, selected_row_ids=rows['row_ids'])
    return chosen, {**rows, 'row_ids_sha256': row_digest(rows['row_ids']), 'source_file_sha256': sha(path)}


def calibration_bundle(config, source):
    """Pure Validation prediction calibration; does not deserialize models."""
    import numpy as np
    from .selection import calibrate_fpr
    from .data import row_digest
    thresholds = []
    for job in source['jobs']:
        require(sha(job['prediction_path']) == job['prediction_sha256'], 'Validation prediction changed')
        with np.load(job['prediction_path'], allow_pickle=False) as saved:
            require(np.array_equal(saved['row_ids'], source['val_ids']) and np.array_equal(saved['y_true'], source['val_y']), 'Calibration population differs')
            for alpha in config['thresholds']['fpr_targets']:
                thresholds.append({'task_id': job['task_id'], 'model_sha256': job['model_sha256'],
                                   'prediction_sha256': job['prediction_sha256'], **calibrate_fpr(saved['y_true'], saved['y_score'], alpha)})
    require(len(thresholds) == 20, 'Exactly twenty thresholds required')
    return {'payload_sha256': digest({k: v for k, v in config.items() if k != 'approvals'}),
            'code_sha256': digest(source['code']), 'source_m1_sha256': source['m1_sha256'],
            'archive_pins': source['archive_pins'], 'inputs': source['inputs'], 'features': source['features'],
            'attack_families': source['families'], 'validation_row_ids_sha256': row_digest(source['val_ids']),
            'validation_labels_sha256': hashlib.sha256(np.asarray(source['val_y'], dtype=np.uint8).tobytes()).hexdigest(),
            'thresholds': thresholds, 'test_accessed': False, 'models_retrained': False}


def calibrate(config, run, source):
    require_approval(config, 'calibrate', launch=True)
    import numpy as np
    chosen, rows = timing_rows(config, source)
    x = np.load(source['cache'] / 'val/X.npy', mmap_mode='r', allow_pickle=False)
    try:
        ready = read(source['cache'] / 'val/READY.json')
        require(list(ready['features']) == source['features'] and x.shape == (len(source['val_y']), 140), 'Timing cache identity differs')
        with (run / 'timing_X.npy').open('xb') as stream:
            np.save(stream, x[chosen], allow_pickle=False)
    finally:
        x._mmap.close()
    rows['benign_rows'] = int((source['val_y'][chosen] == 0).sum())
    rows['attack_rows'] = int((source['val_y'][chosen] == 1).sum())
    write(run / 'timing_rows.json', rows)
    write(run / 'thresholds.json', calibration_bundle(config, source))


def assert_test_isolated(cache, source, run, memory_mb):
    """Exact ID isolation with bounded DuckDB; only after explicit Test authorization."""
    import duckdb
    scratch = run / 'duckdb'
    scratch.mkdir(exist_ok=False)
    with duckdb.connect(config={'memory_limit': f'{memory_mb}MB', 'threads': 1, 'temp_directory': str(scratch)}) as con:
        for name, path in [('heldout', cache.directory / 'metadata.jsonl'),
                           ('training', source['cache'] / 'train/metadata.jsonl'),
                           ('validation', source['cache'] / 'val/metadata.jsonl')]:
            con.read_json(str(path), format='newline_delimited').project('row_id').create_view(name)
        rows, unique = con.execute('SELECT count(*), count(DISTINCT row_id) FROM heldout').fetchone()
        require(rows == unique == cache.rows, 'Duplicate Test row IDs')
        for name in ('training', 'validation'):
            require(con.execute(f'SELECT count(*) FROM heldout JOIN {name} USING(row_id)').fetchone()[0] == 0, 'Test overlaps ' + name)


def evaluate(config, run, source, calibration, *, permit_test=False, loader_factory=None, model_loader=None):
    """Called only by guarded execute; injectable loaders are for synthetic tests."""
    require(permit_test is True, 'Final Test authorization missing')
    require_approval(config, 'evaluate', launch=True, permit_test=permit_test)
    import numpy as np
    frozen = read(calibration / 'thresholds.json')
    require(frozen == calibration_bundle(config, source), 'Frozen Validation thresholds/identity cannot be reproduced')
    chosen, expected_timing = timing_rows(config, source)
    stored_timing = read(calibration / 'timing_rows.json')
    require(all(stored_timing[k] == v for k, v in expected_timing.items()), 'Frozen timing IDs differ')
    timing_x = np.load(calibration / 'timing_X.npy', allow_pickle=False)
    require(timing_x.shape == (512, 140) and np.isfinite(timing_x).all(), 'Timing tensor invalid')
    original_x = np.load(source['cache'] / 'val/X.npy', mmap_mode='r', allow_pickle=False)
    try:
        require(np.array_equal(timing_x, original_x[chosen]), 'Frozen timing features differ')
    finally:
        original_x._mmap.close()
    # No Test loader or file hash may occur above this boundary.
    path = source['data_dir'] / 'test.parquet'
    from .seal_gbdt import freeze_files
    with freeze_files([path]):
        _evaluate_test(config, run, source, frozen, timing_x, stored_timing,
                       loader_factory=loader_factory, model_loader=model_loader)


def _evaluate_test(config, run, source, frozen, timing_x, stored_timing, *, loader_factory=None, model_loader=None):
    import joblib
    import numpy as np
    import pandas as pd
    from .data import _materialize_loaded, row_digest
    from .executor import metadata_arrays, predict_batches, formal_record, summarize_records
    from .models import attack_scores, inference_workers
    from .evaluation import family_coverage, score_metrics
    from eval.metrics import measure_inference
    from src.model_loader import ModelDataset
    path = source['data_dir'] / 'test.parquet'
    expected_test = source['p2_manifest']['output']['test']
    loader = (loader_factory or ModelDataset)(path)
    require(list(loader.feature_names) == source['features'] and sha(path) == expected_test['main']['sha256'], 'Test file/features identity differs')
    cache = _materialize_loaded(loader, path, run / 'cache/test')
    require(cache.rows == expected_test['rows'], 'Test sample count differs')
    assert_test_isolated(cache, source, run, config['resources']['duckdb_memory_limit_mb'])
    ids, families, flags = metadata_arrays(cache)
    metadata = pd.DataFrame({'attack_family': families, 'is_attempted': flags})
    threshold_map = {(x['task_id'], x['target_fpr']): x for x in frozen['thresholds']}
    protocol = read(source['root'] / 'methods/m1/protocol.unified.v1.json')
    records = []
    with cache.model_input() as test:
        coverage = family_coverage(test.y, families, source['families'])
        for job in source['jobs']:
            print('Evaluating', job['task_id'], flush=True)
            require(sha(job['model_path']) == job['model_sha256'], 'Frozen model changed')
            model = (model_loader or joblib.load)(job['model_path'])
            effective = model.get_params(deep=False)
            require(effective.get('random_state') == job['seed'] and model.n_features_in_ == 140 and
                    all(effective.get(k) == v for k, v in job['params'].items()), 'Loaded model parameters/seed differ')
            scores = predict_batches(model, test.X, rows=8192, threads=1)
            with inference_workers(model, 1):
                inference = measure_inference(lambda batch: attack_scores(model, batch), timing_x,
                                              batch_size=512, warmup=5, repeats=30, threads=1)
            base = 'predictions/' + job['task_id'] + '.npz'
            (run / 'predictions').mkdir(exist_ok=True)
            with (run / base).open('xb') as stream:
                np.savez_compressed(stream, row_ids=ids, y_true=test.y, y_score=scores, attack_family=families, is_attempted=flags,
                    y_pred_fpr_01=(scores >= threshold_map[(job['task_id'], .01)]['threshold']).astype(np.uint8),
                    y_pred_fpr_05=(scores >= threshold_map[(job['task_id'], .05)]['threshold']).astype(np.uint8))
            # Re-read exported scores, not just in-memory decisions.
            with np.load(run / base, allow_pickle=False) as saved:
                require(np.array_equal(saved['row_ids'], ids) and np.array_equal(saved['y_true'], test.y) and
                        np.array_equal(saved['y_score'], scores) and np.array_equal(saved['attack_family'], families) and
                        np.array_equal(saved['is_attempted'], flags), 'Saved Test predictions differ')
                for alpha in (0.01, 0.05):
                    threshold = threshold_map[(job['task_id'], alpha)]
                    require(np.array_equal(saved['y_pred_fpr_01' if alpha == .01 else 'y_pred_fpr_05'],
                            (saved['y_score'] >= threshold['threshold']).astype(np.uint8)), 'Saved threshold decisions differ')
                    key = 'records/' + job['task_id'] + '_alpha_' + str(alpha) + '.json'
                    record = formal_record(protocol, config, {**job, 'effective_parameters': effective},
                        {'path': 'data/split/original/stratified/test.parquet', 'sha256': expected_test['main']['sha256']},
                        metadata, saved['y_true'], saved['y_score'], threshold, inference, [],
                        notes='Existing train-fold model reused. Fixed Validation threshold. Empirical Test ROC operating points are descriptive, not threshold calibration. Bootstrap deferred. Timing uses frozen common Validation row IDs.')
                    require(record['metrics'] == score_metrics(saved['y_true'], saved['y_score'], metadata, threshold['threshold'], source['families']), 'Saved-score metrics differ')
                    write(run / key, record)
                    records.append({'path': key, 'method': record['method'], 'seed': job['seed'], 'target_validation_fpr': alpha})
            del model, scores
            gc.collect()
        write(run / 'evaluation_index.json', {'records': records, 'five_seed_summaries': summarize_records(run, records),
              'family_coverage': coverage, 'test_row_ids_sha256': row_digest(ids), 'test_rows': cache.rows,
              'bootstrap': 'deferred_separate_stage', 'training_fits': 0,
              'prediction_format': 'One full score/label/metadata NPZ per model; decisions are score >= frozen threshold.'})
    write(run / 'timing_rows.json', stored_timing)
    write(run / 'thresholds.json', frozen)


def execute(config, request, run, *, project=ROOT, phase=lambda name: None):
    """Production worker: approvals -> identity freeze -> calibration/evaluation."""
    from .final_stage import verify_output
    from .seal_gbdt import artifact_paths, freeze_files
    require_approval(config, request['stage'], launch=request['launch'], permit_test=request['permit_test'], project=project)
    if request['stage'] == 'evaluate':
        require(request.get('calibration') and request.get('calibration_sha256'), 'Missing calibration pin')
    with verified_sources(config, project) as source, ExitStack() as stack:
        require(digest(source['code']) == config['approvals'][request['stage']]['code_sha256'], 'Approved code differs after freeze')
        phase('verified_archives_and_validation')
        if request['stage'] == 'evaluate':
            require(request['calibration'] and request['calibration_sha256'], 'Missing calibration pin')
            calibration = Path(request['calibration']).resolve()
            require(not run.resolve().is_relative_to(calibration), 'Output cannot be inside frozen calibration')
            stack.enter_context(freeze_files([*artifact_paths(calibration).values(), calibration / 'MANIFEST.json']))
            manifest = verify_output(calibration, request['calibration_sha256'])
            require(manifest['status'] == 'validation_thresholds_frozen', 'Calibration stage not frozen')
            calibration_config = read(calibration / 'config.json')
            require_approval(calibration_config, 'calibrate', launch=True, project=project)
            require(digest({k: v for k, v in calibration_config.items() if k != 'approvals'}) ==
                    digest({k: v for k, v in config.items() if k != 'approvals'}), 'Calibration execution payload differs')
            write(run / 'calibration_reference.json', {'local_directory': str(calibration),
                  'manifest_sha256': request['calibration_sha256']})
        else:
            calibration = None
        write(run / 'sources.json', {'code': source['code'], 'm1_sha256': source['m1_sha256'], 'archives': source['archive_pins']})
        for name in source['code']:
            destination = run / 'sources' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write((Path(project) / name).read_bytes())
        from .storage import environment
        write(run / 'environment.json', environment())
        phase('calibrate_validation_only' if calibration is None else 'evaluate_frozen_models')
        if calibration is None:
            calibrate(config, run, source)
        else:
            evaluate(config, run, source, calibration, permit_test=request['permit_test'])
            verify_output(calibration, request['calibration_sha256'])
    write(run / 'worker_result.json', {'status': request['stage'] + '_complete',
          'test_accessed': request['stage'] == 'evaluate', 'training_fits': 0, 'bootstrap_run': False})
