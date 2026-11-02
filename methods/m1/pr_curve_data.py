"""Read-only, hash-bound G05 Validation curve data for figure exports."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 43, 44, 45, 46]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def row_digest(ids):
    result = hashlib.sha256()
    for value in ids:
        require(isinstance(value, str) and value.strip(), 'Invalid row ID')
        encoded = value.encode('utf-8')
        result.update(len(encoded).to_bytes(8, 'little'))
        result.update(encoded)
    return result.hexdigest()


def verified_curve(ids, y, scores, *, expected_rows, expected_row_digest, expected_ap, reference=None):
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve
    require(ids.ndim == y.ndim == scores.ndim == 1 and len(ids) == len(y) == len(scores) == expected_rows,
            'Prediction shape/sample count mismatch')
    require(ids.dtype.kind == 'U' and len(set(ids)) == expected_rows, 'Missing/duplicate row IDs')
    require(row_digest(ids) == expected_row_digest, 'Validation row ID digest mismatch')
    require(np.isin(y, [0, 1]).all() and set(np.unique(y)) == {0, 1}, 'Invalid binary labels')
    require(scores.dtype.kind == 'f' and np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
            'Invalid attack-probability scores')
    require(not np.isin(scores, [0., 1.]).all(), 'Hard prediction labels cannot stand in for continuous scores')
    if reference is not None:
        require(np.array_equal(ids, reference[0]) and np.array_equal(y, reference[1]), 'Cross-seed row order/labels differ')
    ap = float(average_precision_score(y, scores, pos_label=1))
    require(ap == expected_ap, f'Independent AP mismatch: {ap!r} != {expected_ap!r}')
    precision, recall, thresholds = precision_recall_curve(y, scores, pos_label=1)
    return {'precision': precision, 'recall': recall, 'average_precision': ap,
            'curve_points': len(precision), 'unique_score_thresholds': len(thresholds),
            'rows': len(y), 'benign_rows': int(np.count_nonzero(y == 0)),
            'attack_rows': int(np.count_nonzero(y == 1)),
            'row_ids_sha256': row_digest(ids),
            'labels_uint8_sha256': hashlib.sha256(np.asarray(y, dtype=np.uint8).tobytes()).hexdigest()}


def safe_path(run, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe archive path')
    result = run / path
    require(result.resolve().is_relative_to(run.resolve()), 'Archive path escaped root')
    return result


@contextmanager
def verified_g05_curves():
    import numpy as np
    from tools.aggregate_m1 import validate_document
    from .search_gbdt import resource_slot
    from .seal_gbdt import artifact_paths, file_inventory, freeze_files
    root = ROOT
    m1_path = root / 'results/m1.json'
    m1 = json.loads(m1_path.read_bytes())
    validate_document(m1)
    require(m1['gbdt']['selected_candidate_id'] == 'G05' and m1['scope']['validation_only'], 'Expected Validation G05')
    run_relative = 'results/m1_gbdt_search/' + m1['gbdt']['archive']['run_id']
    run = safe_path(root, run_relative)
    with resource_slot(), freeze_files([*artifact_paths(run).values(), run / 'MANIFEST.json', m1_path]):
        require(sha(run / 'MANIFEST.json') == m1['gbdt']['archive']['manifest_sha256'] and
                         sha(run / 'selection.json') == m1['gbdt']['archive']['selection_sha256'], 'Archive pin mismatch')
        manifest = json.loads((run / 'MANIFEST.json').read_bytes())
        require(file_inventory(run) == manifest['files'] and len(manifest['files']) == 302, 'Archive SHA256 mismatch')
        identity = json.loads((run / 'binding.json').read_bytes())
        config = json.loads((run / 'config.json').read_bytes())
        require(config == m1['gbdt']['execution_config'] and identity['config_sha256'] == digest(config) and
                         identity['inputs'] == m1['dataset']['input_file_sha256'], 'Data/config identity differs')
        selection = json.loads((run / 'selection.json').read_bytes())
        require(selection['binding_sha256'] == digest(identity) and selection['selected_candidate_id'] == 'G05',
                         'Selection binding differs')
        aps = {t['seed']: t['validation_ap'] for t in m1['gbdt']['selected_candidate']['tasks']}
        curves, reference = [], None
        for seed in SEEDS:
            job = json.loads((run / 'completed' / f'G05_seed_{seed}.json').read_bytes())
            expected = {'task_id': f'G05_seed_{seed}', 'candidate_id': 'G05', 'seed': seed, 'algorithm': 'hist_gradient_boosting',
                        'params': m1['gbdt']['selected_candidate']['parameters']}
            require(job['task'] == expected and job['binding_sha256'] == digest(identity) and
                             job['validation_rows'] == m1['dataset']['validation_rows'] and job['validation_ap'] == aps[seed] and
                             job['test_accessed'] is False, 'Task identity/AP differs')
            directory = safe_path(run, job['artifact_directory'])
            require(directory.relative_to(run).parts[0] == 'attempts' and json.loads((directory / 'task.json').read_bytes()) == expected,
                             'Saved task/path differs')
            for name, entry in job['files'].items():
                path = safe_path(directory, name)
                require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'Task artifact hash/size differs')
            with np.load(directory / 'validation.npz', allow_pickle=False) as saved:
                ids, labels, scores = saved['row_ids'], saved['y_true'], saved['y_score']
            curve = verified_curve(ids, labels, scores, expected_rows=job['validation_rows'],
                                            expected_row_digest=job['validation_row_ids_sha256'], expected_ap=aps[seed], reference=reference)
            if reference is None:
                reference = (ids.copy(), labels.copy())
            curves.append({**curve, 'seed': seed, 'prediction_sha256': job['files']['validation.npz']['sha256'],
                           'prediction_path': (directory / 'validation.npz').relative_to(run).as_posix()})
            print(f"Seed {seed}: rows={curve['rows']}; AP={curve['average_precision']!r}; exact AP/label/row-ID match", flush=True)
            del ids, labels, scores
        metadata = {'run_relative': run_relative, 'manifest_sha256': sha(run / 'MANIFEST.json'),
                    'selection_sha256': sha(run / 'selection.json'), 'config_sha256': sha(run / 'config.json'),
                    'm1_sha256': sha(m1_path), 'input_file_sha256': identity['inputs'],
                    'parameters': m1['gbdt']['selected_candidate']['parameters'], 'archive_files_verified': len(manifest['files'])}
        yield curves, metadata
        require(file_inventory(run) == manifest['files'], 'Archive changed during plotting')
