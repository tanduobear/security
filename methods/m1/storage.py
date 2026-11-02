"""Exclusive per-run files and a content-addressed manifest, not mutable aliases."""
from datetime import datetime, timezone
from importlib.metadata import distributions
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import uuid

import joblib
import numpy as np

from .data import require


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
            'packages': dict(sorted((d.metadata['Name'], d.version) for d in distributions() if d.metadata['Name']))}


class RunStore:
    def __init__(self, root, *, kind, run_id=None):
        run_id = run_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex)
        require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', run_id)), 'Invalid run ID')
        require(kind in ('smoke', 'prepared', 'search', 'selection', 'evaluation'), 'Unknown run kind')
        self.path = Path(root) / kind / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.kind, self.sealed = kind, False
        self.write_json('run.json', {'run_id': run_id, 'kind': kind, 'created_at': datetime.now(timezone.utc).isoformat()})
        self.write_json('environment.json', environment())

    def file(self, name):
        require(not self.sealed, 'Run is sealed')
        require(name not in ('MANIFEST.json', 'FAILED.json'), 'Reserved run file')
        path = self.path / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.resolve().is_relative_to(self.path.resolve()), 'Invalid artifact path')
        require(not path.exists(), 'Artifact already exists')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, name, value):
        encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
        with self.file(name).open('x', encoding='utf-8') as stream:
            stream.write(encoded)

    def write_model(self, name, model):
        with self.file(name).open('xb') as stream:
            joblib.dump(model, stream)

    def write_predictions(self, arrays):
        with self.file('predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)

    def finish(self):
        require(not self.sealed and not (self.path / 'FAILED.json').exists(), 'Run cannot be completed')
        required = ['run.json', 'environment.json', 'config.json']
        required += {'smoke':['metrics.json','predictions.npz'], 'prepared':['plan.json'],
                     'search':['inputs.json','jobs.json','search_summary.json','events.jsonl','validation.npz','timing_X.npy','timing_rows.json'],
                     'selection':['selection.json','thresholds.json','search_reference.json'],
                     'evaluation':['evaluation_index.json','selection_reference.json','events.jsonl']}[self.kind]
        require(all((self.path / name).is_file() for name in required), 'Required artifacts are missing')
        if self.kind == 'smoke':
            record = json.loads((self.path / 'metrics.json').read_text(encoding='utf-8'))
            require(all((self.path / 'models' / (name + '.joblib')).is_file() for name in record['models']), 'Model artifacts are missing')
        files = {p.relative_to(self.path).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(self.path.rglob('*')) if p.is_file()}
        with (self.path / 'MANIFEST.json').open('x', encoding='utf-8') as stream:
            status = {'smoke':'complete','prepared':'prepared_not_executed','search':'validation_search_complete',
                      'selection':'selection_frozen','evaluation':'test_evaluation_complete'}[self.kind]
            json.dump({'status': status, 'files': files}, stream, indent=2)
        self.sealed = True

    def fail(self, error):
        if not self.sealed:
            with (self.path / 'FAILED.json').open('x', encoding='utf-8') as stream:
                json.dump({'status': 'failed', 'error_type': type(error).__name__, 'message': str(error)}, stream)
            self.sealed = True


def verify_run(directory):
    directory = Path(directory)
    require(not (directory / 'FAILED.json').exists(), 'Failed run cannot be published')
    manifest = json.loads((directory / 'MANIFEST.json').read_text(encoding='utf-8'))
    require(manifest['status'] in ('complete', 'prepared_not_executed', 'validation_search_complete', 'selection_frozen', 'test_evaluation_complete'), 'Incomplete run')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'}
    require(actual == set(manifest['files']), 'Run artifact set disagrees')
    for name, item in manifest['files'].items():
        path = directory / name
        require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()), 'Unsafe artifact path')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Run artifact hash disagrees')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if manifest['status'] == 'complete':
        from .validation import validate_record, validate_saved_predictions
        record = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
        require(record['config'] == config, 'Configuration/result binding disagrees')
        schema = json.loads((Path(__file__).resolve().parents[2] / 'eval/schema.json').read_text(encoding='utf-8-sig'))
        validate_record(record, schema)
        require(all((directory / 'models' / (name + '.joblib')).is_file() for name in record['models']), 'Model artifacts are missing')
        with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
            validate_saved_predictions(record, saved)
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify run hashes, smoke metrics and saved predictions without fitting a model.')
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    print(verify_run(args.run_directory)['status'])
