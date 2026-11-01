"""Verify the original stratified main bundle without refitting preprocessing."""
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model_loader import ModelDataset


def verify():
    directory = ROOT / 'data/split/original/stratified'
    manifest = json.loads((directory / 'preprocess_manifest.json').read_text(encoding='utf-8'))
    report = {'version': 'original', 'protocol': 'stratified', 'variant': 'main', 'splits': {}}
    with duckdb.connect() as con:
        for split in ('train', 'val', 'test'):
            expected = manifest['output'][split]
            hashes = {}
            for entry in (expected['main'], expected['effective_index']):
                path = ROOT / entry['path']
                with path.open('rb') as stream:
                    if stream.read(4) != b'PAR1':
                        raise ValueError(f'{path.name}: real Parquet data is not downloaded')
                    stream.seek(0)
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != entry['sha256']:
                    raise ValueError(f'{path.name}: SHA256 mismatch')
                hashes[path.name] = digest
            dataset = ModelDataset(directory / f'{split}.parquet')
            rows = 0
            counts = [0, 0]
            for X, y, metadata in dataset:
                if X.shape != (len(y), len(dataset.feature_names)) or not np.isfinite(X).all():
                    raise ValueError(f'{split}: invalid feature matrix')
                if not np.isin(y, [0, 1]).all() or len(metadata) != len(y):
                    raise ValueError(f'{split}: invalid labels or metadata')
                rows += len(y)
                for label in (0, 1):
                    counts[label] += int((y == label).sum())
            if rows != expected['rows']:
                raise ValueError(f'{split}: unexpected row count')
            model_path = str(directory / f'{split}.parquet')
            index_path = str(directory / f'model_{split}_indices.parquet')
            con.from_parquet(model_path).project('row_index').create_view(f'{split}_model')
            con.from_parquet(index_path).project('row_index').create_view(f'{split}_idx')
            for view in (f'{split}_model', f'{split}_idx'):
                total, distinct = con.execute(f'SELECT count(*), count(DISTINCT row_index) FROM {view}').fetchall()[0]
                if total != rows or distinct != rows:
                    raise ValueError(f'{view}: duplicate or missing row IDs')
            differences = con.execute(f'SELECT count(*) FROM ((SELECT * FROM {split}_model EXCEPT SELECT * FROM {split}_idx) UNION ALL (SELECT * FROM {split}_idx EXCEPT SELECT * FROM {split}_model))').fetchall()[0][0]
            if differences:
                raise ValueError(f'{split}: model and effective index row IDs differ')
            report['splits'][split] = {'rows': rows, 'benign': counts[0], 'attack': counts[1], 'features': len(dataset.feature_names), 'hashes': hashes}
            print(f'{split}: {rows} rows, {len(dataset.feature_names)} features verified', flush=True)
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            overlap = con.execute(f'SELECT count(*) FROM {left}_idx INNER JOIN {right}_idx USING(row_index)').fetchall()[0][0]
            if overlap:
                raise ValueError(f'{left}/{right}: overlapping row IDs')
    report['status'] = 'passed'
    report['scope'] = 'Hash, full streaming load, feature finiteness, labels, effective indices and row ID disjointness; no model training.'
    output = Path(__file__).with_name('primary_data_check.json')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'PASS: {output}')


if __name__ == '__main__':
    verify()
