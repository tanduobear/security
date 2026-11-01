"""M1 data adapters. P2 owns preprocessing; these adapters never fit it."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.model_loader import ModelDataset
from src.preprocess import METADATA_COLUMNS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f'{name} must be an integer >= {minimum}')
    return value


def row_digest(ids):
    digest = hashlib.sha256()
    for value in ids:
        require(isinstance(value, str) and bool(value.strip()), 'Invalid row ID')
        encoded = value.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return digest.hexdigest()


def check_features(names):
    names = tuple(names)
    forbidden = set(METADATA_COLUMNS) | {'Label', 'Day', 'Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'id'}
    require(bool(names) and len(set(names)) == len(names), 'Invalid feature whitelist')
    require(all(isinstance(x, str) and x for x in names), 'Invalid feature name')
    require(not forbidden.intersection(names), 'Metadata must not enter model features')
    return names


def checked_batch(X, y, metadata, names):
    check_features(names)
    X, y = np.asarray(X), np.asarray(y)
    require(X.ndim == 2 and X.shape == (len(y), len(names)), 'Feature whitelist/shape mismatch')
    require(X.dtype.kind in 'fiu' and np.isfinite(X).all(), 'Invalid feature values')
    require(y.ndim == 1 and np.isin(y, [0, 1]).all(), 'Invalid binary labels')
    require(len(metadata) == len(y) and metadata.columns.is_unique, 'Invalid metadata alignment')
    require({'row_index', 'row_id'}.issubset(metadata.columns), 'Missing row identity')
    require(metadata['row_id'].map(lambda x: isinstance(x, str) and bool(x.strip())).all(), 'Invalid row IDs')
    require(metadata['row_id'].is_unique, 'Duplicate row IDs in batch')
    if 'y_binary' in metadata:
        require(np.array_equal(y, metadata['y_binary'].to_numpy()), 'Metadata labels disagree')
    return X, y


@dataclass
class ModelInput:
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple
    split: str
    class_weights: dict

    def close(self):
        """Release read-only memmap handles explicitly (important on Windows)."""
        for array in (self.X, self.y):
            if isinstance(array, np.memmap) and not array._mmap.closed:
                array._mmap.close()

    def __enter__(self):
        return self.validate()

    def __exit__(self, *exc):
        self.close()

    def validate(self):
        self.feature_names = check_features(self.feature_names)
        require(self.X.ndim == 2 and self.X.shape == (len(self.y), len(self.feature_names)), 'Input shape mismatch')
        require(self.X.dtype.kind in 'fiu' and self.y.ndim == 1, 'Invalid model input dtype/dimension')
        # Chunk checks avoid a full-size temporary boolean matrix on a memmap.
        for offset in range(0, len(self.y), 16384):
            require(np.isfinite(self.X[offset:offset + 16384]).all(), 'Nonfinite model input')
            require(np.isin(self.y[offset:offset + 16384], [0, 1]).all(), 'Invalid model labels')
        require(len(self.y) > 0, 'Empty model input')
        require(set(self.class_weights) == {0, 1} and all(np.isfinite(v) and v > 0 for v in self.class_weights.values()), 'Invalid P2 class weights')
        return self


def assert_isolated(train_ids, validation_ids):
    require(len(set(train_ids)) == len(train_ids), 'Duplicate training row IDs')
    require(len(set(validation_ids)) == len(validation_ids), 'Duplicate validation row IDs')
    require(not set(train_ids).intersection(validation_ids), 'Training and validation overlap')


def sample_batches(batches, names, limit, seed):
    """Uniform random-priority sample; feature/metadata buffers allocated once.

    Only keys and slot indices are concatenated per chunk. Retained feature
    rows are replaced in place, rather than repeatedly copying k x p arrays.
    The selected rows are invariant to batch boundaries for a fixed row order.
    """
    integer(limit, 'sample rows', 1)
    integer(seed, 'sampling seed')
    names = check_features(names)
    rng = np.random.default_rng(seed)
    keys = np.full(limit, np.inf)
    X_kept = np.empty((limit, len(names)), dtype=np.float32)
    y_kept = np.empty(limit, dtype=np.uint8)
    columns, metadata_kept, seen = None, None, 0
    for X, y, metadata in batches:
        X, y = checked_batch(X, y, metadata, names)
        if not len(y):
            continue
        if columns is None:
            columns = tuple(metadata.columns)
            metadata_kept = {c: np.empty(limit, dtype=object) for c in columns}
        require(tuple(metadata.columns) == columns, 'Metadata columns changed between batches')
        priorities = rng.random(len(y))
        selected = np.argpartition(np.concatenate((keys, priorities)), limit - 1)[:limit]
        survived = selected[selected < limit]
        incoming = selected[selected >= limit] - limit
        replace_mask = np.ones(limit, dtype=bool)
        replace_mask[survived] = False
        slots = np.flatnonzero(replace_mask)
        keys[slots] = priorities[incoming]
        X_kept[slots], y_kept[slots] = X[incoming], y[incoming]
        for column in columns:
            metadata_kept[column][slots] = metadata[column].to_numpy()[incoming]
        seen += len(y)
    require(seen >= limit and np.isfinite(keys).all(), 'Requested sample exceeds available rows')
    frame = pd.DataFrame(metadata_kept)
    order = np.argsort(frame['row_index'].to_numpy(), kind='stable')
    frame = frame.iloc[order].reset_index(drop=True)
    require(frame['row_id'].is_unique and np.unique(y_kept).size == 2, 'Sample needs unique IDs and both classes')
    return X_kept[order], y_kept[order], frame, seen


def preparation_path(root, version, protocol, variant, split):
    require(version in ('original', 'corrected') and protocol in ('stratified', 'temporal'), 'Unknown dataset protocol')
    require(variant in ('main', 'no_destination_port'), 'Unknown dataset variant')
    # No preparation API can resolve/open a real test dataset.
    require(split in ('train', 'val'), 'Preparation is restricted to train/val')
    path = Path(root) / 'data/split' / version / protocol
    if variant != 'main':
        path /= variant
    return path / (split + '.parquet')


@dataclass
class DiskDataset:
    directory: Path
    feature_names: tuple
    rows: int
    split: str
    class_weights: dict

    def model_input(self):
        return ModelInput(np.load(self.directory / 'X.npy', mmap_mode='r'),
                          np.load(self.directory / 'y.npy', mmap_mode='r'),
                          self.feature_names, self.split, self.class_weights)

    def metadata_batches(self, rows=16384):
        return pd.read_json(self.directory / 'metadata.jsonl', lines=True, chunksize=rows)


def materialize_preparation(path, directory, *, loader_factory=None):
    """Optional future full-data preparation, not a training operation.

    Preallocates float32/uint8 disk arrays once and streams metadata to JSONL.
    Does not promise zero-copy model fitting: sklearn may allocate internally.
    This function is only exercised against tiny fake loaders by unit tests.
    """
    path, directory = Path(path), Path(directory)
    require(path.name in ('train.parquet', 'val.parquet'), 'Test data access is disabled')
    loader = (loader_factory or ModelDataset)(path)
    return _materialize_loaded(loader, path, directory)


def _materialize_loaded(loader, path, directory):
    """Internal streaming writer; final evaluation must gate before constructing its loader."""
    path, directory = Path(path), Path(directory)
    names = check_features(loader.feature_names)
    split = path.stem
    rows = integer(loader.manifest['output'][split]['rows'], 'source rows', 1)
    directory.mkdir(parents=True, exist_ok=False)
    X_file = y_file = None
    try:
        X_file = np.lib.format.open_memmap(directory / 'X.npy', mode='w+', dtype=np.float32, shape=(rows, len(names)))
        y_file = np.lib.format.open_memmap(directory / 'y.npy', mode='w+', dtype=np.uint8, shape=(rows,))
        offset, last_index = 0, None
        with (directory / 'metadata.jsonl').open('x', encoding='utf-8') as stream:
            for X, y, metadata in loader:
                X, y = checked_batch(X, y, metadata, names)
                if not len(y):
                    continue
                indices = metadata['row_index'].to_numpy(dtype=np.int64)
                require((np.diff(indices) > 0).all() and (last_index is None or indices[0] > last_index), 'Source row indices must increase')
                require(offset + len(y) <= rows, 'Source exceeds manifest count')
                X_file[offset:offset + len(y)], y_file[offset:offset + len(y)] = X, y
                stream.write(metadata.to_json(orient='records', lines=True, force_ascii=False))
                offset += len(y)
                last_index = indices[-1]
        require(offset == rows, 'Source/manifest row count mismatch')
        X_file.flush()
        y_file.flush()
        with (directory / 'READY.json').open('x', encoding='utf-8') as stream:
            json.dump({'rows': rows, 'features': names, 'split': split, 'class_weights': loader.class_weights,
                       'source_path': str(path), 'source_manifest_output': loader.manifest['output'][split]}, stream)
    except Exception:
        (directory / 'FAILED').touch(exist_ok=False)
        raise
    finally:
        for array in (X_file, y_file):
            if isinstance(array, np.memmap) and not array._mmap.closed:
                array._mmap.close()
    return DiskDataset(directory, names, rows, split, loader.class_weights)


def assert_disk_isolated(train, validation, *, memory_limit_mb):
    """Exact ID uniqueness/overlap checks with a bounded DuckDB memory pool."""
    integer(memory_limit_mb, 'DuckDB memory limit MB', 1)
    require(train.split == 'train' and validation.split == 'val', 'Wrong fold roles')
    with duckdb.connect(config={'memory_limit': f'{memory_limit_mb}MB', 'threads': 1}) as con:
        for name, item in [('training_ids', train), ('validation_ids', validation)]:
            con.read_json(str(item.directory / 'metadata.jsonl'), format='newline_delimited').project('row_id').create_view(name)
            count, distinct = con.execute(f'SELECT count(*),count(DISTINCT row_id) FROM {name}').fetchall()[0]
            require(count == distinct == item.rows, 'Duplicate or missing cached row IDs')
        overlap = con.execute('SELECT count(*) FROM training_ids JOIN validation_ids USING(row_id)').fetchall()[0][0]
        require(overlap == 0, 'Training and validation overlap')
