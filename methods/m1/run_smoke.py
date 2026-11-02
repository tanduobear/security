"""Development train/validation integration. No real test data is opened."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tmp/matplotlib'))

import joblib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from eval.metrics import measure_inference, paired_bootstrap
from src.model_loader import ModelDataset
from methods.m1.data import ModelInput, assert_isolated, integer, require, row_digest, sample_batches
from methods.m1.evaluation import family_coverage, score_metrics, timing_indices
from methods.m1.models import attack_scores, fit_tree, inference_workers
from methods.m1.storage import RunStore, sha, verify_run
from methods.m1.validation import validate_record, validate_saved_predictions


def sample_dataset(path, limit, seed):
    """Compatibility wrapper retaining the original smoke function signature."""
    require(Path(path).name in ('train.parquet', 'val.parquet'), 'Smoke cannot open test data')
    dataset = ModelDataset(path)
    X, y, metadata, seen = sample_batches(dataset, dataset.feature_names, limit, seed)
    return X, y, metadata, dataset, seen


def validate_config(cfg):
    require(cfg['purpose'] == 'development_smoke_only', 'This runner is only for smoke checks')
    integer(cfg['seed'], 'model seed')
    for key in ('train_sample_rows', 'validation_sample_rows', 'fit_threads'):
        integer(cfg[key], key, 1)
    require(isinstance(cfg['threshold'], (int, float)) and not isinstance(cfg['threshold'], bool) and np.isfinite(cfg['threshold']), 'Invalid smoke threshold')
    require(set(cfg['models']) == {'random_forest', 'hist_gradient_boosting'}, 'Smoke checks require RF and GBDT')
    for key, minimum in [('batch_size', 1), ('warmup', 0), ('repeats', 1), ('threads', 1)]:
        integer(cfg['inference'][key], key, minimum)
    require(cfg['validation_sample_rows'] >= cfg['inference']['batch_size'], 'Timing batch exceeds validation subset')
    integer(cfg['bootstrap_repetitions'], 'bootstrap repetitions', 2)
    integer(cfg['bootstrap_seed'], 'bootstrap seed')
    sampling = cfg['sampling_seeds']
    for key in ('train', 'validation'):
        integer(sampling[key], key + ' sampling seed')
    integer(cfg['timing_seed'], 'timing seed')
    return sampling


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='methods/m1/config_smoke.json')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/m1_runs')
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    sampling = validate_config(cfg)
    data_dir = (ROOT / cfg['dataset_directory']).resolve()
    protocol_dir = data_dir.parent if data_dir.name == 'no_destination_port' else data_dir
    require(protocol_dir.parent.parent == (ROOT / 'data/split').resolve(), 'Use a P2 version/protocol directory')
    families = json.loads((ROOT / 'docs/label_split_policy.json').read_text(encoding='utf-8'))['family_names']
    families = [name for name in families if name != 'benign']
    store = RunStore(args.output_root, kind='smoke')
    try:
        store.write_json('config.json', cfg)
        print('Sampling train and validation; test data is disabled.', flush=True)
        X_train, y_train, train_meta, train_data, train_total = sample_dataset(data_dir / 'train.parquet', cfg['train_sample_rows'], sampling['train'])
        X_val, y_val, val_meta, val_data, val_total = sample_dataset(data_dir / 'val.parquet', cfg['validation_sample_rows'], sampling['validation'])
        require(train_data.feature_names == val_data.feature_names, 'Feature schemas differ')
        assert_isolated(train_meta['row_id'].tolist(), val_meta['row_id'].tolist())
        selected = timing_indices(val_meta['row_id'].tolist(), batch_size=cfg['inference']['batch_size'], seed=cfg.get('timing_seed', cfg['seed']))
        timing_ids = val_meta.iloc[selected]['row_id'].tolist()
        timing_attack = int(y_val[selected].sum())
        state_dir = data_dir.parent if data_dir.name == 'no_destination_port' else data_dir
        sources = sorted((ROOT / 'methods/m1').glob('*.py')) + [config_path, ROOT / 'src/model_loader.py', ROOT / 'src/preprocess.py', ROOT / 'eval/metrics.py', ROOT / 'eval/schema.json', ROOT / 'docs/label_split_policy.json']
        record = {
            'format_version': '2', 'purpose': cfg['purpose'], 'evaluation_split': 'validation_subset', 'config': cfg,
            'data': {
                'directory': cfg['dataset_directory'], 'features': list(train_data.feature_names),
                'train_source_rows': train_total, 'validation_source_rows': val_total,
                'train_used_rows': len(y_train), 'validation_used_rows': len(y_val),
                'train_sha256': sha(data_dir / 'train.parquet'), 'validation_sha256': sha(data_dir / 'val.parquet'),
                'preprocessor_sha256': sha(state_dir / 'preprocessor.json'),
                'sampling': 'uniform without replacement using independent random priorities',
                'sampling_seeds': sampling, 'training_class_weights': train_data.class_weights,
                'expected_attack_families': families,
                'train_coverage': family_coverage(y_train, train_meta['attack_family'], families),
                'validation_coverage': family_coverage(y_val, val_meta['attack_family'], families),
                'train_row_ids_sha256': row_digest(train_meta['row_id']),
                'validation_row_ids_sha256': row_digest(val_meta['row_id']),
                'timing_row_ids': timing_ids, 'timing_row_ids_sha256': row_digest(timing_ids),
                'timing_composition': {'benign': len(selected) - timing_attack, 'attack': timing_attack}},
            'decision_rule': {'threshold': cfg['threshold'], 'operator': '>=', 'source': 'predefined_for_smoke'},
            'models': {},
            'provenance': {
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files_sha256': {str(p): sha(p) for p in sources},
                'python': platform.python_version(), 'platform': platform.platform(),
                'software': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'duckdb', 'joblib', 'matplotlib', 'threadpoolctl', 'jsonschema']},
                'note': 'Development validation subset only; not formal or test results. Sources may be uncommitted.'}}
        predictions = {'y_true': y_val, 'row_ids': val_meta['row_id'].to_numpy(dtype=str),
                       'attack_family': val_meta['attack_family'].to_numpy(dtype=str),
                       'is_attempted': val_meta['is_attempted'].to_numpy(dtype=bool),
                       'train_row_ids': train_meta['row_id'].to_numpy(dtype=str)}
        training = ModelInput(X_train, y_train, tuple(train_data.feature_names), 'train', train_data.class_weights)
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            for name, params in cfg['models'].items():
                print('Training smoke model: ' + name, flush=True)
                model, seconds = fit_tree(name, params, training, model_seed=cfg['seed'], fit_threads=cfg['fit_threads'])
                with threadpool_limits(limits=1):
                    scores = attack_scores(model, X_val)
                relative_model = 'models/' + name + '.joblib'
                store.write_model(relative_model, model)
                restored = joblib.load(store.path / relative_model)
                with threadpool_limits(limits=1):
                    require(np.array_equal(scores, attack_scores(restored, X_val)), 'Model reload changed predictions')
                metrics = score_metrics(y_val, scores, val_meta, cfg['threshold'], families)
                with inference_workers(model, cfg['inference']['threads']):
                    measured = measure_inference(lambda batch: attack_scores(model, batch), X_val[selected], **cfg['inference'])
                measured['model_worker_limit'] = cfg['inference']['threads']
                record['models'][name] = {'parameters': params, 'effective_parameters': model.get_params(), 'metrics': metrics,
                                          'training_seconds': seconds, 'inference': measured, 'model_reload_prediction_equal': True}
                predictions[name] = scores
                curve = metrics['curves']
                ax.step(curve['pr']['recall'], curve['pr']['precision'], where='post', label=f'{name} (AP={curve["average_precision"]:.4f})')
            ids = val_meta['row_id'].tolist()
            record['comparison'] = {'scope': 'development validation subset only', 'method_a': 'hist_gradient_boosting', 'method_b': 'random_forest',
                'result': paired_bootstrap(y_val, predictions['hist_gradient_boosting'], predictions['random_forest'], row_ids_a=ids, row_ids_b=ids,
                    metric='ap', n_bootstrap=cfg['bootstrap_repetitions'], seed=cfg.get('bootstrap_seed', cfg['seed']))}
            ax.axhline(float(y_val.mean()), color='gray', linestyle='--', label='Validation subset attack prevalence')
            ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02), title='M1 development check - validation subset only')
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            with store.file('pr.png').open('xb') as stream:
                fig.savefig(stream, format='png', dpi=160)
        finally:
            plt.close(fig)
        schema = json.loads((ROOT / 'eval/schema.json').read_text(encoding='utf-8-sig'))
        validate_record(record, schema)
        store.write_predictions(predictions)
        with np.load(store.path / 'predictions.npz', allow_pickle=False) as saved:
            validate_saved_predictions(record, saved)
        store.write_json('metrics.json', record)
        store.finish()
        verify_run(store.path)
        print('PASS: strict smoke validation, independent saved predictions and model reload checks.', flush=True)
        print('Run: ' + str(store.path), flush=True)
        print('Historical results/m1_smoke.json is preserved. No formal m1.json generated.', flush=True)
    except Exception as error:
        store.fail(error)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
