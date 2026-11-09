"""P6 autoencoder pipeline. No test access during smoke/search/calibration."""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def signature(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def dataset_path(cfg, split):
    data = cfg['dataset']
    path = ROOT / 'data/split' / data['version'] / data['protocol']
    if data['variant'] == 'no_destination_port':
        path /= 'no_destination_port'
    return path / (split + '.parquet')


def read_data(cfg, split, *, benign_only=False, limit=None):
    # Use the unmodified P2 loader, including its data and processor hash checks.
    if not hasattr(hashlib, 'file_digest'):
        raise RuntimeError('P2 ModelDataset requires Python >= 3.11; use Python 3.12 or 3.13.')
    from src.model_loader import ModelDataset
    path = dataset_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(f'{path}: download this split using Git LFS first')
    with path.open('rb') as stream:
        if stream.read(50).startswith(b'version https://git-lfs.github.com'):
            raise ValueError(f'{path} is an LFS pointer, not real data')
    source = ModelDataset(path)
    arrays, labels, metas = [], [], []
    count = 0
    for X, y, meta in source:
        mask = (y == 0) if benign_only else np.ones(len(y), dtype=bool)
        X, y, meta = X[mask], y[mask], meta.loc[mask]
        if limit is not None:
            remaining = limit - count
            X, y, meta = X[:remaining], y[:remaining], meta.iloc[:remaining]
        if len(X):
            arrays.append(X)
            labels.append(y)
            metas.append(meta)
            count += len(X)
        if limit is not None and count >= limit:
            break
    if count == 0:
        raise ValueError(f'No eligible samples: {path}')
    import pandas as pd
    X = np.concatenate(arrays).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite model inputs')
    meta = pd.concat(metas, ignore_index=True)
    if not meta.row_id.is_unique:
        raise ValueError('Duplicate row IDs')
    return X, np.concatenate(labels), meta, sha(path)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        widths = [input_dim] + hidden + list(reversed(hidden[:-1])) + [input_dim]
        layers = []
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


def scores(model, X, device='cpu', batch_size=4096):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], device=device)
            result.append(((model(batch) - batch) ** 2).mean(dim=1).cpu().numpy())
    values = np.concatenate(result).astype(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite reconstruction error')
    return values


def train_one(cfg, candidate, seed, train_X, val_X, out, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = Autoencoder(train_X.shape[1], candidate['hidden']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate['learning_rate'],
                                 weight_decay=candidate['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    batches = DataLoader(TensorDataset(torch.from_numpy(train_X)),
                         batch_size=cfg['batch_size'], shuffle=True,
                         num_workers=0, generator=generator)
    best, stale, history, best_state = float('inf'), 0, [], None
    start = time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        loss_total = 0.0
        for (batch,) in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(batch) - batch) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(batch)
        val_mse = float(scores(model, val_X, device).mean())
        history.append({'epoch': epoch + 1, 'train_mse': loss_total / len(train_X),
                        'benign_validation_mse': val_mse})
        print(candidate['id'], seed, history[-1], flush=True)
        if val_mse < best - cfg['min_delta']:
            best, stale = val_mse, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    errors = scores(model, val_X, device)
    thresholds = {str(q): float(np.quantile(errors, q, method=cfg['quantile_method']))
                  for q in cfg['quantiles']}
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'model.pt')
    record = {'candidate': candidate, 'seed': seed, 'config_sha256': signature(cfg),
              'model_sha256': sha(out / 'model.pt'),
              'input_dim': train_X.shape[1], 'training_seconds': elapsed,
              'training_samples': len(train_X), 'benign_validation_samples': len(val_X),
              'best_benign_validation_mse': float(errors.mean()),
              'thresholds': thresholds, 'history': history,
              'threshold_source': 'validation_benign_quantile',
              'test_accessed': False, 'device': device}
    dump(out / 'run.json', record)
    return record


def binding(cfg):
    return {'train_sha256': sha(dataset_path(cfg, 'train')),
            'val_sha256': sha(dataset_path(cfg, 'val'))}


def search(cfg, out, device, smoke=False):
    limit = 2048 if smoke else None
    train_X, _, _, train_hash = read_data(cfg, 'train', benign_only=True, limit=limit)
    val_X, _, val_meta, val_hash = read_data(cfg, 'val', benign_only=True,
                                          limit=512 if smoke else None)
    run_cfg = copy.deepcopy(cfg)
    if smoke:
        run_cfg.update(epochs=2, patience=2)
    expected = {'train_sha256': train_hash, 'val_sha256': val_hash}
    frozen = out / 'plan.json'
    plan = {'config': run_cfg, 'data': expected, 'is_smoke': smoke}
    if frozen.exists() and load(frozen) != plan:
        raise ValueError('Existing plan differs; choose a new output directory')
    dump(frozen, plan)
    candidates = cfg['candidates'][:1] if smoke else cfg['candidates']
    seeds = cfg['seeds'][:1] if smoke else cfg['seeds']
    all_runs = []
    for candidate in candidates:
        for seed in seeds:
            folder = out / candidate['id'] / str(seed)
            record_path = folder / 'run.json'
            if record_path.exists() and (folder / 'model.pt').exists():
                record = load(record_path)
                if (record['config_sha256'] != signature(run_cfg)
                        or record['model_sha256'] != sha(folder / 'model.pt')):
                    raise ValueError('Resume configuration mismatch')
                print('Resume: already complete', folder, flush=True)
            else:
                record = train_one(run_cfg, candidate, seed, train_X, val_X, folder, device)
            all_runs.append(record)
    ranking = []
    for candidate in candidates:
        values = [r['best_benign_validation_mse'] for r in all_runs
                  if r['candidate']['id'] == candidate['id']]
        ranking.append({'candidate_id': candidate['id'],
                        'mean_benign_validation_mse': float(np.mean(values))})
    ranking.sort(key=lambda r: (r['mean_benign_validation_mse'], r['candidate_id']))
    summary = {'status': 'smoke_only' if smoke else 'validation_search_complete',
               'is_smoke': smoke, 'test_accessed': False, 'data': expected,
               'config_sha256': signature(cfg), 'ranking': ranking,
               'selected_candidate': ranking[0]['candidate_id'], 'runs': all_runs,
               'calibration_row_ids_sha256': hashlib.sha256(
                   '\n'.join(val_meta.row_id.astype(str)).encode()).hexdigest()}
    dump(out / 'search.json', summary)
    if not smoke:
        lines = ['# M3 阈值标定记录', '',
                 '训练及配置选择只使用训练集和验证集的 y_binary=0 样本。',
                 '选择规则：五种子平均良性验证 MSE 最小，相同则按配置编号排序。',
                 '阈值使用 numpy linear 分位数；误差 >= 阈值判为攻击。',
                 'y_binary=0 沿用 P2 映射，可能包括 attempted 负类，不自行重标。',
                 '未读取测试集。正式测试结果由独立 evaluate 命令生成。', '',
                 f'配置 SHA256：{signature(cfg)}',
                 f'训练文件 SHA256：{train_hash}', f'验证文件 SHA256：{val_hash}',
                 f'获选配置：{summary["selected_candidate"]}', '',
                 '| Seed | 良性训练数 | 良性标定数 | q95 | q99 |',
                 '|---|---|---|---|---|']
        for r in all_runs:
            if r['candidate']['id'] == summary['selected_candidate']:
                lines.append(f'| {r["seed"]} | {r["training_samples"]} | '
                             f'{r["benign_validation_samples"]} | '
                             f'{r["thresholds"]["0.95"]} | {r["thresholds"]["0.99"]} |')
        (out / 'm3_threshold.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Saved:', out / 'search.json', flush=True)


def export_record(cfg, out, model_name, seed, run, predictor, X, y, meta,
                  data_hash, thresholds, training_seconds, model_config):
    from eval.metrics import (classification_metrics, compute_curves,
                              measure_inference, per_class_recall, tpr_at_fpr)
    from eval.validate import validate_result
    values = predictor(X)
    if values.shape != (len(X),) or not np.isfinite(values).all():
        raise ValueError('Invalid anomaly scores')
    name = f'{model_name}_seed{seed}'
    prediction_dir = out / 'predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    frame = meta.copy()
    frame['score'] = values
    frame.to_parquet(prediction_dir / (name + '.parquet'), index=False)
    latency = measure_inference(predictor, X, **cfg['inference'])
    curves = compute_curves(y, values)
    operating = tpr_at_fpr(y, values)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    notes = {'code_sha256': sha(__file__), 'config_sha256': signature(cfg),
             'prediction_sha256': sha(prediction_dir / (name + '.parquet')),
             'calibration_run': run, 'evaluation_split': 'test',
             'comparisons': 'deferred to P7 with aligned per-row scores'}
    entries = []
    for q, threshold in thresholds.items():
        pred = (values >= threshold).astype(int)
        record = {
            'schema_version': '1.0', 'is_demo': False, 'method': model_name,
            'dataset': {**cfg['dataset'], 'split': 'test',
                        'path': dataset_path(cfg, 'test').relative_to(ROOT).as_posix(),
                        'sha256': data_hash, 'n_samples': len(y)},
            'seed': seed, 'config': {**model_config, 'benign_quantile': float(q)},
            'decision_rule': {'score_direction': 'higher_means_attack', 'operator': '>=',
                              'threshold': threshold, 'source': 'validation_benign_quantile'},
            'metrics': {'curves': curves, 'classification': classification_metrics(
                y, pred, is_attempted=meta.is_attempted.to_numpy(dtype=bool)),
                'operating_points': operating,
                'per_class': per_class_recall(y, pred, meta.attack_family.tolist())},
            'cost': {'training_seconds': training_seconds,
                     'training_time_note': 'Measured fit and validation-selection time for this run.',
                     'hardware': platform.platform() + ' / ' + platform.processor(),
                     'thread_settings': {'torch': 1, 'inference': cfg['inference']},
                     'inference': latency},
            'daily_alerts': None, 'comparisons': [],
            'provenance': {'code_commit': commit or 'unavailable',
                           'created_at': datetime.now(timezone.utc).isoformat(),
                           'software': {'python': platform.python_version(),
                                        'numpy': np.__version__, 'torch': torch.__version__},
                           'notes': json.dumps(notes, ensure_ascii=False)}}
        validate_result(record)
        path = out / 'records' / f'{name}_q{round(float(q)*100)}.json'
        dump(path, record)
        entries.append({'record': path.relative_to(ROOT).as_posix(),
                        'method': model_name, 'seed': seed, 'quantile': float(q),
                        'ap': curves['average_precision'],
                        'roc_auc': curves['roc_auc'],
                        'classification': record['metrics']['classification']})
    return entries, values


def evaluate(cfg, out):
    summary = load(out / 'search.json')
    if summary['is_smoke'] or summary['config_sha256'] != signature(cfg):
        raise ValueError('Requires completed formal search with unchanged configuration')
    if summary['data'] != binding(cfg):
        raise ValueError('Train/validation data changed since calibration')
    selected = [r for r in summary['runs']
                if r['candidate']['id'] == summary['selected_candidate']]
    if len(selected) != len(cfg['seeds']):
        raise ValueError('Incomplete selected candidate')
    for run in selected:
        model_path = out / run['candidate']['id'] / str(run['seed']) / 'model.pt'
        if sha(model_path) != run['model_sha256']:
            raise ValueError('Checkpoint hash changed since calibration')
    # Test is loaded only in this explicit subcommand, after selection is fixed.
    X, y, meta, data_hash = read_data(cfg, 'test')
    entries, distribution = [], None
    torch.set_num_threads(1)
    for run in selected:
        folder = out / run['candidate']['id'] / str(run['seed'])
        model = Autoencoder(run['input_dim'], run['candidate']['hidden'])
        model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
        predictor = lambda batch: scores(model, batch)
        rows, values = export_record(cfg, out, 'Autoencoder', run['seed'], run,
            predictor, X, y, meta, data_hash, run['thresholds'],
            run['training_seconds'], run['candidate'])
        entries.extend(rows)
        if distribution is None:
            distribution = (values, y, run['thresholds'], run['seed'])
    dump(out / 'm3.json', {'status': 'test_evaluation_complete',
                          'selected_candidate': summary['selected_candidate'],
                          'dataset': cfg['dataset'], 'records': entries,
                          'note': 'Index only; validate the individual records with eval.validate.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values, labels, thresholds, seed = distribution
    # Log1p keeps zero errors visible. Histograms show test distributions after calibration.
    for label, name in [(0, 'Benign'), (1, 'Attack')]:
        plt.hist(np.log1p(values[labels == label]), bins=80, alpha=0.5,
                 density=True, label=name)
    for q, threshold in thresholds.items():
        plt.axvline(np.log1p(threshold), linestyle='--', label=f'q={q}')
    plt.xlabel('log(1 + reconstruction MSE)')
    plt.ylabel('Density')
    plt.title(f'Test error distribution; selected AE; seed {seed}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'fig_m3_error_dist.png', dpi=200)
    plt.close()


def auxiliary(cfg, out):
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    import joblib
    # A disclosed fixed training subsample keeps kernel SVM cost bounded.
    train_X, _, _, _ = read_data(cfg, 'train', benign_only=True)
    val_X, _, _, _ = read_data(cfg, 'val', benign_only=True)
    aux = cfg['auxiliary']
    rng = np.random.default_rng(aux['seed'])
    indices = np.sort(rng.choice(len(train_X), min(len(train_X), aux['max_benign_train_rows']),
                                 replace=False))
    models = [('IsolationForest', IsolationForest(random_state=aux['seed'], **aux['isolation_forest'])),
              ('OneClassSVM', OneClassSVM(**aux['ocsvm']))]
    fitted = []
    for name, model in models:
        start = time.perf_counter()
        model.fit(train_X[indices])
        elapsed = time.perf_counter() - start
        # Both estimators return higher decision scores for inliers: reverse sign.
        errors = -model.decision_function(val_X)
        thresholds = {str(q): float(np.quantile(errors, q, method=cfg['quantile_method']))
                      for q in cfg['quantiles']}
        fitted.append((name, model, thresholds, elapsed))
    del train_X, val_X
    X, y, meta, data_hash = read_data(cfg, 'test')
    entries = []
    for name, model, thresholds, elapsed in fitted:
        model_dir = out / 'auxiliary'
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / (name + '.joblib'))
        details = {'training_samples': len(indices), 'train_subset_seed': aux['seed'],
                   'training_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                   'eligible_for_main_selection': False, 'test_used_for_calibration': False}
        rows, _ = export_record(cfg, out, name, aux['seed'], details,
            lambda batch: -model.decision_function(batch), X, y, meta, data_hash,
            thresholds, elapsed, model.get_params())
        entries.extend(rows)
    dump(out / 'auxiliary.json', {'records': entries, 'note': 'One fit per method; no five-seed statistics.'})


def preflight(cfg):
    import importlib.util
    report = {'python': platform.python_version(), 'python_supported': sys.version_info >= (3, 11),
              'cuda': torch.cuda.is_available(),
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'modules': {m: importlib.util.find_spec(m) is not None
                          for m in ['duckdb', 'pyarrow', 'pandas', 'sklearn', 'jsonschema', 'matplotlib']},
              'splits': {}}
    for split in ['train', 'val', 'test']:
        path = dataset_path(cfg, split)
        actual = False
        if path.exists():
            with path.open('rb') as stream:
                actual = stream.read(4) == b'PAR1'
        report['splits'][split] = {'path': str(path), 'real_parquet': actual}
    print(json.dumps(report, indent=2), flush=True)
    if not report['python_supported'] or not all(report['modules'].values()):
        raise SystemExit('Environment incomplete; see README. No experiment executed.')
    if not all(report['splits'][s]['real_parquet'] for s in ['train', 'val']):
        raise SystemExit('Download train/val before smoke/search. Test is not needed for search.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'search', 'evaluate', 'auxiliary'])
    parser.add_argument('--config', type=Path, default=ROOT / 'methods/m3/config.json')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    cfg = load(args.config)
    os.chdir(ROOT)
    if len(cfg['candidates']) != 8 or cfg['seeds'] != [42, 43, 44, 45, 46]:
        raise ValueError('Main search must have 8 candidates and seeds 42..46')
    if args.command == 'preflight':
        return preflight(cfg)
    device = cfg['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(cfg['threads'])
    out = (args.output or ROOT / ('results/m3_smoke' if args.command == 'smoke'
                                  else 'results/m3_runs')).resolve()
    # Keep relative artifact references portable and outputs inside this repository.
    out.relative_to(ROOT)
    if args.command in ['smoke', 'search']:
        search(cfg, out, device, smoke=args.command == 'smoke')
    elif args.command == 'evaluate':
        evaluate(cfg, out)
    else:
        auxiliary(cfg, out)


if __name__ == '__main__':
    main()
