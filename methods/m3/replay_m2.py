"""Replay P5 checkpoints without retraining or overwriting their delivery."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--atol', type=float, default=1e-5)
    args = parser.parse_args()
    import numpy as np
    import torch
    from methods.m2.model import CNN1D
    from methods.m3.run import read_data, dump, sha
    from eval.metrics import compute_curves, tpr_at_fpr, classification_metrics
    if args.atol < 0:
        raise ValueError('atol must be nonnegative')
    original_path = ROOT / 'results/m2.json'
    original = json.loads(original_path.read_text(encoding='utf-8-sig'))
    cfg = {'dataset': {k: original['dataset'][k] for k in ['version', 'protocol', 'variant']}}
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    X, y, meta, data_hash = read_data(cfg, args.split)
    out = ROOT / 'docs/p6_acceptance' / ('m2_replay_' + args.split)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for saved in original['runs']:
        seed = saved['seed']
        checkpoint = ROOT / 'results/preds_m2_corrected' / f'model_seed{seed}.pth'
        model = CNN1D(X.shape[1]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(X), 4096):
                batch = torch.as_tensor(X[start:start+4096], device=device)
                chunks.append(torch.sigmoid(model(batch)).cpu().numpy().reshape(-1))
        values = np.concatenate(chunks)
        curves = compute_curves(y, values)
        points = tpr_at_fpr(y, values, targets=(0.01, 0.05))
        cls = classification_metrics(y, (values >= saved['threshold']).astype(int),
                                     is_attempted=meta.is_attempted.to_numpy(dtype=bool))
        report = {'seed': seed, 'checkpoint_sha256': sha(checkpoint),
                  'roc_auc': curves['roc_auc'], 'average_precision': curves['average_precision'],
                  'operating_points': points, 'classification_at_delivered_threshold': cls}
        comparisons = []
        if args.split == 'test':
            for key in ['roc_auc', 'average_precision']:
                actual, expected = report[key], saved['test'][key]
                comparisons.append({'metric': key, 'actual': actual, 'expected': expected,
                                    'within_atol': abs(actual - expected) <= args.atol})
            for key in ['accuracy', 'macro_f1', 'tpr', 'fpr']:
                actual, expected = cls[key], saved['test']['classification'][key]
                comparisons.append({'metric': key, 'actual': actual, 'expected': expected,
                                    'within_atol': abs(actual - expected) <= args.atol})
            for point in points['operating_points']:
                key = f'tpr_at_fpr_{round(point["target_fpr"]*100)}pct'
                actual, expected = point['tpr'], saved['test'][key]
                comparisons.append({'metric': key, 'actual': actual, 'expected': expected,
                                    'within_atol': abs(actual - expected) <= args.atol})
        else:
            point = next(p for p in points['operating_points'] if p['target_fpr'] == 0.01)
            boundary = point['threshold']
            if boundary is None:
                boundary = float(np.nextafter(float(values.max()), np.inf))
            comparisons.append({'metric': 'validation_selected_threshold',
                                'actual': float(boundary), 'expected': saved['threshold'],
                                'within_atol': abs(boundary - saved['threshold']) <= args.atol})
        report['comparisons'] = comparisons
        report['all_checked_values_within_atol'] = all(c['within_atol'] for c in comparisons)
        frame = meta.copy()
        frame['score'] = values
        frame.to_parquet(out / f'predictions_seed{seed}.parquet', index=False)
        reports.append(report)
        print(f'Seed {seed}: checked values within atol={report["all_checked_values_within_atol"]}', flush=True)
    dump(out / 'report.json', {'split': args.split, 'device': device,
        'dataset': cfg['dataset'], 'data_sha256': data_hash, 'm2_json_sha256': sha(original_path),
        'replay_code_sha256': sha(__file__), 'atol': args.atol, 'runs': reports,
        'acceptance': 'Evidence for P6 review; not an automatic acceptance signature.',
        'scope': 'Delivered checkpoint replay only; no retraining; no latency comparison.'})
    print('Saved:', out / 'report.json')


if __name__ == '__main__':
    main()
