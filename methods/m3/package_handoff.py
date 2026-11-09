"""Package P6 deliverables with relative paths intact. Does not upload or commit."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    report_path = ROOT / 'docs/p6_acceptance/final_audit.json'
    if not report_path.exists() or json.loads(report_path.read_text(encoding='utf-8'))['status'] != 'passed':
        raise SystemExit('Run audit_handoff.py and resolve failures before packaging.')
    selected = set()
    for pattern in ['methods/m3/*', 'results/m3_runs/**/*.json',
                    'results/m3_runs/**/*.pt', 'results/m3_runs/**/*.joblib',
                    'results/m3_runs/predictions/*.parquet', 'results/m3_summary/*',
                    'docs/p6_acceptance/**/*.json',
                    'docs/p6_text.md', 'docs/p6_results_and_handover.md',
                    'docs/m3_threshold.md', 'results/m3.json',
                    'figures/fig_m3_error_dist*.png', 'results/m3_runs/*.md']:
        selected.update(p for p in ROOT.glob(pattern) if p.is_file()
                        and '__pycache__' not in p.parts)
    # Include shared source so the bundle does not require untracked M3 code in Git.
    for folder in ['eval', 'src']:
        selected.update(p for p in (ROOT / folder).glob('*')
                        if p.is_file() and p.suffix in ['.py', '.json', '.txt', '.md'])
    manifest = []
    out = ROOT / 'submission'
    out.mkdir(exist_ok=True)
    archive = out / 'P6_handoff.zip'
    if archive.exists():
        raise SystemExit('P6_handoff.zip already exists; rename the old archive before rebuilding.')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(selected):
            relative = path.relative_to(ROOT).as_posix()
            raw = path.read_bytes()
            manifest.append({'path': relative, 'bytes': len(raw),
                             'sha256': hashlib.sha256(raw).hexdigest()})
            bundle.writestr(relative, raw)
        bundle.writestr('P6_MANIFEST.json', json.dumps(manifest, indent=2))
        bundle.writestr('P6_README.txt',
            'Unpack into a new directory. Relative result-index paths are preserved.\n'
            'Raw train/val/test datasets are not included; obtain the same P2 data via Git LFS.\n'
            'Read methods/m3/README.md and docs/p6_results_and_handover.md.\n'
            'Test was already evaluated: do not tune candidates or thresholds against it.\n'
            'P7 must independently accept/replay calibration. This archive does not sign acceptance.\n')
    with zipfile.ZipFile(archive) as bundle:
        failed = bundle.testzip()
        if failed:
            raise RuntimeError('ZIP CRC check failed: ' + failed)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / 'P6_handoff.sha256.txt').write_text(archive_hash + '  P6_handoff.zip\n', encoding='utf-8')
    print('Saved:', archive)
    print('Files:', len(manifest), 'Size MB:', round(archive.stat().st_size / 1024**2, 2))
    print('ZIP CRC check passed. SHA256:', archive_hash)


if __name__ == '__main__':
    main()
