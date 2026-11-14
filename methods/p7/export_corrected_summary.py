"""Publish compact copies of accepted corrected outputs with raw-file hashes."""
from pathlib import Path
from collect_results import load, sha, require


def export(root, run_root, audit):
    import json

    require(audit['status'] == 'passed', 'Formal acceptance must pass before export')
    sources = {}

    def read(path):
        path = Path(path)
        sources[path.relative_to(run_root).as_posix()] = {
            'bytes': path.stat().st_size, 'sha256': sha(path)}
        return load(path)

    def save(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                        encoding='utf-8', newline='\n')

    def compact(record):
        # Preserve scalar metrics, operating points, confusion counts and family supports.
        metrics = record['metrics']
        curves = metrics.get('curves')
        if curves is not None:
            metrics['curves'] = {key: curves[key] for key in ['average_precision', 'roc_auc']}
        return record

    out = root / 'results'
    evidence = out / 'p7_corrected_summary'
    evidence.mkdir(exist_ok=True)
    m1_dir = run_root / audit['m1_run']
    index = read(m1_dir / 'evaluation_index.json')
    m1 = {'summary_kind': 'compact_formal_test_records', 'audit': audit,
          'records': [compact(read(m1_dir / item['path'])) for item in index['records']]}
    save(out / 'm1_corrected.json', m1)
    save(out / 'm2_corrected.json', read(run_root / 'results/p7_corrected/m2_corrected.json'))
    m3_dir = run_root / 'results/p7_corrected/m3'
    index3 = read(m3_dir / 'm3.json')
    m3 = {'summary_kind': 'compact_formal_test_records',
          'selected_candidate': index3['selected_candidate'], 'dataset': index3['dataset'],
          'records': [compact(read(run_root / item['record'])) for item in index3['records']]}
    save(out / 'm3_corrected.json', m3)
    save(evidence / 'm3_search.json', read(m3_dir / 'search.json'))
    save(evidence / 'm1_validation_frozen.json', read(m1_dir / 'validation_frozen.json'))
    save(evidence / 'm1_execution_approval.json', read(m1_dir / 'execution_approval.json'))
    receipts = []
    for stage in ['Check', 'M1', 'M2Train', 'M2Evaluate', 'M3Search', 'M3Evaluate']:
        receipt = read(run_root / 'results/p7_corrected/receipts' / (stage + '.json'))
        require(receipt['status'] == 'complete' and receipt['git_commit'] == audit['required_training_commit'],
                stage + ' must have a complete receipt at the required commit')
        receipt['log'] = 'results/p7_corrected/logs/' + Path(receipt['log']).name
        receipts.append(receipt)
    save(evidence / 'receipts.json', receipts)
    save(evidence / 'source_manifest.json', {
        'training_commit': audit['required_training_commit'], 'files': sources,
        'note': 'Paths are relative to the frozen training checkout. Full curves, models and row predictions remain in that local archive.'})
    print('Saved compact corrected results and source hashes:', evidence, flush=True)
