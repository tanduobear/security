"""Export small reviewable summaries from saved records, without model execution."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT / 'results/m3_runs'
    out = ROOT / 'results/m3_summary'
    out.mkdir(exist_ok=True)
    records = []
    for path in sorted((folder / 'records').glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        item = {key: record[key] for key in ['method', 'seed', 'dataset', 'config',
                                            'decision_rule', 'cost', 'provenance']}
        item['full_record_after_archive_extraction'] = path.relative_to(ROOT).as_posix()
        item['metrics'] = {key: value for key, value in record['metrics'].items()
                           if key != 'curves'}
        item['metrics']['curves'] = {key: record['metrics']['curves'][key]
                                     for key in ['status', 'average_precision', 'roc_auc']}
        records.append(item)
    if len(records) != 14:
        raise ValueError('Expected 14 completed records')
    aggregates = []
    for q in [0.95, 0.99]:
        selected = [r for r in records if r['method'] == 'Autoencoder'
                    and r['config']['benign_quantile'] == q]
        if len(selected) != 5:
            raise ValueError('Expected five seeds per AE threshold')
        group = {'method': 'Autoencoder', 'quantile': q, 'seeds': [r['seed'] for r in selected]}
        for name in ['average_precision', 'roc_auc', 'tpr', 'fpr', 'accuracy', 'macro_f1', 'precision']:
            values = [r['metrics']['curves'][name] if name in ['average_precision', 'roc_auc']
                      else r['metrics']['classification'][name] for r in selected]
            group[name] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
        aggregates.append(group)
    output = {'format': 'm3-public-summary-v1', 'is_p3_single_record': False,
              'note': 'Curves are omitted from this public summary. Full P3 records are in the LFS ZIP.',
              'archive': 'submission/P6_handoff.zip', 'std_ddof': 1,
              'five_seed_summary': aggregates, 'records': records}
    (out / 'summary.json').write_text(json.dumps(output, indent=2, ensure_ascii=False,
                                                allow_nan=False) + '\n', encoding='utf-8')
    search = json.loads((folder / 'search.json').read_text(encoding='utf-8'))
    (out / 'search.json').write_text(json.dumps(search, indent=2, ensure_ascii=False,
                                               allow_nan=False) + '\n', encoding='utf-8')
    archive = {'path': 'submission/P6_handoff.zip', 'storage': 'git-lfs',
               'extract_to': 'repository root; original relative paths are preserved',
               'public_summary': 'results/m3_summary/summary.json',
               'auxiliary_index_after_extraction': 'results/m3_runs/auxiliary.json'}
    index = json.loads((folder / 'm3.json').read_text(encoding='utf-8'))
    index['archive'] = archive
    for path in [folder / 'm3.json', ROOT / 'results/m3.json']:
        path.write_text(json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                        encoding='utf-8')
    print('Saved: results/m3_summary/summary.json, search.json; archive references added to m3.json')


if __name__ == '__main__':
    main()
