"""Deterministic Validation-only aggregation of public M1 JSON summaries.

No model/data-loader imports, dataset reads, fitting, inference, or Test access.
JSON Schema checks shape; semantic checks bind every value to public sources.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'schemas/m1.schema.json'
GENERATOR = 'tools/aggregate_m1.py'
SOURCES = {
    'gbdt_summary': 'results/m1_gbdt_summary/summary.json',
    'gbdt_selection': 'results/m1_gbdt_summary/selection.json',
    'gbdt_manifest': 'results/m1_gbdt_summary/MANIFEST.local-archive.json',
    'rf_summary': 'results/m1_rf_summary/summary.json',
    'rf_receipt': 'results/m1_rf_summary/acceptance_receipt.json',
    'rf_manifest': 'results/m1_rf_summary/MANIFEST.local-archive.json',
    'rf_verification': 'results/m1_rf_summary/verification.json',
    'rf_environment': 'results/m1_rf_summary/environment.public.json',
    'gbdt_execution': 'methods/m1/execution.gbdt-search.v1.json',
    'rf_execution': 'methods/m1/execution.rf-only.proposed.json',
    'rf_trial_compatibility': 'methods/m1/rf_per_seed_compatibility.v1.json',
    'unified_protocol': 'methods/m1/protocol.unified.v1.json',
}
SEEDS = [42, 43, 44, 45, 46]
IDS = [f'G{i:02d}' for i in range(1, 9)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(raw):
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def source(root, path):
    raw = (root / path).read_bytes()
    return decode(raw), {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def statistics_match(tasks, ap_mean, ap_std, time_mean, time_std, time_key):
    require([t['seed'] for t in tasks] == SEEDS, 'Missing, duplicate, or unordered seeds')
    ap = [t['validation_ap'] for t in tasks]
    seconds = [t[time_key] for t in tasks]
    require(all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in ap), 'Invalid AP')
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in seconds), 'Invalid fit time')
    for actual, expected in [(statistics.mean(ap), ap_mean), (statistics.stdev(ap), ap_std),
                             (statistics.mean(seconds), time_mean), (statistics.stdev(seconds), time_std)]:
        require(type(expected) in (float, int) and math.isfinite(expected) and
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), 'Mean/sample standard deviation mismatch')


def check_sources(s, p):
    g, r = s['gbdt_summary'], s['rf_summary']
    gc, rc = s['gbdt_execution'], s['rf_execution']
    gm, rm = s['gbdt_manifest'], s['rf_manifest']
    selection = s['gbdt_selection']
    require(g['summary_kind'] == 'formal_gbdt_validation_search_only' and not g['is_final_test_evaluation'], 'GBDT scope differs')
    require(g['test_accessed'] is False and g['bootstrap_run'] is False and g['thresholds_calibrated'] is False and g['RF_fits'] == 0, 'GBDT stage boundary differs')
    require(r['status'] == 'rf_validation_acceptance_complete' and r['test_accessed'] is False and
            r['bootstrap'] is False and r['thresholds_calibrated'] is False and r['eligible_for_gbdt_selection'] is False, 'RF stage boundary differs')
    require(g['search_tasks'] == 40 and g['seeds'] == gc['seeds'] == rc['seeds'] == SEEDS, 'Task budget/seeds differ')
    require(g['selected_candidate_id'] == selection['selected_candidate_id'] == r['paired_reference']['selected_candidate_id'] == 'G05', 'Selected GBDT differs')
    require([c['candidate_id'] for c in g['candidates']] == [c['id'] for c in gc['candidates']] == IDS, 'Eight ordered candidates required')
    require([c['candidate_id'] for c in selection['candidates']] == IDS, 'Selection candidates differ')
    require(g['selection_rule'] == {'metric': 'mean_full_validation_AP', 'ap_tolerance': 1e-6,
                                  'tie_rule': 'lowest_candidate_id', 'std_ddof': 1}, 'Selection rule differs')
    require(g['dataset'] == {'version': 'original', 'protocol': 'stratified', 'variant': 'main', 'features': 140,
                            'train_rows': 1764265, 'validation_rows': 367036,
                            'input_file_sha256': r['paired_reference']['inputs']}, 'Data identities differ')
    require(set(g['dataset']['input_file_sha256']) == {'train.parquet', 'val.parquet', 'preprocessor.json', 'preprocess_manifest.json'}, 'Unexpected input identity')
    for c, configured, selected in zip(g['candidates'], gc['candidates'], selection['candidates']):
        require(c['parameters'] == configured['params'], 'Candidate parameters differ')
        statistics_match(c['tasks'], c['mean_validation_ap'], c['sample_std_ap'],
                         c['mean_fit_seconds_raw'], c['sample_std_fit_seconds_raw'], 'fit_seconds_raw')
        require(selected == {k: c[k] for k in ('candidate_id', 'mean_validation_ap', 'sample_std_ap')}, 'Selection summary differs')
    statistics_match(r['tasks'], r['mean_validation_ap'], r['sample_std_ap'], r['mean_fit_seconds'], r['sample_std_fit_seconds'], 'fit_seconds')
    receipt = s['rf_receipt']
    require(receipt['AP_exact_matches'] == 5 and [t['seed'] for t in receipt['per_seed']] == SEEDS, 'RF receipt incomplete')
    for t, proof in zip(r['tasks'], receipt['per_seed']):
        require(proof['ap_exact_match'] is True and proof['original_ap'] == proof['independently_recomputed_ap'] == t['validation_ap']
                and proof['training_seconds'] == t['fit_seconds'], 'RF receipt per-seed mismatch')
        require(not t['monitor_attempt_interrupted'], 'RF training attempt interrupted')
    require(g['execution_config_sha256'] == p['gbdt_execution']['sha256'], 'GBDT config hash mismatch')
    require(rm['files']['sources/methods/m1/execution.rf-only.proposed.json']['sha256'] == p['rf_execution']['sha256'], 'RF config hash mismatch')
    require(g['archive']['manifest_sha256'] == p['gbdt_manifest']['sha256'] == r['paired_reference']['manifest_sha256'], 'GBDT manifest hash mismatch')
    require(g['archive']['selection_sha256'] == p['gbdt_selection']['sha256'] == r['paired_reference']['selection_sha256']
            == gm['files']['selection.json']['sha256'], 'GBDT selection hash mismatch')
    require(rm['files']['summary.json']['sha256'] == p['rf_summary']['sha256'] == receipt['summary_sha256'], 'RF summary hash mismatch')
    require(rm['files']['recovery/venv-acceptance-02/acceptance_receipt.json']['sha256'] == p['rf_receipt']['sha256'], 'RF receipt hash mismatch')
    require(rc['resource_trial_code_compatibility'] == {k: p['rf_trial_compatibility'][k] for k in ('path', 'sha256')}, 'RF compatibility hash mismatch')
    require(len(gm['files']) == 302 and len(rm['files']) == 157, 'Public manifest entry count differs')
    v = s['rf_verification']
    require(v['RF_manifest_sha256'] == p['rf_manifest']['sha256'] and v['summary_sha256'] == p['rf_summary']['sha256'] and
            v['AP_exact_matches'] == 5 and v['max_absolute_AP_difference'] == 0, 'RF public verification differs')
    require(g['independent_AP_recheck']['tasks_passed'] == 40 and g['independent_AP_recheck']['maximum_absolute_difference'] == 0, 'GBDT AP evidence incomplete')
    require(gc['approval']['status'] == rc['execution_approval']['status'] == s['unified_protocol']['approval']['status'] == 'approved', 'Approval scope differs')
    require(gc['unified_protocol_sha256'] == rc['unified_protocol_sha256'] == s['unified_protocol']['sha256'], 'Unified protocol identity differs')
    paired = list(r['paired_reference']['paired_seeds'].values())
    require(list(r['paired_reference']['paired_seeds']) == list(map(str, SEEDS)), 'Pairing seeds incomplete')
    require(all(x['rows'] == 367036 and x['row_ids_sha256'] == paired[0]['row_ids_sha256'] and
                x['labels_sha256'] == paired[0]['labels_sha256'] for x in paired), 'Pairing population differs')


def build(root=ROOT):
    root = Path(root)
    sources, provenance = {}, {}
    for name, path in SOURCES.items():
        sources[name], provenance[name] = source(root, path)
    check_sources(sources, provenance)
    g, r = sources['gbdt_summary'], sources['rf_summary']
    generator = (root / GENERATOR).read_bytes()
    schema = (root / SCHEMA).read_bytes()
    return {'schema_version': '1.0', 'artifact_type': 'm1_validation_summary',
            'scope': {'evaluation_split': 'validation', 'validation_only': True, 'is_final_test_evaluation': False,
                      'test_accessed': False, 'models_retrained_in_this_step': False, 'predictions_read_in_this_step': False,
                      'prediction_AP_recomputed_in_this_step': False, 'bootstrap_run': False, 'thresholds_calibrated': False,
                      'available_metrics': ['average_precision', 'fit_wall_seconds'],
                      'full_local_archives_reverified_in_this_step': False},
            'dataset': deepcopy(g['dataset']), 'seeds': SEEDS.copy(), 'std_ddof': 1,
            'provenance': {'sources': provenance,
                           'generator': {'path': GENERATOR, 'sha256': hashlib.sha256(generator).hexdigest(), 'bytes': len(generator)},
                           'schema': {'path': SCHEMA, 'sha256': hashlib.sha256(schema).hexdigest(), 'bytes': len(schema)},
                           'integrity_scope': 'Public JSON byte hashes and small-file cross-references only; manifest entries identify local files absent from GitHub. Historical AP and full-archive checks are preserved evidence, not rerun here.'},
            'unified_protocol': deepcopy(sources['unified_protocol']),
            'gbdt': {'algorithm': 'HistGradientBoostingClassifier', 'role': 'primary', 'fit_count': 40,
                     'execution_config': deepcopy(sources['gbdt_execution']), 'selection_rule': deepcopy(g['selection_rule']),
                     'candidates': deepcopy(g['candidates']), 'selected_candidate_id': 'G05',
                     'selected_candidate': deepcopy(next(c for c in g['candidates'] if c['candidate_id'] == 'G05')),
                     'historical_AP_recheck': deepcopy(g['independent_AP_recheck']), 'archive': deepcopy(g['archive'])},
            'rf': {'algorithm': 'RandomForestClassifier', 'role': 'auxiliary', 'fit_count': 5,
                   'eligible_for_gbdt_selection': False, 'execution_config': deepcopy(sources['rf_execution']),
                   'validation_results': deepcopy(r), 'historical_acceptance_receipt': deepcopy(sources['rf_receipt']),
                   'historical_verification': deepcopy(sources['rf_verification']), 'environment': deepcopy(sources['rf_environment'])}}


def validate_document(document, root=ROOT):
    schema = decode((Path(root) / SCHEMA).read_bytes())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    require(document == build(root), 'Aggregation differs from source, configuration, statistics, or provenance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate existing results/m1.json without writing')
    args = parser.parse_args()
    target = ROOT / 'results/m1.json'
    if args.check:
        validate_document(decode(target.read_bytes()))
        print('PASS: Schema, sources, statistics, and per-seed identities; Validation-only.')
    else:
        document = build()
        validate_document(document)
        encoded = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        if target.exists():
            require(target.read_bytes() == encoded, 'Existing results/m1.json differs; refusing overwrite')
            print('Unchanged: results/m1.json')
        else:
            with target.open('xb') as stream:
                stream.write(encoded)
            print('Created results/m1.json from public Validation summaries only.')


if __name__ == '__main__':
    main()
