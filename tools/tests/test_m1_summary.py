"""Public JSON aggregation tests; never loads datasets, models, or predictions."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('aggregate_m1', ROOT / 'tools/aggregate_m1.py')
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


class M1SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = aggregate.build(ROOT)
        cls.schema = json.loads((ROOT / aggregate.SCHEMA).read_bytes())

    def validate_mutation(self, change):
        changed = deepcopy(self.document)
        change(changed)
        with self.assertRaises((ValueError, ValidationError)):
            aggregate.validate_document(changed, ROOT)

    def test_committed_artifact_matches_deterministic_build(self):
        self.assertEqual(json.loads((ROOT / 'results/m1.json').read_bytes()), self.document)
        self.assertEqual(aggregate.build(ROOT), self.document)

    def test_schema_and_semantic_validation_pass(self):
        Draft202012Validator.check_schema(self.schema)
        Draft202012Validator(self.schema).validate(self.document)
        aggregate.validate_document(self.document, ROOT)

    def test_all_45_unique_runs_and_g05_exact_copy(self):
        runs = [(c['candidate_id'], t['seed']) for c in self.document['gbdt']['candidates'] for t in c['tasks']]
        self.assertEqual(len(runs), 40)
        self.assertEqual(len(set(runs)), 40)
        g05 = self.document['gbdt']['selected_candidate']
        self.assertEqual(g05, next(c for c in self.document['gbdt']['candidates'] if c['candidate_id'] == 'G05'))
        self.assertEqual([t['seed'] for t in self.document['rf']['validation_results']['tasks']], aggregate.SEEDS)

    def test_all_means_and_sample_stds_independent(self):
        groups = [(c['tasks'], c['mean_validation_ap'], c['sample_std_ap'], c['mean_fit_seconds_raw'],
                   c['sample_std_fit_seconds_raw'], 'fit_seconds_raw') for c in self.document['gbdt']['candidates']]
        r = self.document['rf']['validation_results']
        groups.append((r['tasks'], r['mean_validation_ap'], r['sample_std_ap'], r['mean_fit_seconds'], r['sample_std_fit_seconds'], 'fit_seconds'))
        for tasks, mean, std, seconds_mean, seconds_std, key in groups:
            values = [t['validation_ap'] for t in tasks]
            independent_mean = math.fsum(values) / 5
            independent_std = math.sqrt(math.fsum((v - independent_mean)**2 for v in values) / 4)
            self.assertTrue(math.isclose(independent_mean, mean, rel_tol=1e-12, abs_tol=1e-15))
            self.assertTrue(math.isclose(independent_std, std, rel_tol=1e-12, abs_tol=1e-15))
            self.assertTrue(math.isclose(statistics.mean(t[key] for t in tasks), seconds_mean, rel_tol=1e-12))
            self.assertTrue(math.isclose(statistics.stdev(t[key] for t in tasks), seconds_std, rel_tol=1e-12))

    def test_only_allowlisted_small_public_files_are_opened(self):
        original = Path.read_bytes
        allowed = {(ROOT / p).resolve() for p in [*aggregate.SOURCES.values(), aggregate.GENERATOR, aggregate.SCHEMA]}
        opened = []
        def read(path):
            self.assertIn(path.resolve(), allowed)
            opened.append(path.resolve())
            return original(path)
        with patch.object(Path, 'read_bytes', read):
            aggregate.validate_document(self.document, ROOT)
        self.assertEqual(set(opened), allowed)

    def test_no_ml_or_dataset_loader_imports(self):
        tree = ast.parse((ROOT / aggregate.GENERATOR).read_text(encoding='utf-8'))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        forbidden = ('sklearn', 'numpy', 'duckdb', 'joblib', 'methods', 'src', 'eval')
        self.assertFalse([name for name in imports if name and name.startswith(forbidden)])

    def test_g02_standby_timing_warning_preserved(self):
        candidate = self.document['gbdt']['candidates'][1]
        self.assertIn('Modern Standby', candidate['timing_note'])
        self.assertGreater(candidate['tasks'][0]['fit_seconds_raw'], 1200)

    def test_missing_candidate_rejected_by_schema(self):
        changed = deepcopy(self.document)
        changed['gbdt']['candidates'].pop()
        with self.assertRaises(ValidationError): Draft202012Validator(self.schema).validate(changed)

    def test_missing_seed_rejected_by_schema(self):
        changed = deepcopy(self.document)
        changed['rf']['validation_results']['tasks'].pop()
        with self.assertRaises(ValidationError): Draft202012Validator(self.schema).validate(changed)

    def test_duplicate_candidate_rejected(self):
        self.validate_mutation(lambda d: d['gbdt']['candidates'].__setitem__(1, deepcopy(d['gbdt']['candidates'][0])))

    def test_duplicate_seed_rejected(self):
        self.validate_mutation(lambda d: d['rf']['validation_results']['tasks'][1].__setitem__('seed', 42))

    def test_per_seed_score_change_even_with_same_mean_rejected(self):
        def swap(d):
            tasks = d['gbdt']['candidates'][0]['tasks']
            tasks[0]['validation_ap'], tasks[1]['validation_ap'] = tasks[1]['validation_ap'], tasks[0]['validation_ap']
        self.validate_mutation(swap)

    def test_selected_g05_duplicate_disagreement_rejected(self):
        self.validate_mutation(lambda d: d['gbdt']['selected_candidate']['tasks'][0].__setitem__('validation_ap', 0.5))

    def test_wrong_mean_rejected(self):
        self.validate_mutation(lambda d: d['rf']['validation_results'].__setitem__('mean_validation_ap', 0.9))

    def test_population_std_instead_of_sample_std_rejected(self):
        self.validate_mutation(lambda d: d['rf']['validation_results'].__setitem__('sample_std_ap',
            statistics.pstdev(t['validation_ap'] for t in d['rf']['validation_results']['tasks'])))

    def test_wrong_fit_time_rejected(self):
        self.validate_mutation(lambda d: d['gbdt']['candidates'][0]['tasks'][0].__setitem__('fit_seconds_raw', -1))

    def test_source_sha_change_rejected(self):
        self.validate_mutation(lambda d: d['provenance']['sources']['rf_summary'].__setitem__('sha256', '0'*64))

    def test_config_change_rejected(self):
        self.validate_mutation(lambda d: d['rf']['execution_config']['parameters'].__setitem__('n_estimators', 100))

    def test_validation_only_and_unfinished_metrics_enforced(self):
        for key in ('test_accessed', 'is_final_test_evaluation', 'bootstrap_run', 'thresholds_calibrated', 'prediction_AP_recomputed_in_this_step'):
            with self.subTest(key=key):
                self.validate_mutation(lambda d: d['scope'].__setitem__(key, True))
        self.validate_mutation(lambda d: d['rf']['validation_results'].__setitem__('roc_auc', 1.0))

    def test_nonfinite_or_out_of_range_ap_rejected(self):
        for score in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(score=score):
                self.validate_mutation(lambda d: d['rf']['validation_results']['tasks'][0].__setitem__('validation_ap', score))

    def test_bad_source_statistics_refused_before_generation(self):
        original = aggregate.source
        def corrupted(root, path):
            value, provenance = original(root, path)
            if path == aggregate.SOURCES['gbdt_summary']:
                value['candidates'][0]['sample_std_ap'] = 0
            return value, provenance
        with patch.object(aggregate, 'source', corrupted), self.assertRaisesRegex(ValueError, 'standard deviation'):
            aggregate.build(ROOT)

    def test_source_pairing_and_receipt_mismatch_refused(self):
        original = aggregate.source
        for source_name in ('rf_summary', 'rf_receipt'):
            def corrupted(root, path):
                value, provenance = original(root, path)
                if path == aggregate.SOURCES[source_name]:
                    if source_name == 'rf_summary': value['paired_reference']['inputs']['val.parquet'] = '0'*64
                    else: value['per_seed'][0]['independently_recomputed_ap'] = 0.5
                return value, provenance
            with self.subTest(source=source_name), patch.object(aggregate, 'source', corrupted), self.assertRaises(ValueError):
                aggregate.build(ROOT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
