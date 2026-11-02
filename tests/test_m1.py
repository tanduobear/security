"""M1 engineering tests: synthetic fixtures only, never real train/val/test."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from eval.metrics import paired_bootstrap
from methods.m1 import data, models, run_smoke
from methods.m1.evaluation import family_coverage, score_metrics, timing_indices
from methods.m1.formal import main as formal_main, prepare_run, protocol_digest, protocol_issues
from methods.m1.storage import RunStore, verify_run
from methods.m1.validation import validate_record, validate_saved_predictions

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'eval/schema.json').read_text(encoding='utf-8'))


class FakeDataset:
    feature_names = ['f0', 'f1']
    class_weights = {0: 0.6, 1: 3.0}
    calls = []

    def __init__(self, path, chunks=(13, 28, 59)):
        self.path = Path(path)
        if self.path.name not in ('train.parquet', 'val.parquet'):
            raise AssertionError('Test data must not be opened')
        self.calls.append(self.path.name)
        self.chunks = chunks
        self.manifest = {'output': {self.path.stem: {'rows': 100}}}

    def __iter__(self):
        lo = 0
        for size in self.chunks:
            ids = np.arange(lo, lo + size)
            y = (ids % 2).astype(np.uint8)
            yield (np.column_stack((ids / 100, np.sin(ids))).astype(np.float32), y,
                   pd.DataFrame({'row_index': ids, 'row_id': [self.path.stem + ':' + str(i) for i in ids],
                                 'y_binary': y, 'attack_family': np.where(y == 1, 'ddos', 'benign'),
                                 'is_attempted': np.zeros(size, dtype=bool),
                                 'label_raw': ['never-a-feature'] * size, 'capture_date': ['2017-07-03'] * size}))
            lo += size


def fixture_record():
    """A fully synthetic smoke record, independent of historical result files."""
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.uint8)
    a = np.array([.1, .3, .8, .7, .2, .6])
    b = np.array([.15, .35, .9, .85, .25, .65])
    family = np.where(y == 1, 'ddos', 'benign')
    flags = np.zeros(6, dtype=bool)
    ids, training_ids = np.array(['v' + str(i) for i in range(6)]), np.array(['t' + str(i) for i in range(6)])
    metadata = pd.DataFrame({'attack_family': family, 'is_attempted': flags})
    names = ['ddos', 'heartbleed']
    cfg = {'purpose': 'development_smoke_only', 'dataset_directory': 'synthetic', 'seed': 7,
           'sampling_seeds': {'train': 1, 'validation': 2}, 'bootstrap_seed': 7, 'timing_seed': 3,
           'train_sample_rows': 6, 'validation_sample_rows': 6, 'fit_threads': 1, 'threshold': .5,
           'inference': {'batch_size': 2, 'warmup': 1, 'repeats': 2, 'threads': 1},
           'bootstrap_repetitions': 4, 'models': {'random_forest': {}, 'hist_gradient_boosting': {}}}
    timing = {'status': 'ok', 'device': 'cpu', 'scope': 'prepared_batch_to_numpy_scores',
              'batch_size': 2, 'warmup': 1, 'repeats': 2, 'threadpool_limit': 1,
              'median_batch_ms': 1., 'amortized_ms_per_flow': .5, 'throughput_flows_per_second': 2000., 'batch_times_ms': [1., 1.]}
    record = {'format_version': '2', 'purpose': cfg['purpose'], 'evaluation_split': 'validation_subset', 'config': cfg,
              'data': {'directory': 'synthetic', 'features': ['f0', 'f1'], 'train_source_rows': 10, 'validation_source_rows': 10,
                       'train_used_rows': 6, 'validation_used_rows': 6, 'train_sha256': 'a' * 64, 'validation_sha256': 'b' * 64,
                       'preprocessor_sha256': 'c' * 64, 'sampling': 'synthetic unit fixture', 'training_class_weights': {'0': .6, '1': 3.},
                       'sampling_seeds': cfg['sampling_seeds'],
                       'expected_attack_families': names, 'train_coverage': family_coverage(y, family, names),
                       'validation_coverage': family_coverage(y, family, names),
                       'train_row_ids_sha256': data.row_digest(training_ids), 'validation_row_ids_sha256': data.row_digest(ids),
                       'timing_row_ids': ids[:2].tolist(), 'timing_row_ids_sha256': data.row_digest(ids[:2]),
                       'timing_composition': {'benign': 2, 'attack': 0}},
              'decision_rule': {'threshold': .5, 'operator': '>=', 'source': 'predefined_for_smoke'},
              'models': {name: {'parameters': {}, 'effective_parameters': {'random_state':7,'class_weight':None}, 'training_seconds': .01, 'inference': {**timing,'model_worker_limit':1},
                                'model_reload_prediction_equal': True, 'metrics': score_metrics(y, scores, metadata, .5, names)}
                         for name, scores in [('random_forest', a), ('hist_gradient_boosting', b)]},
              'provenance': {'git_commit': 'synthetic', 'python': '3', 'platform': 'unit-test', 'note': 'synthetic fixture only',
                             'software': {'numpy': np.__version__}, 'files_sha256': {'fixture': 'd' * 64}},
              'comparison': {'scope': 'development validation subset only', 'method_a': 'hist_gradient_boosting', 'method_b': 'random_forest',
                             'result': paired_bootstrap(y, b, a, row_ids_a=ids.tolist(), row_ids_b=ids.tolist(), n_bootstrap=4, seed=7)}}
    saved = {'y_true': y, 'row_ids': ids, 'attack_family': family, 'is_attempted': flags,
             'train_row_ids': training_ids, 'random_forest': a, 'hist_gradient_boosting': b}
    return record, saved


class M1DataTests(unittest.TestCase):
    def test_metadata_rejected_as_features(self):
        for name in ['Label', 'y_binary', 'attack_family', 'capture_date', 'row_id', 'row_index']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                data.check_features(['f0', name])

    def test_feature_dimension_and_metadata_labels_must_match(self):
        X, y, meta = next(iter(FakeDataset('train.parquet')))
        for altered_X, altered_y in [(X[:, :1], y), (X, 1 - y)]:
            with self.assertRaises(ValueError):
                data.checked_batch(altered_X, altered_y, meta, ['f0', 'f1'])

    def test_sampling_is_reproducible_and_batch_boundary_invariant(self):
        a = data.sample_batches(FakeDataset('train.parquet'), ['f0', 'f1'], 20, 42)
        b = data.sample_batches(FakeDataset('train.parquet', (20, 20, 60)), ['f0', 'f1'], 20, 42)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertTrue(a[2].equals(b[2]))
        expected = np.sort(np.argpartition(np.random.default_rng(42).random(100), 19)[:20])
        np.testing.assert_array_equal(a[2].row_index, expected)

    def test_invalid_sample_request_rejected(self):
        for size in [0, -1, 101]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                data.sample_batches(FakeDataset('train.parquet'), ['f0', 'f1'], size, 42)

    def test_model_seed_does_not_change_sampling_seeds(self):
        cfg = json.loads((ROOT / 'methods/m1/config_smoke.json').read_text())
        before = run_smoke.validate_config(cfg)
        cfg['seed'] = 999
        self.assertEqual(before, run_smoke.validate_config(cfg))

    def test_overlap_or_duplicate_ids_rejected(self):
        for a, b in [(['t', 't'], ['v']), (['t'], ['v', 'v']), (['t'], ['t'])]:
            with self.assertRaises(ValueError):
                data.assert_isolated(a, b)

    def test_main_and_ablation_paths_and_test_access_gate(self):
        main = data.preparation_path(ROOT, 'original', 'stratified', 'main', 'train')
        ablation = data.preparation_path(ROOT, 'corrected', 'temporal', 'no_destination_port', 'val')
        self.assertEqual(main.name, 'train.parquet')
        self.assertEqual(ablation.parent.name, 'no_destination_port')
        with self.assertRaises(ValueError):
            data.preparation_path(ROOT, 'original', 'stratified', 'main', 'test')
        with patch.object(data, 'ModelDataset') as loader:
            with self.assertRaises(ValueError):
                data.materialize_preparation('test.parquet', 'unused')
            loader.assert_not_called()

    def test_disk_materialization_is_preallocated_and_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = data.materialize_preparation('train.parquet', Path(tmp) / 'train', loader_factory=FakeDataset)
            b = data.materialize_preparation('val.parquet', Path(tmp) / 'val', loader_factory=FakeDataset)
            model_input = a.model_input().validate()
            self.assertIsInstance(model_input.X, np.memmap)
            self.assertEqual(model_input.X.shape, (100, 2))
            self.assertEqual(model_input.X.dtype, np.float32)
            meta = pd.concat(a.metadata_batches(rows=11), ignore_index=True)
            self.assertEqual(len(meta), 100)
            np.testing.assert_array_equal(meta.y_binary, model_input.y)
            data.assert_disk_isolated(a, b, memory_limit_mb=64)
            model_input.close()
            with self.assertRaises(FileExistsError):
                data.materialize_preparation('train.parquet', a.directory, loader_factory=FakeDataset)

    def test_failed_materialization_is_not_ready(self):
        class BadCount(FakeDataset):
            def __init__(self, path):
                super().__init__(path)
                self.manifest['output'][self.path.stem]['rows'] = 101
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'bad'
            with self.assertRaises(ValueError):
                data.materialize_preparation('train.parquet', target, loader_factory=BadCount)
            self.assertTrue((target / 'FAILED').exists())
            self.assertFalse((target / 'READY.json').exists())


class M1ModelTests(unittest.TestCase):
    def training(self, split='train'):
        return data.ModelInput(np.ones((4, 2), dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.uint8), ('f0', 'f1'), split, {0: .6, 1: 3.})

    def test_only_training_fold_is_fitted(self):
        with patch.object(models, 'RandomForestClassifier') as constructor:
            with self.assertRaises(ValueError):
                models.fit_tree('random_forest', {}, self.training('val'), model_seed=7, fit_threads=1)
            constructor.assert_not_called()

    def test_weights_and_model_seed_reach_fit(self):
        train = self.training()
        for name, symbol in [('random_forest','RandomForestClassifier'), ('hist_gradient_boosting','HistGradientBoostingClassifier')]:
            with self.subTest(name=name), patch.object(models, symbol) as constructor:
                model, _ = models.fit_tree(name, {}, train, model_seed=7, fit_threads=1)
                self.assertIs(model, constructor.return_value)
                call = model.fit.call_args
                self.assertIs(call.args[0], train.X)
                self.assertIs(call.args[1], train.y)
                np.testing.assert_array_equal(call.kwargs['sample_weight'], [.6, 3., .6, 3.])
                self.assertEqual(constructor.call_args.kwargs['random_state'], 7)

    def test_double_weight_and_unbounded_workers_rejected(self):
        for params in [{'class_weight': 'balanced'}, {'n_jobs': -1}, {'n_jobs': 2}, {'random_state': 3}]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                models.fit_tree('random_forest', params, self.training(), model_seed=7, fit_threads=1)

    def test_attack_column_uses_class_identity_not_position(self):
        class Reversed:
            classes_ = np.array([1, 0])
            def predict_proba(self, X):
                return np.tile([.8, .2], (len(X), 1))
        np.testing.assert_array_equal(models.attack_scores(Reversed(), np.ones((3, 2))), [.8, .8, .8])

    def test_malformed_model_scores_rejected(self):
        class Invalid:
            classes_ = np.array([0, 1])
            def __init__(self, output):self.output = output
            def predict_proba(self, X):return self.output
        for output in [np.zeros(2), np.full((2, 2), np.nan), np.ones((2, 2)), np.array([[2., -1.], [2., -1.]])]:
            with self.assertRaises(ValueError):
                models.attack_scores(Invalid(output), np.ones((2, 2)))

    def test_shared_timing_ids_are_order_invariant(self):
        ids = ['v' + str(i) for i in range(30)]
        a = timing_indices(ids, batch_size=6, seed=5)
        b = timing_indices(ids[::-1], batch_size=6, seed=5)
        self.assertEqual(np.asarray(ids)[a].tolist(), np.asarray(ids[::-1])[b].tolist())
        explicit = ['v3', 'v7']
        c = timing_indices(ids, batch_size=2, selected_row_ids=explicit)
        self.assertEqual(np.asarray(ids)[c].tolist(), explicit)
        with self.assertRaises(ValueError):timing_indices(ids, batch_size=1, selected_row_ids=['missing'])

    def test_inference_worker_override_is_restored_even_on_failure(self):
        class WorkerModel:n_jobs = 4
        model = WorkerModel()
        with self.assertRaises(RuntimeError):
            with models.inference_workers(model,1):
                self.assertEqual(model.n_jobs,1)
                raise RuntimeError('synthetic')
        self.assertEqual(model.n_jobs,4)

    def test_absent_families_are_explicit_null_recall(self):
        record, _ = fixture_record()
        entry = record['models']['random_forest']['metrics']['per_class']['classes'][1]
        self.assertEqual(entry, {'attack_family': 'heartbleed', 'support': 0, 'detected': 0, 'recall': None})
        self.assertEqual(record['data']['validation_coverage']['missing'], ['heartbleed'])


class M1ValidationTests(unittest.TestCase):
    def test_valid_synthetic_record_and_predictions_pass(self):
        record, saved = fixture_record()
        self.assertTrue(validate_record(record, SCHEMA))
        self.assertTrue(validate_saved_predictions(record, saved))

    def test_corrupt_records_are_rejected(self):
        record, _ = fixture_record()
        mutations = {
            'accuracy': lambda r:r['models']['random_forest']['metrics']['classification'].__setitem__('accuracy', 0.),
            'missing inference': lambda r:r['models']['random_forest'].__setitem__('inference', {}),
            'median latency': lambda r:r['models']['random_forest']['inference'].__setitem__('median_batch_ms', 5.),
            'data identity': lambda r:r.pop('data'),
            'comparison': lambda r:r.__setitem__('comparison', {}),
            'comparison value': lambda r:r['comparison']['result'].__setitem__('value_a', .1),
            'interval': lambda r:r['comparison']['result'].__setitem__('ci_lower', 1.),
            'family omission': lambda r:r['models']['random_forest']['metrics']['per_class']['classes'].pop(),
            'metadata feature': lambda r:r['data']['features'].__setitem__(0, 'row_id'),
            'threshold': lambda r:r['decision_rule'].__setitem__('threshold', .9),
            'row hash': lambda r:r['data'].__setitem__('timing_row_ids_sha256', 'a' * 64),
            'NaN': lambda r:r['models']['random_forest'].__setitem__('training_seconds', float('nan')),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                damaged = copy.deepcopy(record)
                mutate(damaged)
                with self.assertRaises((ValueError, KeyError)):
                    validate_record(damaged, SCHEMA)

    def test_saved_metadata_scores_and_row_order_are_checked(self):
        record, saved = fixture_record()
        for name in ['row_ids', 'attack_family', 'y_true', 'random_forest', 'train_row_ids', 'is_attempted']:
            with self.subTest(name=name):
                damaged = {k:v.copy() for k,v in saved.items()}
                if name == 'row_ids':damaged[name] = damaged[name][::-1]
                elif name == 'attack_family':damaged[name][2] = 'heartbleed'
                elif name == 'y_true':damaged[name][0] = 1
                elif name == 'random_forest':damaged[name][0] = .9
                elif name == 'train_row_ids':damaged[name][0] = 'v0'
                else:damaged[name] = np.zeros(6, dtype=np.uint8)
                with self.assertRaises(ValueError):validate_saved_predictions(record, damaged)


class M1StorageTests(unittest.TestCase):
    def make_run(self, tmp):
        record, saved = fixture_record()
        store = RunStore(tmp, kind='smoke', run_id='unit')
        store.write_json('config.json', record['config'])
        store.write_json('metrics.json', record)
        store.write_predictions(saved)
        for name in record['models']:
            store.write_model('models/' + name + '.joblib', {'synthetic': True})
        store.finish()
        return store

    def test_saved_files_and_config_hashes_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_run(tmp)
            self.assertEqual(verify_run(store.path)['status'], 'complete')
            self.assertEqual(joblib.load(store.path / 'models/random_forest.joblib'), {'synthetic': True})
            self.assertIn('environment.json', json.loads((store.path / 'MANIFEST.json').read_text())['files'])

    def test_run_and_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp, kind='smoke', run_id='unit')
            store.write_json('config.json', {})
            with self.assertRaises(ValueError):store.write_json('config.json', {'wrong': True})
            with self.assertRaises(FileExistsError):RunStore(tmp, kind='smoke', run_id='unit')
            with self.assertRaises(ValueError):store.file('../escape.json')
            with self.assertRaises(ValueError):store.finish()
            store.fail(ValueError('test incomplete run'))
            with self.assertRaises(ValueError):store.write_json('late.json', {})

    def test_artifact_corruption_or_extra_file_is_rejected(self):
        for extra in [True, False]:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                store = self.make_run(tmp)
                target = store.path / ('extra.json' if extra else 'config.json')
                target.write_text('{}')
                with self.assertRaises(ValueError):verify_run(store.path)

    def test_failed_or_unsealed_run_cannot_be_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp, kind='smoke')
            with self.assertRaises(FileNotFoundError):verify_run(store.path)
            store.fail(ValueError('unit fixture'))
            with self.assertRaises(ValueError):verify_run(store.path)


def approved_fixture():
    """Simulated approval for gate tests, never a real team approval document."""
    p = json.loads((ROOT / 'methods/m1/protocol.pending.json').read_text())
    p.update(dataset={'version':'original','protocol':'stratified','variant':'main'},
             search={'configurations_per_algorithm':1,'budget_scope':'synthetic fixture only','strategy':'fixture','seed':1,
                     'configurations':{'random_forest':[{}],'hist_gradient_boosting':[{}]}},
             model_seeds=[1,2,3,4,5], selection={'metric':'fixture','aggregation':'fixture','tie_rule':'fixture'},
             early_stopping={'rule':'fixture'}, threshold_policy={'rule':'fixture','source_split':'val'},
             inference={'row_ids_path':'not-opened','population_identity':'fixture','batch_size':2,'warmup':0,'repeats':1,'threads':1},
             expected_attack_families=['ddos'], resources={'fit_threads':1,'duckdb_memory_limit_mb':32})
    p['approval'] = {'status':'approved','reviewers':{'P4':'TEST ONLY','P5':'TEST ONLY','P6':'TEST ONLY'},
                     'approved_at':'TEST ONLY','protocol_sha256':protocol_digest(p)}
    return p


class M1FormalGateTests(unittest.TestCase):
    def test_pending_protocol_blocks_before_any_data_access(self):
        p = json.loads((ROOT / 'methods/m1/protocol.pending.json').read_text())
        with tempfile.TemporaryDirectory() as tmp, patch.object(data, 'ModelDataset') as loader:
            target = Path(tmp) / 'no-output'
            with self.assertRaises(ValueError):prepare_run(p, target)
            self.assertFalse(target.exists())
            loader.assert_not_called()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(formal_main(['--protocol', str(ROOT / 'methods/m1/protocol.pending.json')]), 2)

    def test_changed_protocol_invalidates_recorded_approval(self):
        p = approved_fixture()
        self.assertEqual(protocol_issues(p), [])
        p['selection']['metric'] = 'changed'
        self.assertTrue(protocol_issues(p))

    def test_prepared_plan_never_loads_data_or_fits_a_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(data, 'ModelDataset') as loader, patch.object(models, 'fit_tree') as fitter:
            path = prepare_run(approved_fixture(), tmp)
            self.assertEqual(verify_run(path)['status'], 'prepared_not_executed')
            loader.assert_not_called()
            fitter.assert_not_called()
            self.assertFalse((path / 'models').exists())


class M1SmokeIntegrationTests(unittest.TestCase):
    def test_original_cli_flow_with_tiny_synthetic_data(self):
        cfg = json.loads((ROOT / 'methods/m1/config_smoke.json').read_text())
        cfg.update(train_sample_rows=40, validation_sample_rows=24, fit_threads=1, bootstrap_repetitions=4,
                   inference={'batch_size':8,'warmup':1,'repeats':2,'threads':1},
                   models={'random_forest':{'n_estimators':3,'max_depth':3,'n_jobs':1},
                           'hist_gradient_boosting':{'max_iter':3,'max_leaf_nodes':3,'min_samples_leaf':2,'early_stopping':False}})
        FakeDataset.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'config.json'
            config_path.write_text(json.dumps(cfg))
            # Source hashes are mocked, so no real dataset or processor is read.
            with patch.object(run_smoke, 'ModelDataset', FakeDataset), patch.object(run_smoke, 'sha', return_value='a' * 64), patch.object(run_smoke, 'fit_tree', wraps=models.fit_tree) as fitter, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_smoke.main(['--config',str(config_path),'--output-root',str(Path(tmp)/'runs')]),0)
                self.assertEqual(fitter.call_count,2)
                self.assertIs(fitter.call_args_list[0].args[2],fitter.call_args_list[1].args[2])
                fitted_input = fitter.call_args_list[0].args[2]
                self.assertEqual(fitted_input.X.shape,(40,2))
                self.assertEqual(tuple(fitted_input.feature_names),('f0','f1'))
            runs = list((Path(tmp)/'runs/smoke').iterdir())
            self.assertEqual(len(runs),1)
            verify_run(runs[0])
            record = json.loads((runs[0]/'metrics.json').read_text())
            self.assertTrue(validate_record(record,SCHEMA))
            self.assertEqual(FakeDataset.calls,['train.parquet','val.parquet'])
            for item in record['models'].values():
                self.assertTrue(item['model_reload_prediction_equal'])
                self.assertEqual(item['effective_parameters']['random_state'],cfg['seed'])
            with np.load(runs[0]/'predictions.npz',allow_pickle=False) as saved:
                self.assertEqual(len(saved['row_ids']),24)
                self.assertTrue(validate_saved_predictions(record,saved))


if __name__ == '__main__':
    unittest.main()
