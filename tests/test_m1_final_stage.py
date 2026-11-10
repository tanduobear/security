"""Final-stage gates and fake-model integrations use temporary synthetic files only."""
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

from methods.m1 import final_stage as cli, final_stage_core as core
from methods.m1.data import row_digest, materialize_preparation
from methods.m1.executor import metadata_arrays
from methods.m1.seal_gbdt import file_inventory
from test_m1_executor import SyntheticP2, FixtureModel, fixture_files

ROOT = Path(__file__).resolve().parents[1]


def config():
    return cli.read(ROOT / 'methods/m1/execution.final-test.proposed.json')


class FrozenFixtureModel(FixtureModel):
    def get_params(self, deep=True):
        return super().get_params()


def fixture(root):
    """640 fake Validation/Test samples, no estimator fitting, no real data."""
    root = Path(root)
    cfg = config()
    directory = fixture_files(root)
    cache = root / 'cache'
    train = materialize_preparation(directory / 'train.parquet', cache / 'train', loader_factory=SyntheticP2)
    val = materialize_preparation(directory / 'val.parquet', cache / 'val', loader_factory=SyntheticP2)
    ids, families, flags = metadata_arrays(val)
    with val.model_input() as data:
        y = data.y.copy()
        features = list(data.feature_names)
    timing = {'source_split': 'val', 'population_row_ids_sha256': row_digest(ids), 'row_ids': ids[:512].tolist()}
    cli.write(root / 'common.json', timing)
    cfg['timing'].update(row_ids_path='common.json', row_ids_sha256=cli.sha(root / 'common.json'))
    policy = cli.read(ROOT / 'docs/label_split_policy.json')
    cli.write(root / 'docs/label_split_policy.json', policy)
    cli.write(root / 'methods/m1/protocol.unified.v1.json', cli.read(ROOT / 'methods/m1/protocol.unified.v1.json'))
    p2 = {'status': 'verified_train_only_and_model_data_complete', 'features': features,
          'precision_guard': {'checks': {'no_transformed_group_overlap': True}},
          'output': {'test': {'rows': 640, 'main': {'sha256': cli.sha(directory / 'test.parquet')}}}}
    cli.write(directory / 'preprocess_manifest.json', p2)
    inputs = {name: cli.sha(directory / name) for name in ('train.parquet', 'val.parquet', 'preprocessor.json', 'preprocess_manifest.json')}
    m1 = copy.deepcopy(cli.read(ROOT / 'results/m1.json'))
    m1['dataset']['input_file_sha256'] = inputs
    m1['dataset']['validation_rows'] = 640
    m1['provenance']['sources'] = {}
    source = {'jobs': [], 'inputs': inputs, 'features': features, 'families': [x for x in policy['family_names'] if x != 'benign'],
              'val_ids': ids, 'val_y': y, 'cache': cache, 'data_dir': directory, 'p2_manifest': p2,
              'root': root, 'code': {'synthetic.py': '1' * 64}, 'm1_sha256': '2' * 64, 'archive_pins': cfg['archives']}
    dependencies = ('src/model_loader.py', 'src/preprocess.py', 'eval/metrics.py', 'eval/schema.json',
                    'eval/validate.py', 'methods/m1/models.py', 'methods/m1/data.py', 'methods/m1/evaluation.py')
    for name in dependencies:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    for model in cfg['models']:
        kind = 'GBDT' if model == 'G05' else 'RF'
        run = root / cfg['archives'][kind]['path']
        run.mkdir(parents=True)
        archived_config = m1['gbdt' if kind == 'GBDT' else 'rf']['execution_config']
        binding = {'inputs': inputs, 'config_sha256': cli.digest(archived_config),
                   'unified_file_sha256': cli.sha(root / 'methods/m1/protocol.unified.v1.json'),
                   'code': {name: cli.sha(root / name) for name in dependencies}, 'environment': {}}
        cli.write(run / 'config.json', archived_config)
        cli.write(run / 'binding.json', binding)
        for seed in cfg['seeds']:
            params = m1['gbdt']['selected_candidate']['parameters'] if kind == 'GBDT' else archived_config['parameters']
            task = {'task_id': f'{model}_seed_{seed}', 'candidate_id': model, 'seed': seed,
                    'algorithm': 'hist_gradient_boosting' if kind == 'GBDT' else 'random_forest', 'params': params}
            attempt = run / 'attempts' / task['task_id']
            attempt.mkdir(parents=True)
            cli.write(attempt / 'task.json', task)
            joblib.dump(FrozenFixtureModel(params, seed), attempt / 'model.joblib')
            np.savez_compressed(attempt / 'validation.npz', row_ids=ids, y_true=y, y_score=.1 + .8 * y,
                                attack_family=families, is_attempted=flags)
            record = {'task': task, 'artifact_directory': attempt.relative_to(run).as_posix(), 'files': file_inventory(attempt),
                      'binding_sha256': cli.digest(binding), 'validation_ap': 1., 'validation_rows': 640,
                      'validation_row_ids_sha256': row_digest(ids), 'training_seconds': .001, 'test_accessed': False}
            cli.write(run / 'completed' / (task['task_id'] + '.json'), record)
            source['jobs'].append({**task, 'role': 'main' if kind == 'GBDT' else 'auxiliary',
                'model_path': attempt / 'model.joblib', 'model_sha256': cli.sha(attempt / 'model.joblib'),
                'prediction_path': attempt / 'validation.npz', 'prediction_sha256': cli.sha(attempt / 'validation.npz'),
                'training_seconds': .001, 'validation_ap': 1.})
        if kind == 'GBDT':
            cli.write(run / 'selection.json', {'selected_candidate_id': 'G05', 'binding_sha256': cli.digest(binding)})
            cfg['archives'][kind]['selection_sha256'] = cli.sha(run / 'selection.json')
            # This synthetic archive references its own mapped P2 cache copy.
            # hard links avoid any second data construction in the fixture.
            import os
            for path in cache.rglob('*'):
                if path.is_file():
                    target = run / 'cache' / path.relative_to(cache)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(path, target)
            cli.write(run / 'cache.json', {'directory': 'cache'})
        cli.write(run / 'MANIFEST.json', {'status': 'gbdt_search_and_validation_selection_complete' if kind == 'GBDT'
                  else 'rf_auxiliary_validation_complete', 'files': file_inventory(run)})
        cfg['archives'][kind]['manifest_sha256'] = cli.sha(run / 'MANIFEST.json')
        if kind == 'GBDT':
            m1['gbdt']['archive']['manifest_sha256'] = cfg['archives'][kind]['manifest_sha256']
            m1['gbdt']['selected_candidate']['tasks'] = [{'seed': seed, 'validation_ap': 1.} for seed in cfg['seeds']]
        else:
            m1['rf']['historical_verification']['RF_manifest_sha256'] = cfg['archives'][kind]['manifest_sha256']
            m1['rf']['validation_results']['tasks'] = [{'seed': seed, 'validation_ap': 1.} for seed in cfg['seeds']]
    cli.write(root / 'results/m1.json', m1)
    cfg['m1_summary_sha256'] = cli.sha(root / 'results/m1.json')
    source['archive_pins'] = cfg['archives']
    for stage in ('calibrate', 'evaluate'):
        cfg['approvals'][stage] = {'status': 'approved', 'source': 'SYNTHETIC UNIT TEST ONLY',
            'approved_at': 'UNIT TEST ONLY', 'payload_sha256': cli.digest(cli.payload(cfg)),
            'code_sha256': cli.digest(cli.code_identity())}
    return cfg, source


class FinalStageGateTests(unittest.TestCase):
    def test_plan_has_exactly_ten_frozen_models_and_twenty_thresholds(self):
        result = cli.plan(config())
        self.assertEqual(result['task_count'], 10)
        self.assertEqual(result['threshold_record_count'], 20)
        self.assertEqual(len({x['task_id'] for x in result['tasks']}), 10)
        self.assertEqual(result['training_fits'], 0)
        self.assertFalse(result['test_accessed'])

    def test_pending_stages_refuse_before_resources_outputs_or_data(self):
        for stage in ('calibrate', 'evaluate'):
            with patch('methods.m1.resource_trial.WindowsAPI') as api, patch.object(core, 'execute') as execute:
                with self.assertRaisesRegex(ValueError, 'pending'):
                    cli.launch(config(), stage, enabled=True, permit_test=True)
                api.assert_not_called()
                execute.assert_not_called()

    def test_explicit_launch_and_test_access_are_separate_gates(self):
        with self.assertRaisesRegex(ValueError, 'launch'):
            cli.require_approval(config(), 'calibrate')
        with self.assertRaisesRegex(ValueError, 'Test access'):
            cli.require_approval(config(), 'evaluate', launch=True)

    def test_mutated_scope_and_resources_refused(self):
        for key, value in [('seeds', [42]), ('models', ['G01']), ('training_fits', 1), ('bootstrap', 'automatic')]:
            c = config();c[key] = value
            with self.assertRaises(ValueError):cli.validate(c)
        c = config();c['resources']['job_commit_limit_bytes'] += 1
        with self.assertRaises(ValueError):cli.validate(c)

    def test_approval_requires_exact_payload_code_and_shared_ids(self):
        c = config()
        c['approvals']['calibrate'].update(status='approved', source='SYNTHETIC UNIT TEST ONLY', approved_at='TEST')
        with self.assertRaisesRegex(ValueError, 'payload'):
            cli.require_approval(c, 'calibrate', launch=True)
        c['approvals']['calibrate']['payload_sha256'] = cli.digest(cli.payload(c))
        with self.assertRaisesRegex(ValueError, 'code'):
            cli.require_approval(c, 'calibrate', launch=True)
        c['approvals']['calibrate']['code_sha256'] = cli.digest(cli.code_identity())
        with self.assertRaisesRegex(ValueError, 'timing'):
            cli.require_approval(c, 'calibrate', launch=True)

    def test_default_plan_does_not_call_worker_or_loader(self):
        with patch.object(cli, 'launch') as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([]), 0)
            launch.assert_not_called()

    def test_worker_without_guarded_go_cannot_import_execution(self):
        with patch('sys.stdin', io.StringIO('NO\n')), patch.object(core, 'execute') as execute:
            with self.assertRaisesRegex(ValueError, 'GO'):cli.worker('does-not-exist')
            execute.assert_not_called()

    def test_no_fit_or_bootstrap_call_in_new_modules(self):
        import ast
        for module in (cli, core):
            tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
            calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
            names = [getattr(f, 'attr', getattr(f, 'id', '')) for f in calls]
            self.assertTrue(set(names).isdisjoint({'fit', 'fit_tree', 'paired_bootstrap', 'execute_search', 'execute_selection'}))

    def test_path_traversal_refused(self):
        for value in ('../escape', str(ROOT)):
            with self.assertRaises(ValueError):cli.safe_path(ROOT, value)

    def test_insufficient_memory_refuses_before_creating_output(self):
        with patch.object(cli, 'require_approval'), patch('methods.m1.resource_trial.WindowsAPI') as api:
            api.return_value.memory.return_value = {'available_physical_bytes': 13 * cli.GIB}
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, '14 GiB'):
                    cli.launch(config(), 'calibrate', enabled=True, project=tmp)
                self.assertFalse((Path(tmp) / '.tmp').exists())

    def test_resource_abort_preserves_failure_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, 'require_approval'), \
             patch.object(cli, 'resource_slot', contextlib.nullcontext), \
             patch('methods.m1.resource_trial.WindowsAPI') as api, \
             patch('methods.m1.resource_trial.monitored_child') as child, patch.object(cli.shutil, 'disk_usage') as disk:
            api.return_value.memory.return_value = {'available_physical_bytes': 20 * cli.GIB}
            disk.return_value.free = 100 * cli.GIB
            child.return_value = {'exit_code': 71, 'abort_reason': 'rss_above_8_GiB'}
            with self.assertRaisesRegex(ValueError, 'aborted'):
                cli.launch(config(), 'calibrate', enabled=True, project=tmp)
            self.assertEqual(child.call_count, 1)
            limits = child.call_args.args[1]
            self.assertEqual(limits['job_commit_limit_bytes'], 8 * cli.GIB)
            self.assertEqual(limits['logical_cpu_limit'], 4)
            outputs = list((Path(tmp) / '.tmp/m1_final_runs').iterdir())
            self.assertEqual(len(outputs), 1)
            self.assertTrue((outputs[0] / 'FAILED.json').is_file())
            self.assertFalse((outputs[0] / 'MANIFEST.json').exists())


class FinalStageSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config, self.source = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_independent_archives_verify_and_preserve_all_bytes(self):
        before = {name: file_inventory(self.root / spec['path']) for name, spec in self.config['archives'].items()}
        with patch('tools.aggregate_m1.validate_document'), patch.object(core, 'check_environment'), \
             patch.object(core, 'code_identity', return_value={}):
            with core.verified_sources(self.config, self.root) as source:
                self.assertEqual(len(source['jobs']), 10)
                self.assertEqual(len(source['val_ids']), 640)
        self.assertEqual(before, {name: file_inventory(self.root / spec['path']) for name, spec in self.config['archives'].items()})

    def test_manifest_hash_mismatch_refuses_source(self):
        self.config['archives']['RF']['manifest_sha256'] = '0' * 64
        with patch('tools.aggregate_m1.validate_document'), patch.object(core, 'check_environment'), patch.object(core, 'code_identity', return_value={}):
            with self.assertRaisesRegex(ValueError, 'pin'):
                with core.verified_sources(self.config, self.root):pass

    def test_validation_tamper_refuses_calibration(self):
        self.source['jobs'][0]['prediction_path'].write_bytes(b'BROKEN')
        with self.assertRaisesRegex(ValueError, 'prediction changed'):
            core.calibration_bundle(self.config, self.source)

    def test_archive_freeze_denies_concurrent_model_writes(self):
        with patch('tools.aggregate_m1.validate_document'), patch.object(core, 'check_environment'), patch.object(core, 'code_identity', return_value={}):
            with core.verified_sources(self.config, self.root) as source:
                with self.assertRaises(PermissionError):
                    source['jobs'][0]['model_path'].write_bytes(b'ILLEGAL WRITE')

    def test_modified_public_summary_is_rejected(self):
        self.config['m1_summary_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'M1 pin'):
            with core.verified_sources(self.config, self.root):pass

    def test_calibration_creates_twenty_rules_without_model_loading(self):
        run = self.root / 'calibration';run.mkdir()
        with patch('joblib.load') as load:
            core.calibrate(self.config, run, self.source)
            load.assert_not_called()
        bundle = cli.read(run / 'thresholds.json')
        self.assertEqual(len(bundle['thresholds']), 20)
        self.assertFalse(bundle['test_accessed'])
        self.assertTrue(all(x['actual_validation_fpr'] <= x['target_fpr'] for x in bundle['thresholds']))
        self.assertEqual(np.load(run / 'timing_X.npy').shape, (512, 140))

    def test_shared_timing_hash_and_population_are_checked(self):
        self.config['timing']['row_ids_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash'):
            core.timing_rows(self.config, self.source)
        self.config['timing']['row_ids_sha256'] = cli.sha(self.root / 'common.json')
        self.source['val_ids'] = self.source['val_ids'][::-1]
        with self.assertRaisesRegex(ValueError, 'population'):
            core.timing_rows(self.config, self.source)

    def test_eval_permission_refused_before_any_loader(self):
        with patch('src.model_loader.ModelDataset') as loader:
            with self.assertRaisesRegex(ValueError, 'authorization'):
                core.evaluate(self.config, self.root, self.source, self.root)
            loader.assert_not_called()

    def test_threshold_tamper_refused_before_test_loader(self):
        calibration = self.root / 'calibration';calibration.mkdir()
        core.calibrate(self.config, calibration, self.source)
        path = calibration / 'thresholds.json'
        d = cli.read(path);d['thresholds'][0]['threshold'] = .999
        path.write_text(json.dumps(d), encoding='utf-8')
        with patch('src.model_loader.ModelDataset') as loader:
            with self.assertRaisesRegex(ValueError, 'cannot be reproduced'):
                core.evaluate(self.config, self.root, self.source, calibration, permit_test=True)
            loader.assert_not_called()

    def test_fake_model_test_evaluation_outputs_twenty_schema_records(self):
        calibration = self.root / 'calibration';calibration.mkdir()
        core.calibrate(self.config, calibration, self.source)
        run = self.root / 'evaluation';run.mkdir()
        SyntheticP2.calls = []
        core.evaluate(self.config, run, self.source, calibration, permit_test=True, loader_factory=SyntheticP2)
        self.assertEqual(SyntheticP2.calls, ['test.parquet']) # the temporary synthetic placeholder only
        result = cli.read(run / 'evaluation_index.json')
        self.assertEqual(len(result['records']), 20)
        self.assertEqual(len(result['five_seed_summaries']), 4)
        self.assertEqual(result['training_fits'], 0)
        self.assertEqual(len(list((run / 'predictions').glob('*.npz'))), 10)
        from eval.validate import validate_result
        for row in result['records']:
            record = cli.read(run / row['path'])
            self.assertTrue(validate_result(record))
            self.assertEqual(record['comparisons'], [])

    def test_production_core_chains_synthetic_calibration_and_evaluation(self):
        c = self.config
        for stage in ('calibrate', 'evaluate'):
            c['approvals'][stage] = {'status': 'approved', 'source': 'SYNTHETIC UNIT TEST ONLY',
                'approved_at': 'UNIT TEST ONLY', 'payload_sha256': cli.digest(cli.payload(c)), 'code_sha256': cli.digest({})}
        calibration = self.root / 'guarded-calibration';calibration.mkdir()
        cli.write(calibration / 'config.json', c)
        with patch('tools.aggregate_m1.validate_document'), patch.object(core, 'check_environment'), \
             patch.object(core, 'code_identity', return_value={}), patch.object(cli, 'code_identity', return_value={}), \
             patch('src.model_loader.ModelDataset', SyntheticP2), patch('methods.m1.models.fit_tree') as fit, \
             patch('eval.metrics.paired_bootstrap') as bootstrap:
            core.execute(c, {'stage': 'calibrate', 'launch': True, 'permit_test': False}, calibration, project=self.root)
            cli.write(calibration / 'monitor/outcome.json', {'exit_code': 0, 'abort_reason': None, 'guard_verified': True,
                      'job_commit_limit_bytes': 8 * cli.GIB, 'logical_cpu_limit': 4})
            manifest_sha = cli.seal_output(calibration, 'calibrate')
            run = self.root / 'guarded-evaluation';run.mkdir()
            core.execute(c, {'stage': 'evaluate', 'launch': True, 'permit_test': True,
                         'calibration': str(calibration), 'calibration_sha256': manifest_sha}, run, project=self.root)
            self.assertEqual(len(cli.read(run / 'evaluation_index.json')['records']), 20)
            self.assertEqual(cli.read(run / 'calibration_reference.json')['manifest_sha256'], manifest_sha)
            fit.assert_not_called()
            bootstrap.assert_not_called()

    def test_model_hash_mismatch_is_not_repaired(self):
        calibration = self.root / 'calibration';calibration.mkdir()
        core.calibrate(self.config, calibration, self.source)
        self.source['jobs'][0]['model_sha256'] = '0' * 64
        # Reproduce threshold bundle pins deliberately, to exercise model verification itself.
        (calibration / 'thresholds.json').write_text(json.dumps(core.calibration_bundle(self.config, self.source)), encoding='utf-8')
        run = self.root / 'evaluation';run.mkdir()
        with patch('joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'model changed'):
                core.evaluate(self.config, run, self.source, calibration, permit_test=True, loader_factory=SyntheticP2)
            load.assert_not_called()

    def test_test_row_overlap_is_rejected(self):
        from methods.m1.data import _materialize_loaded
        test = _materialize_loaded(SyntheticP2(self.source['data_dir'] / 'test.parquet'),
                                   self.source['data_dir'] / 'test.parquet', self.root / 'heldout')
        metadata = test.directory / 'metadata.jsonl'
        text = metadata.read_text(encoding='utf-8').replace('test:0', 'train:0')
        metadata.write_text(text, encoding='utf-8')
        run = self.root / 'isolation';run.mkdir()
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            core.assert_test_isolated(test, self.source, run, 64)

    def test_manifest_sealing_verification_and_no_overwrite(self):
        run = self.root / 'sealed';run.mkdir()
        cli.write(run / 'worker_result.json', {'status': 'calibrate_complete', 'test_accessed': False,
                  'training_fits': 0, 'bootstrap_run': False})
        cli.write(run / 'monitor/outcome.json', {'exit_code': 0, 'abort_reason': None, 'guard_verified': True,
                  'job_commit_limit_bytes': 8 * cli.GIB, 'logical_cpu_limit': 4})
        digest = cli.seal_output(run, 'calibrate')
        self.assertEqual(cli.verify_output(run, digest)['status'], 'validation_thresholds_frozen')
        with self.assertRaises(ValueError):cli.seal_output(run, 'calibrate')
        (run / 'worker_result.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            cli.verify_output(run, digest)


if __name__ == '__main__':
    unittest.main()
