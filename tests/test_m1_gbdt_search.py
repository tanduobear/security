"""Search acceptance uses synthetic train/val only; never real data or fits."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from methods.m1 import search_gbdt as search
from methods.m1.resource_trial import monitored_child, read_json, GIB, MIB
from test_m1_executor import SyntheticP2, fake_fit


def config():
    return search.read(search.CONFIG)


def synthetic_files(root):
    directory = root/'data/split/original/stratified'
    directory.mkdir(parents=True)
    # Deliberately no test file, so accidental test access cannot succeed.
    for name in ('train.parquet','val.parquet','preprocessor.json','preprocess_manifest.json'):
        (directory/name).write_text('synthetic fixture',encoding='utf-8')


class SearchConfigTests(unittest.TestCase):
    def test_exact_forty_tasks_and_later_details_pending(self):
        cfg = config()
        plan = search.tasks(cfg)
        self.assertEqual(len(plan),40)
        self.assertEqual(len({x['task_id'] for x in plan}),40)
        self.assertTrue(all(x['algorithm']=='hist_gradient_boosting' for x in plan))
        self.assertEqual(plan[0]['task_id'],'G01_seed_42')
        self.assertEqual(plan[-1]['task_id'],'G08_seed_46')
        self.assertEqual(search.read(search.ROOT/'methods/m1/execution.proposed.json')['approval']['status'],'pending')

    def test_approval_and_config_tampering_rejected(self):
        for key in ('approval','candidates','resources','seeds'):
            bad = copy.deepcopy(config())
            if key=='approval':bad[key]['status']='pending'
            elif key=='candidates':bad[key][0]['params']['max_iter']=1
            elif key=='resources':bad[key]['job_commit_limit_bytes']=9*GIB
            else:bad[key]=[42]
            with self.subTest(key=key),self.assertRaises(ValueError):search.tasks(bad)

    def test_worker_requires_go_before_loading(self):
        import io
        with patch('sys.stdin',io.StringIO('NO\n')),patch.object(search,'binding') as binding:
            with self.assertRaises(ValueError):search.worker('missing','missing','G01')
            binding.assert_not_called()

    def test_run_requires_explicit_launch_switch_before_work(self):
        with patch.object(search,'run_search') as runner:
            with self.assertRaises(ValueError):search.main(['run'])
            runner.assert_not_called()

    def test_select_requires_explicit_launch_switch_before_work(self):
        with patch.object(search,'run_selection') as runner:
            with self.assertRaises(ValueError):search.main(['select','--resume','unused'])
            runner.assert_not_called()

    def test_checkpoint_is_manual_and_identity_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertFalse(search.checkpoint(run,'G01',{'v':1},lambda _: 'STOP'))
            self.assertFalse((run/'checkpoints/G01.json').exists())
            self.assertTrue(search.checkpoint(run,'G01',{'v':1},lambda _: 'CONTINUE'))
            self.assertTrue(search.checkpoint(run,'G01',{'v':1},lambda _: self.fail('Already acknowledged')))
            with self.assertRaises(ValueError):search.checkpoint(run,'G01',{'v':2})

    def test_abort_thresholds(self):
        resources = config()['resources']
        normal = {'available_physical_bytes':7*GIB,'rss_bytes':2*GIB}
        self.assertIsNone(search.abort_reason(normal,resources))
        self.assertIsNotNone(search.abort_reason({**normal,'available_physical_bytes':5*GIB},resources))
        self.assertIsNotNone(search.abort_reason({**normal,'rss_bytes':9*GIB},resources))

    def test_selection_uses_five_seed_mean_and_tolerance(self):
        cfg = config()
        records = {t['task_id']:{'validation_ap':.5} for t in search.tasks(cfg)}
        for seed in cfg['seeds']:
            records[f'G02_seed_{seed}']['validation_ap']=.8
            records[f'G03_seed_{seed}']['validation_ap']=.8000005
        records['G01_seed_42']['validation_ap']=1.0  # best seed does not win
        with patch.object(search,'verify_tasks',return_value=records):
            result = search.select(Path('unused'),cfg,{'fixture':True})
        self.assertEqual(result['selected_candidate_id'],'G02')
        self.assertGreater(result['candidates'][0]['sample_std_ap'],0)

    def test_code_identity_changes_when_uncommitted_code_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'methods/m1').mkdir(parents=True)
            (root/'docs').mkdir()
            (root/'docs/label_split_policy.json').write_text('{}')
            (root/'methods/m1/requirements-lock.txt').write_text('fixture')
            code = root/'methods/m1/model.py'
            code.write_text('before')
            before = search.fingerprint(root)
            code.write_text('after')
            self.assertNotEqual(before,search.fingerprint(root))

    @unittest.skipUnless(os.name=='nt','Windows locking')
    def test_concurrent_run_lock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with search.run_lock(Path(tmp)):
                with self.assertRaises(OSError):
                    with search.run_lock(Path(tmp)):pass


class SearchArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        synthetic_files(self.root)
        self.cfg = config()
        self.identity = {'config':search.digest(self.cfg),'data':'SYNTHETIC ONLY','code':'fixture'}
        SyntheticP2.calls = []
        self.train,self.val = search.prepare_cache(self.run,self.cfg,self.root,SyntheticP2)

    def group(self,group='G01',fitter=fake_fit):
        search.execute_group(self.run,group,self.cfg,self.identity,self.train,self.val,fitter=fitter)

    def test_full_forty_then_resume_without_refitting_and_select(self):
        count = []
        def fit(*args,**kwargs):
            count.append(kwargs['model_seed'])
            return fake_fit(*args,**kwargs)
        for group in self.cfg['candidates']:self.group(group['id'],fit)
        self.assertEqual(len(count),40)
        for group in self.cfg['candidates']:
            self.group(group['id'],lambda *a,**k:self.fail('Completed task refitted'))
        result = search.select(self.run,self.cfg,self.identity)
        self.assertEqual(result['selected_candidate_id'],'G01')
        self.assertEqual(len(result['candidates']),8)
        self.assertEqual(result['candidates'][0]['sample_std_ap'],0)
        self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet'])
        self.assertFalse(result['test_accessed'])

    def test_interruption_retains_two_complete_then_only_three_refits(self):
        count = []
        def interrupted(*args,**kwargs):
            if len(count)==2:raise KeyboardInterrupt()
            count.append(kwargs['model_seed'])
            return fake_fit(*args,**kwargs)
        with self.assertRaises(KeyboardInterrupt):self.group(fitter=interrupted)
        self.assertEqual(len(search.verify_tasks(self.run,self.cfg,self.identity)),2)
        resumed = []
        def fit(*args,**kwargs):
            resumed.append(kwargs['model_seed']);return fake_fit(*args,**kwargs)
        self.group(fitter=fit)
        self.assertEqual(resumed,[44,45,46])
        self.assertEqual(len(list((self.run/'attempts').iterdir())),6)  # orphan kept

    def test_output_hash_mismatch_rejected_before_fit(self):
        self.group()
        record = search.read(self.run/'completed/G01_seed_42.json')
        (self.run/record['artifact_directory']/'model.joblib').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.group(fitter=lambda *a,**k:self.fail('Fit before verification'))

    def test_changed_identity_rejected(self):
        self.group()
        for key in ('config','data','code'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                search.verify_tasks(self.run,self.cfg,{**self.identity,key:'changed'})

    def test_cache_corruption_rejected_without_source_reload(self):
        record = search.read(self.run/'cache.json')
        (self.run/record['directory']/'train/y.npy').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):search.prepare_cache(self.run,self.cfg,self.root,SyntheticP2)
        self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet'])

    def test_partial_selection_and_result_tampering_rejected(self):
        self.group()
        with self.assertRaises(ValueError):search.select(self.run,self.cfg,self.identity)
        path = self.run/'completed/G01_seed_42.json'
        record = search.read(path);record['validation_ap']=0
        path.write_text(__import__('json').dumps(record),encoding='utf-8')
        with self.assertRaises(ValueError):search.verify_tasks(self.run,self.cfg,self.identity)

    def test_input_hashes_ignore_test_and_detect_training_changes(self):
        before = search.inputs(self.root)
        directory = self.root/'data/split/original/stratified'
        (directory/'test.parquet').write_text('DO NOT READ')
        self.assertEqual(before,search.inputs(self.root))
        (directory/'train.parquet').write_text('changed')
        self.assertNotEqual(before,search.inputs(self.root))

    def test_worker_go_to_five_saved_tasks_uses_synthetic_data_only(self):
        import io
        identity = {'code':{},'inputs':search.inputs(self.root),'fixture':'SYNTHETIC ONLY'}
        search.write(self.run/'config.json',self.cfg)
        monitor = self.run/'monitor_fixture'
        monitor.mkdir()
        search.write(monitor/'launch.json',{'allow_initialize':True})
        original = search.execute_group
        def execute(*args,**kwargs):
            return original(*args,**kwargs,fitter=fake_fit)
        with patch('sys.stdin',io.StringIO('GO\n')),patch.object(search,'binding',return_value=identity), \
             patch.object(search,'prepare_cache',return_value=(self.train,self.val)), \
             patch.object(search,'execute_group',side_effect=execute),patch('sys.stdout',io.StringIO()):
            search.worker(self.run,monitor,'G01')
        self.assertEqual(len(search.verify_tasks(self.run,self.cfg,identity)),5)
        self.assertEqual(search.read(monitor/'worker_result.json')['status'],'group_complete')
        self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet'])


@unittest.skipUnless(os.name=='nt','Windows native Job Object')
class SearchNativeGuardTests(unittest.TestCase):
    def test_shared_formal_worker_route_enforces_job_memory_and_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={**config()['resources'],'job_commit_limit_bytes':64*MIB,'sample_seconds':.05,'disk_sample_seconds':5}
            outcome=monitored_child(tmp,cfg,worker_module='methods.m1.resource_trial',
                worker_args=('--guard-probe',tmp),reason_function=search.abort_reason)
            probe=read_json(Path(tmp)/'probe.json')
            self.assertEqual(outcome['exit_code'],0)
            self.assertTrue(probe['memory_limit_enforced'])
            self.assertEqual(probe['affinity_mask'].bit_count(),4)
            self.assertEqual(outcome['job_commit_limit_bytes'],64*MIB)

    def test_guard_abort_terminates_child_and_preserves_monitoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={**config()['resources'],'job_commit_limit_bytes':64*MIB,'sample_seconds':.01,'disk_sample_seconds':5}
            outcome=monitored_child(tmp,cfg,worker_module='methods.m1.resource_trial',
                worker_args=('--guard-probe',tmp),reason_function=lambda s,c:'synthetic_abort')
            self.assertEqual(outcome['abort_reason'],'synthetic_abort')
            self.assertNotEqual(outcome['exit_code'],0)
            self.assertTrue((Path(tmp)/'resource_samples.csv').exists())


if __name__=='__main__':unittest.main()
