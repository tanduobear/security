"""Native sealing tests use synthetic byte files only. No ML/test reads."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from methods.m1 import seal_gbdt as seal
from methods.m1.search_gbdt import ROOT,digest,read,run_lock,sha,tasks


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@unittest.skipUnless(os.name=='nt','Windows mandatory file locks')
class SealTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project=Path(self.temporary.name)
        self.run=self.project/'run'
        self.run.mkdir()
        (self.run/'RUN.lock').write_bytes(b'0')
        self.config=read(ROOT/'methods/m1/execution.gbdt-search.v1.json')
        for name in ('execution.gbdt-search.v1.json','protocol.unified.v1.json'):
            path=self.project/'methods/m1'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((ROOT/'methods/m1'/name).read_bytes())
        source=self.project/'src/fixture.py'
        source.parent.mkdir()
        source.write_bytes(b'SYNTHETIC SOURCE ONLY')
        archived=self.run/'sources/src/fixture.py'
        archived.parent.mkdir(parents=True)
        archived.write_bytes(source.read_bytes())
        inputs=self.project/'data/split/original/stratified'
        inputs.mkdir(parents=True)
        for name in ('train.parquet','val.parquet','preprocessor.json','preprocess_manifest.json'):
            (inputs/name).write_bytes(('SYNTHETIC BYTES '+name).encode())
        self.identity={'config_sha256':digest(self.config),'code':{'src/fixture.py':sha(source)},
                       'inputs':{p.name:sha(p) for p in inputs.iterdir()},
                       'unified_file_sha256':sha(self.project/'methods/m1/protocol.unified.v1.json')}
        put(self.run/'binding.json',self.identity)
        put(self.run/'config.json',self.config)
        self.first_model=None
        for task in tasks(self.config):
            directory=self.run/'attempts'/task['task_id']
            directory.mkdir(parents=True)
            (directory/'model.joblib').write_bytes(b'NOT A MODEL: SYNTHETIC')
            (directory/'validation.npz').write_bytes(b'NOT AN ARRAY: SYNTHETIC')
            put(directory/'task.json',task)
            if self.first_model is None:self.first_model=directory/'model.joblib'
            files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in directory.iterdir()}
            put(self.run/'completed'/(task['task_id']+'.json'),{
                'task':task,'binding_sha256':digest(self.identity),'validation_rows':367036,
                'test_accessed':False,'validation_ap':.5,'training_seconds':1,
                'validation_row_ids_sha256':'synthetic-common-order',
                'artifact_directory':directory.relative_to(self.run).as_posix(),'files':files})
        cache=self.run/'cache/fixture'
        cache.mkdir(parents=True)
        (cache/'arrays.bin').write_bytes(b'SYNTHETIC CACHE')
        put(self.run/'cache.json',{'directory':'cache/fixture','files':{'arrays.bin':{'bytes':15,'sha256':sha(cache/'arrays.bin')}}})
        # No threshold/bootstrap/test modules or model loaders can parse these fixtures.
        for group in [c['id'] for c in self.config['candidates']]+['SELECT']:
            monitor=self.run/'monitor'/(group+'-fixture')
            put(monitor/'outcome.json',{'exit_code':0,'abort_reason':None})
            put(monitor/'guard_verified.json',{'job_commit_limit_bytes':8*1024**3,'logical_cpu_limit':4,'verified_before_GO':True})
            put(monitor/'worker_result.json',{'status':'selection_complete' if group=='SELECT' else 'group_complete','test_accessed':False})
            if group!='SELECT':put(self.run/'checkpoints'/(group+'.json'),{'group':group,'binding_sha256':digest(self.identity),'action':'CONTINUE'})
        put(self.run/'selection.json',{'status':'gbdt_validation_selection_only','binding_sha256':digest(self.identity),
            'selected_candidate_id':'G05','test_accessed':False,'rf_fits':0,'thresholds_calibrated':False})
        self.selection_sha=sha(self.run/'selection.json')
        self.baseline=self.project/'baseline.json'
        self.snapshot()

    def snapshot(self):
        put(self.baseline,{'selection_sha256':self.selection_sha,'files':seal.file_inventory(self.run)})
        self.baseline_sha=sha(self.baseline)

    def execute(self,**kwargs):
        return seal.seal(self.run,self.baseline,self.selection_sha,baseline_sha=self.baseline_sha,
                         project=self.project,_process_probe=kwargs.pop('_process_probe',lambda:[]),**kwargs)

    def test_seal_and_full_second_verification_preserve_all_originals(self):
        before=read(self.baseline)['files']
        result=self.execute()
        self.assertEqual(result['manifest_file_count'],len(before))
        self.assertEqual(result['sha256_verified_count'],len(before))
        manifest=read(self.run/'MANIFEST.json')
        self.assertNotIn('RUN.lock',manifest['files'])
        self.assertIn('RUN.lock',manifest['excluded_runtime_files'])
        self.assertEqual(seal.file_inventory(self.run),before)
        self.assertEqual(sha(self.run/'selection.json'),self.selection_sha)
        checked=seal.verify(self.run,self.selection_sha,expected_manifest_sha=result['manifest_sha256'],_process_probe=lambda:[])
        self.assertEqual(checked['sha256_verified_count'],len(before))

    def test_inventory_while_run_lock_held_excludes_only_root_lock(self):
        with run_lock(self.run):
            with self.assertRaises(PermissionError):sha(self.run/'RUN.lock')
            self.assertIn('selection.json',seal.file_inventory(self.run))

    def test_active_process_refused(self):
        with self.assertRaises(ValueError):self.execute(_process_probe=lambda:[{'pid':123}])
        self.assertFalse((self.run/'MANIFEST.json').exists())

    def test_existing_run_lock_holder_refused(self):
        with run_lock(self.run):
            with self.assertRaises(OSError):self.execute()
        self.assertFalse((self.run/'MANIFEST.json').exists())

    def test_existing_global_slot_holder_refused(self):
        slot=self.project/'.tmp/m1_gbdt_resource_slot'
        slot.mkdir(parents=True)
        with run_lock(slot):
            with self.assertRaises(OSError):self.execute()

    def test_existing_writer_refused(self):
        with self.first_model.open('a+b'):
            with self.assertRaises(OSError):self.execute()
        self.assertFalse((self.run/'MANIFEST.json').exists())

    def test_concurrent_thread_cannot_write_or_delete_artifact(self):
        rejected=[]
        def writer():
            for operation in (lambda:self.first_model.write_bytes(b'changed'),lambda:self.first_model.unlink()):
                try:operation()
                except OSError:rejected.append(True)
        def hook():
            thread=threading.Thread(target=writer)
            thread.start();thread.join()
        self.execute(_before_publish=hook)
        self.assertEqual(rejected,[True,True])

    def test_model_hash_change_refused(self):
        self.first_model.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.execute()
        self.assertFalse((self.run/'MANIFEST.json').exists())

    def test_cache_corruption_refused(self):
        (self.run/'cache/fixture/arrays.bin').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.execute()

    def test_original_source_change_refused(self):
        (self.project/'src/fixture.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.execute()

    def test_missing_artifact_refused(self):
        self.first_model.unlink()
        with self.assertRaises(ValueError):self.execute()

    def test_selection_hash_change_refused(self):
        (self.run/'selection.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.execute()

    def test_baseline_change_refused(self):
        self.baseline.write_text('{}')
        with self.assertRaises(ValueError):self.execute()

    def test_new_namespace_file_during_seal_refused(self):
        def hook():(self.run/'unexpected.bin').write_bytes(b'external concurrent creation')
        with self.assertRaises(ValueError):self.execute(_before_publish=hook)
        self.assertFalse((self.run/'MANIFEST.json').exists())

    def test_process_appearing_before_publish_refused(self):
        probes=iter([[],[],[{'pid':123}]])
        with self.assertRaises(ValueError):self.execute(_process_probe=lambda:next(probes))
        self.assertFalse((self.run/'MANIFEST.json').exists())

    def test_never_overwrite_existing_manifest(self):
        target=self.run/'MANIFEST.json'
        target.write_bytes(b'PREEXISTING')
        with self.assertRaises(ValueError):self.execute()
        self.assertEqual(target.read_bytes(),b'PREEXISTING')

    def test_post_seal_tampering_refused(self):
        result=self.execute()
        self.first_model.write_bytes(b'changed after seal')
        with self.assertRaises(ValueError):seal.verify(self.run,self.selection_sha,expected_manifest_sha=result['manifest_sha256'],_process_probe=lambda:[])

    def test_manifest_hash_pin_refuses_manifest_changes(self):
        result=self.execute()
        (self.run/'MANIFEST.json').write_text('{}')
        with self.assertRaises(ValueError):seal.verify(self.run,self.selection_sha,expected_manifest_sha=result['manifest_sha256'],_process_probe=lambda:[])

    def test_nested_lock_is_not_silently_excluded(self):
        nested=self.run/'extra/RUN.lock'
        nested.parent.mkdir()
        nested.write_bytes(b'ordinary diagnostic artifact')
        self.snapshot()
        result=self.execute()
        self.assertIn('extra/RUN.lock',read(self.run/'MANIFEST.json')['files'])


if __name__=='__main__':unittest.main()
