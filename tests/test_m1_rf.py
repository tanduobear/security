"""RF preparation acceptance: synthetic arrays, fake forests, native tiny probes.

No real RF fit, real Test access, or writes to the sealed GBDT archive.
"""
import copy
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from methods.m1 import run_rf as rf
from methods.m1 import models,resource_trial
from methods.m1.data import row_digest,ModelInput
from methods.m1.search_gbdt import prepare_cache
from test_m1_executor import SyntheticP2,FixtureModel


def settings():
    # Approval fixtures must never inherit real launch authorization.
    config=copy.deepcopy(rf.read(rf.CONFIG))
    for key in ('execution_approval','trial_approval'):
        config[key]={'status':'pending','source':None,'approved_at':None,'payload_sha256':None}
    return config


def approved():
    config=copy.deepcopy(settings())
    for key in ('execution_approval','trial_approval'):
        config[key]={'status':'approved','source':'SYNTHETIC TEST ONLY','approved_at':'SYNTHETIC TEST ONLY',
                     'payload_sha256':rf.digest(rf.payload(config))}
    return config


class FakeRF(FixtureModel):
    def __init__(self,params,seed):
        super().__init__(params,seed)
        self.n_jobs=params['n_jobs']
        self.estimators_=[None]*200


def fake_fit(name,params,training,*,model_seed,fit_threads):
    assert name=='random_forest' and params==rf.PARAMS
    assert training.split=='train' and len(training.y)==80 and training.class_weights=={0:.6,1:3.0}
    assert fit_threads==4 and model_seed in [42,43,44,45,46]
    return FakeRF(params,model_seed),.001


def fixture(root):
    directory=root/'data/split/original/stratified'
    directory.mkdir(parents=True)
    for name in ('train.parquet','val.parquet','preprocessor.json','preprocess_manifest.json'):
        (directory/name).write_bytes(('SYNTHETIC '+name).encode())
    # No test.parquet exists.


def reference(rows=640):
    ids=np.array(['val:'+str(i) for i in range(rows)])
    y=(np.arange(rows)%2).astype(np.uint8)
    population={'row_ids_sha256':row_digest(ids),'labels_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'rows':rows}
    return {'paired_seeds':{str(seed):{**population,'scores_file_sha256':'synthetic'} for seed in range(42,47)}}


class RFPlanTests(unittest.TestCase):
    def test_plan_is_five_fixed_rf_tasks_and_both_launch_approvals_pending(self):
        config=settings()
        plan=rf.tasks(config)
        self.assertEqual([t['task_id'] for t in plan],[f'RF_FIXED_seed_{s}' for s in range(42,47)])
        self.assertTrue(all(t['algorithm']=='random_forest' and t['params']==rf.PARAMS and not t['eligible_for_gbdt_selection'] for t in plan))
        self.assertEqual(config['trial_approval']['status'],'pending')
        self.assertEqual(config['execution_approval']['status'],'pending')
        self.assertEqual(rf.read(rf.ROOT/'methods/m1/execution.proposed.json')['approval']['status'],'pending')

    def test_pending_blocks_before_output_and_protected_call(self):
        with patch.object(rf,'protected_call') as call,patch.object(rf,'binding') as bind:
            for stage in ('trial','run','accept'):
                with self.assertRaises(ValueError):rf.launch(settings(),stage)
            call.assert_not_called();bind.assert_not_called()

    def test_explicit_launch_switch_required(self):
        with patch.object(rf,'launch') as launch:
            with self.assertRaises(ValueError):rf.main(['run'])
            launch.assert_not_called()

    def test_configuration_tampering_and_approval_digest_rejected(self):
        for key,value in [('n_estimators',201),('class_weight','balanced'),('n_jobs',8)]:
            bad=approved();bad['parameters'][key]=value
            with self.assertRaises(ValueError):rf.require_execution(bad,'run')
        bad=approved();bad['execution_approval']['payload_sha256']='changed'
        with self.assertRaises(ValueError):rf.require_execution(bad,'run')

    def test_formal_run_requires_trial_reference_before_outputs(self):
        with patch.object(rf,'protected_call') as call:
            with self.assertRaises(ValueError):rf.launch(approved(),'run')
            call.assert_not_called()

    def test_worker_refuses_loading_before_go(self):
        with patch('sys.stdin',io.StringIO('NO\n')),patch.object(rf,'binding') as bind:
            with self.assertRaises(ValueError):rf.worker('none','none','run')
            bind.assert_not_called()

    def test_resource_abort_reuses_approved_limits(self):
        config=settings()['resources']
        normal={'available_physical_bytes':10*rf.GIB,'rss_bytes':3*rf.GIB}
        self.assertIsNone(rf.abort_reason(normal,config))
        self.assertIsNotNone(rf.abort_reason({**normal,'rss_bytes':9*rf.GIB},config))
        self.assertIsNotNone(rf.abort_reason({**normal,'available_physical_bytes':5*rf.GIB},config))

    def test_fit_adapter_passes_rf_seed_and_only_p2_weights(self):
        captured={}
        class Capture:
            def __init__(self,**kwargs):captured['params']=kwargs
            def fit(self,X,y,sample_weight):captured['weights']=sample_weight.copy()
        training=ModelInput(np.zeros((4,140),dtype=np.float32),np.array([0,1,0,1]),tuple('f'+str(i) for i in range(140)), 'train',{0:.6,1:3.0})
        with patch.object(models,'RandomForestClassifier',Capture):
            models.fit_tree('random_forest',rf.PARAMS,training,model_seed=44,fit_threads=4)
        self.assertEqual(captured['params'],{**rf.PARAMS,'random_state':44})
        np.testing.assert_array_equal(captured['weights'],[.6,3,.6,3])


class RFTaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        fixture(self.root)
        self.run=self.root/'rf'
        self.run.mkdir()
        self.monitor=self.run/'monitor/RUN-synthetic'
        self.monitor.mkdir(parents=True)
        self.config=approved()
        self.identity={'config':rf.digest(self.config),'data':'SYNTHETIC','code':'SYNTHETIC'}
        self.reference=reference()
        SyntheticP2.calls=[]
        self.train,self.val=prepare_cache(self.run,self.config,self.root,SyntheticP2)

    def execute(self,*,trial=False,fitter=fake_fit):
        return rf.execute_tasks(self.run,self.config,self.identity,self.reference,self.train,self.val,self.monitor,trial=trial,fitter=fitter)

    def monitoring(self,interrupted=False):
        rf.write(self.monitor/'guard_verified.json',{'job_commit_limit_bytes':8*rf.GIB,'logical_cpu_limit':4,'verified_before_GO':True})
        rf.write(self.monitor/'outcome.json',{'exit_code':71 if interrupted else 0,'abort_reason':'synthetic_abort' if interrupted else None})
        with (self.monitor/'resource_samples.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['phase','rss_bytes'])
            writer.writeheader()
            for task in rf.tasks(self.config):writer.writerow({'phase':task['task_id']+':fit','rss_bytes':12345})

    def test_five_fake_fits_save_reload_independent_ap_and_summary(self):
        calls=[]
        def fit(*args,**kwargs):calls.append(kwargs['model_seed']);return fake_fit(*args,**kwargs)
        self.execute(fitter=fit)
        self.assertEqual(calls,[42,43,44,45,46])
        self.monitoring()
        result=rf.summarize(self.run,self.config,self.identity,self.reference)
        self.assertEqual(result['mean_validation_ap'],1)
        self.assertEqual(result['sample_std_ap'],0)
        self.assertEqual(len(result['tasks']),5)
        self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet'])
        self.assertFalse(result['eligible_for_gbdt_selection'])

    def test_original_forest_released_before_model_reload(self):
        import joblib
        weak=[]
        def fit(*args,**kwargs):
            model,seconds=fake_fit(*args,**kwargs)
            weak.append(weakref.ref(model))
            return model,seconds
        original_load=joblib.load
        def load(*args,**kwargs):
            gc.collect()
            self.assertIsNone(weak[-1]())
            return original_load(*args,**kwargs)
        with patch.object(joblib,'load',side_effect=load):self.execute(fitter=fit)

    def test_explicit_resume_skips_completed_prefix(self):
        seeds=[]
        def fit(*args,**kwargs):
            if len(seeds)==2:raise KeyboardInterrupt()
            seeds.append(kwargs['model_seed']);return fake_fit(*args,**kwargs)
        with self.assertRaises(KeyboardInterrupt):self.execute(fitter=fit)
        self.assertEqual(len(rf.verify_tasks(self.run,self.config,self.identity,self.reference)),2)
        resumed=[]
        def resume(*args,**kwargs):resumed.append(kwargs['model_seed']);return fake_fit(*args,**kwargs)
        self.execute(fitter=resume)
        self.assertEqual(resumed,[44,45,46])
        self.assertEqual(len(list((self.run/'attempts').iterdir())),6)
        self.execute(fitter=lambda *a,**k:self.fail('Completed RF refitted'))

    def test_completed_tasks_from_interrupted_monitor_still_reported(self):
        self.execute()
        self.monitoring(interrupted=True)
        result=rf.summarize(self.run,self.config,self.identity,self.reference)
        self.assertTrue(all(t['monitor_attempt_interrupted'] for t in result['tasks']))

    def test_output_corruption_rejected_before_fit(self):
        self.execute()
        record=rf.read(self.run/'completed/RF_FIXED_seed_42.json')
        (self.run/record['artifact_directory']/'model.joblib').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.execute(fitter=lambda *a,**k:self.fail('Fit before verification'))

    def test_binding_mismatch_rejected(self):
        self.execute()
        for key in ('config','data','code'):
            identity={**self.identity,key:'changed'}
            with self.assertRaises(ValueError):rf.verify_tasks(self.run,self.config,identity,self.reference)

    def test_recorded_ap_tampering_rejected_by_saved_predictions(self):
        self.execute()
        path=self.run/'completed/RF_FIXED_seed_42.json'
        record=rf.read(path);record['validation_ap']=.1
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):rf.verify_tasks(self.run,self.config,self.identity,self.reference)

    def test_pairing_order_and_labels_mismatch_rejected_before_fit(self):
        for key in ('row_ids_sha256','labels_sha256'):
            original=self.reference['paired_seeds']['42'][key]
            self.reference['paired_seeds']['42'][key]='wrong'
            with self.assertRaises(ValueError):self.execute(fitter=lambda *a,**k:self.fail('Fit before pairing'))
            self.reference['paired_seeds']['42'][key]=original

    def test_trial_runs_one_fit_and_computes_no_ap(self):
        with patch('eval.metrics.compute_curves',side_effect=AssertionError('Trial AP forbidden')):
            self.execute(trial=True)
            records=rf.verify_tasks(self.run,self.config,self.identity,self.reference,trial=True)
        self.assertEqual(list(records),['RF_FIXED_seed_42'])
        self.assertIsNone(records['RF_FIXED_seed_42']['validation_ap'])
        self.assertEqual(records['RF_FIXED_seed_42']['purpose'],'resource_trial_only')

    @unittest.skipUnless(os.name=='nt','Windows trial parent/worker integration')
    def test_trial_entry_end_to_end_one_fake_fit_pending_formal_no_ap(self):
        from contextlib import nullcontext
        config=copy.deepcopy(self.config)
        config['execution_approval']=copy.deepcopy(settings()['execution_approval'])
        (self.root/'methods/m1').mkdir(parents=True)
        (self.root/'methods/m1/protocol.unified.v1.json').write_bytes((rf.ROOT/'methods/m1/protocol.unified.v1.json').read_bytes())
        from methods.m1.search_gbdt import inputs
        identity={'inputs':inputs(self.root),'code':{},'fixture':'SYNTHETIC TRIAL ENTRY ONLY'}
        paired={**self.reference,'inputs':identity['inputs']}
        fit_seeds=[]
        original=rf.execute_tasks
        def fit(*args,**kwargs):
            fit_seeds.append(kwargs['model_seed']);return fake_fit(*args,**kwargs)
        def execute(*args,**kwargs):return original(*args,**kwargs,fitter=fit)
        def protected(run,cfg,stage,initialize):
            self.assertEqual(stage,'trial')
            self.assertEqual(cfg['execution_approval']['status'],'pending')
            monitor=run/'monitor/TRIAL-synthetic';monitor.mkdir(parents=True)
            rf.write(monitor/'launch.json',{'initialize':initialize})
            with patch('sys.stdin',io.StringIO('GO\n')):
                rf.worker(run,monitor,stage)
            result={'exit_code':0,'abort_reason':None,'guard_verified':True,
                    'job_commit_limit_bytes':8*rf.GIB,'logical_cpu_limit':4}
            rf.write(monitor/'outcome.json',result)
            return result
        with patch.object(rf,'ROOT',self.root),patch.object(rf,'binding',return_value=identity), \
             patch.object(rf,'reference_identity',return_value=paired), \
             patch.object(rf,'prepare_cache',side_effect=lambda run,cfg:prepare_cache(run,cfg,self.root,SyntheticP2)), \
             patch.object(rf,'execute_tasks',side_effect=execute),patch.object(rf,'protected_call',side_effect=protected), \
             patch.object(rf,'resource_slot',return_value=nullcontext()), \
             patch('eval.metrics.compute_curves',side_effect=AssertionError('AP forbidden throughout trial')), \
             patch('sys.stdout',io.StringIO()):
            run=rf.launch(config,'trial')
        self.assertEqual(fit_seeds,[42])
        self.assertEqual(len(list((run/'completed').glob('*.json'))),1)
        self.assertFalse(rf.read(run/'resource_report.json')['AP_computed'])
        self.assertEqual(rf.read(run/'MANIFEST.json')['status'],'rf_resource_trial_complete')
        self.assertFalse((self.root/'results/m1_rf_runs').exists())

    def test_partial_five_seed_acceptance_refused(self):
        self.execute(trial=True)
        with self.assertRaises(ValueError):rf.summarize(self.run,self.config,self.identity,self.reference)

    @unittest.skipUnless(os.name=='nt','Windows RF archive protection')
    def test_five_task_acceptance_and_seal_keep_reference_untouched(self):
        from contextlib import nullcontext
        self.identity={'code':{},'fixture':'SYNTHETIC ONLY'}
        (self.root/'methods/m1').mkdir(parents=True)
        (self.root/'methods/m1/protocol.unified.v1.json').write_bytes((rf.ROOT/'methods/m1/protocol.unified.v1.json').read_bytes())
        gbdt=self.root/'gbdt';gbdt.mkdir()
        (gbdt/'MANIFEST.json').write_bytes(b'SYNTHETIC IMMUTABLE GBDT REFERENCE')
        before=rf.sha(gbdt/'MANIFEST.json')
        self.config['reference'].update(run='gbdt',manifest_sha256=before)
        for key in ('trial_approval','execution_approval'):self.config[key]['payload_sha256']=rf.digest(rf.payload(self.config))
        self.execute();self.monitoring()
        summary=rf.summarize(self.run,self.config,self.identity,self.reference)
        for name,value in [('summary.json',summary),('config.json',self.config),('binding.json',self.identity),
                           ('paired_reference.json',self.reference),('checkpoint.json',{'action':'CONTINUE','binding_sha256':rf.digest(self.identity)})]:
            rf.write(self.run/name,value)
        for task in rf.tasks(self.config):
            rf.write(self.run/'checkpoints'/(task['task_id']+'.json'),{'action':'CONTINUE','task_id':task['task_id'],
                'binding_sha256':rf.digest(self.identity),'completed_record_sha256':rf.sha(self.run/'completed'/(task['task_id']+'.json'))})
        accept=self.run/'monitor/ACCEPT-synthetic';accept.mkdir()
        rf.write(accept/'outcome.json',{'exit_code':0})
        rf.write(accept/'worker_result.json',{'status':'rf_accept_complete','summary_sha256':rf.sha(self.run/'summary.json')})
        with patch.object(rf,'ROOT',self.root),patch.object(rf,'binding',return_value=self.identity), \
             patch.object(rf,'resource_slot',return_value=nullcontext()):
            manifest_sha=rf.seal_run(self.run,self.config)
        with patch.object(rf,'resource_slot',return_value=nullcontext()):
            result=rf.verify_run(self.run,manifest_sha)
        self.assertGreater(result['files_verified'],30)
        self.assertEqual(before,rf.sha(gbdt/'MANIFEST.json'))
        self.assertEqual(summary['mean_validation_ap'],1)


@unittest.skipUnless(os.name=='nt','Read-only Windows sharing protection')
class RFReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.config=settings()
        self.gbdt=self.root/'gbdt'
        self.gbdt.mkdir()
        self.inputs={'train.parquet':'a','val.parquet':'b','preprocessor.json':'c','preprocess_manifest.json':'d'}
        original={'inputs':self.inputs,'fixture':'SYNTHETIC GBDT ONLY'}
        rf.write(self.gbdt/'binding.json',original)
        rf.write(self.gbdt/'selection.json',{'selected_candidate_id':'G05','binding_sha256':rf.digest(original)})
        ids=np.array(['val:'+str(i) for i in range(20)])
        y=(np.arange(20)%2).astype(np.uint8)
        for seed in range(42,47):
            directory=self.gbdt/'attempts'/f'G05_seed_{seed}'
            directory.mkdir(parents=True)
            with (directory/'validation.npz').open('xb') as stream:np.savez_compressed(stream,row_ids=ids,y_true=y,y_score=y*.8+.1)
            path=self.gbdt/'completed'/f'G05_seed_{seed}.json'
            path.parent.mkdir(exist_ok=True)
            rf.write(path,{'task':{'candidate_id':'G05','seed':seed},'validation_rows':20,
                'validation_row_ids_sha256':row_digest(ids),'artifact_directory':directory.relative_to(self.gbdt).as_posix(),
                'files':{'validation.npz':{'sha256':rf.sha(directory/'validation.npz')}}})
        from methods.m1.seal_gbdt import file_inventory,STATUS,EXCLUDED
        rf.write(self.gbdt/'MANIFEST.json',{'status':STATUS,'files':file_inventory(self.gbdt),'excluded_runtime_files':EXCLUDED})
        self.config['reference']={'run':'gbdt','selected_candidate_id':'G05','manifest_sha256':rf.sha(self.gbdt/'MANIFEST.json'),
                                  'selection_sha256':rf.sha(self.gbdt/'selection.json')}

    def test_five_seed_reference_is_verified_without_loading_gbdt_models(self):
        import joblib
        before=rf.sha(self.gbdt/'MANIFEST.json')
        with patch.object(joblib,'load',side_effect=AssertionError('GBDT model load forbidden')):
            ref=rf.reference_identity(self.config,self.root)
        self.assertEqual(set(ref['paired_seeds']),{'42','43','44','45','46'})
        self.assertEqual(ref['inputs'],self.inputs)
        self.assertEqual(before,rf.sha(self.gbdt/'MANIFEST.json'))

    def test_reference_corruption_is_rejected(self):
        (self.gbdt/'attempts/G05_seed_42/validation.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):rf.reference_identity(self.config,self.root)

    def test_reference_manifest_pin_is_required(self):
        self.config['reference']['manifest_sha256']='wrong'
        with self.assertRaises(ValueError):rf.reference_identity(self.config,self.root)

    def test_rf_trial_cannot_accept_gbdt_archive_as_evidence(self):
        with self.assertRaises(ValueError):rf.require_trial(self.gbdt,self.config['reference']['manifest_sha256'],self.config,{'inputs':self.inputs})


class RFProtectedRoutingTests(unittest.TestCase):
    def test_protected_call_covers_rf_worker_and_approved_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            def observed(monitor,resources,**kwargs):
                self.assertEqual(resources['job_commit_limit_bytes'],8*rf.GIB)
                self.assertEqual(resources['logical_cpu_limit'],4)
                self.assertEqual(kwargs['worker_module'],'methods.m1.run_rf')
                self.assertEqual(kwargs['worker_args'][-1],'run_42')
                self.assertIs(kwargs['reason_function'],rf.abort_reason)
                self.assertEqual(os.environ['PYTHONUTF8'],'1')
                rf.write(monitor/'worker_result.json',{'status':'rf_run_42_complete'})
                return {'exit_code':0,'abort_reason':None}
            with patch.object(resource_trial,'WindowsAPI') as api,patch.object(resource_trial,'monitored_child',side_effect=observed):
                api.return_value.memory.return_value={'available_physical_bytes':15*rf.GIB}
                result=rf.protected_call(run,approved(),'run_42',True)
                self.assertEqual(result['exit_code'],0)

    def test_low_start_memory_blocks_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(resource_trial,'WindowsAPI') as api,patch.object(resource_trial,'monitored_child') as child:
                api.return_value.memory.return_value={'available_physical_bytes':13*rf.GIB}
                with self.assertRaises(ValueError):rf.protected_call(Path(tmp),approved(),'run',True)
                child.assert_not_called()

    def test_manual_checkpoint_missing_blocks_acceptance_before_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);monitor=run/'monitor';monitor.mkdir()
            config=approved();identity={'inputs':{'fixture':'same'}}
            ref={'inputs':{'fixture':'same'}}
            rf.write(run/'config.json',config)
            rf.write(run/'binding.json',identity)
            rf.write(run/'paired_reference.json',ref)
            rf.write(run/'resource_trial_reference.json',{'run':'.tmp/synthetic','manifest_sha256':'fixture'})
            with patch('sys.stdin',io.StringIO('GO\n')),patch.object(rf,'binding',return_value=identity), \
                 patch.object(rf,'reference_identity',return_value=ref),patch.object(rf,'require_trial'), \
                 patch.object(rf,'summarize') as summary,patch('sys.stdout',io.StringIO()):
                with self.assertRaises(FileNotFoundError):rf.worker(run,monitor,'accept')
                summary.assert_not_called()


@unittest.skipUnless(os.name=='nt','Windows immutable evidence reading')
class RFTrialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.run=Path(self.temporary.name)
        self.config=approved()
        self.identity={'inputs':{name:name+'-SYNTHETIC-HASH' for name in ('train.parquet','val.parquet','preprocessor.json','preprocess_manifest.json')},
                       'environment':{'fixture':'SYNTHETIC ONLY'},'code':{'methods/m1/run_rf.py':'SYNTHETIC-CODE'}}
        rf.write(self.run/'binding.json',self.identity)
        for split,rows in [('train',1764265),('val',367036)]:
            path=self.run/'cache'/split;path.mkdir(parents=True)
            rf.write(path/'READY.json',{'rows':rows,'source_manifest_output':{'main':{'sha256':self.identity['inputs'][split+'.parquet']}}})
        rf.write(self.run/'cache.json',{'directory':'cache'})
        self.report={'status':'rf_resource_trial_complete','resource_spec_sha256':rf.digest(rf.resource_spec(self.config)),
                     'rows':{'train':1764265,'val':367036},'AP_computed':False,
                     'monitor_outcome':{'exit_code':0,'abort_reason':None,'job_commit_limit_bytes':8*rf.GIB,'logical_cpu_limit':4,'guard_verified':True}}
        rf.write(self.run/'resource_report.json',self.report)
        (self.run/'completed').mkdir()
        rf.write(self.run/'completed/RF_FIXED_seed_42.json',{'task':{'seed':42,'params':rf.PARAMS},'purpose':'resource_trial_only',
            'validation_ap':None,'reload_predictions_equal':True,'saved_predictions_equal':True})
        self.publish_manifest()

    def publish_manifest(self):
        from methods.m1.seal_gbdt import file_inventory
        manifest={'status':'rf_resource_trial_complete','files':file_inventory(self.run)}
        (self.run/'MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
        self.manifest_sha=rf.sha(self.run/'MANIFEST.json')

    def test_well_formed_synthetic_evidence_is_accepted_not_actual_feasibility(self):
        rf.require_trial(self.run,self.manifest_sha,self.config,self.identity)

    def test_wrong_evidence_hash_is_rejected(self):
        with self.assertRaises(ValueError):rf.require_trial(self.run,'wrong',self.config,self.identity)

    def test_trial_environment_data_or_code_mismatch_is_rejected(self):
        for key in ('inputs','environment','code'):
            identity={**self.identity,key:{'changed':True}}
            with self.assertRaises(ValueError):rf.require_trial(self.run,self.manifest_sha,self.config,identity)

    def test_failed_resource_trial_cannot_unlock_formal_run(self):
        self.report['monitor_outcome']['exit_code']=71
        (self.run/'resource_report.json').write_text(json.dumps(self.report))
        self.publish_manifest()
        with self.assertRaises(ValueError):rf.require_trial(self.run,self.manifest_sha,self.config,self.identity)

    def test_wrong_trial_parameters_are_rejected(self):
        self.report['resource_spec_sha256']='another model'
        (self.run/'resource_report.json').write_text(json.dumps(self.report))
        self.publish_manifest()
        with self.assertRaises(ValueError):rf.require_trial(self.run,self.manifest_sha,self.config,self.identity)


@unittest.skipUnless(os.name=='nt','Windows native mechanisms')
class RFNativeTests(unittest.TestCase):
    def test_native_memory_cap_and_four_cpu_affinity_shared_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={**settings()['resources'],'job_commit_limit_bytes':64*1024**2,'sample_seconds':.05,'disk_sample_seconds':5}
            result=resource_trial.monitored_child(tmp,config,worker_module='methods.m1.resource_trial',
                worker_args=('--guard-probe',tmp),reason_function=rf.abort_reason)
            self.assertEqual(result['exit_code'],0)
            self.assertTrue(rf.read(Path(tmp)/'probe.json')['memory_limit_enforced'])
            self.assertEqual(result['logical_cpu_limit'],4)

    def test_native_resource_abort_kills_child_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={**settings()['resources'],'job_commit_limit_bytes':64*1024**2,'sample_seconds':.01,'disk_sample_seconds':5}
            result=resource_trial.monitored_child(tmp,config,worker_module='methods.m1.resource_trial',
                worker_args=('--guard-probe',tmp),reason_function=lambda s,c:'rf_synthetic_abort')
            self.assertNotEqual(result['exit_code'],0)
            self.assertEqual(result['abort_reason'],'rf_synthetic_abort')

    def test_rf_entry_pending_worker_fails_inside_job_before_data_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root/'run';run.mkdir()
            rf.write(run/'config.json',settings())
            monitor=root/'monitor';monitor.mkdir()
            config={**settings()['resources'],'job_commit_limit_bytes':64*1024**2,'sample_seconds':.05,'disk_sample_seconds':5}
            with patch.dict(os.environ,{'PYTHONUTF8':'1'}):
                result=resource_trial.monitored_child(monitor,config,worker_module='methods.m1.run_rf',
                    worker_args=('--worker',run,monitor,'run_42'),reason_function=rf.abort_reason)
            self.assertNotEqual(result['exit_code'],0)
            self.assertIn('approval pending',(monitor/'worker.log').read_text(encoding='utf-8'))
            self.assertFalse((run/'binding.json').exists())
            self.assertFalse((run/'cache').exists())

    def test_atomic_rf_seal_excludes_lock_and_refuses_overwrite(self):
        from methods.m1.search_gbdt import run_lock
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            (run/'data.bin').write_bytes(b'synthetic')
            with run_lock(run):
                digest=rf.atomic_manifest(run,'rf_synthetic_complete',lambda files:None)
                self.assertEqual(rf.sha(run/'MANIFEST.json'),digest)
                self.assertNotIn('RUN.lock',rf.read(run/'MANIFEST.json')['files'])
                with self.assertRaises(ValueError):rf.atomic_manifest(run,'rf_synthetic_complete',lambda files:None)


class RFPerSeedControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        (self.root/'methods/m1').mkdir(parents=True)
        (self.root/'methods/m1/protocol.unified.v1.json').write_bytes((rf.ROOT/'methods/m1/protocol.unified.v1.json').read_bytes())
        self.config=approved();self.config['checkpoint']='manual_after_each_seed'
        for key in ('trial_approval','execution_approval'):self.config[key]['payload_sha256']=rf.digest(rf.payload(self.config))
        self.trial=self.root/'trial';self.trial.mkdir()
        self.identity={'fixture':'SYNTHETIC CONTROLLER ONLY'}
        self.calls=[];self.fits=[]

    def protected(self,run,config,stage,initialize):
        seed=int(stage.split('_')[1]);self.calls.append(seed)
        if not (run/'binding.json').exists():rf.write(run/'binding.json',self.identity)
        task=next(t for t in rf.tasks(config) if t['seed']==seed)
        record_path=run/'completed'/(task['task_id']+'.json')
        if record_path.exists():return {}
        self.fits.append(seed)
        directory=run/'attempts'/task['task_id'];directory.mkdir(parents=True)
        (directory/'model.joblib').write_bytes(b'SYNTHETIC MODEL BYTES')
        (directory/'validation.npz').write_bytes(b'SYNTHETIC SCORE BYTES')
        rf.write(directory/'task.json',task)
        monitor=run/'monitor'/stage;monitor.mkdir(parents=True)
        outcome={'exit_code':0,'abort_reason':None,'job_peak_commit_bytes':2*rf.GIB,'sampled_peak_rss_bytes':rf.GIB}
        rf.write(monitor/'outcome.json',outcome)
        rf.write(record_path,{'task':task,'binding_sha256':rf.digest(self.identity),'artifact_directory':directory.relative_to(run).as_posix(),
            'monitor_directory':monitor.relative_to(run).as_posix(),'files':rf.inventory(directory),'validation_ap':.9,'training_seconds':1})
        return outcome

    def launch(self,answers,resume=None,callback=None):
        from contextlib import nullcontext
        iterator=iter(answers)
        with patch.object(rf,'ROOT',self.root),patch.object(rf,'resource_slot',return_value=nullcontext()), \
             patch.object(rf,'protected_call',side_effect=callback or self.protected),patch('sys.stdout',io.StringIO()):
            return rf.launch(self.config,'run',resume,self.trial,'SYNTHETIC',ask=lambda _:next(iterator))

    def test_no_continue_stops_before_second_seed(self):
        run=self.launch(['STOP'])
        self.assertEqual(self.fits,[42])
        self.assertFalse((run/'completed/RF_FIXED_seed_43.json').exists())
        self.assertFalse((run/'checkpoint.json').exists())

    def test_one_continue_allows_only_one_more_seed(self):
        run=self.launch(['CONTINUE','STOP'])
        self.assertEqual(self.fits,[42,43])
        self.assertEqual([p.name for p in (run/'checkpoints').glob('*.json')],['RF_FIXED_seed_42.json'])

    def test_five_explicit_acknowledgments_required(self):
        run=self.launch(['CONTINUE']*5)
        self.assertEqual(self.fits,[42,43,44,45,46])
        rf.require_seed_checkpoints(run,self.config,self.identity)
        self.assertTrue((run/'checkpoint.json').exists())

    def test_resume_does_not_refit_completed_seed(self):
        run=self.launch(['STOP'])
        self.launch(['CONTINUE','STOP'],resume=run)
        self.assertEqual(self.fits,[42,43])

    def test_corrupted_acknowledgment_blocks_next_fit(self):
        run=self.launch(['CONTINUE','STOP'])
        path=run/'checkpoints/RF_FIXED_seed_42.json'
        record=rf.read(path);record['completed_record_sha256']='changed'
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):self.launch(['CONTINUE'],resume=run)
        self.assertEqual(self.fits,[42,43])

    def test_failure_never_starts_next_seed_or_retries(self):
        calls=[]
        def failed(run,cfg,stage,initialize):calls.append(stage);raise RuntimeError('synthetic resource abort')
        with self.assertRaises(RuntimeError):self.launch(['CONTINUE']*5,callback=failed)
        self.assertEqual(calls,['run_42'])

    def test_ast_proof_ignores_only_target_filter(self):
        old='def execute_tasks(config, *, trial=False):\n for task in tasks(config):\n  fit_tree(4)\n'
        new='def execute_tasks(config, *, trial=False, target_seed=None):\n for task in tasks(config):\n  if target_seed is not None and task[\'seed\']!=target_seed:continue\n  fit_tree(4)\n'
        self.assertEqual(rf.core_code_digest(old,'execute_tasks'),rf.core_code_digest(new,'execute_tasks'))
        self.assertNotEqual(rf.core_code_digest(old,'execute_tasks'),rf.core_code_digest(new.replace('fit_tree(4)','fit_tree(8)'),'execute_tasks'))


if __name__=='__main__':unittest.main()
