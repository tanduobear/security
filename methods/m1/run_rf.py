"""RF-only auxiliary execution. Default plan has no dataset access.

Actual trial/run/accept need separate recorded approval and explicit launch.
All ML/data/model loading happens after guarded GO. GBDT artifacts are read-only.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import uuid

from .search_gbdt import (ROOT,GIB,read,write,sha,digest,require,binding,run_lock,resource_slot,
                         prepare_cache,verify_inventory,inventory,abort_reason)

CONFIG=ROOT/'methods/m1/execution.rf-only.proposed.json'
PARAMS={'n_estimators':200,'max_depth':20,'max_leaf_nodes':4096,'min_samples_leaf':2,
        'max_features':'sqrt','bootstrap':True,'n_jobs':4,'class_weight':None}


def payload(config):
    return {k:v for k,v in config.items() if k not in ('trial_approval','execution_approval')}


def validate(config):
    require(config['stage']=='rf_auxiliary_only' and config['algorithm']=='RandomForestClassifier','RF-only scope required')
    require(config['parameters']==PARAMS and config['seeds']==[42,43,44,45,46],'Fixed RF parameters/seeds changed')
    require(config['eligible_for_gbdt_selection'] is False,'RF cannot enter GBDT selection')
    require(config['dataset']=={'version':'original','protocol':'stratified','variant':'main','features':140,'allowed_splits':['train','val']},'RF data contract changed')
    expected={'outer_parallel_jobs':1,'fit_threads':4,'inference_threads':1,'prediction_batch_rows':8192,
              'job_commit_limit_bytes':8*GIB,'logical_cpu_limit':4,'min_start_free_bytes':14*GIB,
              'min_running_free_bytes':6*GIB,'rss_abort_bytes':8*GIB,'duckdb_memory_limit_mb':512}
    require(config['resources']==expected,'RF resource limits changed')
    require(config['reporting']=={'metric':'full_validation_AP','std_ddof':1,'thresholds':False,'bootstrap':False},'Reporting scope changed')
    require(config['checkpoint'] in ('manual_after_all_five_seeds','manual_after_each_seed') and config['automatic_retries']==0,'Checkpoint/retry policy changed')
    require(config['resource_trial']=={'required_before_formal':True,'seed':42,'counts_as_formal':False,'AP_computed':False},'Resource trial scope changed')
    require(config['specification_approval']['status']=='approved','Specification approval missing')
    unified=read(ROOT/'methods/m1/protocol.unified.v1.json')
    require(unified['approval']['status']=='approved' and unified['sha256']==config['unified_protocol_sha256'],'Unified protocol mismatch')
    require(digest({k:v for k,v in unified.items() if k!='sha256'})==unified['sha256'],'Unified protocol modified')
    require(config['reference']['selected_candidate_id']=='G05','Paired reference must be G05')
    return config


def require_execution(config,stage):
    validate(config)
    approval=config['trial_approval' if stage=='trial' else 'execution_approval']
    require(approval['status']=='approved' and bool(approval['source']) and bool(approval['approved_at']),
            'RF '+stage+' approval pending; no output or data access allowed')
    require(approval['payload_sha256']==digest(payload(config)),'RF approved payload hash mismatch')


def tasks(config):
    validate(config)
    return [{'task_id':f'RF_FIXED_seed_{seed}','algorithm':'random_forest','seed':seed,
             'params':dict(config['parameters']),'role':'auxiliary','eligible_for_gbdt_selection':False} for seed in config['seeds']]


def safe_path(run,value):
    path=Path(value)
    require(not path.is_absolute() and '..' not in path.parts,'Unsafe artifact path')
    path=Path(run)/path
    require(path.resolve().is_relative_to(Path(run).resolve()),'Artifact escaped run')
    return path


def reference_identity(config,project=ROOT):
    """Verify sealed G05 identities and validation order, never load GBDT models."""
    import numpy as np
    from .seal_gbdt import artifact_paths,freeze_files,verify_manifest
    from .data import row_digest
    spec=config['reference']
    run=safe_path(project,spec['run'])
    require(sha(run/'MANIFEST.json')==spec['manifest_sha256'] and sha(run/'selection.json')==spec['selection_sha256'],'GBDT reference SHA256 mismatch')
    paths=artifact_paths(run)
    with freeze_files([*paths.values(),run/'MANIFEST.json']):
        verify_manifest(run,read(run/'MANIFEST.json'))
        selection=read(run/'selection.json')
        require(selection['selected_candidate_id']=='G05','GBDT selected reference changed')
        original=read(run/'binding.json')
        require(selection['binding_sha256']==digest(original),'GBDT binding mismatch')
        paired={}
        for seed in config['seeds']:
            record=read(run/'completed'/f'G05_seed_{seed}.json')
            directory=safe_path(run,record['artifact_directory'])
            require(record['task']['candidate_id']=='G05' and record['task']['seed']==seed,'G05 seed identity changed')
            with np.load(directory/'validation.npz',allow_pickle=False) as saved:
                y=np.asarray(saved['y_true'],dtype=np.uint8)
                require(len(y)==record['validation_rows'] and row_digest(saved['row_ids'])==record['validation_row_ids_sha256'],'G05 prediction identity mismatch')
                paired[str(seed)]={'row_ids_sha256':row_digest(saved['row_ids']),
                                   'labels_sha256':__import__('hashlib').sha256(y.tobytes()).hexdigest(),
                                   'rows':len(y),'scores_file_sha256':record['files']['validation.npz']['sha256']}
        require(len({digest({k:v for k,v in value.items() if k!='scores_file_sha256'}) for value in paired.values()})==1,'G05 validation populations differ')
    return {'manifest_sha256':spec['manifest_sha256'],'selection_sha256':spec['selection_sha256'],
            'selected_candidate_id':'G05','inputs':original['inputs'],'paired_seeds':paired}


def compare_population(task,ids,y,reference):
    import hashlib
    import numpy as np
    from .data import row_digest
    expected=reference['paired_seeds'][str(task['seed'])]
    require(expected['rows']==len(y) and expected['row_ids_sha256']==row_digest(ids) and
            expected['labels_sha256']==hashlib.sha256(np.asarray(y,dtype=np.uint8).tobytes()).hexdigest(),
            'RF/G05 validation row order or labels differ')


def verify_tasks(run,config,identity,reference,*,trial=False):
    import numpy as np
    from eval.metrics import compute_curves
    allowed={t['task_id']:t for t in tasks(config) if not trial or t['seed']==42}
    result={}
    for path in sorted((run/'completed').glob('*.json')):
        record=read(path)
        require(path.stem in allowed and record['task']==allowed[path.stem],'RF task/parameter mismatch')
        require(record['binding_sha256']==digest(identity) and record['purpose']==('resource_trial_only' if trial else 'rf_formal_validation'),'RF task identity mismatch')
        directory=safe_path(run,record['artifact_directory'])
        require(directory.relative_to(run).parts[0]=='attempts','Wrong attempt root')
        verify_inventory(directory,record['files'])
        require(read(directory/'task.json')==record['task'],'Saved RF task changed')
        with np.load(directory/'validation.npz',allow_pickle=False) as saved:
            compare_population(record['task'],saved['row_ids'],saved['y_true'],reference)
            require(saved['y_score'].shape==saved['y_true'].shape,'Saved RF scores misaligned')
            if trial:require(record['validation_ap'] is None,'Resource trial cannot compute AP')
            else:require(compute_curves(saved['y_true'],saved['y_score'])['average_precision']==record['validation_ap'],'Independent saved RF AP mismatch')
        require(record['reload_predictions_equal'] and record['saved_predictions_equal'],'RF save consistency missing')
        result[path.stem]=record
    return result


def execute_tasks(run,config,identity,reference,training,validation,monitor,*,trial=False,fitter=None,phase=lambda x:None,target_seed=None):
    """Production called only after GO; tests inject a non-training RF model."""
    import joblib
    import numpy as np
    from eval.metrics import compute_curves
    from .models import fit_tree
    from .executor import metadata_arrays,predict_batches
    complete=verify_tasks(run,config,identity,reference,trial=trial)
    ids,families,flags=metadata_arrays(validation)
    with training.model_input() as train,validation.model_input() as val:
        for task in tasks(config):
            if target_seed is not None and task['seed']!=target_seed:continue
            if trial and task['seed']!=42:continue
            compare_population(task,ids,val.y,reference)
            if task['task_id'] in complete:
                phase(task['task_id']+':verified_skip');continue
            name=task['task_id']
            directory=run/'attempts'/(name+'-'+uuid.uuid4().hex)
            directory.mkdir(parents=True,exist_ok=False)
            phase(name+':fit')
            model,seconds=(fitter or fit_tree)('random_forest',task['params'],train,model_seed=task['seed'],fit_threads=4)
            require(model.n_features_in_==140 and len(model.estimators_)==200,'RF feature/tree count mismatch')
            phase(name+':predict_validation')
            scores=predict_batches(model,val.X,rows=8192,threads=1)
            phase(name+':save_and_release_original_model')
            joblib.dump(model,directory/'model.joblib')
            del model
            gc.collect()  # never hold original and restored 200-tree forests together
            phase(name+':reload_model_and_predict')
            restored=joblib.load(directory/'model.joblib')
            require(np.array_equal(scores,predict_batches(restored,val.X,rows=8192,threads=1)),'Reloaded RF predictions differ')
            del restored
            gc.collect()
            phase(name+':save_and_check_predictions')
            with (directory/'validation.npz').open('xb') as stream:
                np.savez_compressed(stream,row_ids=ids,y_true=val.y,y_score=scores,attack_family=families,is_attempted=flags)
            ap=None if trial else compute_curves(val.y,scores)['average_precision']
            with np.load(directory/'validation.npz',allow_pickle=False) as saved:
                require(np.array_equal(saved['row_ids'],ids) and np.array_equal(saved['y_true'],val.y)
                        and np.array_equal(saved['y_score'],scores),'Saved RF predictions differ')
                if not trial:
                    phase(name+':independent_saved_AP')
                    require(compute_curves(saved['y_true'],saved['y_score'])['average_precision']==ap,'Independent RF AP differs')
            write(directory/'task.json',task)
            write(run/'completed'/(name+'.json'),{'task':task,'purpose':'resource_trial_only' if trial else 'rf_formal_validation',
                'binding_sha256':digest(identity),'artifact_directory':directory.relative_to(run).as_posix(),
                'files':inventory(directory),'monitor_directory':monitor.relative_to(run).as_posix(),
                'validation_ap':ap,'training_seconds':seconds,'reload_predictions_equal':True,
                'saved_predictions_equal':True,'test_accessed':False})
            phase(name+':complete')
            del scores


def resource_spec(config):
    return {key:config[key] for key in ('algorithm','parameters','dataset','resources','reference','resource_trial')}


def require_trial(run,manifest_sha,config,current_identity):
    from .seal_gbdt import artifact_paths,freeze_files
    require(sha(run/'MANIFEST.json')==manifest_sha,'RF trial manifest hash mismatch')
    with freeze_files([*artifact_paths(run).values(),run/'MANIFEST.json']):
        return _require_trial_locked(run,manifest_sha,config,current_identity)


def core_code_digest(source,name):
    """Ignore only the explicit task-subset filter; fitting body must be identical."""
    import ast
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    if name=='execute_tasks':
        keep=[i for i,arg in enumerate(node.args.kwonlyargs) if arg.arg!='target_seed']
        node.args.kwonlyargs=[node.args.kwonlyargs[i] for i in keep]
        node.args.kw_defaults=[node.args.kw_defaults[i] for i in keep]
        expected=ast.dump(ast.parse("if target_seed is not None and task['seed']!=target_seed:continue").body[0],include_attributes=False)
        for child in ast.walk(node):
            if isinstance(child,ast.For):child.body=[n for n in child.body if ast.dump(n,include_attributes=False)!=expected]
    return digest(ast.dump(node,include_attributes=False))


def trial_code_compatible(run,manifest_sha,config,previous,current):
    essential=lambda identity:{k:v for k,v in identity['code'].items() if k.endswith('.py') or k.endswith('requirements-lock.txt')}
    old,new=essential(previous),essential(current)
    if old==new:return
    require(set(old)==set(new),'RF trial code file set changed')
    changed={name for name in old if old[name]!=new[name]}
    require(changed=={'methods/m1/run_rf.py','tests/test_m1_rf.py'},'Unapproved RF runtime code change')
    pointer=config.get('resource_trial_code_compatibility') or {}
    path=safe_path(ROOT,pointer.get('path','missing'))
    require(path.is_file() and sha(path)==pointer.get('sha256'),'Pinned RF checkpoint compatibility record missing')
    proof=read(path)
    require(proof['scope']=='per_seed_manual_checkpoints_only' and proof['trial_manifest_sha256']==manifest_sha,
            'Wrong RF compatibility scope/trial')
    for name in changed:
        require(proof['files'][name]=={'trial_sha256':old[name],'formal_sha256':new[name]},'RF compatibility byte hash mismatch')
    archived=(run/'sources/methods/m1/run_rf.py').read_text(encoding='utf-8')
    current_source=(ROOT/'methods/m1/run_rf.py').read_text(encoding='utf-8')
    for name in ('execute_tasks','reference_identity','compare_population','verify_tasks','resource_spec','atomic_manifest','protected_call','summarize'):
        require(core_code_digest(archived,name)==core_code_digest(current_source,name)==proof['unchanged_core_ast_sha256'][name],
                'RF fitting/prediction/storage core changed: '+name)


def _require_trial_locked(run,manifest_sha,config,current_identity):
    """A user-pinned successful full-size RF trial, never a GBDT trial."""
    from .seal_gbdt import file_inventory
    require(sha(run/'MANIFEST.json')==manifest_sha,'RF trial manifest hash mismatch')
    manifest=read(run/'MANIFEST.json')
    require(manifest['status']=='rf_resource_trial_complete' and file_inventory(run)==manifest['files'],'RF trial archive invalid')
    report=read(run/'resource_report.json')
    require(report['status']=='rf_resource_trial_complete' and report['resource_spec_sha256']==digest(resource_spec(config)), 'RF trial not applicable')
    previous=read(run/'binding.json')
    cache=read(run/'cache.json')
    for split in ('train','val'):
        ready=read(safe_path(run,cache['directory'])/split/'READY.json')
        require(ready['rows']==report['rows'][split] and ready['source_manifest_output']['main']['sha256']==previous['inputs'][split+'.parquet'],'RF trial cache scale/source mismatch')
    require(previous['inputs']==current_identity['inputs'] and previous['environment']==current_identity['environment'],'RF trial data/environment mismatch')
    trial_code_compatible(run,manifest_sha,config,previous,current_identity)
    outcome=report['monitor_outcome']
    require(outcome['exit_code']==0 and outcome['abort_reason'] is None and outcome['job_commit_limit_bytes']==8*GIB
            and outcome['logical_cpu_limit']==4 and outcome['guard_verified'],'RF trial resource guard failed')
    require(report['rows']=={'train':1764265,'val':367036} and report['AP_computed'] is False,'RF trial scale/purpose mismatch')
    jobs=list((run/'completed').glob('*.json'))
    require(len(jobs)==1,'RF trial must have exactly one fit')
    trial_record=read(jobs[0])
    require(trial_record['task']['seed']==42 and trial_record['task']['params']==config['parameters'] and
            trial_record['purpose']=='resource_trial_only' and trial_record['validation_ap'] is None and
            trial_record['reload_predictions_equal'] and trial_record['saved_predictions_equal'],'RF trial fit/reload proof invalid')


def summarize(run,config,identity,reference):
    import numpy as np
    jobs=verify_tasks(run,config,identity,reference)
    require(set(jobs)=={t['task_id'] for t in tasks(config)},'Five RF tasks required')
    output=[]
    for task in tasks(config):
        record=jobs[task['task_id']]
        monitor=safe_path(run,record['monitor_directory'])
        guard=read(monitor/'guard_verified.json')
        require(guard['job_commit_limit_bytes']==8*GIB and guard['logical_cpu_limit']==4 and guard['verified_before_GO'],'RF task guard proof missing')
        outcome=read(monitor/'outcome.json') if (monitor/'outcome.json').exists() else None
        with (monitor/'resource_samples.csv').open(encoding='utf-8') as stream:
            rows=[row for row in csv.DictReader(stream) if row['phase'].startswith(task['task_id']+':')]
        require(rows,'Missing RF per-task resource samples')
        output.append({'seed':task['seed'],'validation_ap':record['validation_ap'],'fit_seconds':record['training_seconds'],
            'sampled_task_peak_rss_bytes':max(int(row['rss_bytes']) for row in rows),
            'resource_sampling_scope':'whole task including fit, prediction, save and reload; not exact OS peak',
            'monitor_attempt_interrupted':outcome is None or outcome['exit_code']!=0 or outcome['abort_reason'] is not None,
            'completed_record_sha256':sha(run/'completed'/(task['task_id']+'.json')),
            'model_prediction_hashes':record['files']})
    ap=[item['validation_ap'] for item in output]
    seconds=[item['fit_seconds'] for item in output]
    return {'status':'rf_validation_acceptance_complete','tasks':output,'mean_validation_ap':float(np.mean(ap)),
            'sample_std_ap':float(np.std(ap,ddof=1)),'mean_fit_seconds':float(np.mean(seconds)),
            'sample_std_fit_seconds':float(np.std(seconds,ddof=1)), 'eligible_for_gbdt_selection':False,
            'binding_sha256':digest(identity),'paired_reference':reference,'test_accessed':False,'bootstrap':False,'thresholds_calibrated':False}


def worker(run,monitor,stage):
    require(sys.stdin.readline().strip()=='GO','Protected GO missing')
    target_seed=int(stage.split('_')[1]) if stage.startswith('run_') else None
    require(stage in ('trial','accept') or target_seed in (42,43,44,45,46),'Unknown RF worker stage')
    run,monitor=Path(run),Path(monitor)
    config=read(run/'config.json')
    require_execution(config,'trial' if stage=='trial' else 'run')
    def phase(name):
        (monitor/'state.json').write_text(json.dumps({'phase':name}),encoding='utf-8')
        from .resource_trial import WindowsAPI
        api=WindowsAPI()
        with (monitor/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'phase':name,
                                    'worker_stats_cumulative':api.stats(api.k.GetCurrentProcess())})+'\n')
        print(datetime.now(timezone.utc).isoformat(),name,flush=True)
    phase('binding_and_reference_verification')
    identity=binding(config)
    reference=reference_identity(config)
    require(identity['inputs']==reference['inputs'],'RF and G05 P2 data identity differs')
    if not (run/'binding.json').exists():
        require(read(monitor/'launch.json')['initialize'] and stage!='accept','RF initialization forbidden on resume')
        write(run/'binding.json',identity)
        write(run/'paired_reference.json',reference)
        for name in identity['code']:
            destination=run/'sources'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    require(read(run/'binding.json')==identity and read(run/'paired_reference.json')==reference,'RF resume identity/reference mismatch')
    if stage!='trial':
        trial=read(run/'resource_trial_reference.json')
        require_trial(safe_path(ROOT,trial['run']),trial['manifest_sha256'],config,identity)
    if stage=='accept':
        require(read(run/'checkpoint.json')=={'action':'CONTINUE','binding_sha256':digest(identity)},'RF manual checkpoint missing')
        require_seed_checkpoints(run,config,identity)
        phase('independent_full_validation_acceptance')
        write(run/'summary.json',summarize(run,config,identity,reference))
    else:
        phase('prepare_or_verify_train_val_cache')
        training,validation=prepare_cache(run,config)
        for split,item in (('train',training),('val',validation)):
            require(read(item.directory/'READY.json')['source_manifest_output']['main']['sha256']==identity['inputs'][split+'.parquet'],'RF cache P2 source identity differs')
        execute_tasks(run,config,identity,reference,training,validation,monitor,trial=stage=='trial',phase=phase,target_seed=target_seed)
    result={'status':'rf_'+stage+'_complete','test_accessed':False}
    if stage=='accept':result['summary_sha256']=sha(run/'summary.json')
    write(monitor/'worker_result.json',result)


def protected_call(run,config,stage,initialize):
    from .resource_trial import WindowsAPI,monitored_child
    require(WindowsAPI().memory()['available_physical_bytes']>=14*GIB,'RF start requires 14 GiB free')
    monitor=run/'monitor'/(stage.upper()+'-'+uuid.uuid4().hex)
    monitor.mkdir(parents=True,exist_ok=False)
    write(monitor/'launch.json',{'initialize':initialize})
    previous_utf8=os.environ.get('PYTHONUTF8')
    os.environ['PYTHONUTF8']='1'
    try:
        outcome=monitored_child(monitor,{**config['resources'],'sample_seconds':1,'disk_sample_seconds':5},
            worker_module='methods.m1.run_rf',worker_args=('--worker',run,monitor,stage),reason_function=abort_reason)
    finally:
        if previous_utf8 is None:os.environ.pop('PYTHONUTF8',None)
        else:os.environ['PYTHONUTF8']=previous_utf8
    write(monitor/'outcome.json',outcome)
    require(outcome['exit_code']==0 and outcome['abort_reason'] is None and
            read(monitor/'worker_result.json')['status']=='rf_'+stage+'_complete','RF interrupted/aborted; no automatic retry')
    return outcome


def atomic_manifest(run,status,checks):
    """Caller holds global/run locks; reuse mandatory sharing protection."""
    import os
    import tempfile
    from .seal_gbdt import artifact_paths,file_inventory,freeze_files,EXCLUDED
    require(not (run/'MANIFEST.json').exists(),'RF manifest cannot be overwritten')
    paths=artifact_paths(run)
    with freeze_files(paths.values()):
        files=file_inventory(run)
        checks(files)
        require(file_inventory(run)==files,'RF artifacts changed during sealing')
        manifest={'status':status,'files':files,'excluded_runtime_files':EXCLUDED,'test_accessed':False,
                  'self_hash_policy':'MANIFEST excluded from itself; pin its SHA256 externally.'}
        staging=ROOT/'.tmp'
        staging.mkdir(exist_ok=True)
        fd,name=tempfile.mkstemp(dir=staging,prefix='rf-manifest-')
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as stream:
                json.dump(manifest,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.link(name,run/'MANIFEST.json')
        finally:Path(name).unlink(missing_ok=True)
        with freeze_files([run/'MANIFEST.json']):
            require(read(run/'MANIFEST.json')==manifest,'RF published manifest differs')
            require(file_inventory(run)==files,'RF post-publication hash mismatch')
            result=sha(run/'MANIFEST.json')
    return result


def seed_checkpoint(run,config,identity,task,ask=input):
    record_path=run/'completed'/(task['task_id']+'.json')
    record=read(record_path)
    require(record['task']==task and record['binding_sha256']==digest(identity),'RF checkpoint task mismatch')
    verify_inventory(safe_path(run,record['artifact_directory']),record['files'])
    expected={'action':'CONTINUE','task_id':task['task_id'],'binding_sha256':digest(identity),
              'completed_record_sha256':sha(record_path)}
    path=run/'checkpoints'/(task['task_id']+'.json')
    if path.exists():
        require(read(path)==expected,'RF per-seed checkpoint changed');return True
    outcome=read(safe_path(run,record['monitor_directory'])/'outcome.json')
    print(f"{task['task_id']}: AP={record['validation_ap']:.9f}; fit={record['training_seconds']:.2f}s",flush=True)
    print(f"Protected attempt: peak Job commit={outcome['job_peak_commit_bytes']/GIB:.2f} GiB; sampled tree RSS={outcome['sampled_peak_rss_bytes']/GIB:.2f} GiB; CSV={record['monitor_directory']}/resource_samples.csv",flush=True)
    counters=outcome.get('last_process_counters') or {}
    if outcome.get('elapsed_seconds') and 'user_cpu_seconds' in counters:
        cores=(counters['user_cpu_seconds']+counters['kernel_cpu_seconds'])/outcome['elapsed_seconds']
        print(f'Whole protected attempt mean CPU: {cores:.2f} logical-core equivalent',flush=True)
    print(f"CHECKPOINT {task['task_id']}: no next seed starts without CONTINUE.",flush=True)
    if ask('Type CONTINUE after reviewing this seed; anything else stops safely: ').strip()!='CONTINUE':return False
    write(path,expected)
    return True


def require_seed_checkpoints(run,config,identity):
    if config['checkpoint']!='manual_after_each_seed':return
    for task in tasks(config):
        expected={'action':'CONTINUE','task_id':task['task_id'],'binding_sha256':digest(identity),
                  'completed_record_sha256':sha(run/'completed'/(task['task_id']+'.json'))}
        require(read(run/'checkpoints'/(task['task_id']+'.json'))==expected,'RF seed not acknowledged')


def launch(config,stage,resume=None,trial_run=None,trial_sha=None,ask=input):
    require_execution(config,stage)  # approval before creating outputs/importing ML
    require(stage in ('trial','run','accept'),'Unknown launch stage')
    if stage=='run' and not resume:
        require(trial_run is not None and trial_sha,'Successful RF resource trial and pinned manifest required')
    if resume:
        run=Path(resume).resolve()
        require(read(run/'config.json')==config and not (run/'MANIFEST.json').exists(),'RF configuration changed or run sealed')
    else:
        require(stage!='accept','RF accept requires existing run')
        folder='m1_rf_trials' if stage=='trial' else 'm1_rf_runs'
        run=ROOT/'results'/folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex)
        run.mkdir(parents=True,exist_ok=False)
        write(run/'config.json',config)
        if stage=='run':
            relative=Path(trial_run).resolve().relative_to(ROOT)
            write(run/'resource_trial_reference.json',{'run':relative.as_posix(),'manifest_sha256':trial_sha})
    with resource_slot(),run_lock(run):
        initialize=not resume
        require((run/'binding.json').exists() or initialize,'Incomplete RF initialization; manual review required')
        if stage=='run':
            require(config['checkpoint']=='manual_after_each_seed','Formal RF requires per-seed manual checkpoints')
            if not (run/'controller.json').exists():write(run/'controller.json',{'pid':os.getpid(),'purpose':'foreground_RF_five_seed_controller'})
            for task in tasks(config):
                checkpoint=run/'checkpoints'/(task['task_id']+'.json')
                if not checkpoint.exists():
                    protected_call(run,config,'run_'+str(task['seed']),initialize)
                    initialize=False
                identity=read(run/'binding.json')
                if not seed_checkpoint(run,config,identity,task,ask):
                    print('RF paused at seed checkpoint; completed results preserved.',flush=True);return run
            acknowledgment={'action':'CONTINUE','binding_sha256':digest(read(run/'binding.json'))}
            if (run/'checkpoint.json').exists():require(read(run/'checkpoint.json')==acknowledgment,'RF final checkpoint changed')
            else:write(run/'checkpoint.json',acknowledgment)
            print('Five RF seeds and manual checkpoints complete. Run accept and seal separately. RF archive:',run,flush=True)
            return run
        outcome=protected_call(run,config,stage,initialize)
        if stage=='trial':
            cache=read(run/'cache.json')
            directory=run/cache['directory']
            write(run/'resource_report.json',{'status':'rf_resource_trial_complete','resource_spec_sha256':digest(resource_spec(config)),
                'rows':{split:read(directory/split/'READY.json')['rows'] for split in ('train','val')},
                'AP_computed':False,'monitor_outcome':outcome})
            print('RF trial MANIFEST SHA256:',atomic_manifest(run,'rf_resource_trial_complete',lambda files:None),flush=True)
        print('RF archive:',run,flush=True)
    return run


def seal_run(run,config):
    require_execution(config,'run')
    run=Path(run).resolve()
    require(read(run/'config.json')==config,'RF seal configuration mismatch')
    with resource_slot(),run_lock(run):
        def checks(files):
            identity=read(run/'binding.json')
            require(identity==binding(config),'RF seal current identity mismatch')
            summary=read(run/'summary.json')
            require(summary['status']=='rf_validation_acceptance_complete' and summary['binding_sha256']==digest(identity),'RF acceptance missing')
            require(summary['paired_reference']==read(run/'paired_reference.json'),'RF pairing summary changed')
            require(read(run/'checkpoint.json')=={'action':'CONTINUE','binding_sha256':digest(identity)},'RF checkpoint not confirmed')
            require_seed_checkpoints(run,config,identity)
            require({p.stem for p in (run/'completed').glob('*.json')}=={t['task_id'] for t in tasks(config)},'RF seal must contain exactly five completed tasks')
            for task in tasks(config):
                job=read(run/'completed'/(task['task_id']+'.json'))
                require(job['task']==task and job['binding_sha256']==digest(identity),'RF seal task mismatch')
                directory=safe_path(run,job['artifact_directory'])
                require(inventory(directory)==job['files'],'RF task changed after acceptance')
                accepted=next(r for r in summary['tasks'] if r['seed']==task['seed'])
                require(accepted['model_prediction_hashes']==job['files'] and accepted['validation_ap']==job['validation_ap'],'RF accepted result changed')
                require(accepted['completed_record_sha256']==sha(run/'completed'/(task['task_id']+'.json')),'RF completion record changed after acceptance')
            accepts=list((run/'monitor').glob('ACCEPT-*'))
            require(len(accepts)==1 and read(accepts[0]/'outcome.json')['exit_code']==0 and
                    read(accepts[0]/'outcome.json').get('abort_reason') is None and
                    read(accepts[0]/'worker_result.json')['summary_sha256']==sha(run/'summary.json'),'RF acceptance worker/summary incomplete')
            cache=read(run/'cache.json')
            require(inventory(safe_path(run,cache['directory']))==cache['files'],'RF cache changed')
            for name,expected in identity['code'].items():require(sha(run/'sources'/name)==expected,'RF source snapshot changed')
            require(sha(safe_path(ROOT,config['reference']['run'])/'MANIFEST.json')==config['reference']['manifest_sha256'],'GBDT reference changed')
        return atomic_manifest(run,'rf_auxiliary_validation_complete',checks)


def verify_run(run,manifest_sha):
    from .seal_gbdt import artifact_paths,file_inventory,freeze_files,EXCLUDED
    run=Path(run).resolve()
    with resource_slot(),run_lock(run):
        require(sha(run/'MANIFEST.json')==manifest_sha,'RF manifest pinned hash mismatch')
        with freeze_files([*artifact_paths(run).values(),run/'MANIFEST.json']):
            manifest=read(run/'MANIFEST.json')
            require(manifest['status']=='rf_auxiliary_validation_complete' and manifest['excluded_runtime_files']==EXCLUDED,'Not an accepted RF archive')
            require(file_inventory(run)==manifest['files'],'RF archive file set/hash mismatch')
            return {'files_verified':len(manifest['files']),'manifest_sha256':manifest_sha,'test_accessed':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','trial','run','accept','seal','verify'],nargs='?',default='plan')
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resource-trial',type=Path)
    parser.add_argument('--trial-manifest-sha256')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--confirm-start',action='store_true')
    parser.add_argument('--worker',nargs=3)
    args=parser.parse_args(argv)
    if args.worker:worker(*args.worker);return
    if args.stage=='verify':
        require(args.resume is not None and args.manifest_sha256,'RF verify needs archive and pinned manifest hash')
        print(json.dumps(verify_run(args.resume,args.manifest_sha256),indent=2));return
    config=validate(read(args.config))
    if args.stage=='plan':
        print(json.dumps({'config_sha256':sha(args.config),'proposed_payload_sha256':digest(payload(config)),
            'formal_tasks':tasks(config),'trial_approval':config['trial_approval']['status'],
            'formal_execution_approval':config['execution_approval']['status'],'training_started':False},indent=2))
        return
    require(args.confirm_start,'Explicit RF phase launch confirmation required')
    if args.stage=='seal':
        require(args.resume is not None,'RF seal requires --resume')
        print(seal_run(args.resume,config))
    else:launch(config,args.stage,args.resume,args.resource_trial,args.trial_manifest_sha256)


if __name__=='__main__':main()
