"""One explicitly authorized resource-only fit; no search, AP or test access.

Parent: Windows Job Object, CPU affinity, one-second monitoring and aborts.
Worker: waits for parent authorization before importing/loading the ML stack.
This entry does not approve execution.proposed.json or call the formal runner.
"""
import argparse
from contextlib import contextmanager, ExitStack
import csv
import ctypes as ct
from ctypes import wintypes as wt
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
MIB = 1024**2


def checked(condition, message):
    if not condition:raise ValueError(message)


def write_json(path, value, *, exclusive=True):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class MemoryStatus(ct.Structure):
    _fields_ = [('length',wt.DWORD),('load',wt.DWORD)] + [(x,ct.c_ulonglong) for x in
        ['total_physical','available_physical','total_pagefile','available_pagefile','total_virtual','available_virtual','available_extended']]


class ProcessMemory(ct.Structure):
    _fields_ = [('cb',wt.DWORD),('page_faults',wt.DWORD)] + [(x,ct.c_size_t) for x in
        ['peak_rss','rss','peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool','pagefile','peak_pagefile','private_bytes']]


class IOCounters(ct.Structure):
    _fields_ = [(x,ct.c_ulonglong) for x in ['read_operations','write_operations','other_operations','read_bytes','write_bytes','other_bytes']]


class JobBasic(ct.Structure):
    _fields_ = [('process_user_limit',ct.c_longlong),('job_user_limit',ct.c_longlong),('flags',wt.DWORD),
                ('min_working_set',ct.c_size_t),('max_working_set',ct.c_size_t),('active_process_limit',wt.DWORD),
                ('affinity',ct.c_size_t),('priority_class',wt.DWORD),('scheduling_class',wt.DWORD)]


class JobExtended(ct.Structure):
    _fields_ = [('basic',JobBasic),('io',IOCounters),('process_memory_limit',ct.c_size_t),
                ('job_memory_limit',ct.c_size_t),('peak_process_memory',ct.c_size_t),('peak_job_memory',ct.c_size_t)]


class JobAccounting(ct.Structure):
    _fields_ = [(x,ct.c_longlong) for x in ['user_time','kernel_time','period_user_time','period_kernel_time']] + [
        (x,wt.DWORD) for x in ['page_faults','total_processes','active_processes','terminated_processes']]


class JobAccountingIO(ct.Structure):
    _fields_ = [('basic',JobAccounting),('io',IOCounters)]


class WindowsAPI:
    def __init__(self):
        checked(os.name == 'nt','Resource trial requires Windows native guards')
        self.k = ct.WinDLL('kernel32',use_last_error=True)
        self.p = ct.WinDLL('psapi',use_last_error=True)
        signatures = {
            'GetCurrentProcess':([],wt.HANDLE),
            'CreateJobObjectW':([ct.c_void_p,wt.LPCWSTR],wt.HANDLE),
            'SetInformationJobObject':([wt.HANDLE,ct.c_int,ct.c_void_p,wt.DWORD],wt.BOOL),
            'QueryInformationJobObject':([wt.HANDLE,ct.c_int,ct.c_void_p,wt.DWORD,ct.c_void_p],wt.BOOL),
            'AssignProcessToJobObject':([wt.HANDLE,wt.HANDLE],wt.BOOL),
            'IsProcessInJob':([wt.HANDLE,wt.HANDLE,ct.POINTER(wt.BOOL)],wt.BOOL),
            'TerminateJobObject':([wt.HANDLE,wt.UINT],wt.BOOL),
            'CloseHandle':([wt.HANDLE],wt.BOOL),
            'OpenProcess':([wt.DWORD,wt.BOOL,wt.DWORD],wt.HANDLE),
            'GlobalMemoryStatusEx':([ct.POINTER(MemoryStatus)],wt.BOOL),
            'GetProcessAffinityMask':([wt.HANDLE,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)],wt.BOOL),
            'SetProcessAffinityMask':([wt.HANDLE,ct.c_size_t],wt.BOOL),
            'GetProcessTimes':([wt.HANDLE]+[ct.POINTER(wt.FILETIME)]*4,wt.BOOL),
            'GetSystemTimes':([ct.POINTER(wt.FILETIME)]*3,wt.BOOL),
            'GetProcessIoCounters':([wt.HANDLE,ct.POINTER(IOCounters)],wt.BOOL),
        }
        for name,(args,result) in signatures.items():
            fn=getattr(self.k,name);fn.argtypes=args;fn.restype=result
        self.p.GetProcessMemoryInfo.argtypes=[wt.HANDLE,ct.POINTER(ProcessMemory),wt.DWORD]
        self.p.GetProcessMemoryInfo.restype=wt.BOOL

    def ok(self, value):
        if not value:raise ct.WinError(ct.get_last_error())
        return value

    def memory(self):
        value=MemoryStatus();value.length=ct.sizeof(value)
        self.ok(self.k.GlobalMemoryStatusEx(ct.byref(value)))
        return {'total_physical_bytes':value.total_physical,'available_physical_bytes':value.available_physical}

    def affinity(self, handle):
        process,system=ct.c_size_t(),ct.c_size_t()
        self.ok(self.k.GetProcessAffinityMask(handle,ct.byref(process),ct.byref(system)))
        return process.value

    def stats(self, handle):
        memory=ProcessMemory();memory.cb=ct.sizeof(memory)
        self.ok(self.p.GetProcessMemoryInfo(handle,ct.byref(memory),ct.sizeof(memory)))
        times=[wt.FILETIME() for _ in range(4)]
        self.ok(self.k.GetProcessTimes(handle,*[ct.byref(x) for x in times]))
        seconds=lambda value:((value.dwHighDateTime<<32)|value.dwLowDateTime)/10_000_000
        io=IOCounters();self.ok(self.k.GetProcessIoCounters(handle,ct.byref(io)))
        return {'rss_bytes':memory.rss,'os_peak_rss_bytes':memory.peak_rss,'private_bytes':memory.private_bytes,
                'user_cpu_seconds':seconds(times[3]),'kernel_cpu_seconds':seconds(times[2]),
                'read_io_bytes':io.read_bytes,'write_io_bytes':io.write_bytes}

    def system_times(self):
        values=[wt.FILETIME() for _ in range(3)]
        self.ok(self.k.GetSystemTimes(*[ct.byref(x) for x in values]))
        return [((x.dwHighDateTime<<32)|x.dwLowDateTime) for x in values]


class GuardedJob:
    def __init__(self, api, memory_bytes):
        self.api,self.handle=api,api.ok(api.k.CreateJobObjectW(None,None))
        limits=JobExtended()
        limits.basic.flags=0x200|0x2000  # JOB_MEMORY and KILL_ON_JOB_CLOSE
        limits.job_memory_limit=memory_bytes
        try:
            api.ok(api.k.SetInformationJobObject(self.handle,9,ct.byref(limits),ct.sizeof(limits)))
            checked(self.info().job_memory_limit==memory_bytes,'Job memory limit verification failed')
        except BaseException:
            self.close();raise

    def info(self):
        info=JobExtended()
        self.api.ok(self.api.k.QueryInformationJobObject(self.handle,9,ct.byref(info),ct.sizeof(info),None))
        return info

    def stats(self):
        """Query the entire Job tree, including Windows venv redirector children.

        A Popen handle can identify a 4 MiB launcher rather than the interpreter.
        Job accounting retains CPU/I/O totals even for terminated descendants.
        RSS is an instantaneous sum; individual peak sums are NOT a tree peak.
        """
        size=4096
        while True:
            buffer=ct.create_string_buffer(size)
            if self.api.k.QueryInformationJobObject(self.handle,3,buffer,size,None):break
            error=ct.get_last_error()
            if error!=234 or size>=1024*1024:raise ct.WinError(error)
            size*=2
        count=wt.DWORD.from_buffer(buffer,ct.sizeof(wt.DWORD)).value
        pids=[ct.c_size_t.from_buffer(buffer,2*ct.sizeof(wt.DWORD)+i*ct.sizeof(ct.c_size_t)).value for i in range(count)]
        values=[]
        for pid in pids:
            handle=self.api.k.OpenProcess(0x0400|0x0010,False,pid)
            if not handle:
                error=ct.get_last_error()
                if error in (87,1168):continue # process exited between enumeration and open
                raise ct.WinError(error)
            try:
                values.append(self.api.stats(handle))
            except OSError as error:
                if error.winerror not in (6,87,1168):raise
            finally:self.api.k.CloseHandle(handle)
        accounting=JobAccountingIO()
        self.api.ok(self.api.k.QueryInformationJobObject(self.handle,8,ct.byref(accounting),ct.sizeof(accounting),None))
        return {'rss_bytes':sum(x['rss_bytes'] for x in values),
            'sum_individual_os_peak_rss_bytes':sum(x['os_peak_rss_bytes'] for x in values),
            'private_bytes':sum(x['private_bytes'] for x in values),
            'user_cpu_seconds':accounting.basic.user_time/10_000_000,
            'kernel_cpu_seconds':accounting.basic.kernel_time/10_000_000,
            'read_io_bytes':accounting.io.read_bytes,'write_io_bytes':accounting.io.write_bytes,
            'monitored_process_count':len(values),'job_active_process_count':accounting.basic.active_processes,
            'job_peak_commit_bytes':self.info().peak_job_memory,'job_process_ids':str(pids)}

    def attach(self, process, logical_cpus):
        handle=wt.HANDLE(int(process._handle))
        self.api.ok(self.api.k.AssignProcessToJobObject(self.handle,handle))
        member=wt.BOOL()
        self.api.ok(self.api.k.IsProcessInJob(handle,self.handle,ct.byref(member)))
        checked(member.value,'Worker is not in the guarded Job Object')
        allowed=self.api.affinity(self.api.k.GetCurrentProcess())
        bits=[1<<i for i in range(allowed.bit_length()) if allowed&(1<<i)]
        checked(len(bits)>=logical_cpus,'Insufficient allowed logical CPUs')
        mask=sum(bits[:logical_cpus])
        self.api.ok(self.api.k.SetProcessAffinityMask(handle,mask))
        checked(self.api.affinity(handle)==mask,'Worker CPU affinity was not applied')
        return mask

    def terminate(self):
        self.api.ok(self.api.k.TerminateJobObject(self.handle,71))

    def close(self):
        if self.handle:
            self.api.k.CloseHandle(self.handle);self.handle=None


def directory_bytes(path):
    total=0
    for entry in Path(path).rglob('*'):
        try:
            if entry.is_file():total+=entry.stat().st_size
        except FileNotFoundError:pass
    return total


def resource_reason(sample, config):
    if sample['available_physical_bytes'] < config['min_running_free_bytes']:return 'system_available_memory_below_6_GiB'
    if sample['rss_bytes'] > config['rss_abort_bytes']:return 'worker_RSS_above_8_GiB'
    if sample['disk_free_bytes'] < config['min_disk_free_bytes']:return 'disk_free_below_20_GiB'
    if sample['trial_directory_bytes'] > config['directory_abort_bytes']:return 'trial_directory_above_10_GiB'
    if sample['elapsed_seconds'] >= config['timeout_seconds']:return 'wall_time_above_60_minutes'
    return None


def trial_config():
    proposal=read_json(ROOT/'methods/m1/execution.proposed.json')
    unified=read_json(ROOT/'methods/m1/protocol.unified.v1.json')
    candidate=next(x for x in proposal['gbdt_candidates'] if x['id']=='G01')
    checked(candidate['params']=={'max_iter':200,'learning_rate':.05,'max_leaf_nodes':15,'min_samples_leaf':20,
        'l2_regularization':1.0,'max_bins':255,'early_stopping':False,'class_weight':None},'G01 no longer matches the approved resource-trial parameters')
    return {'purpose':'resource_trial_only','authorization':'User explicitly confirmed the resource-only G01/seed42 plan; not approval of formal execution details',
        'unified_protocol_sha256':unified['sha256'],'candidate':candidate,'seed':42,
        'dataset':{'version':'original','protocol':'stratified','variant':'main','features':140,'allowed_splits':['train','val']},
        'fit_threads':4,'inference_threads':1,'prediction_batch_rows':8192,
        'inference':{'batch_size':512,'warmup':5,'repeats':30,'threads':1},'timing_seed':2026,
        'job_commit_limit_bytes':8*GIB,'rss_abort_bytes':8*GIB,'min_start_free_bytes':14*GIB,
        'min_running_free_bytes':6*GIB,'min_disk_free_bytes':20*GIB,'directory_abort_bytes':10*GIB,
        'timeout_seconds':3600,'sample_seconds':1,'disk_sample_seconds':5,'logical_cpu_limit':4,
        'performance_metrics_forbidden':['AP','ROC_AUC','F1','recall','threshold_calibration','bootstrap','model_selection'],
        'trial_model_is_not_a_formal_result':True,'retry_count':0}


def validate_trial_config(config):
    checked(config['purpose']=='resource_trial_only' and config['seed']==42,'Unexpected trial scope')
    expected=trial_config()
    checked(config==expected,'Trial configuration differs from the explicitly authorized plan')


def allowed_source(path, directory):
    path,directory=Path(path),Path(directory).resolve()
    checked(path.name in ('train.parquet','val.parquet'),'Resource trial forbids test data')
    checked(not path.is_symlink() and path.resolve().parent==directory,'Resource input escaped the approved directory')
    return path


def worker(run_dir):
    # Child starts with stdlib only. Parent sends GO only after assigning guards.
    if sys.stdin.readline().strip()!='GO':return 72
    run_dir=Path(run_dir).resolve();config=read_json(run_dir/'config.json')
    validate_trial_config(config)
    api=WindowsAPI();self_handle=api.k.GetCurrentProcess()
    phases={};started=time.perf_counter()
    with (run_dir/'events.jsonl').open('x',encoding='utf-8') as events:
        @contextmanager
        def phase(name):
            phase_start=time.perf_counter()
            write_json(run_dir/'state.json',{'phase':name,'started_at':datetime.now(timezone.utc).isoformat()},exclusive=False)
            events.write(json.dumps({'phase':name,'event':'start','elapsed_seconds':phase_start-started})+'\n');events.flush()
            try:yield
            finally:
                duration=time.perf_counter()-phase_start
                phases[name]=duration
                events.write(json.dumps({'phase':name,'event':'end','seconds':duration,**api.stats(self_handle)})+'\n');events.flush()
                print(f'{name}: {duration:.2f}s',flush=True)
        try:
            with phase('imports'):
                import joblib
                import numpy as np
                from threadpoolctl import threadpool_limits
                from src.model_loader import ModelDataset
                from eval.metrics import measure_inference
                from methods.m1.data import materialize_preparation,assert_disk_isolated,row_digest
                from methods.m1.evaluation import timing_indices
                from methods.m1.models import fit_tree,attack_scores,inference_workers
                from methods.m1.protocol import validate_unified
                from methods.m1.storage import environment
                validate_unified(read_json(ROOT/'methods/m1/protocol.unified.v1.json'))
            write_json(run_dir/'environment.json',{**environment(),'resource_trial_affinity_mask':api.affinity(self_handle),
                'logical_processor_count':os.cpu_count(),'working_directory':str(Path.cwd()),
                'cache_state':'uncontrolled existing OS cache; not a cold-cache benchmark'})
            sources=list((ROOT/'methods/m1').glob('*.py'))+[ROOT/'src/model_loader.py',ROOT/'src/preprocess.py',ROOT/'eval/metrics.py',ROOT/'methods/m1/requirements-lock.txt']
            for source in sources:
                target=run_dir/'sources'/source.relative_to(ROOT)
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as out:out.write(source.read_bytes())
            data_dir=ROOT/'data/split/original/stratified'
            accesses=[]
            def factory(path):
                path=allowed_source(path,data_dir)
                accesses.append(str(path))
                return ModelDataset(path)
            with phase('load_train'):
                training=materialize_preparation(data_dir/'train.parquet',run_dir/'cache/train',loader_factory=factory)
            with phase('load_validation'):
                validation=materialize_preparation(data_dir/'val.parquet',run_dir/'cache/val',loader_factory=factory)
            with phase('isolation_check'):
                assert_disk_isolated(training,validation,memory_limit_mb=512)
                checked(training.feature_names==validation.feature_names and len(training.feature_names)==140,'Wrong P2 feature whitelist')
            with ExitStack() as stack:
                with phase('open_maps_and_check'):
                    train=stack.enter_context(training.model_input())
                    val=stack.enter_context(validation.model_input())
                with phase('input_identity_snapshot'):
                    write_json(run_dir/'inputs.json',{'sources_opened':accesses,'test_accessed':False,'features':list(training.feature_names),
                        'train_rows':training.rows,'validation_rows':validation.rows,'training_class_weights':training.class_weights,
                        'train_sha256':file_sha(data_dir/'train.parquet'),'validation_sha256':file_sha(data_dir/'val.parquet'),
                        'preprocessor_sha256':file_sha(data_dir/'preprocessor.json')})
                with phase('fit_G01_seed42'):
                    model,fit_seconds=fit_tree('hist_gradient_boosting',config['candidate']['params'],train,model_seed=42,fit_threads=4)
                with phase('predict_validation'):
                    scores=np.empty(validation.rows,dtype=np.float64)
                    with threadpool_limits(limits=1):
                        for offset in range(0,len(scores),8192):
                            scores[offset:offset+8192]=attack_scores(model,val.X[offset:offset+8192])
                with phase('save_model'):
                    with (run_dir/'model.joblib').open('xb') as out:joblib.dump(model,out)
                with phase('reload_and_verify_predictions'):
                    restored=joblib.load(run_dir/'model.joblib')
                    with threadpool_limits(limits=1):
                        for offset in range(0,len(scores),8192):
                            checked(np.array_equal(scores[offset:offset+8192],attack_scores(restored,val.X[offset:offset+8192])),'Model reload changed scores')
                    del restored
                with phase('save_predictions'):
                    ids=np.empty(validation.rows,dtype=object);offset=0
                    for metadata in validation.metadata_batches():
                        ids[offset:offset+len(metadata)]=metadata.row_id.to_numpy();offset+=len(metadata)
                    checked(offset==validation.rows,'Validation row identity count mismatch')
                    ids=ids.astype(str)
                    with (run_dir/'validation_predictions.npz').open('xb') as out:
                        np.savez_compressed(out,row_ids=ids,y_score=scores)
                    with np.load(run_dir/'validation_predictions.npz',allow_pickle=False) as saved:
                        checked(np.array_equal(saved['row_ids'],ids) and np.array_equal(saved['y_score'],scores),'Saved predictions differ')
                with phase('inference_resource_measurement'):
                    indices=timing_indices(ids,batch_size=512,seed=2026)
                    with inference_workers(model,1):
                        inference=measure_inference(lambda batch:attack_scores(model,batch),val.X[indices],**config['inference'])
                    write_json(run_dir/'inference.json',inference)
                    write_json(run_dir/'timing_rows.json',{'purpose':'resource_trial_only_not_the_formally_approved_common_list',
                        'row_ids':ids[indices].tolist(),'row_ids_sha256':row_digest(ids[indices]),'selection_seed':2026})
                checked(len(accesses)==2 and all(Path(x).name in ('train.parquet','val.parquet') for x in accesses),'Unexpected data access')
            write_json(run_dir/'worker_result.json',{'status':'worker_complete','purpose':'resource_trial_only','phases_seconds':phases,
                'fit_seconds':fit_seconds,'model_iterations':int(model.n_iter_),'train_rows':training.rows,'validation_rows':validation.rows,
                'worker_total_seconds_before_result':time.perf_counter()-started,
                'final_native_process_stats':api.stats(self_handle),'test_accessed':False,'performance_metrics_computed':False,
                'model_reload_prediction_equal':True,'saved_predictions_equal':True,'sample_weight_source':'P2 training fold',
                'model_is_not_a_formal_result':True})
            return 0
        except BaseException as error:
            traceback.print_exc()
            write_json(run_dir/'worker_result.json',{'status':'worker_failed','purpose':'resource_trial_only','phases_seconds':phases,
                'error_type':type(error).__name__,'message':str(error),'final_native_process_stats':api.stats(self_handle)})
            return 1


def guard_probe(run_dir):
    if sys.stdin.readline().strip()!='GO':return 72
    api=WindowsAPI()
    # Stay observable long enough to verify tree RSS/CPU, not just the launcher.
    small=bytearray(16*MIB)
    until=time.perf_counter()+.25
    while time.perf_counter()<until:
        sum(i*i for i in range(2000))
    del small
    enforced=False
    try:
        allocation=bytearray(128*MIB)  # tiny guard test under a 64 MiB job cap
        del allocation
    except MemoryError:enforced=True
    write_json(Path(run_dir)/'probe.json',{'memory_limit_enforced':enforced,'affinity_mask':api.affinity(api.k.GetCurrentProcess()),
        'native_process_stats':api.stats(api.k.GetCurrentProcess())})
    return 0 if enforced else 73


def preflight(config, memory, disk_free):
    if memory['available_physical_bytes'] < config['min_start_free_bytes']:return 'available_memory_below_14_GiB'
    if disk_free < config['min_disk_free_bytes']:return 'disk_free_below_20_GiB'
    return None


def monitored_child(run_dir, config, *, probe=False, worker_module=None, worker_args=(), reason_function=None):
    api=WindowsAPI();run_dir=Path(run_dir).resolve()
    job=GuardedJob(api,64*MIB if probe else config['job_commit_limit_bytes'])
    child=None;mask=None;samples=[];reason=None;last_stats=None;peak_job=0
    started=time.perf_counter();last_phase=None;last_disk_check=-10.;folder_size=directory_bytes(run_dir)
    previous_cpu=None;previous_system=None;previous_time=None
    try:
        env=dict(os.environ);env['PYTHONPATH']=str(ROOT);env['PYTHONDONTWRITEBYTECODE']='1';env['MPLCONFIGDIR']=str(run_dir/'worker/mplconfig')
        cwd=run_dir/'worker';cwd.mkdir(exist_ok=False)
        command=([sys.executable,'-B','-m',worker_module,*map(str,worker_args)] if worker_module else
                 [sys.executable,'-B','-m','methods.m1.resource_trial','--guard-probe' if probe else '--worker',str(run_dir)])
        with (run_dir/'worker.log').open('xb') as log,(run_dir/'resource_samples.csv').open('x',encoding='utf-8',newline='') as csv_file:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
            mask=job.attach(child,config['logical_cpu_limit'])
            checked(job.info().job_memory_limit==(64*MIB if probe else config['job_commit_limit_bytes']),'Job cap mismatch')
            write_json(run_dir/'guard_verified.json',{'job_commit_limit_bytes':job.info().job_memory_limit,
                'cpu_affinity_mask':mask,'logical_cpu_limit':mask.bit_count(),
                'monitoring_scope':'entire Job process tree, including venv launcher descendants',
                'verified_before_GO':True})
            child.stdin.write(b'GO\n');child.stdin.flush();child.stdin.close()
            writer=None
            while child.poll() is None:
                now=time.perf_counter();elapsed=now-started
                try:stats=job.stats()
                except OSError:
                    if child.poll() is not None:break
                    raise
                memory=api.memory();system=api.system_times()
                cpu=stats['user_cpu_seconds']+stats['kernel_cpu_seconds']
                cores=0. if previous_time is None else max(0.,(cpu-previous_cpu)/(now-previous_time))
                host=0.
                if previous_system is not None:
                    idle=system[0]-previous_system[0];total=(system[1]-previous_system[1])+(system[2]-previous_system[2])
                    host=100.*(total-idle)/total if total else 0.
                if elapsed-last_disk_check>=config['disk_sample_seconds']:
                    folder_size=directory_bytes(run_dir);last_disk_check=elapsed
                phase='worker_startup'
                if (run_dir/'state.json').exists():
                    try:phase=read_json(run_dir/'state.json')['phase']
                    except (json.JSONDecodeError,PermissionError):phase='phase_update'
                sample={'elapsed_seconds':elapsed,'phase':phase,**stats,**memory,
                    'cpu_core_equivalent':cores,'worker_cpu_percent_machine':100*cores/(os.cpu_count() or 1),'host_cpu_percent':host,
                    'disk_free_bytes':shutil.disk_usage(run_dir).free,'trial_directory_bytes':folder_size}
                samples.append(sample);last_stats=stats
                if writer is None:writer=csv.DictWriter(csv_file,fieldnames=list(sample));writer.writeheader()
                writer.writerow(sample);csv_file.flush()
                if phase!=last_phase or int(elapsed)//30 != int((previous_time or started)-started)//30:
                    print(f"{phase}: {elapsed:.0f}s, RSS {stats['rss_bytes']/GIB:.2f} GiB, system free {memory['available_physical_bytes']/GIB:.2f} GiB",flush=True)
                    last_phase=phase
                previous_cpu,previous_time,previous_system=cpu,now,system
                reason=None if probe else (reason_function or resource_reason)(sample,config)
                if reason:
                    job.terminate();break
                if probe and elapsed>30:
                    reason='guard_probe_timeout';job.terminate();break
                time.sleep(config['sample_seconds'])
            child.wait(timeout=30)
        peak_job=job.info().peak_job_memory
        return {'exit_code':child.returncode,'abort_reason':reason,'elapsed_seconds':time.perf_counter()-started,
            'guard_verified':True,'job_commit_limit_bytes':job.info().job_memory_limit,'job_peak_commit_bytes':peak_job,
            'affinity_mask':mask,'logical_cpu_limit':mask.bit_count(),'sample_count':len(samples),
            'sampled_peak_rss_bytes':max((x['rss_bytes'] for x in samples),default=0),
            'max_sum_individual_os_peak_rss_bytes':max((x['sum_individual_os_peak_rss_bytes'] for x in samples),default=0),
            'sampled_peak_private_bytes':max((x['private_bytes'] for x in samples),default=0),
            'min_available_physical_bytes':min((x['available_physical_bytes'] for x in samples),default=0),
            'peak_cpu_core_equivalent':max((x['cpu_core_equivalent'] for x in samples),default=0),
            'sampled_peak_directory_logical_bytes':max((x['trial_directory_bytes'] for x in samples),default=0),
            'last_process_counters':last_stats}
    finally:
        if child is not None and child.poll() is None:
            try:job.terminate();child.wait(timeout=30)
            except Exception:child.kill();child.wait(timeout=30)
        job.close()


def seal(run_dir):
    files={p.relative_to(run_dir).as_posix():{'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(run_dir.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
    report=read_json(run_dir/'resource_report.json')
    write_json(run_dir/'MANIFEST.json',{'purpose':'resource_trial_only','status':report['status'],'files':files})
    for name,item in files.items():checked(file_sha(run_dir/name)==item['sha256'],'Trial artifact hash mismatch')


def run_trial(output_root=None):
    config=trial_config();validate_trial_config(config)
    api=WindowsAPI();root=Path(output_root or ROOT/'results/m1_resource_trials')
    run_dir=root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex)
    run_dir.mkdir(parents=True,exist_ok=False)
    write_json(run_dir/'config.json',config)
    initial={**api.memory(),'disk_free_bytes':shutil.disk_usage(run_dir).free,'created_at':datetime.now(timezone.utc).isoformat()}
    reason=preflight(config,initial,initial['disk_free_bytes'])
    print('Resource trial archive: '+str(run_dir),flush=True)
    if reason:
        report={'status':'blocked_before_data_access','reason':reason,'initial':initial,'test_accessed':False,'training_started':False}
    else:
        try:
            observed=monitored_child(run_dir,config)
            result=read_json(run_dir/'worker_result.json') if (run_dir/'worker_result.json').exists() else None
            completed=observed['exit_code']==0 and result is not None and result['status']=='worker_complete'
            report={'status':'resource_trial_complete' if completed else 'resource_trial_aborted',
                'initial':initial,'monitor':observed,'worker':result,'test_accessed':False,
                'formal_execution_approval_unchanged':True,'performance_metrics_computed':False,
                'final_directory_logical_bytes_before_report':directory_bytes(run_dir),
                'final_disk_free_bytes':shutil.disk_usage(run_dir).free}
        except BaseException as error:
            report={'status':'resource_trial_aborted','reason':'guard_or_monitor_failure','error_type':type(error).__name__,
                'message':str(error),'initial':initial,'test_accessed':False,'formal_execution_approval_unchanged':True}
    write_json(run_dir/'resource_report.json',report)
    seal(run_dir)
    print(json.dumps({'status':report['status'],'report':str(run_dir/'resource_report.json')},ensure_ascii=False),flush=True)
    return 0 if report['status']=='resource_trial_complete' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-approved',action='store_true')
    group.add_argument('--worker',type=Path)
    group.add_argument('--guard-probe',type=Path)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args(argv)
    if args.worker:return worker(args.worker)
    if args.guard_probe:return guard_probe(args.guard_probe)
    return run_trial(args.output_root)


if __name__=='__main__':raise SystemExit(main())
