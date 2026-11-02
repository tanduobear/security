"""Repair-only, Windows-safe sealing of an already selected GBDT archive.

No ML imports, prediction loading, AP calculation, model loading, or test I/O.
Original binding/source snapshots stay unchanged; this repair has separate provenance.
"""
import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .search_gbdt import ROOT, digest, read, require, run_lock, sha, tasks, validate

SELECTION_SHA = '859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9'
CONFIG_SHA = '931d2d6e6639a825b62dd7e21f23f4a63e0e30a6a631b58ffd3469afca81b08b'
STATUS = 'gbdt_search_and_validation_selection_complete'
EXCLUDED = {'RUN.lock':'Runtime OS byte-lock coordination file, held exclusively during sealing; no experimental content.'}


def artifact_paths(run):
    run = Path(run).resolve()
    result = {}
    for path in sorted(run.rglob('*')):
        require(not path.is_symlink() and not path.is_junction(), 'Symlink/junction in archive: '+str(path))
        require(path.resolve().is_relative_to(run), 'Archive path escaped root')
        if path.is_file():
            name = path.relative_to(run).as_posix()
            if name not in EXCLUDED and name != 'MANIFEST.json':
                result[name] = path
    return result


def file_inventory(run):
    return {name:{'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in artifact_paths(run).items()}


@contextmanager
def freeze_files(paths):
    """FILE_SHARE_READ rejects existing writers and prevents write/delete opens.

Locks are on file objects, including hard-link aliases, and are released by OS
on process exit. No ACLs, attributes or existing content are changed.
"""
    require(os.name=='nt', 'Windows file-sharing protection is required')
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                                   ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles=[]
    try:
        for path in paths:
            handle = kernel.CreateFileW(str(Path(path).resolve()),0x80000000,1,None,3,0x80,None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):kernel.CloseHandle(handle)


def active_jobs():
    """Exclude this tool and its launch ancestry; reject any other formal job.

Only PID/parent/name/relevance are returned; no command lines are logged.
"""
    command = r"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
Get-CimInstance Win32_Process | ForEach-Object {
 $relevant = $_.ProcessId -ne $PID -and $_.Name -match '^(python|pythonw|powershell|pwsh)(\.exe)?$' -and $_.CommandLine -match 'methods[.]m1[.](search_gbdt|seal_gbdt|executor)|methods[\\/]m1[\\/](search_gbdt|seal_gbdt|executor)[.]py'
 [pscustomobject]@{pid=[int]$_.ProcessId;parent=[int]$_.ParentProcessId;name=$_.Name;relevant=[bool]$relevant}
} | ConvertTo-Json -Compress"""
    process = subprocess.run(['powershell.exe','-NoProfile','-Command',command],check=True,
                             capture_output=True,encoding='utf-8',errors='strict')
    rows=json.loads(process.stdout)
    if isinstance(rows,dict):rows=[rows]
    by_pid={r['pid']:r for r in rows}
    ancestry=set()
    pid=os.getpid()
    while pid and pid not in ancestry:
        ancestry.add(pid)
        pid=by_pid.get(pid,{}).get('parent',0)
    return [r for r in rows if r['relevant'] and r['pid'] not in ancestry]


def assert_quiet(probe):
    jobs=probe()
    require(not jobs, 'Active search/selection/archive process: '+json.dumps(jobs))


def bound_path(run, value):
    path=Path(value)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe recorded path')
    result=run/path
    require(result.resolve().is_relative_to(run.resolve()), 'Recorded path escaped archive')
    return result


def subset_inventory(run, directory, expected, actual):
    prefix=directory.relative_to(run).as_posix()+'/'
    found={name[len(prefix):]:value for name,value in actual.items() if name.startswith(prefix)}
    require(found==expected, 'Bound artifact hash/set mismatch: '+prefix)


def validate_archive(run, actual, project, expected_selection_sha):
    """Verify recorded identities and hashes only; never recompute AP or winner."""
    require(actual['selection.json']['sha256']==expected_selection_sha, 'selection.json hash mismatch')
    config=validate(read(run/'config.json'))
    require(read(project/'methods/m1/execution.gbdt-search.v1.json')==config, 'Search configuration mismatch')
    require(sha(project/'methods/m1/execution.gbdt-search.v1.json')==CONFIG_SHA, 'Approved configuration file changed')
    identity=read(run/'binding.json')
    require(identity['config_sha256']==digest(config), 'Configuration binding mismatch')
    require(identity['unified_file_sha256']==sha(project/'methods/m1/protocol.unified.v1.json'), 'Unified protocol identity changed')
    selection=read(run/'selection.json')
    require(selection['binding_sha256']==digest(identity), 'Selection identity mismatch')
    require(selection['status']=='gbdt_validation_selection_only' and selection['test_accessed'] is False
            and selection['rf_fits']==0 and selection['thresholds_calibrated'] is False, 'Unexpected selection scope')
    require(selection['selected_candidate_id']=='G05', 'Pinned selection winner changed')
    allowed={t['task_id']:t for t in tasks(config)}
    recorded={p.stem:p for p in (run/'completed').glob('*.json')}
    require(set(recorded)==set(allowed), 'Missing/extra completed tasks')
    row_orders=set()
    attempt_directories=set()
    for task_id,path in recorded.items():
        job=read(path)
        require(job['task']==allowed[task_id] and job['binding_sha256']==digest(identity), 'Task identity mismatch')
        require(job['validation_rows']==367036 and job['test_accessed'] is False, 'Validation scope mismatch')
        require(0<=job['validation_ap']<=1 and job['training_seconds']>=0, 'Invalid recorded result')
        directory=bound_path(run,job['artifact_directory'])
        require(directory.parts[len(run.parts)]=='attempts', 'Unexpected task directory')
        subset_inventory(run,directory,job['files'],actual)
        require(set(job['files'])=={'model.joblib','validation.npz','task.json'}, 'Unexpected task artifact types')
        require(read(directory/'task.json')==allowed[task_id], 'Saved task configuration mismatch')
        row_orders.add(job['validation_row_ids_sha256'])
        attempt_directories.add(str(directory))
    require(len(attempt_directories)==40 and len(row_orders)==1, 'Duplicate attempts or validation orders')
    require({p.name for p in (run/'attempts').iterdir()}=={Path(p).name for p in attempt_directories}, 'Unaccounted attempt directory')
    cache=read(run/'cache.json')
    subset_inventory(run,bound_path(run,cache['directory']),cache['files'],actual)
    for name,expected in identity['code'].items():
        require(actual['sources/'+name]['sha256']==expected, 'Archived source hash mismatch: '+name)
        require(sha(project/name)==expected, 'Original source changed: '+name)
    require(set(identity['inputs'])=={'train.parquet','val.parquet','preprocessor.json','preprocess_manifest.json'}, 'Input allowlist changed')
    for name,expected in identity['inputs'].items():
        require(sha(project/'data/split/original/stratified'/name)==expected, 'Train/val input identity mismatch: '+name)
    for candidate in config['candidates']:
        group=candidate['id']
        require(read(run/'checkpoints'/(group+'.json'))=={'group':group,'binding_sha256':digest(identity),'action':'CONTINUE'}, 'Unconfirmed checkpoint')
    monitors=list((run/'monitor').iterdir())
    expected_groups={c['id'] for c in config['candidates']}|{'SELECT'}
    require(len(monitors)==9 and {p.name.split('-')[0] for p in monitors}==expected_groups, 'Unexpected monitor directories')
    for monitor in monitors:
        group=monitor.name.split('-')[0]
        outcome=read(monitor/'outcome.json')
        guard=read(monitor/'guard_verified.json')
        worker=read(monitor/'worker_result.json')
        require(outcome['exit_code']==0 and outcome['abort_reason'] is None, 'Worker/monitor incomplete')
        require(guard['job_commit_limit_bytes']==8*1024**3 and guard['logical_cpu_limit']==4
                and guard['verified_before_GO'], 'Resource guard identity mismatch')
        require(worker['status']==('selection_complete' if group=='SELECT' else 'group_complete')
                and worker['test_accessed'] is False, 'Unexpected worker status')
    return {'tasks_verified':40,'task_files_verified':120,'selection_hash_verified':True,
            'source_files_verified':len(identity['code']),'cache_files_verified':len(cache['files']),
            'AP_recomputed':False,'winner_reselected':False,'models_loaded':False,'test_accessed':False}


def verify_manifest(run, manifest):
    require(manifest['status']==STATUS and manifest['excluded_runtime_files']==EXCLUDED, 'Invalid manifest policy/status')
    actual=file_inventory(run)
    require(actual==manifest['files'], 'Manifest file set/hash mismatch')
    return len(actual)


def seal(run, baseline_path, expected_selection_sha=SELECTION_SHA, *, baseline_sha, project=ROOT, _process_probe=active_jobs,
         _validator=validate_archive, _before_publish=None):
    run,project=Path(run).resolve(),Path(project).resolve()
    require(run.is_dir() and not run.is_symlink(), 'Invalid archive directory')
    require(not (run/'MANIFEST.json').exists(), 'MANIFEST already exists; never overwrite')
    require((run/'RUN.lock').is_file(), 'Original RUN.lock missing; do not recreate')
    require(sha(run/'selection.json')==expected_selection_sha, 'Pinned selection hash mismatch')
    require(sha(baseline_path)==baseline_sha, 'Preflight baseline hash mismatch')
    baseline=read(baseline_path)
    require(baseline['selection_sha256']==expected_selection_sha, 'Baseline selection mismatch')
    assert_quiet(_process_probe)
    slot=project/'.tmp/m1_gbdt_resource_slot'
    slot.mkdir(parents=True,exist_ok=True)
    with run_lock(slot),run_lock(run):
        assert_quiet(_process_probe)
        paths=artifact_paths(run)
        original_identity=read(run/'binding.json')
        require(set(original_identity['inputs'])=={'train.parquet','val.parquet','preprocessor.json','preprocess_manifest.json'}, 'Input allowlist changed')
        external=[Path(baseline_path),Path(__file__),project/'methods/m1/protocol.unified.v1.json',
                  project/'methods/m1/execution.gbdt-search.v1.json']
        external += [project/name for name in original_identity['code']]
        external += [project/'data/split/original/stratified'/name for name in original_identity['inputs']]
        with freeze_files([*paths.values(),*external]):
            require(sha(baseline_path)==baseline_sha, 'Preflight baseline changed')
            require(set(artifact_paths(run))==set(paths), 'Concurrent archive namespace change')
            before=file_inventory(run)
            require(before==baseline['files'], 'Original artifact changed since preflight snapshot')
            evidence=_validator(run,before,project,expected_selection_sha)
            manifest={'status':STATUS,'schema_version':'repair-seal-1','created_at':datetime.now(timezone.utc).isoformat(),
                      'files':before,'excluded_runtime_files':EXCLUDED,
                      'self_hash_policy':'MANIFEST.json is excluded from its own files; its SHA256 is recorded externally.',
                      'selection_sha256':expected_selection_sha,'validation':evidence,
                      'repair_provenance':{'tool':'methods/m1/seal_gbdt.py','tool_sha256':sha(Path(__file__)),
                          'preflight_snapshot_sha256':sha(baseline_path),
                          'original_code_identity_preserved':True,
                          'purpose':'Seal existing selected archive only; repair RUN.lock hashing failure.'},
                      'concurrency_protection':'Global and run byte locks plus FILE_SHARE_READ handles on every original artifact; namespace and hashes rechecked.',
                      'test_accessed':False,'rf_fits':0,'retraining':False,'AP_recomputed':False,'thresholds_calibrated':False}
            if _before_publish:_before_publish()
            assert_quiet(_process_probe)
            require(set(artifact_paths(run))==set(paths), 'Concurrent archive namespace change')
            require(file_inventory(run)==before, 'Concurrent artifact modification')
            # Stage outside the experiment. Exclusive atomic hard-link publication
            # fails if MANIFEST appeared concurrently; never replace an existing file.
            staging=project/'.tmp'
            staging.mkdir(parents=True,exist_ok=True)
            descriptor,temporary=tempfile.mkstemp(prefix='gbdt-manifest-',suffix='.json',dir=staging)
            temporary=Path(temporary)
            try:
                with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                    json.dump(manifest,stream,indent=2,allow_nan=False)
                    stream.write('\n');stream.flush();os.fsync(stream.fileno())
                os.link(temporary,run/'MANIFEST.json')
            finally:
                temporary.unlink(missing_ok=True)
            with freeze_files([run/'MANIFEST.json']):
                require(read(run/'MANIFEST.json')==manifest, 'Published manifest mismatch')
                count=verify_manifest(run,manifest)  # independent complete second SHA256 pass
                require(sha(run/'selection.json')==expected_selection_sha, 'Selection changed')
                return {'status':STATUS,'manifest_file_count':len(before),'sha256_verified_count':count,
                        'manifest_sha256':sha(run/'MANIFEST.json'),'selection_sha256':sha(run/'selection.json'),
                        'excluded_runtime_files':EXCLUDED,'validation':evidence}


def verify(run, expected_selection_sha=SELECTION_SHA, *, expected_manifest_sha, _process_probe=active_jobs):
    run=Path(run).resolve()
    assert_quiet(_process_probe)
    with run_lock(run):
        require(sha(run/'MANIFEST.json')==expected_manifest_sha, 'Manifest pinned SHA256 mismatch')
        paths=artifact_paths(run)
        with freeze_files([*paths.values(),run/'MANIFEST.json']):
            require(sha(run/'MANIFEST.json')==expected_manifest_sha, 'Manifest changed before verification')
            manifest=read(run/'MANIFEST.json')
            count=verify_manifest(run,manifest)
            require(sha(run/'selection.json')==expected_selection_sha,'Selection hash mismatch')
            return {'manifest_file_count':len(manifest['files']),'sha256_verified_count':count,
                    'selection_sha256':sha(run/'selection.json'),'manifest_sha256':sha(run/'MANIFEST.json')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['seal','verify'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--baseline-sha256')
    parser.add_argument('--manifest-sha256')
    args=parser.parse_args(argv)
    if args.stage=='seal':
        require(args.baseline is not None and args.baseline_sha256,'Explicit preflight snapshot and pinned hash required')
        result=seal(args.run,args.baseline,baseline_sha=args.baseline_sha256)
    else:
        require(args.manifest_sha256,'Pinned manifest SHA256 required')
        result=verify(args.run,expected_manifest_sha=args.manifest_sha256)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
