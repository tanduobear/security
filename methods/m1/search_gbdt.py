"""Stage-scoped GBDT search. Parent is stdlib-only until the guarded GO signal.

No RF, test loader, thresholds or bootstrap. Each group is a guarded worker;
the foreground parent requires an explicit CONTINUE after all five seeds.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'methods/m1/execution.gbdt-search.v1.json'
GIB = 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    """Publish once, only after complete bytes are flushed; never replace a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Exclusive destination creation; an interrupted write is rejected on resume.
    with path.open('xb') as target:
        target.write(temporary.read_bytes())
        target.flush()
        os.fsync(target.fileno())
    temporary.unlink()


def validate(config):
    require(config.get('stage')=='gbdt_search_only', 'Wrong execution stage')
    approval = config['approval']
    require(approval['status']=='approved' and approval['scope']=='gbdt_search_only', 'Search-only approval required')
    require(bool(approval['source']) and bool(approval['approved_at']), 'Approval provenance missing')
    require(approval['sha256']==digest({k:v for k,v in config.items() if k!='approval'}), 'Approved configuration hash mismatch')
    unified = read(ROOT/'methods/m1/protocol.unified.v1.json')
    require(unified['approval']['status']=='approved' and unified['sha256']==config['unified_protocol_sha256'], 'Unified protocol mismatch')
    require(digest({k:v for k,v in unified.items() if k!='sha256'})==unified['sha256'], 'Unified protocol was changed')
    require(config['dataset']=={'version':'original','protocol':'stratified','variant':'main','features':140,'allowed_splits':['train','val']}, 'Dataset scope changed')
    require(config['seeds']==[42,43,44,45,46], 'Seed scope changed')
    rows = [(200,.05,15,20,1),(400,.05,15,20,1),(200,.1,15,20,1),(200,.05,31,20,1),
            (400,.05,31,20,1),(200,.05,15,5,1),(200,.05,15,50,1),(200,.05,15,20,10)]
    expected = [{'id':f'G{i:02}', 'params':dict(zip(('max_iter','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization'), row),
                 max_bins=255, early_stopping=False, class_weight=None)} for i,row in enumerate(rows,1)]
    require(config['candidates']==expected, 'Approved eight candidates changed')
    require(config['selection']=={'metric':'mean_full_validation_AP','ap_tolerance':1e-6,'tie_rule':'lowest_candidate_id','std_ddof':1}, 'Selection changed')
    require(config['resources']=={'outer_parallel_jobs':1,'fit_threads':4,'inference_threads':1,'prediction_batch_rows':8192,
        'job_commit_limit_bytes':8*GIB,'logical_cpu_limit':4,'min_start_free_bytes':14*GIB,
        'min_running_free_bytes':6*GIB,'rss_abort_bytes':8*GIB}, 'Resource approval changed')
    require(config['checkpoint']=='manual_after_each_five_seed_group' and config['automatic_retries']==0, 'Checkpoint/retry policy changed')
    return config


def tasks(config):
    validate(config)
    return [{'task_id':f"{c['id']}_seed_{s}", 'candidate_id':c['id'], 'seed':s,
             'algorithm':'hist_gradient_boosting','params':c['params']} for c in config['candidates'] for s in config['seeds']]


def fingerprint(project=ROOT):
    """Includes uncommitted code bytes and environment, not just HEAD."""
    files = [p for area in ('methods/m1','src','eval','tests') for p in (project/area).rglob('*')
             if p.is_file() and p.suffix in ('.py','.json','.txt') and '__pycache__' not in p.parts]
    files += [project/'docs/label_split_policy.json',project/'methods/m1/requirements-lock.txt']
    return {p.relative_to(project).as_posix():sha(p) for p in sorted(files)}


def inputs(project=ROOT):
    directory = project/'data/split/original/stratified'
    # Explicit allowlist: never glob datasets or hash/open test.
    return {name:sha(directory/name) for name in ('train.parquet','val.parquet','preprocessor.json','preprocess_manifest.json')}


def binding(config, project=ROOT):
    from methods.m1.storage import environment
    env = environment()
    return {'config_sha256':digest(config),'code':fingerprint(project),'inputs':inputs(project),
            'environment':env,'unified_file_sha256':sha(ROOT/'methods/m1/protocol.unified.v1.json')}


def inventory(directory):
    return {p.relative_to(directory).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def verify_inventory(directory, expected):
    directory = Path(directory)
    for name in expected:
        path = directory/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                path.resolve().is_relative_to(directory.resolve()) and not path.is_symlink(), 'Unsafe artifact')
    require(inventory(directory)==expected, 'Artifact hash/file set mismatch')


def verify_tasks(run, config, identity):
    import numpy as np
    from eval.metrics import compute_curves
    from methods.m1.data import row_digest
    allowed = {t['task_id']:t for t in tasks(config)}
    completed = {}
    for path in sorted((run/'completed').glob('*.json')):
        record = read(path)
        task_id = path.stem
        require(task_id in allowed and record['task']==allowed[task_id], 'Unexpected task/config')
        require(record['binding_sha256']==digest(identity), 'Task identity mismatch')
        relative = Path(record['artifact_directory'])
        require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[:1]==('attempts',), 'Unsafe task path')
        require((run/relative).resolve().is_relative_to(run.resolve()), 'Task directory escaped run')
        verify_inventory(run/relative, record['files'])
        require(read(run/relative/'task.json')==allowed[task_id], 'Saved task parameters mismatch')
        require(0<=record['validation_ap']<=1 and record['training_seconds']>=0, 'Invalid task result')
        with np.load(run/relative/'validation.npz',allow_pickle=False) as saved:
            require(len(saved['y_true'])==record['validation_rows'] and
                    row_digest(saved['row_ids'])==record['validation_row_ids_sha256'], 'Saved validation identity mismatch')
            require(compute_curves(saved['y_true'],saved['y_score'])['average_precision']==record['validation_ap'], 'Saved AP mismatch')
        completed[task_id] = record
    return completed


@contextmanager
def run_lock(run):
    """OS-released lock survives process crashes without stale lock bypasses."""
    import msvcrt
    with (run/'RUN.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def resource_slot():
    directory = ROOT/'.tmp/m1_gbdt_resource_slot'
    directory.mkdir(parents=True,exist_ok=True)
    with run_lock(directory):
        yield


def prepare_cache(run, config, project=ROOT, loader_factory=None):
    from methods.m1.data import materialize_preparation, assert_disk_isolated, DiskDataset
    if (run/'cache.json').exists():
        record = read(run/'cache.json')
        relative = Path(record['directory'])
        require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[:1]==('cache',), 'Unsafe cache path')
        directory = run/record['directory']
        require(directory.resolve().is_relative_to(run.resolve()), 'Cache directory escaped run')
        verify_inventory(directory, record['files'])
        datasets = []
        for split in ('train','val'):
            ready = read(directory/split/'READY.json')
            datasets.append(DiskDataset(directory/split, tuple(ready['features']), ready['rows'], split,
                                        {int(k):v for k,v in ready['class_weights'].items()}))
    else:
        directory = run/'cache'/uuid.uuid4().hex
        source = project/'data/split/original/stratified'
        datasets = [materialize_preparation(source/(s+'.parquet'),directory/s,loader_factory=loader_factory) for s in ('train','val')]
        require(datasets[0].feature_names==datasets[1].feature_names and len(datasets[0].feature_names)==140, 'P2 features mismatch')
        assert_disk_isolated(*datasets,memory_limit_mb=512)
        write(run/'cache.json', {'directory':directory.relative_to(run).as_posix(), 'files':inventory(directory)})
    require([d.split for d in datasets]==['train','val'], 'Wrong dataset roles')
    require(datasets[0].feature_names==datasets[1].feature_names and len(datasets[0].feature_names)==140, 'Cached feature contract changed')
    return datasets


def execute_group(run, group, config, identity, training, validation, *, fitter=None, phase=None):
    """All real invocations happen inside a protected worker. Tests inject tiny fits."""
    from contextlib import ExitStack
    import joblib
    import numpy as np
    from eval.metrics import compute_curves
    from methods.m1.models import fit_tree
    from methods.m1.executor import predict_batches, metadata_arrays
    from methods.m1.data import row_digest
    complete = verify_tasks(run, config, identity)
    ids, _, _ = metadata_arrays(validation)
    order_hash = row_digest(ids)
    with ExitStack() as stack:
        train = stack.enter_context(training.model_input())
        val = stack.enter_context(validation.model_input())
        for task in [t for t in tasks(config) if t['candidate_id']==group]:
            name = task['task_id']
            if name in complete:
                require(complete[name]['validation_row_ids_sha256']==order_hash, 'Validation order mismatch')
                if phase: phase('verified_skip_'+name)
                continue
            ordinal = next(i for i,t in enumerate(tasks(config),1) if t['task_id']==name)
            if phase: phase(f'fit_{ordinal:02d}_of_40_{name}')
            directory = run/'attempts'/(name+'-'+uuid.uuid4().hex)
            directory.mkdir(parents=True,exist_ok=False)
            model, seconds = (fitter or fit_tree)(task['algorithm'],task['params'],train,model_seed=task['seed'],fit_threads=4)
            scores = predict_batches(model,val.X,rows=8192,threads=1)
            ap = compute_curves(val.y,scores)['average_precision']
            require(ap is not None and np.isfinite(ap) and 0<=ap<=1, 'Invalid validation AP')
            joblib.dump(model,directory/'model.joblib')
            restored = joblib.load(directory/'model.joblib')
            require(np.array_equal(scores,predict_batches(restored,val.X,rows=8192,threads=1)), 'Reload mismatch')
            del restored, model
            with (directory/'validation.npz').open('xb') as stream:
                np.savez_compressed(stream,row_ids=ids,y_true=val.y,y_score=scores)
            with np.load(directory/'validation.npz',allow_pickle=False) as saved:
                require(np.array_equal(saved['row_ids'],ids) and np.array_equal(saved['y_true'],val.y) and
                        np.array_equal(saved['y_score'],scores), 'Saved predictions mismatch')
            write(directory/'task.json',task)
            record = {'task':task,'binding_sha256':digest(identity),'artifact_directory':directory.relative_to(run).as_posix(),
                      'files':inventory(directory),'validation_ap':float(ap),'training_seconds':seconds,
                      'validation_row_ids_sha256':order_hash,'validation_rows':len(val.y),'test_accessed':False}
            write(run/'completed'/(name+'.json'),record)
            if phase: phase(f'completed_{name}_AP_{ap:.9f}_fit_seconds_{seconds:.2f}')
            del scores


def select(run, config, identity):
    import numpy as np
    completed = verify_tasks(run,config,identity)
    require(len(completed)==40, 'All forty tasks are required')
    summaries = []
    for candidate in config['candidates']:
        ap = [completed[f"{candidate['id']}_seed_{s}"]['validation_ap'] for s in config['seeds']]
        summaries.append({'candidate_id':candidate['id'],'mean_validation_ap':float(np.mean(ap)),
                          'sample_std_ap':float(np.std(ap,ddof=1))})
    best = max(x['mean_validation_ap'] for x in summaries)
    winner = min(x['candidate_id'] for x in summaries if best-x['mean_validation_ap']<=1e-6)
    return {'status':'gbdt_validation_selection_only','selected_candidate_id':winner,'candidates':summaries,
            'binding_sha256':digest(identity),'test_accessed':False,'rf_fits':0,'thresholds_calibrated':False}


def abort_reason(sample, config):
    if sample['available_physical_bytes']<config['min_running_free_bytes']: return 'system_memory_below_6_GiB'
    if sample['rss_bytes']>config['rss_abort_bytes']: return 'tree_RSS_above_8_GiB'
    return None


def checkpoint(run, group, identity, ask=input):
    path = run/'checkpoints'/(group+'.json')
    acknowledgment = {'group':group,'binding_sha256':digest(identity),'action':'CONTINUE'}
    if path.exists():
        require(read(path)==acknowledgment, 'Checkpoint identity mismatch')
        return True
    print(f'CHECKPOINT {group}: five seeds complete. Review task files, AP and monitoring CSV before continuing.',flush=True)
    if ask('Type CONTINUE to acknowledge this group; anything else stops safely: ').strip()!='CONTINUE':
        return False
    write(path,acknowledgment)
    return True


def worker(run, monitor, group):
    # A child cannot import ML or touch datasets before protected parent GO.
    require(sys.stdin.readline().strip()=='GO','Protected GO missing')
    run, monitor = Path(run), Path(monitor)
    config = validate(read(run/'config.json'))
    identity = binding(config)
    if not (run/'binding.json').exists():
        require(read(monitor/'launch.json')['allow_initialize'] and group=='G01', 'Identity initialization forbidden on resume')
        write(run/'binding.json',identity)
        for name in identity['code']:
            destination = run/'sources'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream: stream.write((ROOT/name).read_bytes())
    require(identity==read(run/'binding.json'), 'Resume code/data/environment identity mismatch')
    def phase(name):
        # Monitor tolerates this small state update window.
        (monitor/'state.json').write_text(json.dumps({'phase':name}),encoding='utf-8')
        print(datetime.now(timezone.utc).isoformat(),name,flush=True)
        with (monitor/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'phase':name})+'\n')
    phase('verify_completed_tasks')
    verify_tasks(run,config,identity)
    if group=='SELECT':
        require(all(read(run/'checkpoints'/(c['id']+'.json'))==
                    {'group':c['id'],'binding_sha256':digest(identity),'action':'CONTINUE'} for c in config['candidates']), 'Manual review missing/mismatched')
        write(run/'selection.json',select(run,config,identity))
        write(monitor/'worker_result.json',{'status':'selection_complete','test_accessed':False})
        return
    phase('prepare_or_verify_train_val_cache')
    training, validation = prepare_cache(run,config)
    for split, data in (('train',training),('val',validation)):
        ready = read(data.directory/'READY.json')
        require(ready['source_manifest_output']['main']['sha256']==identity['inputs'][split+'.parquet'], 'Cached data source changed')
    execute_group(run,group,config,identity,training,validation,phase=phase)
    write(monitor/'worker_result.json',{'status':'group_complete','group':group,'test_accessed':False})


def run_search(config, resume=None, ask=input):
    from methods.m1.resource_trial import WindowsAPI, monitored_child
    validate(config)
    resources = config['resources']
    require(WindowsAPI().memory()['available_physical_bytes']>=14*GIB,'Start requires 14 GiB free')
    if resume:
        run = Path(resume).resolve()
        require(read(run/'config.json')==config,'Resume configuration mismatch')
        require(not (run/'MANIFEST.json').exists() and not (run/'selection.json').exists(), 'Completed selection is immutable; do not resume')
    else:
        run = ROOT/'results/m1_gbdt_search'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex)
        run.mkdir(parents=True,exist_ok=False)
        write(run/'config.json',config)
    with resource_slot(), run_lock(run):
        # Identity validation reads only train/val hashes; performed in protected worker.
        if not (run/'binding.json').exists():
            require(not resume,'Interrupted initialization: do not reuse incomplete identity')
        for candidate in config['candidates']:
            group = candidate['id']
            monitor = run/'monitor'/(group+'-'+uuid.uuid4().hex)
            monitor.mkdir(parents=True,exist_ok=False)
            write(monitor/'launch.json',{'allow_initialize':not resume and group=='G01'})
            require(WindowsAPI().memory()['available_physical_bytes']>=14*GIB,'Group start requires 14 GiB free')
            # binding creation itself must be guarded; worker initializes once.
            outcome = monitored_child(monitor,{**resources,'sample_seconds':1,'disk_sample_seconds':5},
                worker_module='methods.m1.search_gbdt', worker_args=('--worker',run,monitor,group), reason_function=abort_reason)
            write(monitor/'outcome.json',outcome)
            require(outcome['exit_code']==0 and outcome['abort_reason'] is None and (monitor/'worker_result.json').exists(),
                    'Worker interrupted/aborted. No automatic retry; inspect archive before explicit resume.')
            identity = read(run/'binding.json')
            for seed in config['seeds']:
                record = read(run/'completed'/f'{group}_seed_{seed}.json')
                print(f"{group} seed={seed}: AP={record['validation_ap']:.9f}; fit={record['training_seconds']:.2f}s",flush=True)
            print(f"Job peak commit={outcome['job_peak_commit_bytes']/GIB:.2f} GiB; sampled tree RSS peak={outcome['sampled_peak_rss_bytes']/GIB:.2f} GiB; CSV={monitor/'resource_samples.csv'}",flush=True)
            require(checkpoint(run,group,identity,ask),'Paused at manual checkpoint; use explicit resume')
        # Selection in guarded worker is not performed here: parent remains stdlib-only.
        print(f'40 task checks complete. Search archive: {run}. Run select separately.',flush=True)
    return run


def run_selection(run, config):
    from methods.m1.resource_trial import WindowsAPI, monitored_child
    run = Path(run).resolve()
    require(read(run/'config.json')==config,'Selection configuration mismatch')
    with resource_slot(), run_lock(run):
        require(not (run/'MANIFEST.json').exists() and not (run/'selection.json').exists(), 'Selection already exists')
        require(WindowsAPI().memory()['available_physical_bytes']>=14*GIB,'Start requires 14 GiB free')
        monitor = run/'monitor'/('SELECT-'+uuid.uuid4().hex)
        monitor.mkdir(parents=True,exist_ok=False)
        outcome = monitored_child(monitor,{**config['resources'],'sample_seconds':1,'disk_sample_seconds':5},
            worker_module='methods.m1.search_gbdt',worker_args=('--worker',run,monitor,'SELECT'),reason_function=abort_reason)
        write(monitor/'outcome.json',outcome)
        require(outcome['exit_code']==0 and outcome['abort_reason'] is None and (monitor/'worker_result.json').exists(), 'Selection worker failed; inspect archive')
        write(run/'MANIFEST.json',{'status':'gbdt_search_and_validation_selection_complete','files':inventory(run),
                                  'test_accessed':False,'rf_fits':0})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',choices=['plan','run','select'],default='plan')
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--confirm-start',action='store_true',help='Explicit launch switch; configuration approval alone does not authorize a run')
    parser.add_argument('--worker',nargs=3,metavar=('RUN','MONITOR','GROUP'))
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker);return
    config = validate(read(args.config))
    if args.stage=='plan':
        print(json.dumps({'config_sha256':sha(args.config),'approved_payload_sha256':config['approval']['sha256'],
                          'tasks':tasks(config),'training_started':False},indent=2))
    elif args.stage=='run':
        require(args.confirm_start,'Explicit launch confirmation required; use --confirm-start only after authorization')
        run_search(config,args.resume)
    else:
        require(args.confirm_start,'Explicit selection launch confirmation required')
        require(args.resume is not None,'select requires --resume RUN')
        run_selection(args.resume,config)


if __name__=='__main__':
    main()
