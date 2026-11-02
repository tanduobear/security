"""Sequential formal search -> validation selection -> separately gated test evaluation.

No function is called automatically. All execution needs approved detailed
settings. Search/selection cannot construct a test loader. Evaluation also
needs a verified sealed selection and an explicit test-access switch.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT / '.tmp/matplotlib'))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from eval.metrics import compute_curves, measure_inference, paired_bootstrap
from eval.validate import validate_result
from src.model_loader import ModelDataset
from methods.m1.data import (_materialize_loaded, assert_disk_isolated,
                             materialize_preparation, preparation_path, require, row_digest)
from methods.m1.evaluation import score_metrics, timing_indices
from methods.m1.models import attack_scores, fit_tree, inference_workers
from methods.m1.protocol import digest, require_execution, execution_issues
from methods.m1.selection import calibrate_fpr, select_configuration
from methods.m1.storage import RunStore, environment, sha, verify_run


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_npy(store, name, array):
    with store.file(name).open('xb') as stream:np.save(stream,array,allow_pickle=False)


def snapshot_sources(store):
    files = list((ROOT / 'methods/m1').glob('*.py')) + [ROOT / 'eval/metrics.py',ROOT / 'eval/schema.json',ROOT / 'eval/validate.py',ROOT / 'src/model_loader.py',ROOT / 'src/preprocess.py',ROOT / 'docs/label_split_policy.json',ROOT / 'methods/m1/requirements-lock.txt']
    for path in files:
        with store.file('sources/' + path.relative_to(ROOT).as_posix()).open('xb') as target:
            target.write(path.read_bytes())


def event(stream, stage, **fields):
    stream.write(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'stage':stage,**fields},allow_nan=False)+'\n')
    stream.flush()


def metadata_arrays(cached):
    ids, families = np.empty(cached.rows,dtype=object), np.empty(cached.rows,dtype=object)
    flags = np.empty(cached.rows,dtype=bool)
    offset = 0
    for frame in cached.metadata_batches():
        n = len(frame)
        ids[offset:offset+n] = frame['row_id'].to_numpy()
        families[offset:offset+n] = frame['attack_family'].to_numpy()
        require(frame['is_attempted'].dtype.kind == 'b', 'Attempted flags must be boolean')
        flags[offset:offset+n] = frame['is_attempted'].to_numpy()
        offset += n
    require(offset == cached.rows and len(set(ids)) == len(ids), 'Cached metadata identity invalid')
    return ids.astype(str), families.astype(str), flags


def predict_batches(model, X, *, rows, threads):
    """Allocate only one score vector; no full-data 2-column probabilities."""
    output = np.empty(len(X),dtype=np.float64)
    with inference_workers(model,threads), threadpool_limits(limits=threads):
        for offset in range(0,len(X),rows):
            output[offset:offset+rows] = attack_scores(model,X[offset:offset+rows])
    return output


def identity(cache, directory, split, variant):
    ready = read_json(cache.directory/'READY.json')
    source = ready['source_manifest_output']['main' if variant == 'main' else 'no_destination_port']
    return {'path':str(directory/(split+'.parquet')),'sha256':source['sha256'],
            'rows':cache.rows,'effective_index':ready['source_manifest_output']['effective_index']}


def task_plan(protocol, details):
    jobs = []
    for candidate in details['gbdt_candidates']:
        for seed in protocol['model_seeds']:
            jobs.append({'role':'main','algorithm':'hist_gradient_boosting','candidate_id':candidate['id'],'seed':seed,'params':candidate['params']})
    rf = details['rf_auxiliary']
    for seed in protocol['model_seeds']:
        jobs.append({'role':'auxiliary','algorithm':'random_forest','candidate_id':rf['id'],'seed':seed,'params':rf['params']})
    return jobs


def execute_search(protocol, details, *, project_root=ROOT, output_root=None, loader_factory=None, fit_function=None):
    require_execution(protocol,details)  # before any output/data access
    project_root = Path(project_root)
    output_root = Path(output_root or project_root/'results/m1_runs')
    factory, fitter = loader_factory or ModelDataset, fit_function or fit_tree
    store = RunStore(output_root,kind='search')
    started = perf_counter()
    try:
        store.write_json('config.json',{'unified':protocol,'execution':details})
        snapshot_sources(store)
        spec = protocol['dataset']
        directory = preparation_path(project_root,spec['version'],spec['protocol'],spec['variant'],'train').parent
        cache_root = project_root/'.tmp/m1_formal_cache'/store.path.name
        training = materialize_preparation(directory/'train.parquet',cache_root/'train',loader_factory=factory)
        validation = materialize_preparation(directory/'val.parquet',cache_root/'val',loader_factory=factory)
        assert_disk_isolated(training,validation,memory_limit_mb=details['resources']['duckdb_memory_limit_mb'])
        require(training.feature_names == validation.feature_names and len(training.feature_names) == spec['features'], 'P2 feature contract mismatch')
        families = [x for x in read_json(ROOT/'docs/label_split_policy.json')['family_names'] if x != 'benign']
        ids, attack_family, flags = metadata_arrays(validation)
        chosen = timing_indices(ids,batch_size=protocol['inference']['batch_size'],seed=details['timing']['seed'])
        state_dir = directory.parent if spec['variant'] != 'main' else directory
        inputs = {'version':spec['version'],'protocol':spec['protocol'],'variant':spec['variant'],
                  'features':list(training.feature_names),'families':families,
                  'train':identity(training,directory,'train',spec['variant']),
                  'val':identity(validation,directory,'val',spec['variant']),
                  'preprocessor_sha256':sha(state_dir/'preprocessor.json'),
                  'validation_row_ids_sha256':row_digest(ids),
                  'training_class_weights':training.class_weights,
                  'test_accessed':False}
        store.write_json('inputs.json',inputs)
        store.write_json('timing_rows.json',{'population':inputs['val'],'row_ids':ids[chosen].tolist(),'row_ids_sha256':row_digest(ids[chosen]),'selection':details['timing']})
        jobs = []
        with training.model_input() as train, validation.model_input() as val:
            require(np.unique(val.y).size == 2, 'AP selection needs both validation classes')
            with store.file('validation.npz').open('xb') as stream:
                np.savez_compressed(stream,y_true=val.y,row_ids=ids,attack_family=attack_family,is_attempted=flags)
            save_npy(store,'timing_X.npy',val.X[chosen])
            with store.file('events.jsonl').open('x',encoding='utf-8') as log:
                for task in task_plan(protocol,details):
                    name = f"{task['candidate_id']}_seed_{task['seed']}"
                    event(log,'fit_started',task=name)
                    model, seconds = fitter(task['algorithm'],task['params'],train,model_seed=task['seed'],fit_threads=details['resources']['fit_threads'])
                    require(model.n_features_in_ == len(inputs['features']), 'Fitted feature contract mismatch')
                    scores = predict_batches(model,val.X,rows=details['resources']['prediction_batch_rows'],threads=details['resources']['inference_threads'])
                    ap = compute_curves(val.y,scores)['average_precision']
                    require(ap is not None, 'Validation AP undefined')
                    model_path, score_path = 'models/'+name+'.joblib', 'validation_scores/'+name+'.npy'
                    store.write_model(model_path,model)
                    save_npy(store,score_path,scores)
                    restored = joblib.load(store.path/model_path)
                    require(np.array_equal(scores,predict_batches(restored,val.X,rows=details['resources']['prediction_batch_rows'],threads=details['resources']['inference_threads'])), 'Reloaded model predictions differ')
                    job = {**task,'task_id':name,'evaluation_split':'val','validation_rows':len(val.y),
                           'validation_row_ids_sha256':inputs['validation_row_ids_sha256'],
                           'validation_ap':ap,'training_seconds':seconds,'effective_parameters':model.get_params(),
                           'model_path':model_path,'model_sha256':sha(store.path/model_path),
                           'scores_path':score_path,'scores_sha256':sha(store.path/score_path)}
                    jobs.append(job)
                    store.write_json('jobs/'+name+'.json',job)
                    event(log,'fit_completed',task=name,validation_ap=ap,training_seconds=seconds)
                    del model, restored, scores
        store.write_json('jobs.json',jobs)
        store.write_json('search_summary.json',{'main_fits':40,'auxiliary_fits':5,'outer_parallel_jobs':1,
                                              'validation_rows':validation.rows,'test_accessed':False,
                                              'pipeline_seconds_before_archive_seal':perf_counter()-started,
                                              'protocol_sha256':protocol['sha256'],'stage':'validation_only'})
        store.finish()
        verify_run(store.path)
    except Exception as error:
        store.fail(error)
        raise
    return store.path


def bound_archive(directory, expected_status):
    manifest = verify_run(directory)
    require(manifest['status'] == expected_status, 'Wrong or incomplete execution stage')
    config = read_json(Path(directory)/'config.json')
    require_execution(config['unified'],config['execution'])
    return config['unified'], config['execution']


def reference(parent, target):
    return {'relative_path':os.path.relpath(target,parent),'manifest_sha256':sha(Path(target)/'MANIFEST.json')}


def resolve_reference(parent, value, expected_status):
    target = (Path(parent)/value['relative_path']).resolve()
    require(sha(target/'MANIFEST.json') == value['manifest_sha256'], 'Linked archive has changed')
    bound_archive(target,expected_status)
    return target


def execute_selection(search_run, *, output_root=None):
    search_run = Path(search_run).resolve()
    protocol, details = bound_archive(search_run,'validation_search_complete')
    jobs = read_json(search_run/'jobs.json')
    inputs = read_json(search_run/'inputs.json')
    expected_jobs = {(j['candidate_id'],j['seed']):j for j in task_plan(protocol,details)}
    require(len(jobs) == len(expected_jobs) == 45, 'Main/auxiliary job count mismatch')
    require(len({(j['candidate_id'],j['seed']) for j in jobs}) == 45, 'Duplicate search tasks')
    for job in jobs:
        expected_job = expected_jobs.get((job['candidate_id'],job['seed']))
        require(expected_job is not None and all(job[k] == v for k,v in expected_job.items()), 'Unexpected search task configuration')
        require(job['validation_rows'] == inputs['val']['rows'] and job['validation_row_ids_sha256'] == inputs['validation_row_ids_sha256'], 'Search tasks used different validation populations')
        require(read_json(search_run/('jobs/'+job['task_id']+'.json')) == job, 'Search job index disagrees with saved job')
    selected = select_configuration(jobs,protocol,details)
    selected.update(protocol_sha256=protocol['sha256'],inputs=inputs)
    store = RunStore(output_root or search_run.parent.parent,kind='selection')
    try:
        store.write_json('config.json',{'unified':protocol,'execution':details})
        store.write_json('search_reference.json',reference(store.path,search_run))
        thresholds = []
        with np.load(search_run/'validation.npz',allow_pickle=False) as val:
            require(row_digest(val['row_ids']) == inputs['validation_row_ids_sha256'], 'Validation order changed')
            require(len(val['y_true']) == inputs['val']['rows'], 'Incomplete validation population')
            chosen_jobs = [job for job in jobs if job['candidate_id'] in (selected['selected_candidate_id'],'RF_FIXED')]
            require(len(chosen_jobs) == 10, 'Selected/auxiliary five-seed runs missing')
            for job in chosen_jobs:
                scores = np.load(search_run/job['scores_path'],allow_pickle=False)
                require(compute_curves(val['y_true'],scores)['average_precision'] == job['validation_ap'], 'Validation metric cannot be reproduced')
                for alpha in protocol['thresholds']['fpr_targets']:
                    thresholds.append({'task_id':job['task_id'],**calibrate_fpr(val['y_true'],scores,alpha)})
        selected['selected_jobs'] = chosen_jobs
        store.write_json('selection.json',selected)
        store.write_json('thresholds.json',thresholds)
        store.finish()
        verify_run(store.path)
    except Exception as error:
        store.fail(error)
        raise
    return store.path


def formal_record(protocol, details, job, dataset_identity, metadata, y, scores, threshold, inference, comparisons, *, notes):
    record = {'schema_version':'1.0','is_demo':False,'method':'M1_GBDT' if job['role'] == 'main' else 'M1_RF_auxiliary',
              'dataset':{'version':protocol['dataset']['version'],'protocol':protocol['dataset']['protocol'],
                         'variant':protocol['dataset']['variant'],'split':'test','path':dataset_identity['path'],
                         'sha256':dataset_identity['sha256'],'n_samples':len(y)},
              'seed':job['seed'],'config':{'candidate_id':job['candidate_id'],'algorithm':job['algorithm'],'parameters':job['params'],
                                         'effective_parameters':job['effective_parameters'],'unified_protocol_sha256':protocol['sha256'],
                                         'target_validation_fpr':threshold['target_fpr'],'threshold_calibration':threshold,
                                         'model_sha256':job['model_sha256'],'resources':details['resources']},
              'decision_rule':{'score_direction':'higher_means_attack','operator':'>=','threshold':threshold['threshold'],'source':'validation_benign_quantile'},
              'metrics':score_metrics(y,scores,metadata,threshold['threshold'],[x for x in read_json(ROOT/'docs/label_split_policy.json')['family_names'] if x != 'benign']),
              'cost':{'training_seconds':job['training_seconds'],'training_time_note':'Selected training-fold fit reused; no refit on train+validation. Excludes loading, validation scoring and serialization.',
                      'hardware':platform.platform()+' / '+(platform.processor() or 'unspecified CPU'),
                      'thread_settings':details['resources'],'inference':inference},
              'daily_alerts':None,'comparisons':comparisons,
              'provenance':{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                            'created_at':datetime.now(timezone.utc).isoformat(),
                            'software':{'python':platform.python_version(),**environment()['packages']},'notes':notes}}
    validate_result(record)
    return record


def execute_evaluation(selection_run, *, allow_test=False, project_root=ROOT, output_root=None, loader_factory=None):
    require(allow_test is True, 'Explicit final test evaluation authorization is required')
    selection_run = Path(selection_run).resolve()
    protocol, details = bound_archive(selection_run,'selection_frozen')
    selected = read_json(selection_run/'selection.json')
    require(selected['protocol_sha256'] == protocol['sha256'] and not selected['rf_used_for_selection'], 'Selection protocol mismatch')
    search_run = resolve_reference(selection_run,read_json(selection_run/'search_reference.json'),'validation_search_complete')
    require(read_json(search_run/'config.json') == read_json(selection_run/'config.json'), 'Selection/search settings mismatch')
    expected = select_configuration(read_json(search_run/'jobs.json'),protocol,details)
    require(expected['selected_candidate_id'] == selected['selected_candidate_id'], 'Frozen winner cannot be reproduced')
    require(selected['inputs'] == read_json(search_run/'inputs.json'), 'Frozen data identity mismatch')
    require(selected['selected_jobs'] == [job for job in read_json(search_run/'jobs.json') if job['candidate_id'] in (selected['selected_candidate_id'],'RF_FIXED')], 'Frozen model list mismatch')
    thresholds = read_json(selection_run/'thresholds.json')
    require(len(thresholds) == 20, 'Incomplete threshold freeze')
    threshold_map = {(x['task_id'],x['target_fpr']):x for x in thresholds}
    # Recheck thresholds against validation predictions before any test access.
    with np.load(search_run/'validation.npz',allow_pickle=False) as val:
        for job in selected['selected_jobs']:
            val_scores = np.load(search_run/job['scores_path'],allow_pickle=False)
            for alpha in protocol['thresholds']['fpr_targets']:
                require(threshold_map[(job['task_id'],alpha)] == {'task_id':job['task_id'],**calibrate_fpr(val['y_true'],val_scores,alpha)}, 'Frozen threshold cannot be reproduced')
    store = RunStore(output_root or selection_run.parent.parent,kind='evaluation')
    try:
        store.write_json('config.json',{'unified':protocol,'execution':details})
        store.write_json('selection_reference.json',reference(store.path,selection_run))
        snapshot_sources(store)
        spec = protocol['dataset']
        # This is the ONLY real-test loader construction, after all gates above.
        directory = preparation_path(project_root,spec['version'],spec['protocol'],spec['variant'],'train').parent
        test_path = directory/'test.parquet'
        loader = (loader_factory or ModelDataset)(test_path)
        require(tuple(loader.feature_names) == tuple(selected['inputs']['features']), 'Test feature contract mismatch')
        state_dir = directory.parent if spec['variant'] != 'main' else directory
        require(sha(state_dir/'preprocessor.json') == selected['inputs']['preprocessor_sha256'], 'Test processor differs from frozen train/val processor')
        cache = _materialize_loaded(loader,test_path,Path(project_root)/'.tmp/m1_formal_cache'/store.path.name/'test')
        ids, family, flags = metadata_arrays(cache)
        meta = pd.DataFrame({'attack_family':family,'is_attempted':flags})
        timing_X = np.load(search_run/'timing_X.npy',allow_pickle=False)
        timing_rows = read_json(search_run/'timing_rows.json')
        require(row_digest(timing_rows['row_ids']) == timing_rows['row_ids_sha256'] and len(timing_X) == 512, 'Common timing identity invalid')
        scores_by_task, measured = {}, {}
        with cache.model_input() as test, store.file('events.jsonl').open('x',encoding='utf-8') as log:
            test_identity = identity(cache,directory,'test',spec['variant'])
            for job in selected['selected_jobs']:
                require(sha(search_run/job['model_path']) == job['model_sha256'], 'Selected model has changed')
                model = joblib.load(search_run/job['model_path'])
                scores = predict_batches(model,test.X,rows=details['resources']['prediction_batch_rows'],threads=details['resources']['inference_threads'])
                path = 'test_scores/'+job['task_id']+'.npy'
                save_npy(store,path,scores)
                scores_by_task[job['task_id']] = path
                settings = {k:protocol['inference'][k] for k in ('batch_size','warmup','repeats')}
                with inference_workers(model,details['resources']['inference_threads']):
                    measured[job['task_id']] = measure_inference(lambda batch:attack_scores(model,batch),timing_X,threads=details['resources']['inference_threads'],**settings)
                event(log,'test_scores_saved',task=job['task_id'],rows=len(scores))
                del model, scores
            records, comparison_records = [], []
            for seed in protocol['model_seeds']:
                main = next(j for j in selected['selected_jobs'] if j['role'] == 'main' and j['seed'] == seed)
                aux = next(j for j in selected['selected_jobs'] if j['role'] == 'auxiliary' and j['seed'] == seed)
                a = np.load(store.path/scores_by_task[main['task_id']],mmap_mode='r')
                b = np.load(store.path/scores_by_task[aux['task_id']],mmap_mode='r')
                comparisons = []
                bootstrap = details['bootstrap']
                for metric, target in [('ap',None)] + [('tpr_at_fpr',x) for x in bootstrap['fpr_targets']]:
                    result = paired_bootstrap(test.y,a,b,row_ids_a=ids.tolist(),row_ids_b=ids.tolist(),metric=metric,
                        target_fpr=.01 if target is None else target,n_bootstrap=bootstrap['repetitions'],confidence=bootstrap['confidence'],seed=bootstrap['seed'])
                    comparisons.append({'method_a':'M1_GBDT','method_b':'M1_RF_auxiliary','seed_a':seed,'seed_b':seed,'result':result})
                comparison_records.extend(comparisons)
                for job, scores in [(main,a),(aux,b)]:
                    for alpha in protocol['thresholds']['fpr_targets']:
                        threshold = threshold_map[(job['task_id'],alpha)]
                        record = formal_record(protocol,details,job,test_identity,meta,test.y,scores,threshold,measured[job['task_id']],comparisons,
                            notes='Final test evaluation after validation-only selection. Bootstrap describes empirical ROC/AP differences, not uncertainty for a fixed deployment threshold. Timing uses the sealed common validation batch. Selection manifest SHA256='+sha(selection_run/'MANIFEST.json')+'; search manifest SHA256='+sha(search_run/'MANIFEST.json'))
                        key = job['task_id']+'_alpha_'+str(alpha)
                        store.write_json('records/'+key+'.json',record)
                        with store.file('predictions/'+key+'.npz').open('xb') as stream:
                            np.savez_compressed(stream,row_ids=ids,y_true=test.y,y_score=scores,y_pred=(scores>=threshold['threshold']).astype(np.uint8),attack_family=family,is_attempted=flags)
                        with np.load(store.path/('predictions/'+key+'.npz'),allow_pickle=False) as saved:
                            require(row_digest(saved['row_ids']) == row_digest(ids), 'Saved test order changed')
                            require(np.array_equal(saved['y_pred'],(saved['y_score']>=threshold['threshold']).astype(np.uint8)), 'Saved decisions inconsistent')
                            require(score_metrics(saved['y_true'],saved['y_score'],pd.DataFrame({'attack_family':saved['attack_family'],'is_attempted':saved['is_attempted']}),threshold['threshold'],selected['inputs']['families']) == record['metrics'], 'Saved test metrics cannot be reproduced')
                        records.append({'path':'records/'+key+'.json','method':record['method'],'seed':seed,'target_validation_fpr':alpha})
                plot_pr(store,seed,test.y,a,b)
                del a,b
            store.write_json('paired_comparisons.json',comparison_records)
            store.write_json('timing_rows.json',timing_rows)
            store.write_json('evaluation_index.json',{'records':records,'dataset':test_identity,'test_row_ids_sha256':row_digest(ids),
                'five_seed_summaries':summarize_records(store.path,records),'selection_manifest_sha256':sha(selection_run/'MANIFEST.json')})
            event(log,'evaluation_complete',records=len(records))
        store.finish()
        verify_run(store.path)
    except Exception as error:
        store.fail(error)
        raise
    return store.path


def plot_pr(store, seed, y, a, b):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7,5))
    try:
        for name, scores in [('GBDT',a),('RF auxiliary',b)]:
            curves = compute_curves(y,scores)
            ax.step(curves['pr']['recall'],curves['pr']['precision'],where='post',label=name)
        ax.set(xlabel='Recall',ylabel='Precision',title=f'Frozen M1 models - seed {seed}',xlim=(0,1),ylim=(0,1.02))
        ax.legend();fig.tight_layout()
        with store.file(f'figures/pr_seed_{seed}.png').open('xb') as stream:fig.savefig(stream,format='png',dpi=160)
    finally:plt.close(fig)


def summarize_records(directory, records):
    summary = []
    for method in ('M1_GBDT','M1_RF_auxiliary'):
        for alpha in (.01,.05):
            values = [read_json(directory/x['path']) for x in records if x['method'] == method and x['target_validation_fpr'] == alpha]
            require(len(values) == 5, 'Five-seed final records missing')
            item = {'method':method,'target_validation_fpr':alpha,'seeds':[v['seed'] for v in values],'std_ddof':1}
            for label, getter in [('ap',lambda r:r['metrics']['curves']['average_precision']),('roc_auc',lambda r:r['metrics']['curves']['roc_auc']),
                                  ('macro_f1',lambda r:r['metrics']['classification']['macro_f1']),('test_fpr',lambda r:r['metrics']['classification']['fpr']),
                                  ('test_tpr',lambda r:r['metrics']['classification']['tpr']),('training_seconds',lambda r:r['cost']['training_seconds'])]:
                numbers = np.array([getter(v) for v in values],dtype=float)
                item[label] = {'mean':float(numbers.mean()),'sample_std':float(numbers.std(ddof=1))}
            summary.append(item)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','search','select','evaluate'])
    parser.add_argument('--unified',type=Path,default=ROOT/'methods/m1/protocol.unified.v1.json')
    parser.add_argument('--execution',type=Path,default=ROOT/'methods/m1/execution.proposed.json')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--permit-final-test',action='store_true')
    args = parser.parse_args(argv)
    if args.stage in ('check','search'):
        protocol, details = read_json(args.unified),read_json(args.execution)
        issues = execution_issues(protocol,details)
        if issues:
            print('BLOCKED before data access: ' + '; '.join(issues));return 2
        if args.stage == 'search':print(execute_search(protocol,details,output_root=args.output_root))
        else:print('Execution settings complete and recorded approval valid; no datasets opened.')
    elif args.stage == 'select':
        require(args.run is not None,'--run must identify a sealed search archive')
        print(execute_selection(args.run,output_root=args.output_root))
    else:
        require(args.run is not None,'--run must identify a sealed selection archive')
        print(execute_evaluation(args.run,allow_test=args.permit_final_test,output_root=args.output_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
