"""Bootstrap acceptance: synthetic arrays, temporary files, no real Test reads."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eval.metrics import paired_bootstrap
from methods.m1 import bootstrap as boot, bootstrap_stats as stats, bootstrap_delivery as delivery


def arrays():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
    a = np.array([.1, .2, .8, .3, .7, .4, .4, .5, .1, .8, .7, .1])
    b = np.array([.2, .2, .7, .1, .6, .8, .3, .2, .4, .7, .6, .1])
    ids = np.array([f'synthetic:{i}' for i in range(len(y))])
    return y, a, b, ids


def approved():
    c = boot.make_config()
    c['approval'] = {'status': 'approved', 'source': 'SYNTHETIC UNIT TEST ONLY',
        'approved_at': 'UNIT TEST ONLY', 'payload_sha256': boot.digest(boot.payload(c))}
    return c


class P3EquivalenceTests(unittest.TestCase):
    def test_rare_positive_and_random_tied_scores_match_p3(self):
        rng=np.random.default_rng(171)
        for positives in (1,2):
            y=np.r_[np.zeros(35,dtype=np.uint8),np.ones(positives,dtype=np.uint8)]
            a=np.round(rng.random(len(y)),1);b=np.round(rng.random(len(y)),1)
            ids=np.array([f'rare:{i}' for i in range(len(y))])
            for metric,target in boot.METRICS:
                rows=list(stats.replicates(y,a,b,metric=metric,target=target,seed=2026,count=17))
                result=stats.summarize([x[2] for x in rows],value_a=stats.metric_value(y,a,metric,target),value_b=stats.metric_value(y,b,metric,target),metric=metric,target=target,y=y)
                ref=paired_bootstrap(y,a,b,row_ids_a=ids,row_ids_b=ids,metric=metric,target_fpr=.01 if target is None else target,n_bootstrap=17,seed=2026)
                self.assertEqual({k:result[k] for k in ref},ref)

    def test_all_four_metrics_strictly_match_every_p3_result_field(self):
        y, a, b, ids = arrays()
        for metric, target in boot.METRICS:
            for seed in (0, 2026, 93):
                values = list(stats.replicates(y, a, b, metric=metric, target=target, seed=seed, count=35))
                own = stats.summarize([v[2] for v in values], value_a=stats.metric_value(y, a, metric, target),
                    value_b=stats.metric_value(y, b, metric, target), metric=metric, target=target, y=y, seed=seed)
                ref = paired_bootstrap(y, a, b, row_ids_a=ids, row_ids_b=ids, metric=metric,
                    target_fpr=.01 if target is None else target, n_bootstrap=35, confidence=.95, seed=seed)
                self.assertEqual({k: own[k] for k in ref}, ref)

    def test_tied_and_constant_scores_match_p3(self):
        y, a, b, ids = arrays();a[:] = .5;b[:] = .5
        for metric, target in boot.METRICS:
            values = list(stats.replicates(y, a, b, metric=metric, target=target, seed=2026, count=20))
            self.assertTrue(all(x[2] == 0 for x in values))
            ref = paired_bootstrap(y, a, b, row_ids_a=ids, row_ids_b=ids, metric=metric,
                target_fpr=.01 if target is None else target, n_bootstrap=20, seed=2026)
            self.assertEqual(ref['ci_lower'], 0)
            self.assertEqual(ref['ci_upper'], 0)

    def test_same_seed_reproduces_every_replicate(self):
        y, a, b, _ = arrays()
        kwargs = dict(metric='ap', target=None, seed=2026, count=21)
        self.assertEqual(list(stats.replicates(y, a, b, **kwargs)), list(stats.replicates(y, a, b, **kwargs)))

    def test_seed_change_changes_rng_state_and_distribution(self):
        y, a, b, _ = arrays()
        first = list(stats.replicates(y, a, b, metric='ap', target=None, seed=2026, count=20))
        second = list(stats.replicates(y, a, b, metric='ap', target=None, seed=42, count=20))
        self.assertNotEqual(first, second)

    def test_rng_resume_is_bitwise_identical_to_uninterrupted(self):
        y, a, b, _ = arrays()
        full = list(stats.replicates(y, a, b, metric='ap', target=None, seed=2026, count=31))
        prefix = list(stats.replicates(y, a, b, metric='ap', target=None, seed=2026, count=11))
        suffix = list(stats.replicates(y, a, b, metric='ap', target=None, seed=2026, count=20, state=prefix[-1][3]))
        self.assertEqual(full, prefix + suffix)

    def test_pair_uses_one_identical_index_vector_and_preserves_counts(self):
        y, a, b, _ = arrays();seen=[];calls=[]
        def sample(indices):seen.append(indices.copy())
        def metric(labels, scores, name, target):calls.append((labels.copy(), scores.copy()));return float(scores.mean())
        with patch.object(stats, 'metric_value', side_effect=metric):
            list(stats.replicates(y, a, b, metric='ap', target=None, seed=2026, count=8, on_indices=sample))
        for i, indices in enumerate(seen):
            self.assertEqual(int((y[indices] == 0).sum()), int((y == 0).sum()))
            self.assertTrue(np.array_equal(calls[2*i][0], calls[2*i+1][0]))
            self.assertTrue(np.array_equal(calls[2*i][1], a[indices]))
            self.assertTrue(np.array_equal(calls[2*i+1][1], b[indices]))

    def test_se_is_sample_sd_not_monte_carlo_mean_error(self):
        y, _, _, _ = arrays();values=[-.1, .1, .2, .3]
        result=stats.summarize(values,value_a=.5,value_b=.4,metric='ap',target=None,y=y)
        self.assertEqual(result['bootstrap_standard_error'], float(np.std(values, ddof=1)))
        self.assertNotEqual(result['bootstrap_standard_error'], float(np.std(values, ddof=1)/2))


class BootstrapInputTests(unittest.TestCase):
    def test_valid_arrays(self):
        y,a,b,ids=arrays();stats.validate_arrays(y,a,b,ids,ids)

    def test_row_order_mismatch_rejected(self):
        y,a,b,ids=arrays()
        with self.assertRaises(ValueError):stats.validate_arrays(y,a,b,ids,ids[::-1])

    def test_duplicate_ids_rejected(self):
        y,a,b,ids=arrays();ids[0]=ids[1]
        with self.assertRaises(ValueError):stats.validate_arrays(y,a,b,ids,ids)

    def test_missing_class_rejected(self):
        y,a,b,ids=arrays();y[:]=0
        with self.assertRaises(ValueError):stats.validate_arrays(y,a,b,ids,ids)

    def test_nan_infinity_and_out_of_range_rejected(self):
        for value in (float('nan'),float('inf'),-.1,1.1):
            y,a,b,ids=arrays();a[0]=value
            with self.assertRaises(ValueError):stats.validate_arrays(y,a,b,ids,ids)

    def test_integer_predictions_and_shape_mismatch_rejected(self):
        y,a,b,ids=arrays()
        with self.assertRaises(ValueError):stats.validate_arrays(y,a.astype(int),b,ids,ids)
        with self.assertRaises(ValueError):stats.validate_arrays(y,a[:-1],b,ids,ids)
        with self.assertRaises(ValueError):stats.validate_arrays(y,a[:,None],b,ids,ids)

    def test_unknown_metric_or_target_rejected(self):
        y,a,_,_=arrays()
        for m,t in [('macro_f1',None),('tpr_at_fpr',.2),('ap',.01)]:
            with self.assertRaises(ValueError):stats.metric_value(y,a,m,t)

    def test_invalid_seed_count_and_summary_rejected(self):
        y,a,b,_=arrays()
        for seed,count in [(-1,20),(True,20),(2026,0)]:
            with self.assertRaises(ValueError):list(stats.replicates(y,a,b,metric='ap',target=None,seed=seed,count=count))
        with self.assertRaises(ValueError):stats.summarize([float('nan'),.1],value_a=.5,value_b=.4,metric='ap',target=None,y=y)


class BootstrapEngineeringTests(unittest.TestCase):
    def test_no_seed_metric_or_repetition_can_be_removed(self):
        for key,value in [('model_seeds',[42]),('repetitions',20),('metrics',[{'metric':'ap','fpr_target':None}])]:
            c=approved();c['statistics'][key]=value
            with self.assertRaises(ValueError):boot.validate(c)

    def test_low_memory_blocks_before_worker_creation(self):
        with patch('methods.m1.resource_trial.WindowsAPI') as api,patch('methods.m1.resource_trial.monitored_child') as child:
            api.return_value.memory.return_value={'available_physical_bytes':13*boot.GIB}
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):boot.protected(Path(tmp),approved(),'trial','trial_seed42_ap')
                self.assertFalse((Path(tmp)/'monitor').exists());child.assert_not_called()

    def test_independent_rng_and_all_csv_summary_checks(self):
        c=approved();task={'id':'synthetic2000','seed':42,'metric':'ap','target':None,'count':2000,'trial':False}
        y,a,b,_=arrays();expected={('M1_GBDT','ap',None):stats.metric_value(y,a,'ap'),('M1_RF_auxiliary','ap',None):stats.metric_value(y,b,'ap')}
        with tempfile.TemporaryDirectory() as tmp,patch.object(boot,'load_pair',return_value=(y,a,b,expected)):
            run=Path(tmp);boot.execute_task(run,c,task,None,None,lambda x:None)
            directory=run/'tasks'/task['id'];rows,_,_,_=boot.completed_chunks(directory,c,task)
            self.assertEqual(delivery.verify_rng_and_samples(directory,task,rows,y,a,b),[1,1000,2000])
            checked=delivery.independent_summary(rows,task,y)
            result=boot.read(directory/'result.json')['result']
            self.assertTrue(all(result[k]==v for k,v in checked.items()))
            cp=directory/'checkpoints/0000.json';d=boot.read(cp);d['rng_state']['state']['state']+=1
            cp.write_text(json.dumps(d),encoding='utf-8')
            with self.assertRaises(ValueError):delivery.verify_rng_and_samples(directory,task,rows,y,a,b)

    def test_complete_record_reuse_and_file_set_tamper_refusal(self):
        c=approved();task={'id':'tiny','seed':42,'metric':'ap','target':None,'count':10,'trial':False}
        y,a,b,_=arrays();expected={('M1_GBDT','ap',None):stats.metric_value(y,a,'ap'),('M1_RF_auxiliary','ap',None):stats.metric_value(y,b,'ap')}
        with tempfile.TemporaryDirectory() as tmp,patch.object(boot,'load_pair',return_value=(y,a,b,expected)):
            run=Path(tmp);boot.execute_task(run,c,task,None,None,lambda x:None)
            monitor=run/'monitor/test';monitor.mkdir(parents=True)
            boot.record_completion(run,c,task,{'exit_code':0,'abort_reason':None,'guard_verified':True},monitor)
            self.assertTrue(boot.verify_completed(run,c,task))
            (run/'tasks/tiny/unexpected.txt').write_text('tamper')
            with self.assertRaises(ValueError):boot.verify_completed(run,c,task)

    def test_twenty_task_synthetic_pipeline_does_not_read_real_sources(self):
        c=approved();y,a,b,_=arrays()
        points={(name,m,t):stats.metric_value(y,s,m,t) for name,s in [('M1_GBDT',a),('M1_RF_auxiliary',b)] for m,t in boot.METRICS}
        with tempfile.TemporaryDirectory() as tmp,patch.object(boot,'load_pair',return_value=(y,a,b,points)) as pair:
            run=Path(tmp)
            for task in boot.tasks(c):
                tiny={**task,'count':7};boot.execute_task(run,c,tiny,None,None,lambda x:None)
                rows,_,_,_=boot.completed_chunks(run/'tasks'/tiny['id'],c,tiny)
                self.assertEqual(len(rows),7)
            self.assertEqual(pair.call_count,20)
            self.assertEqual(len(list((run/'tasks').iterdir())),20)

    def test_three_real_pngs_from_synthetic_distributions(self):
        result=[]
        for task in boot.tasks(approved()):
            result.append({'task':task,'result':{'difference':.01,'ci_lower':-.02,'ci_upper':.03},'deltas':np.linspace(-.02,.03,20)})
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);delivery.plots(result,directory)
            for name in ('ap','tpr','distribution'):
                path=directory/f'fig_m1_bootstrap_{name}.png'
                self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertGreater(path.stat().st_size,10000)

    def test_public_manifest_full_set_hash_validation(self):
        names=[f'results/m1_bootstrap/{x}' for x in ['bootstrap_config.json','bootstrap_results.csv','bootstrap_summary.json','confidence_intervals.json','execution_log.csv']]
        names+=['docs/m1_bootstrap_report.md','figures/fig_m1_bootstrap_ap.png','figures/fig_m1_bootstrap_tpr.png','figures/fig_m1_bootstrap_distribution.png']
        with tempfile.TemporaryDirectory() as tmp,patch.object(boot,'ROOT',Path(tmp)):
            root=Path(tmp)
            for name in names:boot.publish_bytes(root/name,b'SYNTHETIC')
            manifest={'status':'complete','task_count':20,'replicate_rows':40000,'files':{n:{'bytes':(root/n).stat().st_size,'sha256':boot.sha(root/n)} for n in names}}
            path=root/'manifest.json';boot.write(path,manifest)
            self.assertEqual(delivery.verify_public(path,boot.sha(path)),9)
            (root/names[0]).write_bytes(b'CHANGED')
            with self.assertRaises(ValueError):delivery.verify_public(path,boot.sha(path))

    def test_exact_twenty_tasks_and_trial_separation(self):
        c=approved();tasks=boot.tasks(c)
        self.assertEqual(len(tasks),20);self.assertEqual(sum(t['count'] for t in tasks),40000)
        self.assertEqual({t['seed'] for t in tasks},{42,43,44,45,46})
        self.assertEqual({t['metric'] for t in tasks},{'ap','tpr_at_fpr'})
        trial=boot.tasks(c,trial=True)[0]
        self.assertEqual(trial['count'],20);self.assertNotIn(trial['id'],{t['id'] for t in tasks})
        self.assertNotEqual(boot.task_binding(c,trial),boot.task_binding(c,tasks[0]))

    def test_pending_and_changed_configuration_refused(self):
        with self.assertRaises(ValueError):boot.validate(boot.make_config())
        c=approved();self.assertEqual(boot.validate(c),c)
        c['statistics']['bootstrap_seed']=42
        with self.assertRaises(ValueError):boot.validate(c)

    def test_code_or_input_hash_change_refused(self):
        c=approved();c['code']['eval/metrics.py']='0'*64
        with self.assertRaises(ValueError):boot.validate(c)
        c=approved();c['input']['manifest_sha256']='0'*64
        with self.assertRaises(ValueError):boot.validate(c)

    def test_cli_requires_launch_before_real_work(self):
        with patch.object(boot,'run_pipeline') as run:
            with self.assertRaises(ValueError):boot.main(['run'])
            run.assert_not_called()

    def test_no_training_or_parquet_calls_in_new_modules(self):
        import ast
        for module in (boot,stats,delivery):
            tree=ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
            names=[getattr(n.func,'attr',getattr(n.func,'id','')) for n in ast.walk(tree) if isinstance(n,ast.Call)]
            self.assertTrue(set(names).isdisjoint({'fit','fit_tree','ModelDataset','calibrate_fpr','read_parquet','execute_evaluation'}))

    def test_exclusive_publication_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file';boot.publish_bytes(p,b'first')
            with self.assertRaises(FileExistsError):boot.publish_bytes(p,b'second')
            self.assertEqual(p.read_bytes(),b'first')

    def test_resource_abort_conditions(self):
        sample={'available_physical_bytes':20*boot.GIB,'rss_bytes':boot.GIB,'disk_free_bytes':100*boot.GIB}
        self.assertIsNone(boot.abort_reason(sample,boot.RESOURCES))
        for field,value,reason in [('available_physical_bytes',5*boot.GIB,'free_memory'),('rss_bytes',9*boot.GIB,'RSS'),('disk_free_bytes',19*boot.GIB,'disk')]:
            self.assertIn(reason,boot.abort_reason({**sample,field:value},boot.RESOURCES))

    def test_guarded_go_required(self):
        with patch('sys.stdin',io.StringIO('NO\n')):
            with self.assertRaises(ValueError):boot.worker('missing','missing','task','missing')

    def test_synthetic_checkpoint_interruption_resume_and_hash_rejection(self):
        c=approved();task={'id':'synthetic_ap','seed':42,'metric':'ap','target':None,'count':200,'trial':False}
        y,a,b,_=arrays();expected={('M1_GBDT','ap',None):stats.metric_value(y,a,'ap'),('M1_RF_auxiliary','ap',None):stats.metric_value(y,b,'ap')}
        original=stats.metric_value;counter=[0]
        def interrupted(*args):
            counter[0]+=1
            if counter[0]>250:raise RuntimeError('SYNTHETIC INTERRUPTION')
            return original(*args)
        with tempfile.TemporaryDirectory() as tmp,patch.object(boot,'load_pair',return_value=(y,a,b,expected)):
            run=Path(tmp)
            with patch.object(stats,'metric_value',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'INTERRUPTION'):boot.execute_task(run,c,task,None,None,lambda x:None)
            directory=run/'tasks'/task['id'];rows,_,_,_=boot.completed_chunks(directory,c,task)
            self.assertEqual(len(rows),100)
            first_sha=boot.sha(directory/'chunks/0000.csv')
            boot.execute_task(run,c,task,None,None,lambda x:None)
            self.assertEqual(boot.sha(directory/'chunks/0000.csv'),first_sha)
            rows,_,_,_=boot.completed_chunks(directory,c,task)
            self.assertEqual(len(rows),200)
            full=list(stats.replicates(y,a,b,metric='ap',target=None,seed=2026,count=200))
            self.assertEqual([float(r['delta']) for r in rows],[r[2] for r in full])
            self.assertEqual(boot.read(directory/'result.json')['resumed_replicates'],100)
            changed=copy.deepcopy(task);changed['seed']=43
            with self.assertRaises(ValueError):boot.completed_chunks(directory,c,changed)
            (directory/'chunks/0000.csv').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):boot.completed_chunks(directory,c,task)

    def test_orphan_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'chunks').mkdir();(p/'chunks/0000.csv').write_text('orphan')
            with self.assertRaises(ValueError):boot.completed_chunks(p,approved(),boot.tasks(approved())[0])

    def test_independent_summary_rejects_missing_duplicate_and_invalid_rows(self):
        y,a,b,_=arrays();task={'seed':42,'metric':'ap','target':None,'count':3}
        values=list(stats.replicates(y,a,b,metric='ap',target=None,seed=2026,count=3))
        data=[[42,'ap','',i+1,*v[:3]] for i,v in enumerate(values)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';boot.publish_bytes(p,boot.csv_bytes(data));rows=boot.read_rows(p)
            checked=delivery.independent_summary(rows,task,y)
            self.assertEqual(checked['valid_replicates'],3)
            with self.assertRaises(ValueError):delivery.independent_summary(rows[:-1],task,y)
            bad=copy.deepcopy(rows);bad[1]['bootstrap_iteration']='1'
            with self.assertRaises(ValueError):delivery.independent_summary(bad,task,y)
            bad=copy.deepcopy(rows);bad[1]['delta']='0.9999'
            with self.assertRaises(ValueError):delivery.independent_summary(bad,task,y)

    def test_csv_exact_columns_and_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';p.write_text(','.join(boot.COLUMNS)+'\n42,ap\n')
            with self.assertRaises(ValueError):boot.read_rows(p)

    def test_manifest_wrong_hash_or_missing_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';boot.write(p,{'status':'complete','files':{}})
            with self.assertRaises(ValueError):delivery.verify_public(p,'0'*64)
            with self.assertRaises(ValueError):delivery.verify_public(p,boot.sha(p))


if __name__=='__main__':unittest.main()
