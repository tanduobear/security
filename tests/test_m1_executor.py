"""Formal executor acceptance: synthetic files/fake fits only, no real test data.

The full-stage fixture intentionally exercises the official non-demo schema
branch inside a temporary test sandbox. Those records are NOT experiments,
are never exported to results/, and are deleted with the fixture.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eval.validate import validate_result
from methods.m1 import executor
from methods.m1.data import ModelInput
from methods.m1.protocol import execution_digest, execution_issues, unified_digest, validate_unified
from methods.m1.selection import calibrate_fpr, select_configuration
from methods.m1.storage import verify_run

ROOT = Path(__file__).resolve().parents[1]


def settings():
    unified = executor.read_json(ROOT/'methods/m1/protocol.unified.v1.json')
    details = executor.read_json(ROOT/'methods/m1/execution.proposed.json')
    return unified, details


def approved_fixture():
    """Only a simulated approval object in memory, never a saved real approval."""
    u, d = settings()
    d['resources'].update(fit_threads=1,inference_threads=1,prediction_batch_rows=128,duckdb_memory_limit_mb=64)
    d['rf_auxiliary']['params']['n_jobs'] = 1
    d['bootstrap'].update(repetitions=4)
    d['approval'] = {'status':'approved','source':'UNIT TEST SIMULATION ONLY','approved_at':'UNIT TEST ONLY','execution_sha256':execution_digest(d)}
    return u,d


class SyntheticP2:
    calls = []
    feature_names = ['f'+str(i) for i in range(140)]
    class_weights = {0: .6,1:3.0}

    def __init__(self,path):
        self.path = Path(path)
        type(self).calls.append(self.path.name)
        self.rows = 80 if self.path.stem == 'train' else 640
        self.manifest = {'output':{self.path.stem:{'rows':self.rows,
            'main':{'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()},
            'effective_index':{'rows':self.rows,'sha256':'f'*64}}}}

    def __iter__(self):
        for offset in range(0,self.rows,100):
            positions = np.arange(offset,min(offset+100,self.rows))
            y = (positions%2).astype(np.uint8)
            X = np.zeros((len(y),140),dtype=np.float32)
            X[:,0] = y
            X[:,1] = positions/max(self.rows,1)
            yield X,y,pd.DataFrame({'row_index':positions,'row_id':[self.path.stem+':'+str(x) for x in positions],
                'y_binary':y,'attack_family':np.where(y==1,'ddos','benign'),'is_attempted':np.zeros(len(y),dtype=bool)})


class FixtureModel:
    classes_ = np.array([0,1])
    n_features_in_ = 140

    def __init__(self,params,seed):self.params,self.seed,self.n_jobs = params,seed,1
    def predict_proba(self,X):
        p = .1+.8*np.asarray(X[:,0],dtype=float)
        return np.column_stack((1-p,p))
    def get_params(self):return {**self.params,'random_state':self.seed}


def fake_fit(name,params,training,*,model_seed,fit_threads):
    if training.split != 'train' or len(training.y)!=80:
        raise AssertionError('Only the fixed synthetic training fold may reach fit')
    training.validate()
    return FixtureModel(params,model_seed), .001


def fixture_files(root):
    directory = Path(root)/'data/split/original/stratified'
    directory.mkdir(parents=True)
    for name in ('train','val','test'):
        (directory/(name+'.parquet')).write_bytes(('SYNTHETIC PLACEHOLDER '+name).encode())
    (directory/'preprocessor.json').write_text('{"synthetic_test_fixture":true}')
    return directory


class M1FrozenProtocolTests(unittest.TestCase):
    def test_unified_terms_frozen_but_details_still_pending(self):
        u,d = settings()
        self.assertTrue(validate_unified(u))
        self.assertEqual(u['sha256'],unified_digest(u))
        self.assertEqual(u['model_seeds'],[42,43,44,45,46])
        self.assertEqual(len(d['gbdt_candidates']),8)
        self.assertEqual(d['approval']['status'],'pending')
        issues = execution_issues(u,d)
        self.assertTrue(any('not approved' in issue for issue in issues))

    def test_pending_search_blocks_before_output_or_loader(self):
        u,d = settings()
        with tempfile.TemporaryDirectory() as tmp, patch.object(executor,'ModelDataset') as loader:
            target = Path(tmp)/'no-output'
            with self.assertRaises(ValueError):executor.execute_search(u,d,project_root=tmp,output_root=target)
            self.assertFalse(target.exists())
            loader.assert_not_called()
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(executor.main(['check']),2)

    def test_locked_details_and_unified_changes_are_rejected(self):
        u,d = approved_fixture()
        self.assertEqual(execution_issues(u,d),[])
        d['gbdt_candidates'][0]['params']['max_iter'] += 1
        self.assertTrue(execution_issues(u,d))
        changed = copy.deepcopy(u)
        changed['model_seeds'][0] = 99
        changed['sha256'] = unified_digest(changed)
        with self.assertRaises(ValueError):validate_unified(changed)

    def test_plan_is_40_main_and_5_fixed_auxiliary_jobs(self):
        u,d = settings()
        plan = executor.task_plan(u,d)
        self.assertEqual(len(plan),45)
        self.assertEqual(sum(x['role']=='main' for x in plan),40)
        self.assertEqual(sum(x['role']=='auxiliary' for x in plan),5)
        self.assertEqual(len({x['candidate_id'] for x in plan if x['role']=='auxiliary'}),1)
        for c in d['gbdt_candidates']:
            self.assertEqual([j['seed'] for j in plan if j['candidate_id']==c['id']],u['model_seeds'])

    def test_undefined_or_changed_stopping_rejected(self):
        u,d = approved_fixture()
        d['stopping'] = {'mode':'unspecified_early_stopping'}
        d['approval']['execution_sha256']=execution_digest(d)
        self.assertTrue(execution_issues(u,d))


class M1SelectionCalibrationTests(unittest.TestCase):
    def jobs(self):
        u,d=settings()
        return u,d,[{**j,'evaluation_split':'val','validation_ap':.8+(int(j['candidate_id'][1:])*.001 if j['role']=='main' else .19)} for j in executor.task_plan(u,d)]

    def test_five_seed_mean_not_best_seed_or_rf_selects_winner(self):
        u,d,jobs=self.jobs()
        for j in jobs:
            if j['candidate_id']=='G01':j['validation_ap']=1.0 if j['seed']==42 else .1
            if j['role']=='auxiliary':j['validation_ap']=1.0
        result=select_configuration(jobs,u,d)
        self.assertEqual(result['selected_candidate_id'],'G08')
        self.assertFalse(result['rf_used_for_selection'])
        self.assertTrue(all('sample_std_ap' in x for x in result['candidate_summaries']))

    def test_tie_rule_is_deterministic_and_missing_seed_rejected(self):
        u,d,jobs=self.jobs()
        for j in jobs:j['validation_ap']=.8
        self.assertEqual(select_configuration(jobs,u,d)['selected_candidate_id'],'G01')
        with self.assertRaises(ValueError):select_configuration(jobs[1:],u,d)

    def test_threshold_ties_and_zero_allowed_false_alarms(self):
        y=np.array([0,0,0,0,1,1])
        scores=np.array([1.,1.,.5,.5,.9,.8])
        for alpha in (0.,.01,.05,.25,.5):
            c=calibrate_fpr(y,scores,alpha)
            false=int(((scores>=c['threshold'])&(y==0)).sum())
            self.assertEqual(false,c['validation_false_positives'])
            self.assertLessEqual(false,int(np.floor(alpha*4)))
            self.assertTrue(np.isfinite(c['threshold']))

    def test_calibration_uses_only_benign_validation_scores(self):
        y=np.array([0,0,0,0,1,1]);a=np.array([.1,.2,.3,.4,.8,.9]);b=a.copy();b[y==1]=.01
        self.assertEqual(calibrate_fpr(y,a,.25),calibrate_fpr(y,b,.25))
        with self.assertRaises(ValueError):calibrate_fpr([1,1],[.1,.2],.01)
        with self.assertRaises(ValueError):calibrate_fpr([0,1],[np.nan,.1],.01)

    def test_fixed_validation_threshold_may_have_higher_test_fpr(self):
        rule=calibrate_fpr(np.array([0,0,1]),np.array([.1,.2,.9]),.01)
        # Only synthetic vectors; never recalibrate the threshold on test scores.
        test_benign_scores=np.array([.8,.9])
        self.assertGreater(float((test_benign_scores>=rule['threshold']).mean()),.01)


class M1ExecutorIntegrationTests(unittest.TestCase):
    def test_final_evaluation_requires_explicit_permission_before_any_io(self):
        with patch.object(executor,'ModelDataset') as loader:
            with self.assertRaises(ValueError):executor.execute_evaluation('nonexistent-selection')
            loader.assert_not_called()

    def test_synthetic_full_archive_chain_and_official_schema(self):
        u,d=approved_fixture()
        SyntheticP2.calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            fixture_files(tmp)
            output=Path(tmp)/'archives'
            with patch.object(executor,'fit_tree',wraps=fake_fit) as fitter:
                search=executor.execute_search(u,d,project_root=tmp,output_root=output,loader_factory=SyntheticP2)
                self.assertEqual(fitter.call_count,45) # scheduling/fake fits, not 40 actual training runs
            self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet'])
            self.assertEqual(verify_run(search)['status'],'validation_search_complete')
            selected=executor.execute_selection(search,output_root=output)
            self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet'])
            result=executor.read_json(selected/'selection.json')
            self.assertEqual(result['selected_candidate_id'],'G01')
            self.assertEqual(len(executor.read_json(selected/'thresholds.json')),20)
            with self.assertRaises(ValueError):executor.execute_evaluation(search,allow_test=True,project_root=tmp,loader_factory=SyntheticP2)
            self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet'])
            final=executor.execute_evaluation(selected,allow_test=True,project_root=tmp,output_root=output,loader_factory=SyntheticP2)
            self.assertEqual(SyntheticP2.calls,['train.parquet','val.parquet','test.parquet'])
            index=executor.read_json(final/'evaluation_index.json')
            self.assertEqual(len(index['records']),20)
            self.assertEqual(len(index['five_seed_summaries']),4)
            for item in index['records']:
                record=executor.read_json(final/item['path'])
                self.assertTrue(validate_result(record))
                self.assertEqual(record['decision_rule']['source'],'validation_benign_quantile')
            self.assertEqual(verify_run(final)['status'],'test_evaluation_complete')
            self.assertEqual(len(list((final/'figures').glob('*.png'))),5)
            (search/'search_summary.json').write_text('{}')
            with self.assertRaises(ValueError):executor.execute_evaluation(selected,allow_test=True,project_root=tmp,loader_factory=SyntheticP2)
            self.assertEqual(SyntheticP2.calls.count('test.parquet'),1)


if __name__ == '__main__':unittest.main()
