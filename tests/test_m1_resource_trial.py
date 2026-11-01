"""Resource guard checks only; no ML imports/fits or real Parquet access here."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

from methods.m1.resource_trial import (GIB,MIB,allowed_source,monitored_child,preflight,
    read_json,resource_reason,trial_config,validate_trial_config,WindowsAPI)


class ResourcePlanTests(unittest.TestCase):
    def test_scope_is_exactly_approved_g01_seed42(self):
        cfg=trial_config()
        validate_trial_config(cfg)
        self.assertEqual(cfg['candidate']['id'],'G01')
        self.assertEqual(cfg['candidate']['params']['max_iter'],200)
        self.assertEqual(cfg['seed'],42)
        self.assertEqual(cfg['dataset']['allowed_splits'],['train','val'])
        bad=copy.deepcopy(cfg);bad['seed']=43
        with self.assertRaises(ValueError):validate_trial_config(bad)
        bad=copy.deepcopy(cfg);bad['job_commit_limit_bytes']=20*GIB
        with self.assertRaises(ValueError):validate_trial_config(bad)

    def test_source_gate_rejects_test_or_outside_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'approved'
            self.assertEqual(allowed_source(directory/'train.parquet',directory).name,'train.parquet')
            for path in [directory/'test.parquet',Path(tmp)/'other/train.parquet']:
                with self.assertRaises(ValueError):allowed_source(path,directory)

    def test_preflight_never_weakens_thresholds(self):
        cfg=trial_config()
        self.assertIsNone(preflight(cfg,{'available_physical_bytes':15*GIB},30*GIB))
        self.assertIsNotNone(preflight(cfg,{'available_physical_bytes':13*GIB},30*GIB))
        self.assertIsNotNone(preflight(cfg,{'available_physical_bytes':15*GIB},19*GIB))

    def test_abort_conditions(self):
        cfg=trial_config()
        normal={'available_physical_bytes':10*GIB,'rss_bytes':2*GIB,'disk_free_bytes':100*GIB,
            'trial_directory_bytes':2*GIB,'elapsed_seconds':10}
        self.assertIsNone(resource_reason(normal,cfg))
        for key,value in [('available_physical_bytes',5*GIB),('rss_bytes',9*GIB),
                          ('disk_free_bytes',19*GIB),('trial_directory_bytes',11*GIB),('elapsed_seconds',3601)]:
            with self.subTest(key=key):self.assertIsNotNone(resource_reason({**normal,key:value},cfg))

    @unittest.skipUnless(os.name=='nt','Windows native Job Object test')
    def test_actual_job_cap_is_enforced_and_affinity_is_limited(self):
        # Attempts at most a 128 MiB allocation under a verified 64 MiB cap.
        # No dataset is opened, and no sklearn fit is executed.
        with tempfile.TemporaryDirectory() as tmp:
            cfg=trial_config();cfg['sample_seconds']=.05
            result=monitored_child(tmp,cfg,probe=True)
            probe=read_json(Path(tmp)/'probe.json')
            self.assertEqual(result['exit_code'],0)
            self.assertTrue(probe['memory_limit_enforced'])
            self.assertEqual(result['job_commit_limit_bytes'],64*MIB)
            self.assertEqual(result['logical_cpu_limit'],4)
            self.assertEqual(probe['affinity_mask'].bit_count(),4)
            self.assertGreater(result['sampled_peak_rss_bytes'],16*MIB)
            self.assertGreater(result['peak_cpu_core_equivalent'],0)
            self.assertGreaterEqual(result['last_process_counters']['monitored_process_count'],1)


if __name__=='__main__':unittest.main()
