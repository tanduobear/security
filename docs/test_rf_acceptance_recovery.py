"""Synthetic files only; no dataset, model, or prediction loading."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).with_name('rf_acceptance_recovery.py'))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
rf = recovery.rf


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=recovery.ROOT / '.tmp')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.failed = self.run / 'monitor' / recovery.FAILED
        self.failed.mkdir(parents=True)
        rf.write(self.failed / 'outcome.json', {'exit_code': 1, 'abort_reason': None})
        (self.failed / 'worker.log').write_text("ModuleNotFoundError: No module named 'duckdb'", encoding='utf-8')
        self.original = rf.inventory(self.failed)
        self.ok = self.run / 'monitor/ACCEPT-SYNTHETIC-NEW'
        self.ok.mkdir()
        rf.write(self.ok / 'outcome.json', {'exit_code': 0, 'abort_reason': None, 'guard_verified': True,
                 'job_commit_limit_bytes': 8 * rf.GIB, 'logical_cpu_limit': 4})
        rf.write(self.ok / 'guard_verified.json', {'verified_before_GO': True,
                 'job_commit_limit_bytes': 8 * rf.GIB, 'logical_cpu_limit': 4})
        rf.write(self.ok / 'worker_result.json', {'status': 'rf_accept_complete', 'test_accessed': False, 'summary_sha256': 'SYNTHETIC'})

    def check(self):
        return recovery.accepted_attempt(self.run, self.original, 'SYNTHETIC')

    def test_preserved_failure_plus_one_success(self):
        self.assertEqual(self.check(), self.ok)

    def test_failed_log_changed_rejected(self):
        (self.failed / 'worker.log').write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError): self.check()

    def test_second_success_rejected(self):
        (self.run / 'monitor/ACCEPT-SYNTHETIC-EXTRA').mkdir()
        with self.assertRaises(ValueError): self.check()

    def test_summary_hash_mismatch_rejected(self):
        with self.assertRaises(ValueError): recovery.accepted_attempt(self.run, self.original, 'WRONG')

    def test_resource_abort_rejected(self):
        path = self.ok / 'outcome.json'
        value = rf.read(path); value['abort_reason'] = 'synthetic_low_memory'
        path.write_text(__import__('json').dumps(value), encoding='utf-8')
        with self.assertRaises(ValueError): self.check()

    def test_missing_guard_rejected(self):
        path = self.ok / 'guard_verified.json'
        value = rf.read(path); value['verified_before_GO'] = False
        path.write_text(__import__('json').dumps(value), encoding='utf-8')
        with self.assertRaises(ValueError): self.check()

    def test_baseline_allows_only_new_files(self):
        before = recovery.file_inventory(self.run)
        (self.run / 'new-recovery-record.json').write_text('{}')
        recovery.baseline_unchanged(self.run, before)

    def test_baseline_modified_original_rejected(self):
        before = recovery.file_inventory(self.run)
        (self.failed / 'worker.log').write_text('changed')
        with self.assertRaises(ValueError): recovery.baseline_unchanged(self.run, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
