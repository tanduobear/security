"""No real data, fits or model inference; exercise final-result acceptance rules."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'methods/p7/collect_results.py'
spec = importlib.util.spec_from_file_location('p7_collection', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metric_fixture():
    return {'average_precision': .8, 'roc_auc': .9,
            'tpr_at_fpr_1pct': .5, 'tpr_at_fpr_5pct': .75,
            'classification': {'accuracy': .8, 'macro_f1': .8, 'fpr': .25,
                               'tpr': .875, 'precision': .7777777777777778,
                               'n_samples': 20,
                               'confusion_matrix': {'tn': 9, 'fp': 3, 'fn': 1, 'tp': 7}},
            'per_class_recall': {'classes': [{'attack_family': 'seen', 'support': 8, 'recall': .875},
                                           {'attack_family': 'absent', 'support': 0, 'recall': None}]}}


class CollectionRules(unittest.TestCase):
    def search_fixture(self):
        config = {'candidates': [{'id': f'A{i:02}', 'hidden': [i, 1]} for i in range(1, 9)]}
        search = {'status': 'validation_search_complete', 'is_smoke': False,
                  'test_accessed': False, 'selected_candidate': 'A01',
                  'runs': [{'candidate': c, 'seed': seed, 'best_benign_validation_mse': i}
                           for i, c in enumerate(config['candidates']) for seed in module.SEEDS]}
        return config, search

    def test_search_winner_must_follow_validation_scores(self):
        config, search = self.search_fixture()
        module.validate_m3_search(search, config)
        search['selected_candidate'] = 'A08'
        with self.assertRaisesRegex(ValueError, 'winner'):
            module.validate_m3_search(search, config)

    def test_smoke_cannot_become_formal_result(self):
        config, search = self.search_fixture()
        search['is_smoke'] = True
        with self.assertRaisesRegex(ValueError, 'formal search'):
            module.validate_m3_search(search, config)

    def test_duplicate_search_task_rejected_even_when_count_is_40(self):
        config, search = self.search_fixture()
        search['runs'][-1] = search['runs'][0]
        with self.assertRaisesRegex(ValueError, '40 unique'):
            module.validate_m3_search(search, config)

    def test_missing_or_duplicate_seed_rejected(self):
        for values in [[42, 43, 44, 45], [42, 43, 44, 45, 45]]:
            with self.assertRaises(ValueError):
                module.five([{'seed': s} for s in values], 'fixture')

    def test_absent_family_remains_null(self):
        result = module.metrics(metric_fixture())
        rows = [{'method': 'M3', 'version': 'corrected', 'seed': seed,
                 'configuration': 'A06', **result} for seed in module.SEEDS]
        group = module.aggregate(rows)[0]
        self.assertIsNone(group['family']['absent']['mean'])
        self.assertEqual(group['family']['absent']['support'], 0)

    def test_inconsistent_confusion_counts_rejected(self):
        values = metric_fixture()
        values['classification']['fpr'] = .01
        with self.assertRaises(ValueError):
            module.metrics(values)

    def test_partial_null_seed_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            module.mean_std([None, .1, .2, .3, .4])

    def test_sample_standard_deviation(self):
        result = module.mean_std([1, 2, 3, 4, 5])
        self.assertEqual(result['mean'], 3)
        self.assertAlmostEqual(result['sample_std'], 1.5811388300841898)

    def test_no_corrected_results_means_no_final_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'completed M1'):
                module.corrected(root)
            self.assertFalse((root / 'results/summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
