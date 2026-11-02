"""Synthetic-only PR checks; no local experiment archive access."""
from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from methods.m1 import pr_curve_data as plot


class PRCurveTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array(['synthetic-a', 'synthetic-b', 'synthetic-c', 'synthetic-d'])
        self.y = np.array([0, 1, 0, 1], dtype=np.uint8)
        self.scores = np.array([.1, .9, .2, .8], dtype=float)
        self.digest = plot.row_digest(self.ids)

    def curve(self, *, ids=None, y=None, scores=None, expected_ap=1., reference=None):
        return plot.verified_curve(self.ids if ids is None else ids, self.y if y is None else y,
                                   self.scores if scores is None else scores, expected_rows=4,
                                   expected_row_digest=self.digest, expected_ap=expected_ap, reference=reference)

    def test_uses_full_labels_scores_and_sklearn_curve(self):
        with patch('sklearn.metrics.precision_recall_curve', wraps=precision_recall_curve) as compute:
            result = self.curve()
        np.testing.assert_array_equal(compute.call_args.args[0], self.y)
        np.testing.assert_array_equal(compute.call_args.args[1], self.scores)
        self.assertEqual(compute.call_args.kwargs, {'pos_label': 1})
        self.assertEqual(result['average_precision'], 1.)
        self.assertEqual(result['rows'], 4)
        self.assertEqual(result['attack_rows'], 2)

    def test_ap_mismatch_stops(self):
        with self.assertRaisesRegex(ValueError, 'AP mismatch'): self.curve(expected_ap=.9)

    def test_duplicate_row_ids_stop(self):
        with self.assertRaisesRegex(ValueError, 'row IDs'): self.curve(ids=np.array(['a', 'a', 'c', 'd']))

    def test_changed_row_order_digest_stops(self):
        with self.assertRaisesRegex(ValueError, 'digest'): self.curve(ids=self.ids[::-1])

    def test_wrong_sample_count_stops(self):
        with self.assertRaisesRegex(ValueError, 'sample count'): self.curve(scores=self.scores[:3])

    def test_cross_seed_labels_must_match_exactly(self):
        with self.assertRaisesRegex(ValueError, 'Cross-seed'): self.curve(reference=(self.ids, self.y[::-1]))

    def test_cross_seed_order_must_match_exactly(self):
        with self.assertRaisesRegex(ValueError, 'Cross-seed'): self.curve(reference=(self.ids[::-1], self.y))

    def test_nonfinite_scores_stop(self):
        with self.assertRaisesRegex(ValueError, 'scores'): self.curve(scores=np.array([.1, np.nan, .2, .8]))

    def test_hard_labels_cannot_replace_continuous_scores(self):
        with self.assertRaisesRegex(ValueError, 'Hard prediction'): self.curve(scores=self.y.astype(float))

    def test_invalid_label_class_stops(self):
        with self.assertRaisesRegex(ValueError, 'binary labels'): self.curve(y=np.array([0, 2, 0, 1]))

    def test_unsafe_archive_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'): plot.safe_path(ROOT, '../test.npz')



if __name__ == '__main__':
    unittest.main(verbosity=2)
