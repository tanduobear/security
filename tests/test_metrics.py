import json
import unittest

from eval.metrics import compute_curves, tpr_at_fpr


class CurveTests(unittest.TestCase):
    def test_known_example(self):
        result = compute_curves(
            [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        )
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)

    def test_perfect_ranking(self):
        result = compute_curves([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["average_precision"], 1.0)

    def test_constant_scores(self):
        result = compute_curves([0, 0, 0, 1], [0.5] * 4)
        self.assertAlmostEqual(result["roc_auc"], 0.5)
        self.assertAlmostEqual(result["average_precision"], 0.25)

    def test_single_class(self):
        for labels in ([0, 0], [1, 1]):
            with self.subTest(labels=labels):
                result = compute_curves(labels, [0.1, 0.2])
                self.assertEqual(result["status"], "single_class")
                self.assertIsNone(result["roc_auc"])

    def test_invalid_inputs(self):
        cases = [
            ([], []),
            ([0, 1], [0.2]),
            ([0, 2], [0.1, 0.9]),
            ([0, 1], [0.1, float("nan")]),
            ([0, 1], [0.1, float("inf")]),
            ([[0, 1]], [[0.1, 0.9]]),
        ]
        for labels, scores in cases:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    compute_curves(labels, scores)

    def test_standard_json_serialization(self):
        result = compute_curves([0, 1], [0.1, 0.9])
        encoded = json.dumps(result, allow_nan=False)
        self.assertEqual(json.loads(encoded), result)


class OperatingPointTests(unittest.TestCase):
    def test_known_operating_points(self):
        result = tpr_at_fpr(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.35, 0.8],
            targets=[0, 0.5],
        )
        first, second = result["operating_points"]
        self.assertAlmostEqual(first["actual_fpr"], 0.0)
        self.assertAlmostEqual(first["tpr"], 0.5)
        self.assertAlmostEqual(first["threshold"], 0.8)
        self.assertAlmostEqual(second["actual_fpr"], 0.5)
        self.assertAlmostEqual(second["tpr"], 1.0)
        self.assertAlmostEqual(second["threshold"], 0.35)

    def test_tied_scores_are_not_split(self):
        result = tpr_at_fpr([0, 1], [0.5, 0.5], targets=[0.05])
        point = result["operating_points"][0]
        self.assertEqual(point["actual_fpr"], 0.0)
        self.assertEqual(point["tpr"], 0.0)
        self.assertEqual(point["threshold_kind"], "above_max_score")

    def test_single_class(self):
        result = tpr_at_fpr([0, 0], [0.1, 0.2])
        self.assertEqual(result["status"], "single_class")
        self.assertIsNone(result["operating_points"][0]["tpr"])

    def test_invalid_targets(self):
        for targets in ([], [-0.1], [1.1], [float("nan")]):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    tpr_at_fpr([0, 1], [0.1, 0.9], targets=targets)

from eval.metrics import per_class_recall


class PerClassTests(unittest.TestCase):
    def test_known_family_recalls(self):
        result = per_class_recall(
            [0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1],
            ["benign", "benign", "ddos", "ddos", "botnet"],
            families=["ddos", "botnet"],
        )
        ddos, botnet = result["classes"]

        self.assertEqual(ddos["support"], 2)
        self.assertEqual(ddos["detected"], 1)
        self.assertAlmostEqual(ddos["recall"], 0.5)
        self.assertEqual(botnet["support"], 1)
        self.assertAlmostEqual(botnet["recall"], 1.0)

    def test_absent_family_has_null_recall(self):
        result = per_class_recall(
            [0, 1],
            [0, 1],
            ["benign", "ddos"],
            families=["ddos", "heartbleed"],
        )
        missing = result["classes"][1]
        self.assertEqual(missing["support"], 0)
        self.assertEqual(missing["detected"], 0)
        self.assertIsNone(missing["recall"])
        json.dumps(result, allow_nan=False)

    def test_normal_only_sample(self):
        result = per_class_recall(
            [0, 0], [0, 1], ["benign", "benign"],
            families=["ddos"],
        )
        self.assertIsNone(result["classes"][0]["recall"])

    def test_inconsistent_family_metadata_rejected(self):
        with self.assertRaises(ValueError):
            per_class_recall([0, 1], [0, 1], ["ddos", "benign"])

    def test_invalid_inputs_rejected(self):
        cases = [
            ([0, 1], [0], ["benign", "ddos"], None),
            ([0, 1], [0, 2], ["benign", "ddos"], None),
            ([0, 1], [0, 1], ["benign", None], None),
            ([0, 1], [0, 1], ["benign", "ddos"], ["botnet"]),
            ([0, 1], [0, 1], ["benign", "ddos"], ["ddos", "ddos"]),
        ]
        for y, pred, family, names in cases:
            with self.subTest(case=(y, pred, family, names)):
                with self.assertRaises(ValueError):
                    per_class_recall(y, pred, family, families=names)


from unittest.mock import patch
import numpy as np

from eval.metrics import measure_inference


class InferenceTests(unittest.TestCase):
    def test_known_timing_and_call_count(self):
        calls = []

        def predict(batch):
            calls.append(len(batch))
            return batch.sum(axis=1)

        # Two measured durations: 4 ms and 8 ms. Median = 6 ms.
        with patch(
            "time.perf_counter_ns",
            side_effect=[0, 4_000_000, 10_000_000, 18_000_000],
        ):
            result = measure_inference(
                predict,
                np.ones((4, 3)),
                batch_size=2,
                warmup=1,
                repeats=2,
            )

        self.assertEqual(calls, [2, 2, 2])
        self.assertEqual(result["batch_times_ms"], [4.0, 8.0])
        self.assertAlmostEqual(result["median_batch_ms"], 6.0)
        self.assertAlmostEqual(result["amortized_ms_per_flow"], 3.0)
        self.assertAlmostEqual(
            result["throughput_flows_per_second"], 2000 / 6
        )
        json.dumps(result, allow_nan=False)

    def test_invalid_settings_rejected(self):
        X = np.ones((4, 3))
        cases = [
            {"batch_size": 0},
            {"batch_size": 5},
            {"warmup": -1},
            {"repeats": 0},
            {"threads": 0},
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    measure_inference(
                        lambda batch: batch.sum(axis=1), X, **settings
                    )

    def test_wrong_prediction_shape_rejected(self):
        with self.assertRaises(ValueError):
            measure_inference(
                lambda batch: np.zeros((2, 2)),
                np.ones((2, 3)),
                batch_size=2,
                warmup=1,
                repeats=1,
            )

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            measure_inference(
                lambda batch: batch.sum(axis=1),
                np.array([[float("nan")]]),
                batch_size=1,
            )

from eval.metrics import paired_bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.y = [0, 0, 1, 1]
        self.a = [0.1, 0.4, 0.35, 0.8]
        self.b = [0.2, 0.3, 0.8, 0.9]
        self.ids = ["r1", "r2", "r3", "r4"]

    def compare(self, **kwargs):
        return paired_bootstrap(
            self.y, self.a, self.b,
            row_ids_a=self.ids,
            row_ids_b=self.ids,
            **kwargs,
        )

    def test_identical_models_have_zero_difference(self):
        for metric in ("ap", "tpr_at_fpr"):
            with self.subTest(metric=metric):
                result = paired_bootstrap(
                    self.y, self.a, self.a,
                    row_ids_a=self.ids,
                    row_ids_b=self.ids,
                    metric=metric,
                    n_bootstrap=30,
                )
                self.assertEqual(result["difference"], 0.0)
                self.assertEqual(result["ci_lower"], 0.0)
                self.assertEqual(result["ci_upper"], 0.0)
                json.dumps(result, allow_nan=False)

    def test_fixed_seed_is_reproducible(self):
        first = self.compare(n_bootstrap=30, seed=42)
        second = self.compare(n_bootstrap=30, seed=42)
        self.assertEqual(first, second)

        # AP(A)=5/6; AP(B)=1, so A-B=-1/6.
        self.assertAlmostEqual(first["difference"], -1 / 6)
        self.assertLessEqual(first["ci_lower"], first["ci_upper"])

    def test_misaligned_or_duplicate_ids_rejected(self):
        for wrong_ids in (
            ["r2", "r1", "r3", "r4"],
            ["r1", "r1", "r3", "r4"],
        ):
            with self.subTest(ids=wrong_ids):
                with self.assertRaises(ValueError):
                    paired_bootstrap(
                        self.y, self.a, self.b,
                        row_ids_a=self.ids,
                        row_ids_b=wrong_ids,
                        n_bootstrap=10,
                    )

    def test_invalid_settings_rejected(self):
        for settings in (
            {"n_bootstrap": 1},
            {"confidence": 1.0},
            {"target_fpr": -0.1},
            {"metric": "unknown"},
            {"seed": -1},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self.compare(**settings)
from eval.metrics import classification_metrics


class ClassificationTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        result = classification_metrics(
            [0, 0, 1, 1], [0, 0, 0, 1]
        )
        self.assertEqual(
            result["confusion_matrix"],
            {"tn": 2, "fp": 0, "fn": 1, "tp": 1},
        )
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 11 / 15)
        self.assertAlmostEqual(result["tpr"], 0.5)
        self.assertAlmostEqual(result["fpr"], 0.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_zero_denominators(self):
        result = classification_metrics([0, 0], [0, 0])
        self.assertIsNone(result["tpr"])
        self.assertIsNone(result["precision"])
        self.assertEqual(result["fpr"], 0.0)
        self.assertFalse(result["both_true_classes_present"])
        self.assertAlmostEqual(result["macro_f1"], 0.5)

    def test_attempted_subgroup(self):
        result = classification_metrics(
            [0, 0, 1], [0, 1, 1],
            is_attempted=[False, True, False],
        )
        self.assertEqual(result["attempted"]["support"], 1)
        self.assertEqual(result["attempted"]["false_positives"], 1)
        self.assertEqual(result["attempted"]["fpr"], 1.0)

        empty = classification_metrics(
            [0, 1], [0, 1], is_attempted=[False, False]
        )
        self.assertEqual(empty["attempted"]["status"], "no_samples")
        self.assertIsNone(empty["attempted"]["fpr"])

    def test_invalid_inputs_rejected(self):
        cases = [
            ([], [], {}),
            ([0, 1], [0], {}),
            ([0, 2], [0, 1], {}),
            ([0, 1], [0, 2], {}),
            ([0, 1], [0, 1], {"is_attempted": [0, 1]}),
            ([0, 1], [0, 1], {"is_attempted": [False, True]}),
            ([0, 1], [0, 1], {"is_attempted": [False]}),
        ]
        for y, pred, options in cases:
            with self.subTest(case=(y, pred, options)):
                with self.assertRaises(ValueError):
                    classification_metrics(y, pred, **options)
if __name__ == "__main__":
    unittest.main()
