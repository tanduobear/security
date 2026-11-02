import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from eval.demo import build_demo


ROOT = Path(__file__).resolve().parents[1]


class ResultSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(
            (ROOT / "eval/schema.json").read_text(encoding="utf-8-sig")
        )
        cls.validator = Draft202012Validator(cls.schema)
        cls.example = build_demo()

    def test_generated_demo_is_valid(self):
        Draft202012Validator.check_schema(self.schema)
        encoded = json.dumps(self.example, allow_nan=False)
        self.validator.validate(json.loads(encoded))

    def test_missing_required_field_rejected(self):
        broken = copy.deepcopy(self.example)
        del broken["seed"]
        with self.assertRaises(ValidationError):
            self.validator.validate(broken)

    def test_out_of_range_metric_rejected(self):
        broken = copy.deepcopy(self.example)
        broken["metrics"]["classification"]["accuracy"] = 1.5
        with self.assertRaises(ValidationError):
            self.validator.validate(broken)

    def test_synthetic_record_cannot_be_relabelled_as_real(self):
        broken = copy.deepcopy(self.example)
        broken["is_demo"] = False
        with self.assertRaises(ValidationError):
            self.validator.validate(broken)


from eval.validate import validate_result


class ResultConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = build_demo()

    def test_valid_result_passes(self):
        self.assertTrue(validate_result(self.example))

    def test_inconsistent_sample_count_rejected(self):
        broken = copy.deepcopy(self.example)
        broken["dataset"]["n_samples"] += 1
        with self.assertRaises(ValueError):
            validate_result(broken)

    def test_incorrect_family_recall_rejected(self):
        broken = copy.deepcopy(self.example)
        broken["metrics"]["per_class"]["classes"][0]["recall"] = 0.9
        with self.assertRaises(ValueError):
            validate_result(broken)

    def test_inconsistent_totals_and_costs_rejected(self):
        for case in ("daily_alerts", "latency", "interval"):
            with self.subTest(case=case):
                broken = copy.deepcopy(self.example)
                if case == "daily_alerts":
                    broken["daily_alerts"]["expected_total_alerts"] += 1
                elif case == "latency":
                    broken["cost"]["inference"]["median_batch_ms"] *= 2
                else:
                    result = broken["comparisons"][0]["result"]
                    result["ci_lower"] = 0.5
                    result["ci_upper"] = -0.5

                with self.assertRaises(ValueError):
                    validate_result(broken)


class CurveIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = build_demo()

    def test_missing_or_inconsistent_curve_summary_rejected(self):
        for value in (None, 0.1):
            with self.subTest(value=value):
                broken = copy.deepcopy(self.example)
                broken["metrics"]["curves"]["roc_auc"] = value
                with self.assertRaises(ValueError):
                    validate_result(broken)

    def test_wrong_operating_threshold_rejected(self):
        broken = copy.deepcopy(self.example)
        points = broken["metrics"]["operating_points"]["operating_points"]
        points[0]["threshold"] = 999.0
        with self.assertRaises(ValueError):
            validate_result(broken)
if __name__ == "__main__":
    unittest.main()
