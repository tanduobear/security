"""Validate result structure and cross-field numerical consistency."""

import argparse
import json
import math
import statistics
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def validate_result(record):
    # Reject nonstandard JSON numbers, including nested NaN and Infinity.
    json.dumps(record, allow_nan=False)

    schema = json.loads(
        (ROOT / "eval/schema.json").read_text(encoding="utf-8-sig")
    )
    Draft202012Validator(schema).validate(record)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def equal(actual, expected, name):
        if expected is None:
            require(actual is None, f"{name}: expected null")
        else:
            require(
                actual is not None
                and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10),
                f"{name}: inconsistent value",
            )

    def ratio(a, b):
        return a / b if b else None

    c = record["metrics"]["classification"]
    cm = c["confusion_matrix"]
    tn, fp, fn, tp = (cm[k] for k in ("tn", "fp", "fn", "tp"))
    n = tn + fp + fn + tp

    require(n == c["n_samples"] == record["dataset"]["n_samples"],
            "Sample counts disagree")
    require(c["n_negative"] == tn + fp, "Negative count disagrees")
    require(c["n_positive"] == tp + fn, "Positive count disagrees")

    equal(c["accuracy"], (tn + tp) / n, "accuracy")
    equal(c["tpr"], ratio(tp, tp + fn), "tpr")
    equal(c["fpr"], ratio(fp, fp + tn), "fpr")
    equal(c["precision"], ratio(tp, tp + fp), "precision")

    f1_normal = ratio(2 * tn, 2 * tn + fp + fn)
    f1_attack = ratio(2 * tp, 2 * tp + fp + fn)
    equal(c["macro_f1"],
          ((f1_normal or 0) + (f1_attack or 0)) / 2, "macro_f1")

    both = bool((tn + fp) and (tp + fn))
    require(c["both_true_classes_present"] == both,
            "Class-presence flag disagrees")
    expected_status = "ok" if both else "single_class"
    require(record["metrics"]["curves"]["status"] == expected_status,
            "Curve status disagrees with class counts")

    curves = record["metrics"]["curves"]
    operating = record["metrics"]["operating_points"]
    require(operating["status"] == expected_status,
            "Operating-point status disagrees")

    if both:
        require(
            curves["roc_auc"] is not None
            and curves["average_precision"] is not None
            and curves["roc"] is not None
            and curves["pr"] is not None,
            "Two-class evaluation requires curve summaries and arrays",
        )

        roc = curves["roc"]
        xs, ys, thresholds = roc["fpr"], roc["tpr"], roc["thresholds"]
        require(len(xs) == len(ys) == len(thresholds) and len(xs) >= 2,
                "ROC array lengths disagree")
        require(xs[0] == ys[0] == 0 and xs[-1] == ys[-1] == 1,
                "ROC endpoints are invalid")
        require(
            all(a <= b for a, b in zip(xs, xs[1:]))
            and all(a <= b for a, b in zip(ys, ys[1:])),
            "ROC rates must be nondecreasing",
        )
        require(
            thresholds[0] is None
            and all(t is not None for t in thresholds[1:]),
            "ROC threshold sentinel is invalid",
        )
        require(
            all(a > b for a, b in zip(thresholds[1:], thresholds[2:])),
            "Finite ROC thresholds must strictly decrease",
        )

        area = sum(
            (x1 - x0) * (y0 + y1) / 2
            for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:])
        )
        equal(curves["roc_auc"], area, "ROC-AUC from saved curve")

        pr = curves["pr"]
        precision, recall = pr["precision"], pr["recall"]
        require(
            len(precision) == len(recall) == len(pr["thresholds"]) + 1
            and len(recall) >= 2,
            "PR array lengths disagree",
        )
        require(
            recall[0] == 1 and recall[-1] == 0
            and all(a >= b for a, b in zip(recall, recall[1:])),
            "PR recall order or endpoints are invalid",
        )
        require(
            pr["thresholds"] == list(reversed(thresholds[1:])),
            "ROC and PR thresholds disagree",
        )
        ap = sum(
            (r0 - r1) * p
            for r0, r1, p in zip(recall, recall[1:], precision)
        )
        equal(curves["average_precision"], ap, "AP from saved curve")

        for point in operating["operating_points"]:
            eligible = [
                i for i, value in enumerate(xs)
                if value <= point["target_fpr"]
            ]
            index = max(eligible, key=lambda i: ys[i])
            equal(point["actual_fpr"], xs[index], "Operating-point FPR")
            equal(point["tpr"], ys[index], "Operating-point TPR")
            equal(point["threshold"], thresholds[index],
                  "Operating-point threshold")
            expected_kind = (
                "above_max_score" if thresholds[index] is None else "finite"
            )
            require(point["threshold_kind"] == expected_kind,
                    "Operating-point threshold kind disagrees")
    else:
        require(
            all(curves[k] is None
                for k in ("roc_auc", "average_precision", "roc", "pr")),
            "Single-class curve fields must be null",
        )
        for point in operating["operating_points"]:
            require(
                all(point[k] is None
                    for k in ("actual_fpr", "tpr", "threshold"))
                and point["threshold_kind"] == "undefined",
                "Single-class operating points must be undefined",
            )

    attempted = c["attempted"]
    if attempted["status"] == "not_provided":
        require(all(attempted[k] is None
                    for k in ("support", "false_positives", "fpr")),
                "Missing attempted metadata must have null values")
    else:
        support = attempted["support"]
        false = attempted["false_positives"]
        require(support is not None and false is not None,
                "Attempted counts are missing")
        require(0 <= false <= support <= c["n_negative"] and false <= fp,
                "Attempted subgroup counts are impossible")
        require(attempted["status"] == ("ok" if support else "no_samples"),
                "Attempted status disagrees")
        equal(attempted["fpr"], ratio(false, support), "attempted FPR")

    classes = record["metrics"]["per_class"]["classes"]
    names = [item["attack_family"] for item in classes]
    require(len(names) == len(set(names)) and "benign" not in names,
            "Duplicate or benign attack-family entries")
    require(sum(item["support"] for item in classes) == c["n_positive"],
            "Attack-family supports do not cover all attacks")
    require(sum(item["detected"] for item in classes) == tp,
            "Attack-family detections disagree with TP")

    for item in classes:
        require(item["detected"] <= item["support"],
                "Detected count exceeds family support")
        equal(item["recall"], ratio(item["detected"], item["support"]),
              f"Recall for {item['attack_family']}")

    for point in record["metrics"]["operating_points"]["operating_points"]:
        if both:
            require(point["actual_fpr"] is not None and point["tpr"] is not None,
                    "Operating-point rates are missing")
            require(point["actual_fpr"] <= point["target_fpr"],
                    "Operating point exceeds its FPR constraint")

    timing = record["cost"]["inference"]
    require(len(timing["batch_times_ms"]) == timing["repeats"],
            "Timing repeat count disagrees")
    median = statistics.median(timing["batch_times_ms"])
    equal(timing["median_batch_ms"], median, "median batch latency")
    equal(timing["amortized_ms_per_flow"],
          median / timing["batch_size"], "amortized latency")
    equal(timing["throughput_flows_per_second"],
          timing["batch_size"] * 1000 / median, "throughput")

    daily = record["daily_alerts"]
    if daily is not None:
        equal(daily["fpr"], c["fpr"], "daily FPR")
        equal(daily["tpr"], c["tpr"], "daily TPR")
        false = daily["assumed_benign_flows"] * daily["fpr"]
        true = daily["assumed_attack_flows"] * daily["tpr"]
        equal(daily["expected_false_alerts"], false, "daily false alerts")
        equal(daily["expected_true_alerts"], true, "daily true alerts")
        equal(daily["expected_total_alerts"], false + true, "daily total alerts")

    for comparison in record["comparisons"]:
        result = comparison["result"]
        require(
            result["n_samples"] == n
            and result["n_negative"] == c["n_negative"]
            and result["n_positive"] == c["n_positive"],
            "Comparison uses different sample counts",
        )
        equal(result["difference"], result["value_a"] - result["value_b"],
              "Comparison difference")
        require(result["ci_lower"] <= result["ci_upper"],
                "Confidence interval is reversed")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=Path)
    args = parser.parse_args()
    record = json.loads(args.result_file.read_text(encoding="utf-8-sig"))
    validate_result(record)
    print("Result structure and numerical consistency: OK")


if __name__ == "__main__":
    main()
