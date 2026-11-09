
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export M2 five-seed results to P3 single-model/single-seed JSON records.

Does NOT retrain models or run model inference.
Uses:
- results/m2.json
- results/preds_m2_corrected/predictions/*.npz
- data/split/original/stratified/test.parquet
- eval/schema.json (if available)

Run from project root:
    python methods/m2/export_p3_records.py
"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]

SOURCE_JSON = ROOT / "results" / "m2.json"
SCHEMA_JSON = ROOT / "eval" / "schema.json"

TEST_FILE = (
    ROOT / "data" / "split" / "original"
    / "stratified" / "test.parquet"
)

OUTPUT_DIR = ROOT / "results" / "p3_m2_records"

TARGET_FPRS = [0.01, 0.05]


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def calculate_row_hash(row_ids):
    content = "\n".join(row_ids.astype(str).tolist())
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "not_recorded"


def json_array(array):
    return np.asarray(array).tolist()


# --------------------------------------------------
# ROC and PR curves
# --------------------------------------------------


def calculate_curves(y_true, y_score):

    if len(np.unique(y_true)) < 2:
        return {
            "status": "single_class",
            "roc_auc": None,
            "average_precision": None,
            "roc": None,
            "pr": None,
        }

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        y_score,
        drop_intermediate=True,
    )

    # sklearn may use +inf as the first ROC threshold.
    # Replace it with the next representable float above max score.
    max_score = float(np.max(y_score))
    above_max = float(np.nextafter(max_score, np.inf))

    roc_thresholds = np.asarray(
        roc_thresholds,
        dtype=np.float64,
    )

    roc_thresholds[np.isposinf(roc_thresholds)] = above_max

    precision, recall, pr_thresholds = precision_recall_curve(
        y_true,
        y_score,
    )

    return {
        "status": "ok",

        "roc_auc": float(
            roc_auc_score(y_true, y_score)
        ),

        "average_precision": float(
            average_precision_score(y_true, y_score)
        ),

        "roc": {
            "fpr": json_array(fpr),
            "tpr": json_array(tpr),
            "thresholds": json_array(roc_thresholds),
        },

        "pr": {
            "precision": json_array(precision),
            "recall": json_array(recall),
            "thresholds": json_array(pr_thresholds),
        },
    }



# --------------------------------------------------
# Empirical ROC operating points
# --------------------------------------------------


def calculate_operating_points(y_true, y_score):

    if len(np.unique(y_true)) < 2:
        return {
            "status": "single_class",
            "protocol": "empirical_roc",
            "operating_points": [],
        }

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        drop_intermediate=False,
    )

    max_score = float(np.max(y_score))
    above_max = float(np.nextafter(max_score, np.inf))

    points = []

    for target in TARGET_FPRS:

        eligible = np.flatnonzero(fpr <= target)

        if len(eligible) == 0:
            idx = 0
        else:
            best_tpr = np.max(tpr[eligible])

            candidates = eligible[
                tpr[eligible] == best_tpr
            ]

            idx = int(candidates[-1])

        threshold = float(thresholds[idx])

        if np.isposinf(threshold):
            threshold = above_max
            threshold_kind = "above_max_score"
        elif np.isnan(threshold):
            raise ValueError("ROC threshold is NaN.")
        else:
            threshold_kind = "finite"

        points.append({
            "target_fpr": float(target),
            "actual_fpr": float(fpr[idx]),
            "tpr": float(tpr[idx]),
            "threshold": threshold,
            "threshold_kind": threshold_kind,
        })

    return {
        "status": "ok",
        "protocol": "empirical_roc",
        "operating_points": points,
    }


# --------------------------------------------------
# Classification metrics
# --------------------------------------------------

def calculate_classification(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    n_negative = int(np.sum(y_true == 0))
    n_positive = int(np.sum(y_true == 1))

    tpr = (
        float(tp / n_positive)
        if n_positive > 0 else 0.0
    )

    fpr = (
        float(fp / n_negative)
        if n_negative > 0 else 0.0
    )

    precision = (
        float(tp / (tp + fp))
        if (tp + fp) > 0 else 0.0
    )

    return {
        "status": "ok",

        "n_samples": int(len(y_true)),
        "n_negative": n_negative,
        "n_positive": n_positive,

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),

        "macro_f1_convention":
            "fixed_labels_0_1_zero_division_0",

        "both_true_classes_present": bool(
            n_negative > 0 and n_positive > 0
        ),

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,

        "attempted": {
            "status": "not_provided",
            "support": None,
            "false_positives": None,
            "fpr": None,
        },
    }


# --------------------------------------------------
# Per attack-family recall
# --------------------------------------------------

def calculate_per_class(y_true, y_pred, attack_family):

    attack_family = attack_family.astype(str)

    families = sorted(
        set(attack_family[y_true == 1])
    )

    classes = []

    for family in families:

        mask = (
            (y_true == 1)
            & (attack_family == family)
        )

        support = int(np.sum(mask))

        detected = int(
            np.sum(y_pred[mask] == 1)
        )

        recall = (
            float(detected / support)
            if support > 0 else 0.0
        )

        classes.append({
            "attack_family": family,
            "support": support,
            "detected": detected,
            "recall": recall,
        })

    return {
        "status": "ok",
        "classes": classes,
    }


# --------------------------------------------------
# Schema validation
# --------------------------------------------------

def validate_schema(record):

    if not SCHEMA_JSON.exists():
        print(
            "[WARN] eval/schema.json not found. "
            "Formal schema validation skipped."
        )
        return

    try:
        import jsonschema
    except ImportError:
        print(
            "[WARN] jsonschema not installed. "
            "Install with: pip install jsonschema"
        )
        return

    with open(SCHEMA_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)

    errors = list(validator.iter_errors(record))

    if errors:
        messages = []

        for error in errors[:10]:
            location = ".".join(
                str(x) for x in error.absolute_path
            )

            messages.append(
                f"{location}: {error.message}"
            )

        raise ValueError(
            "P3 Schema validation failed:\n"
            + "\n".join(messages)
        )


# --------------------------------------------------
# Main export
# --------------------------------------------------

def main():

    if not SOURCE_JSON.exists():
        raise FileNotFoundError(SOURCE_JSON)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        source = json.load(f)

    runs = source.get("runs", [])

    if len(runs) != 5:
        raise ValueError(
            f"Expected 5 seeds, found {len(runs)}"
        )

    dataset = source["dataset"]

    dataset_hash = sha256_file(TEST_FILE)

    expected_hash = dataset.get(
        "shared_test_row_ids_sha256"
    )

    for run in runs:
        if expected_hash and run.get("row_ids_sha256") != expected_hash:
            raise ValueError(
                f"Seed {run['seed']}: row ID hash mismatch."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)

    except Exception:
        torch_version = "not_available"
        cuda_version = "not_available"

    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pytorch": torch_version,
        "cuda": cuda_version,
    }

    success_count = 0

    for run in runs:

        seed = int(run["seed"])

        print(f"\nProcessing seed {seed}...")

        prediction_path = ROOT / Path(
            run["prediction_path"]
        )

        if not prediction_path.exists():
            raise FileNotFoundError(
                prediction_path
            )

        # No pickle required. NPZ must use numeric/Unicode arrays.
        with np.load(
            prediction_path,
            allow_pickle=False,
        ) as data:

            required = {
                "row_ids",
                "y_true",
                "y_score",
                "y_pred",
                "attack_family",
            }

            missing = required.difference(data.files)

            if missing:
                raise ValueError(
                    f"Seed {seed}: missing fields {missing}"
                )

            row_ids = np.asarray(
                data["row_ids"]
            ).astype(str)

            y_true = np.asarray(
                data["y_true"],
                dtype=np.int8,
            ).reshape(-1)

            y_score = np.asarray(
                data["y_score"],
                dtype=np.float64,
            ).reshape(-1)

            y_pred = np.asarray(
                data["y_pred"],
                dtype=np.int8,
            ).reshape(-1)

            attack_family = np.asarray(
                data["attack_family"]
            ).astype(str).reshape(-1)

        lengths = {
            len(row_ids),
            len(y_true),
            len(y_score),
            len(y_pred),
            len(attack_family),
        }

        if len(lengths) != 1:
            raise ValueError(
                f"Seed {seed}: array lengths differ."
            )

        if not np.isfinite(y_score).all():
            raise ValueError(
                f"Seed {seed}: y_score contains NaN/Inf."
            )

        if not set(np.unique(y_true)).issubset({0, 1}):
            raise ValueError(
                f"Seed {seed}: invalid y_true labels."
            )

        if not set(np.unique(y_pred)).issubset({0, 1}):
            raise ValueError(
                f"Seed {seed}: invalid y_pred labels."
            )

        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError(
                f"Seed {seed}: duplicate row IDs."
            )

        row_hash = calculate_row_hash(row_ids)

        if run.get("row_ids_sha256") != row_hash:
            raise ValueError(
                f"Seed {seed}: recomputed row hash mismatch."
            )

        # Reconstruct metrics from saved predictions.
        curves = calculate_curves(
            y_true,
            y_score,
        )

        classification = calculate_classification(
            y_true,
            y_pred,
        )

        operating = calculate_operating_points(
            y_true,
            y_score,
        )

        per_class = calculate_per_class(
            y_true,
            y_pred,
            attack_family,
        )

        # Preserve the exact recorded configuration.
        config = dict(
            source.get("training_config", {})
        )

        config["input_features"] = dataset.get(
            "input_features"
        )

        config["model"] = source.get(
            "model",
            "CNN1D",
        )

        # Copy actual measured CPU latency.
        latency = run.get("latency_cpu")

        if not latency:
            raise ValueError(
                f"Seed {seed}: latency_cpu missing."
            )

        required_latency = [
            "status",
            "device",
            "scope",
            "batch_size",
            "warmup",
            "repeats",
            "threadpool_limit",
            "median_batch_ms",
            "amortized_ms_per_flow",
            "throughput_flows_per_second",
            "batch_times_ms",
        ]

        missing_latency = [
            k for k in required_latency
            if k not in latency
        ]

        if missing_latency:
            raise ValueError(
                f"Seed {seed}: missing latency fields "
                f"{missing_latency}"
            )

        inference = {
            k: latency[k]
            for k in required_latency
        }

        record = {

            "schema_version": "1.0",

            "is_demo": False,

            "method": "1D-CNN",

            "dataset": {
                "version": dataset["version"],
                "protocol": dataset["protocol"],
                "variant": dataset["variant"],
                "split": "test",
                "path": str(
                    TEST_FILE.relative_to(ROOT)
                ).replace("\\", "/"),
                "sha256": dataset_hash,
                "n_samples": int(len(y_true)),
            },

            "seed": seed,

            "config": config,

            "decision_rule": {
                "score_direction": "higher_means_attack",
                "operator": ">=",
                "threshold": float(run["threshold"]),
                "source": "validation_selected",
            },

            "metrics": {
                "curves": curves,
                "classification": classification,
                "operating_points": operating,
                "per_class": per_class,
            },

            "cost": {
                "training_seconds": float(
                    run["training_seconds"]
                ),

                "training_time_note":
                    "Measured training duration from existing M2 training record.",

                "hardware": str(
                    source.get(
                        "hardware", {}
                    ).get(
                        "gpu",
                        "GPU not recorded",
                    )
                ),

                "thread_settings": {
                    "inference_threads": int(
                        source.get(
                            "evaluation_config", {}
                        ).get(
                            "latency_threads",
                            1,
                        )
                    )
                },

                "inference": inference,
            },

            "daily_alerts": None,

            "comparisons": [],

            "provenance": {
                "code_commit": get_git_commit(),
                "created_at": created_at,
                "software": software,
                "notes": (
                    "Exported from existing M2 results and saved "
                    "per-sample predictions. No retraining or "
                    "test-time model inference was performed. "
                    "ROC/PR curves and classification metrics "
                    "were reconstructed from saved predictions."
                ),
            },
        }

        # Validate before writing.
        validate_schema(record)

        output_file = (
            OUTPUT_DIR / f"m2_seed{seed}.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                record,
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            f.write("\n")

        print(
            f"[OK] Seed {seed} exported: "
            f"{output_file.relative_to(ROOT)}"
        )

        success_count += 1

    print("\n" + "=" * 60)
    print("M2 P3 EXPORT FINISHED")
    print("=" * 60)

    print(f"Successfully exported: {success_count}/5")

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.relative_to(ROOT)}"
    )

    print(
        "Original results/m2.json was NOT modified."
    )


if __name__ == "__main__":
    main()
