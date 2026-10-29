"""Synthetic integration example. Not an experimental model result."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from .metrics import (
    classification_metrics,
    compute_curves,
    measure_inference,
    paired_bootstrap,
    per_class_recall,
    tpr_at_fpr,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "eval" / "examples"

FIXTURE = {
    "row_ids": [f"demo:{i}" for i in range(8)],
    "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
    "scores_a": [0.1, 0.4, 0.2, 0.6, 0.35, 0.8, 0.7, 0.9],
    "scores_b": [0.2, 0.3, 0.1, 0.4, 0.6, 0.9, 0.7, 0.8],
    "attack_family": [
        "benign", "benign", "benign", "benign",
        "ddos", "ddos", "botnet", "botnet",
    ],
    "is_attempted": [False, True, False, False, False, False, False, False],
}


def fixture_bytes():
    return (
        json.dumps(FIXTURE, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def build_demo():
    y = np.asarray(FIXTURE["y_true"])
    scores = np.asarray(FIXTURE["scores_a"])
    threshold = 0.5
    predictions = (scores >= threshold).astype(int)

    classification = classification_metrics(
        y, predictions, is_attempted=FIXTURE["is_attempted"]
    )

    # A synchronous CPU toy scorer, not a trained detection model.
    X = scores.reshape(-1, 1)

    def toy_predict(batch):
        return batch[:, 0].copy()

    inference = measure_inference(
        toy_predict, X, batch_size=4, warmup=2, repeats=5, threads=1
    )

    comparisons = []
    for metric in ("ap", "tpr_at_fpr"):
        comparisons.append({
            "method_a": "synthetic_A",
            "method_b": "synthetic_B",
            "seed_a": 42,
            "seed_b": 42,
            "result": paired_bootstrap(
                y, scores, FIXTURE["scores_b"],
                row_ids_a=FIXTURE["row_ids"],
                row_ids_b=FIXTURE["row_ids"],
                metric=metric,
                target_fpr=0.01,
                n_bootstrap=50,
                seed=42,
            ),
        })

    # Explicit hypothetical daily volumes at the SAME fixed threshold.
    benign_flows = 100000
    attack_flows = 100
    fpr = classification["fpr"]
    tpr = classification["tpr"]
    false_alerts = benign_flows * fpr
    true_alerts = attack_flows * tpr

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    commit_id = commit.stdout.strip() if commit.returncode == 0 else "unavailable"

    file_hashes = {
        name: hashlib.sha256(
            (ROOT / "eval" / name).read_bytes()
        ).hexdigest()
        for name in ("metrics.py", "schema.json", "demo.py")
    }

    return {
        "schema_version": "1.0",
        "is_demo": True,
        "method": "synthetic_A",
        "dataset": {
            "version": "synthetic",
            "protocol": "synthetic",
            "variant": "synthetic",
            "split": "test",
            "path": "eval/examples/synthetic_input.json",
            "sha256": hashlib.sha256(fixture_bytes()).hexdigest(),
            "n_samples": len(y),
        },
        "seed": 42,
        "config": {
            "model": "none; manually specified synthetic scores",
            "bootstrap_repetitions_for_demo": 50,
        },
        "decision_rule": {
            "score_direction": "higher_means_attack",
            "operator": ">=",
            "threshold": threshold,
            "source": "synthetic_demo",
        },
        "metrics": {
            "curves": compute_curves(y, scores),
            "classification": classification,
            "operating_points": tpr_at_fpr(y, scores),
            "per_class": per_class_recall(
                y, predictions, FIXTURE["attack_family"],
                families=["ddos", "botnet", "heartbleed"],
            ),
        },
        "cost": {
            "training_seconds": None,
            "training_time_note": "No model was trained; synthetic demo only.",
            "hardware": f"{platform.system()} / {platform.machine()} / "
                        f"{platform.processor() or 'CPU details unavailable'}",
            "thread_settings": {
                "supported_blas_openmp_limit": 1,
                "framework": "NumPy toy scorer only",
            },
            "inference": inference,
        },
        "daily_alerts": {
            "assumed_benign_flows": benign_flows,
            "assumed_attack_flows": attack_flows,
            "fpr": fpr,
            "tpr": tpr,
            "expected_false_alerts": false_alerts,
            "expected_true_alerts": true_alerts,
            "expected_total_alerts": false_alerts + true_alerts,
        },
        "comparisons": comparisons,
        "provenance": {
            "code_commit": commit_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in (
                        "numpy", "scikit-learn", "jsonschema", "threadpoolctl"
                    )
                },
            },
            "notes": (
                "SYNTHETIC DEMO ONLY. Timings describe array copying, not a "
                "trained detector. Daily volumes are hypothetical. The Git "
                "commit may exclude uncommitted files; current file hashes: "
                + json.dumps(file_hashes, sort_keys=True)
            ),
        },
    }


def main():
    schema = json.loads(
        (ROOT / "eval" / "schema.json").read_text(encoding="utf-8-sig")
    )
    Draft202012Validator.check_schema(schema)

    result = build_demo()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    Draft202012Validator(schema).validate(json.loads(encoded))

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    (EXAMPLES / "synthetic_input.json").write_bytes(fixture_bytes())
    (EXAMPLES / "result_demo.json").write_bytes(encoded.encode("utf-8"))

    print("Demo result: schema validation passed")
    print("Saved: eval/examples/synthetic_input.json")
    print("Saved: eval/examples/result_demo.json")
    print("SYNTHETIC DEMO ONLY - not experimental findings")


if __name__ == "__main__":
    main()
