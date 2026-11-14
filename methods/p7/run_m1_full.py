"""Approved, isolated P7 M1 corrected-dataset full rerun.

The default action is read-only planning.  Training requires both an approved
content-addressed execution record and the explicit ``--confirm-start`` flag.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from methods.m1.data import (DiskDataset, _materialize_loaded, assert_disk_isolated,
                             materialize_preparation, require, row_digest)
from methods.m1.evaluation import family_coverage, score_metrics
from methods.m1.executor import metadata_arrays, predict_batches
from methods.m1.models import fit_tree
from methods.m1.selection import calibrate_fpr
from methods.m1.storage import sha
from src.model_loader import ModelDataset


CONFIG = ROOT / "methods/p7/m1_corrected.json"
APPROVAL = ROOT / "methods/p7/execution.m1-corrected.v2.json"
PROPOSAL = ROOT / "methods/p7/execution.m1-corrected.proposed.v2.json"
LEGACY_APPROVAL = ROOT / "methods/p7/execution.m1-corrected.v1.json"
RUNNER = ROOT / "methods/p7/run_m1_full.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def approval_payload(record):
    return {key: value for key, value in record.items() if key != "approval"}


def issues(config, record):
    found = []
    expected_models = {
        "G05": {"algorithm": "HistGradientBoostingClassifier", "params": {
            "max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 1.0, "max_bins": 255,
            "early_stopping": False, "class_weight": None}},
        "RF_FIXED": {"algorithm": "RandomForestClassifier", "params": {
            "n_estimators": 200, "max_depth": 20, "max_leaf_nodes": 4096,
            "min_samples_leaf": 2, "max_features": "sqrt", "bootstrap": True,
            "n_jobs": 4, "class_weight": None}},
    }
    checks = [
        (config.get("dataset") == {"version": "corrected", "protocol": "stratified", "variant": "main", "features": 140}, "dataset contract changed"),
        (config.get("seeds") == [42, 43, 44, 45, 46], "seed contract changed"),
        (config.get("models") == expected_models, "model parameters changed"),
        (config.get("thresholds") == {"source_split": "val", "fpr_targets": [0.01, 0.05], "boundary_rule": "strict_nextafter_benign_boundary"}, "threshold contract changed"),
        (config.get("output_root") == "results/p7_corrected/m1", "output isolation changed"),
        (record.get("stage") == "p7_m1_corrected_full_rerun", "approval stage changed"),
        (record.get("config_sha256") == sha(CONFIG), "configuration hash differs"),
        (record.get("runner_sha256") == sha(RUNNER), "runner hash differs"),
        (record.get("dataset") == config.get("dataset"), "approval/config dataset mismatch"),
        (record.get("seeds") == config.get("seeds"), "approval/config seeds mismatch"),
        (record.get("models") == config.get("models"), "approval/config model mismatch"),
        (record.get("test_policy") == "open_test_only_after_all_validation_predictions_and_thresholds_are_frozen", "test gate changed"),
    ]
    found.extend(message for ok, message in checks if not ok)
    approval = record.get("approval") or {}
    if approval.get("status") != "approved":
        found.append("execution record is not approved")
    if not approval.get("source") or not approval.get("approved_at"):
        found.append("approval source/date missing")
    if approval.get("payload_sha256") != digest(approval_payload(record)):
        found.append("approved payload hash differs")
    return found


def plan():
    config = read(CONFIG)
    record_path = APPROVAL if APPROVAL.exists() else PROPOSAL if PROPOSAL.exists() else LEGACY_APPROVAL
    record = read(record_path)
    payload_hash = digest(approval_payload(record))
    result = {
        "status": "approved_ready" if not issues(config, record) else "blocked_pending_approval",
        "scope": "plan_only_no_dataset_opened_no_training",
        "config": str(CONFIG.relative_to(ROOT)),
        "execution_record": str(record_path.relative_to(ROOT)),
        "payload_sha256": payload_hash,
        "models": ["G05", "RF_FIXED"],
        "seeds": config["seeds"],
        "fits": 10,
        "output_root": config["output_root"],
        "issues": issues(config, record),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "approved_ready" else 2


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not path.exists() and not temporary.exists(), f"Refusing to overwrite {path}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not path.exists() and not temporary.exists(), f"Refusing to overwrite {path}")
    with temporary.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def atomic_model(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not path.exists() and not temporary.exists(), f"Refusing to overwrite {path}")
    with temporary.open("xb") as stream:
        joblib.dump(model, stream)
    os.replace(temporary, path)


def open_or_build_cache(source, directory):
    ready_path = directory / "READY.json"
    if ready_path.exists():
        ready = read(ready_path)
        return DiskDataset(directory, tuple(ready["features"]), int(ready["rows"]), ready["split"], {int(k): v for k, v in ready["class_weights"].items()})
    require(not directory.exists(), f"Incomplete cache exists: {directory}")
    if Path(source).name == "test.parquet":
        # This branch is called only after validation_frozen.json has been written.
        return _materialize_loaded(ModelDataset(source), source, directory)
    return materialize_preparation(source, directory)


def job_identity(model_id, seed):
    return f"{model_id}_seed_{seed}"


def run(run_id):
    config = read(CONFIG)
    require(APPROVAL.exists(), "Approved execution record is missing; proposal files cannot authorize training")
    record = read(APPROVAL)
    found = issues(config, record)
    require(not found, "Formal execution blocked: " + "; ".join(found))

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
    require(run_id.replace("-", "").replace("_", "").isalnum(), "Invalid run ID")
    run_dir = ROOT / config["output_root"] / "formal" / run_id
    require(not run_dir.exists(), "Run directory already exists; choose a new --run-id")
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "run.json", {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "status": "running"})
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "execution_approval.json", record)

    try:
        data_dir = ROOT / "data/split/corrected/stratified"
        cache_dir = run_dir / "cache"
        train_cache = open_or_build_cache(data_dir / "train.parquet", cache_dir / "train")
        val_cache = open_or_build_cache(data_dir / "val.parquet", cache_dir / "val")
        assert_disk_isolated(train_cache, val_cache, memory_limit_mb=512)
        require(train_cache.feature_names == val_cache.feature_names and len(train_cache.feature_names) == 140, "Feature contract mismatch")
        val_ids, val_family, val_attempted = metadata_arrays(val_cache)
        families = [x for x in read(ROOT / "docs/label_split_policy.json")["family_names"] if x != "benign"]
        jobs = []
        with train_cache.model_input() as training, val_cache.model_input() as validation:
            for model_id, spec in config["models"].items():
                algorithm = "hist_gradient_boosting" if model_id == "G05" else "random_forest"
                for seed in config["seeds"]:
                    name = job_identity(model_id, seed)
                    print(f"Training {name}", flush=True)
                    model, seconds = fit_tree(algorithm, spec["params"], training, model_seed=seed, fit_threads=4)
                    scores = predict_batches(model, validation.X, rows=8192, threads=1)
                    model_path = run_dir / "models" / f"{name}.joblib"
                    scores_path = run_dir / "validation_scores" / f"{name}.npy"
                    atomic_model(model_path, model)
                    atomic_npy(scores_path, scores)
                    restored = joblib.load(model_path)
                    require(np.array_equal(scores, predict_batches(restored, validation.X, rows=8192, threads=1)), "Reloaded model predictions differ")
                    thresholds = [calibrate_fpr(validation.y, scores, alpha) for alpha in config["thresholds"]["fpr_targets"]]
                    job = {"task_id": name, "model_id": model_id, "seed": seed, "algorithm": spec["algorithm"], "params": spec["params"],
                           "training_seconds": seconds, "model_path": model_path.relative_to(run_dir).as_posix(), "model_sha256": sha(model_path),
                           "validation_scores_path": scores_path.relative_to(run_dir).as_posix(), "validation_scores_sha256": sha(scores_path),
                           "validation_row_ids_sha256": row_digest(val_ids), "thresholds": thresholds}
                    atomic_json(run_dir / "jobs" / f"{name}.json", job)
                    jobs.append(job)
                    del model, restored, scores
            atomic_json(run_dir / "validation_frozen.json", {"jobs": jobs, "test_accessed": False,
                "validation_rows": len(validation.y), "validation_row_ids_sha256": row_digest(val_ids),
                "validation_coverage": family_coverage(validation.y, val_family, families)})

        # The real test loader is constructed only after every model and threshold is frozen.
        test_cache = open_or_build_cache(data_dir / "test.parquet", cache_dir / "test")
        test_ids, test_family, test_attempted = metadata_arrays(test_cache)
        metadata = pd.DataFrame({"attack_family": test_family, "is_attempted": test_attempted})
        records = []
        with test_cache.model_input() as test:
            test_coverage = family_coverage(test.y, test_family, families)
            for job in jobs:
                model = joblib.load(run_dir / job["model_path"])
                scores = predict_batches(model, test.X, rows=8192, threads=1)
                score_path = run_dir / "test_scores" / f"{job['task_id']}.npy"
                atomic_npy(score_path, scores)
                for threshold in job["thresholds"]:
                    key = f"{job['task_id']}_fpr_{threshold['target_fpr']}"
                    metrics = score_metrics(test.y, scores, metadata, threshold["threshold"], families)
                    result = {"schema_version": "1", "method": "M1_GBDT" if job["model_id"] == "G05" else "M1_RF_auxiliary",
                              "dataset": {**config["dataset"], "split": "test", "rows": len(test.y)}, "seed": job["seed"],
                              "model_id": job["model_id"], "params": job["params"], "threshold_calibration": threshold,
                              "metrics": metrics, "training_seconds": job["training_seconds"], "test_row_ids_sha256": row_digest(test_ids),
                              "model_sha256": job["model_sha256"], "test_scores_sha256": sha(score_path)}
                    atomic_json(run_dir / "records" / f"{key}.json", result)
                    records.append({"path": f"records/{key}.json", "method": result["method"], "seed": job["seed"], "target_validation_fpr": threshold["target_fpr"]})
                del model, scores
        index = {"status": "complete", "records": records, "test_rows": test_cache.rows, "test_row_ids_sha256": row_digest(test_ids),
                 "test_coverage": test_coverage, "completed_at": datetime.now(timezone.utc).isoformat(),
                 "environment": {"python": platform.python_version(), "platform": platform.platform()},
                 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        atomic_json(run_dir / "evaluation_index.json", index)
        files = {p.relative_to(run_dir).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in run_dir.rglob("*") if p.is_file()}
        atomic_json(run_dir / "MANIFEST.json", {"status": "p7_m1_corrected_full_rerun_complete", "files": files})
        print(json.dumps({"status": "complete", "run": str(run_dir.relative_to(ROOT)), "records": len(records)}, indent=2))
        return 0
    except Exception as error:
        atomic_json(run_dir / "FAILED.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--confirm-start", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.action == "plan":
        return plan()
    require(args.confirm_start is True, "Full rerun requires the literal --confirm-start flag")
    return run(args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
