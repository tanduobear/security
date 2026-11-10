
import os
import sys
import json
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.m2.data import get_dataset
from methods.m2.model import CNN1D

from eval.metrics import (
    compute_curves,
    tpr_at_fpr,
    classification_metrics,
    per_class_recall,
    measure_inference,
)


# ============================================================
# 2. Utility functions
# ============================================================

def to_builtin(obj):
    """Convert NumPy values and nested structures to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, torch.Tensor):
        return to_builtin(obj.detach().cpu().tolist())

    return obj


def get_metric_value(data, possible_names, description):
    """Read a metric using one of several supported field names."""
    for name in possible_names:
        if isinstance(data, dict) and name in data:
            return data[name]

    raise KeyError(
        f"Cannot find {description}. "
        f"Available fields: {list(data.keys()) if isinstance(data, dict) else type(data)}"
    )


def get_operating_point(result, target_fpr):
    """Find an empirical ROC operating point."""
    if result.get("status") != "ok":
        raise ValueError(
            f"Cannot calculate operating point: {result.get('status')}"
        )

    for point in result["operating_points"]:
        if np.isclose(point["target_fpr"], target_fpr):
            return point

    raise ValueError(
        f"Operating point for FPR={target_fpr} not found."
    )


def numeric_summary(values):
    """Calculate mean and sample standard deviation."""
    valid = [
        float(v)
        for v in values
        if v is not None and np.isfinite(float(v))
    ]

    if not valid:
        return {"mean": None, "std": None}

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
    }


# ============================================================
# 3. Prediction
# ============================================================

def predict_scores(model, dataset, device, batch_size=4096):
    """Predict sigmoid probabilities in batches."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    outputs = []

    with torch.inference_mode():
        for X, _, _ in loader:
            X = X.to(
                device,
                dtype=torch.float32,
                non_blocking=(device.type == "cuda"),
            )

            logits = model(X)
            scores = torch.sigmoid(logits)

            outputs.append(scores.cpu().numpy())

    if not outputs:
        raise ValueError("Prediction dataset is empty.")

    return np.concatenate(outputs).reshape(-1)


def predict_cpu(model, X):
    """CPU prediction function used for latency measurement."""

    # measure_inference passes NumPy arrays
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    # Ensure CPU float32 tensor
    X = X.to(
        device="cpu",
        dtype=torch.float32
    )

    model.eval()

    with torch.inference_mode():
        logits = model(X)
        scores = torch.sigmoid(logits)

    # Return NumPy array for the benchmark
    return scores.cpu().numpy().reshape(-1)


# ============================================================
# 4. Evaluate one seed
# ============================================================

def evaluate_one_seed(
    seed,
    cfg,
    train_summary,
    val_ds,
    test_ds,
    device,
):
    print("\n" + "=" * 65)
    print(f"Evaluating Seed {seed}")
    print("=" * 65)

    model_path = (
        Path(cfg["output"]["model_dir"])
        / f"model_seed{seed}.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = CNN1D(val_ds.X.shape[1]).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # --------------------------------------------------------
    # Validation predictions
    # Threshold is selected ONLY on validation data
    # --------------------------------------------------------

    print(f"\nSeed {seed}: predicting validation set...")

    val_scores = predict_scores(
        model,
        val_ds,
        device,
    )

    val_y = val_ds.y.cpu().numpy().astype(int)

    val_operating = tpr_at_fpr(
        val_y,
        val_scores,
        targets=(0.01, 0.05),
    )

    threshold_info = get_operating_point(
        val_operating,
        0.01,
    )

    threshold = threshold_info["threshold"]

    # None means the selected ROC point is above the maximum score.
    if threshold is None:
        threshold = float(
            np.nextafter(np.max(val_scores), np.inf)
        )

    threshold = float(threshold)

    print(f"Validation-selected threshold: {threshold:.10f}")

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    print(f"\nSeed {seed}: predicting test set...")

    test_scores = predict_scores(
        model,
        test_ds,
        device,
    )

    test_y = test_ds.y.cpu().numpy().astype(int)

    if len(test_scores) != len(test_y):
        raise ValueError("Test prediction and label lengths do not match.")

    test_pred = (test_scores >= threshold).astype(int)

    # --------------------------------------------------------
    # ROC-AUC and Average Precision
    # --------------------------------------------------------

    curves = compute_curves(
        test_y,
        test_scores,
    )

    if curves.get("status") != "ok":
        raise ValueError(
            f"Test ROC calculation failed: {curves.get('status')}"
        )

    roc_auc = get_metric_value(
        curves,
        ["roc_auc", "roc_auc_score", "auc_roc"],
        "ROC-AUC",
    )

    average_precision = get_metric_value(
        curves,
        ["average_precision", "ap", "avg_precision"],
        "Average Precision",
    )

    # --------------------------------------------------------
    # Test empirical operating points
    # --------------------------------------------------------

    test_operating = tpr_at_fpr(
        test_y,
        test_scores,
        targets=(0.01, 0.05),
    )

    test_point_1 = get_operating_point(
        test_operating,
        0.01,
    )

    test_point_5 = get_operating_point(
        test_operating,
        0.05,
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    cls_metrics = classification_metrics(
        test_y,
        test_pred,
    )

    family_recall = per_class_recall(
        test_y,
        test_pred,
        test_ds.meta["attack_family"].tolist(),
    )

    # --------------------------------------------------------
    # CPU single-thread inference latency
    # --------------------------------------------------------

    print(f"\nSeed {seed}: measuring CPU latency...")

    cpu_model = CNN1D(test_ds.X.shape[1]).cpu()

    cpu_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    cpu_model.load_state_dict(cpu_state)
    cpu_model.eval()

    cpu_X = test_ds.X.cpu()

    latency = measure_inference(
        lambda X: predict_cpu(cpu_model, X),
        cpu_X,
        batch_size=512,
        warmup=5,
        repeats=30,
        threads=1,
    )

    # --------------------------------------------------------
    # Training time
    # --------------------------------------------------------

    training_record = next(
        (
            item for item in train_summary["results"]
            if int(item["seed"]) == int(seed)
        ),
        None,
    )

    if training_record is None:
        raise ValueError(
            f"Training time for seed {seed} not found."
        )

    training_seconds = float(
        training_record["training_seconds"]
    )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = {
        "seed": int(seed),

        "model_path": str(model_path),

        "threshold": threshold,
        "threshold_source": "validation",
        "validation_operating_point_1pct": threshold_info,

        "test": {
            "roc_auc": roc_auc,
            "average_precision": average_precision,

            "tpr_at_fpr_1pct": test_point_1["tpr"],
            "actual_fpr_at_1pct": test_point_1["actual_fpr"],

            "tpr_at_fpr_5pct": test_point_5["tpr"],
            "actual_fpr_at_5pct": test_point_5["actual_fpr"],

            "classification": cls_metrics,
            "per_class_recall": family_recall,
        },

        "latency_cpu": latency,

        "training_seconds": training_seconds,
    }

    result = to_builtin(result)

    print("\nSeed evaluation results")
    print("-----------------------")
    print("Seed:", seed)
    print("ROC-AUC:", roc_auc)
    print("Average Precision:", average_precision)
    print("TPR@1% FPR:", test_point_1["tpr"])
    print("TPR@5% FPR:", test_point_5["tpr"])
    print("Training time:", round(training_seconds, 2), "seconds")

    del model
    del cpu_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 5. Main evaluation
# ============================================================

def main():

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    config_path = Path("methods/m2/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(cfg["output"]["model_dir"])
    summary_path = model_dir / "training_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        train_summary = json.load(f)

    seeds = [int(s) for s in train_summary["seeds"]]

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    requested_device = cfg.get("device", "cuda")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    device = torch.device(requested_device)

    torch.set_num_threads(1)

    print("Evaluation device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Load validation and test datasets
    # --------------------------------------------------------

    data_cfg = cfg["data"]

    print("\nLoading validation dataset...")

    val_ds = get_dataset(
        data_cfg["version"],
        data_cfg["protocol"],
        data_cfg["variant"],
        "val",
    )

    print("\nLoading test dataset...")

    test_ds = get_dataset(
        data_cfg["version"],
        data_cfg["protocol"],
        data_cfg["variant"],
        "test",
    )

    print("\nDataset information")
    print("-------------------")
    print("Validation samples:", len(val_ds))
    print("Test samples:", len(test_ds))
    print("Input features:", val_ds.X.shape[1])

    if val_ds.X.shape[1] != test_ds.X.shape[1]:
        raise ValueError(
            "Validation and test feature dimensions do not match."
        )

    if len(test_ds) < 512:
        raise ValueError(
            "Test dataset has fewer than 512 samples. "
            "Cannot use the configured latency batch size."
        )

    # --------------------------------------------------------
    # Evaluate all seeds
    # --------------------------------------------------------

    all_results = []

    for seed in seeds:

        result = evaluate_one_seed(
            seed=seed,
            cfg=cfg,
            train_summary=train_summary,
            val_ds=val_ds,
            test_ds=test_ds,
            device=device,
        )

        all_results.append(result)

    # --------------------------------------------------------
    # Aggregate metrics across seeds
    # --------------------------------------------------------

    metric_getters = {
        "roc_auc": lambda r: r["test"]["roc_auc"],

        "average_precision": lambda r: (
            r["test"]["average_precision"]
        ),

        "tpr_at_fpr_1pct": lambda r: (
            r["test"]["tpr_at_fpr_1pct"]
        ),

        "tpr_at_fpr_5pct": lambda r: (
            r["test"]["tpr_at_fpr_5pct"]
        ),

        "accuracy": lambda r: (
            r["test"]["classification"]["accuracy"]
        ),

        "macro_f1": lambda r: (
            r["test"]["classification"]["macro_f1"]
        ),

        "training_seconds": lambda r: (
            r["training_seconds"]
        ),

        "cpu_latency_ms_per_flow": lambda r: (
            get_metric_value(
                r["latency_cpu"],
                [
                    "amortized_ms_per_flow",
                    "median_ms_per_flow",
                    "ms_per_flow",
                ],
                "CPU latency per flow",
            )
        ),

        "cpu_throughput_flows_per_second": lambda r: (
            get_metric_value(
                r["latency_cpu"],
                [
                    "throughput_flows_per_second",
                    "flows_per_second",
                    "throughput",
                ],
                "CPU throughput",
            )
        ),
    }

    aggregate = {}

    for metric_name, getter in metric_getters.items():

        values = []

        for result in all_results:
            try:
                values.append(getter(result))
            except KeyError:
                raise

        aggregate[metric_name] = numeric_summary(values)

    # --------------------------------------------------------
    # Build final JSON
    # --------------------------------------------------------

    output = {
        "experiment_name": "M2_CNN1D",
        "schema_version": 1,
        "model": "CNN1D",

        "seeds": seeds,

        "training_config": {
            "epochs": 40,
            "batch_size": 256,
            "learning_rate": 0.001,
            "pos_weight": train_summary["pos_weight"],
            "device": train_summary["device"],
            "use_amp": True,
        },

        "hardware": {
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
        },

        "evaluation_config": {
            "evaluation_device": str(device),
            "threshold_selection": "validation",
            "latency_device": "CPU",
            "latency_threads": 1,
            "latency_batch_size": 512,
            "latency_warmup": 5,
            "latency_repeats": 30,
        },

        "dataset": {
            "version": data_cfg["version"],
            "protocol": data_cfg["protocol"],
            "variant": data_cfg["variant"],
            "validation_samples": len(val_ds),
            "test_samples": len(test_ds),
            "input_features": int(val_ds.X.shape[1]),
        },

        "runs": all_results,

        "summary": aggregate,
    }

    output = to_builtin(output)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output_path = Path("results/m2.json")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("M2 EVALUATION COMPLETED")
    print("=" * 65)

    print("\nMean ± Std across seeds")

    for name, stats in aggregate.items():
        print(
            f"{name}: "
            f"{stats['mean']} ± {stats['std']}"
        )

    print("\nOutput file:", output_path.resolve())


if __name__ == "__main__":
    main()
