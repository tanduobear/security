"""Shared evaluation utilities: normal=0, attack=1."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_curves(y_true, y_score):
    """Compute ROC/PR curves. Higher scores must mean more suspicious."""
    y = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)

    # 1. Reject invalid inputs.
    if y.ndim != 1 or scores.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y) == 0 or len(y) != len(scores):
        raise ValueError("Inputs must be nonempty and have equal lengths.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must not contain NaN or infinity.")

    # 2. A one-class sample cannot support a full ROC comparison.
    if np.unique(y).size < 2:
        return {
            "status": "single_class",
            "roc_auc": None,
            "average_precision": None,
            "roc": None,
            "pr": None,
        }

    # 3. Compute curves without dropping intermediate ROC points.
    fpr, tpr, roc_thresholds = roc_curve(
        y, scores, pos_label=1, drop_intermediate=False
    )
    precision, recall, pr_thresholds = precision_recall_curve(
        y, scores, pos_label=1
    )

    # 4. Convert NumPy values into JSON-compatible Python values.
    return {
        "status": "ok",
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": [
                float(t) if np.isfinite(t) else None
                for t in roc_thresholds
            ],
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }

def tpr_at_fpr(y_true, y_score, targets=(0.001, 0.01, 0.05)):
    """Describe empirical ROC operating points; do not calibrate deployment."""
    targets = np.asarray(targets, dtype=float)

    if (
        targets.ndim != 1
        or targets.size == 0
        or not np.isfinite(targets).all()
        or ((targets < 0) | (targets > 1)).any()
    ):
        raise ValueError("FPR targets must be a nonempty 1D sequence in [0, 1].")

    curves = compute_curves(y_true, y_score)

    if curves["status"] != "ok":
        return {
            "status": curves["status"],
            "protocol": "empirical_roc",
            "operating_points": [
                {
                    "target_fpr": float(target),
                    "actual_fpr": None,
                    "tpr": None,
                    "threshold": None,
                    "threshold_kind": "undefined",
                }
                for target in targets
            ],
        }

    fpr = np.asarray(curves["roc"]["fpr"])
    tpr = np.asarray(curves["roc"]["tpr"])
    thresholds = curves["roc"]["thresholds"]
    points = []

    for target in targets:
        # All selected points must satisfy the FPR constraint.
        eligible = np.flatnonzero(fpr <= target)

        # Maximize TPR; the first tied point has the lowest FPR.
        index = int(eligible[np.argmax(tpr[eligible])])
        threshold = thresholds[index]

        points.append({
            "target_fpr": float(target),
            "actual_fpr": float(fpr[index]),
            "tpr": float(tpr[index]),
            "threshold": threshold,
            "threshold_kind": (
                "above_max_score" if threshold is None else "finite"
            ),
        })

    return {
        "status": "ok",
        "protocol": "empirical_roc",
        "operating_points": points,
    }

def per_class_recall(y_true, y_pred, attack_family, families=None):
    """Measure binary attack detection recall within each true attack family."""
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    family = np.asarray(attack_family, dtype=object)

    if any(a.ndim != 1 for a in (y, pred, family)):
        raise ValueError("Inputs must be one-dimensional.")

    if len(y) == 0 or not (len(y) == len(pred) == len(family)):
        raise ValueError("Inputs must be nonempty and have equal lengths.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("True labels must contain only 0 and 1.")

    if not np.isin(pred, [0, 1]).all():
        raise ValueError("Predictions must contain only 0 and 1.")

    if not all(isinstance(f, str) and f.strip() for f in family):
        raise ValueError("Each attack family must be a nonempty string.")

    # Follow the project's mapping: benign=0; actual attack families=1.
    if np.any((family == "benign") != (y == 0)):
        raise ValueError("Attack-family metadata disagrees with binary labels.")

    observed = set(family[y == 1].tolist())

    if families is None:
        names = sorted(observed)
    else:
        if isinstance(families, str):
            raise ValueError("families must be a sequence, not one string.")
        names = list(families)

        if not all(
            isinstance(f, str) and f.strip() and f != "benign"
            for f in names
        ):
            raise ValueError("families must contain valid attack-family names.")

        if len(names) != len(set(names)):
            raise ValueError("Duplicate attack-family names.")

        if not observed.issubset(set(names)):
            raise ValueError("families omits an observed attack family.")

    results = []

    for name in names:
        mask = (y == 1) & (family == name)
        support = int(mask.sum())
        detected = int((pred[mask] == 1).sum())

        results.append({
            "attack_family": name,
            "support": support,
            "detected": detected,
            "recall": detected / support if support else None,
        })

    return {
        "status": "ok",
        "classes": results,
    }

def measure_inference(
    predict_fn, X, batch_size=512, warmup=5, repeats=30, threads=1
):
    """Benchmark synchronous CPU prediction returning one NumPy score per row."""
    from time import perf_counter_ns
    from threadpoolctl import threadpool_limits

    if not callable(predict_fn):
        raise ValueError("predict_fn must be callable.")

    settings = {
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "threads": threads,
    }
    for name, value in settings.items():
        minimum = 0 if name == "warmup" else 1
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"Invalid {name}.")

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0 or len(X) < batch_size:
        raise ValueError("X must be 2D and contain at least one full batch.")

    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("X must contain finite numeric values.")

    # Prepare input before timing; do not include slicing/copying in latency.
    batch = np.ascontiguousarray(X[:batch_size])

    def check_output(output):
        if not isinstance(output, np.ndarray):
            raise ValueError("Prediction must return a materialized NumPy array.")
        if output.shape != (batch_size,):
            raise ValueError("Prediction must return one score per input row.")
        if output.dtype.kind not in "fiu" or not np.isfinite(output).all():
            raise ValueError("Prediction scores must be finite numeric values.")

    elapsed_ms = []

    # Limits supported BLAS/OpenMP pools, not every framework's own workers.
    with threadpool_limits(limits=int(threads)):
        for _ in range(warmup):
            check_output(predict_fn(batch))

        for _ in range(repeats):
            start = perf_counter_ns()
            output = predict_fn(batch)
            stop = perf_counter_ns()

            # Validate outside the timed interval.
            check_output(output)
            duration = (stop - start) / 1_000_000
            if duration <= 0:
                raise RuntimeError("Nonpositive duration; use a larger workload.")
            elapsed_ms.append(duration)

    median_ms = float(np.median(elapsed_ms))

    return {
        "status": "ok",
        "device": "cpu",
        "scope": "prepared_batch_to_numpy_scores",
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "threadpool_limit": int(threads),
        "median_batch_ms": median_ms,
        "amortized_ms_per_flow": median_ms / batch_size,
        "throughput_flows_per_second": batch_size * 1000 / median_ms,
        "batch_times_ms": elapsed_ms,
    }

def paired_bootstrap(
    y_true, scores_a, scores_b, *,
    row_ids_a, row_ids_b,
    metric="ap", target_fpr=0.01,
    n_bootstrap=2000, confidence=0.95, seed=42
):
    """Paired, class-stratified percentile bootstrap for metric A minus B."""
    # Reuse input validation. Both classes are required.
    result_a = compute_curves(y_true, scores_a)
    result_b = compute_curves(y_true, scores_b)

    if result_a["status"] != "ok" or result_b["status"] != "ok":
        raise ValueError("Bootstrap comparison requires both binary classes.")

    y = np.asarray(y_true)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    ids_a = np.asarray(row_ids_a, dtype=object)
    ids_b = np.asarray(row_ids_b, dtype=object)

    for ids in (ids_a, ids_b):
        if ids.ndim != 1 or len(ids) != len(y):
            raise ValueError("Row IDs must be 1D and match the sample count.")
        if not all(isinstance(x, str) and x.strip() for x in ids):
            raise ValueError("Row IDs must be nonempty strings.")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("Row IDs must be unique.")

    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Models must use identical row IDs in identical order.")

    if metric not in ("ap", "tpr_at_fpr"):
        raise ValueError("metric must be 'ap' or 'tpr_at_fpr'.")

    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, (int, np.integer))
        or n_bootstrap < 2
    ):
        raise ValueError("n_bootstrap must be an integer of at least 2.")

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")

    if not np.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be in [0, 1].")

    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer.")

    def evaluate(labels, scores):
        if metric == "ap":
            return float(average_precision_score(labels, scores))

        fpr, tpr, _ = roc_curve(
            labels, scores, pos_label=1, drop_intermediate=False
        )
        return float(np.max(tpr[fpr <= target_fpr]))

    point_a = evaluate(y, a)
    point_b = evaluate(y, b)

    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap, dtype=float)

    for iteration in range(n_bootstrap):
        # The SAME indices are used for A and B.
        indices = np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        differences[iteration] = (
            evaluate(y[indices], a[indices])
            - evaluate(y[indices], b[indices])
        )

    tail = (1 - confidence) / 2
    lower, upper = np.quantile(
        differences, [tail, 1 - tail], method="linear"
    )

    return {
        "status": "ok",
        "metric": metric,
        "target_fpr": float(target_fpr) if metric == "tpr_at_fpr" else None,
        "difference_definition": "A_minus_B",
        "value_a": point_a,
        "value_b": point_b,
        "difference": point_a - point_b,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "resampling": "paired_class_stratified_rows",
        "interval_method": "percentile",
        "n_samples": int(len(y)),
        "n_negative": int(len(negative)),
        "n_positive": int(len(positive)),
    }

def classification_metrics(y_true, y_pred, *, is_attempted=None):
    """Summarize binary decisions at one externally chosen operating point."""
    from sklearn.metrics import confusion_matrix, f1_score

    y = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if y.ndim != 1 or pred.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y) == 0 or len(y) != len(pred):
        raise ValueError("Inputs must be nonempty and have equal lengths.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("True labels must contain only 0 and 1.")

    if not np.isin(pred, [0, 1]).all():
        raise ValueError("Predictions must contain only 0 and 1.")

    tn, fp, fn, tp = [
        int(value)
        for value in confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ]

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    if is_attempted is None:
        attempted = {
            "status": "not_provided",
            "support": None,
            "false_positives": None,
            "fpr": None,
        }
    else:
        mask = np.asarray(is_attempted)

        if mask.ndim != 1 or len(mask) != len(y) or mask.dtype.kind != "b":
            raise ValueError("is_attempted must be a matching 1D boolean array.")

        if np.any(mask & (y != 0)):
            raise ValueError("Project policy requires attempted samples to be benign.")

        support = int(mask.sum())
        false_positives = int((pred[mask] == 1).sum())

        attempted = {
            "status": "ok" if support else "no_samples",
            "support": support,
            "false_positives": false_positives,
            "fpr": ratio(false_positives, support),
        }

    return {
        "status": "ok",
        "n_samples": int(len(y)),
        "n_negative": tn + fp,
        "n_positive": tp + fn,
        "accuracy": (tp + tn) / len(y),
        "macro_f1": float(f1_score(
            y, pred, labels=[0, 1], average="macro", zero_division=0
        )),
        "macro_f1_convention": "fixed_labels_0_1_zero_division_0",
        "both_true_classes_present": bool((tn + fp) and (tp + fn)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "tpr": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
        "precision": ratio(tp, tp + fp),
        "attempted": attempted,
    }
