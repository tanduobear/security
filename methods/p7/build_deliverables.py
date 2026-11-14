"""Build P7's auditable cross-method summary and final figures.

Only metrics present in the repository are plotted.  The corrected full rerun
was reported as successful on another computer, but its metric files have not
been copied into this checkout; corrected metric cells therefore remain blank.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = [42, 43, 44, 45, 46]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values):
    values = [float(value) for value in values]
    return statistics.mean(values), statistics.stdev(values)


def metric_at_fpr(record, target):
    points = record["metrics"]["operating_points"]["operating_points"]
    return next(float(point["tpr"]) for point in points if float(point["target_fpr"]) == target)


def class_map(record, key_path):
    value = record
    for key in key_path:
        value = value[key]
    return {item["attack_family"]: float(item["recall"]) for item in value["classes"]}


def save_figure(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    m1 = load_json(RESULTS / "m1_test_summary" / "summary.json")
    m2 = load_json(RESULTS / "m2.json")
    m3 = load_json(RESULTS / "m3_summary" / "summary.json")

    m1_summary = next(
        item
        for item in m1["five_seed_summaries"]
        if item["method"] == "M1_GBDT" and item["target_validation_fpr"] == 0.01
    )
    m1_records = [
        item
        for item in m1["per_seed_threshold_records"]
        if item["method"] == "M1_GBDT" and item["config"]["target_validation_fpr"] == 0.01
    ]
    m3_summary = next(item for item in m3["five_seed_summary"] if item["quantile"] == 0.99)
    m3_records = [
        item
        for item in m3["records"]
        if item["method"] == "Autoencoder" and item["config"]["benign_quantile"] == 0.99
    ]

    m1_accuracy = mean_std(item["metrics"]["classification"]["accuracy"] for item in m1_records)
    m3_training = mean_std(item["cost"]["training_seconds"] for item in m3_records)
    m3_latency = mean_std(
        item["cost"]["inference"]["amortized_ms_per_flow"] for item in m3_records
    )

    tpr = {
        "GBDT G05": {
            0.01: mean_std(metric_at_fpr(item, 0.01) for item in m1_records),
            0.05: mean_std(metric_at_fpr(item, 0.05) for item in m1_records),
        },
        "1D-CNN": {
            0.01: (
                float(m2["summary"]["tpr_at_fpr_1pct"]["mean"]),
                float(m2["summary"]["tpr_at_fpr_1pct"]["std"]),
            ),
            0.05: (
                float(m2["summary"]["tpr_at_fpr_5pct"]["mean"]),
                float(m2["summary"]["tpr_at_fpr_5pct"]["std"]),
            ),
        },
        "Autoencoder A06": {
            0.01: mean_std(metric_at_fpr(item, 0.01) for item in m3_records),
            0.05: mean_std(metric_at_fpr(item, 0.05) for item in m3_records),
        },
    }

    original_rows = [
        {
            "method": "GBDT G05",
            "role": "main_supervised_tree",
            "dataset_version": "original",
            "protocol": "stratified",
            "variant": "main",
            "evaluation_split": "test",
            "seed_count": 5,
            "configuration": "G05",
            "threshold_policy": "validation benign quantile; 1% target for classification columns",
            "average_precision_mean": m1_summary["ap"]["mean"],
            "average_precision_std": m1_summary["ap"]["sample_std"],
            "roc_auc_mean": m1_summary["roc_auc"]["mean"],
            "roc_auc_std": m1_summary["roc_auc"]["sample_std"],
            "tpr_at_empirical_fpr_1pct_mean": tpr["GBDT G05"][0.01][0],
            "tpr_at_empirical_fpr_1pct_std": tpr["GBDT G05"][0.01][1],
            "tpr_at_empirical_fpr_5pct_mean": tpr["GBDT G05"][0.05][0],
            "tpr_at_empirical_fpr_5pct_std": tpr["GBDT G05"][0.05][1],
            "fixed_threshold_test_fpr_mean": m1_summary["test_fpr"]["mean"],
            "fixed_threshold_test_fpr_std": m1_summary["test_fpr"]["sample_std"],
            "fixed_threshold_test_tpr_mean": m1_summary["test_tpr"]["mean"],
            "fixed_threshold_test_tpr_std": m1_summary["test_tpr"]["sample_std"],
            "accuracy_mean": m1_accuracy[0],
            "accuracy_std": m1_accuracy[1],
            "macro_f1_mean": m1_summary["macro_f1"]["mean"],
            "macro_f1_std": m1_summary["macro_f1"]["sample_std"],
            "training_seconds_mean": m1_summary["training_seconds"]["mean"],
            "training_seconds_std": m1_summary["training_seconds"]["sample_std"],
            "cpu_latency_ms_per_flow_mean": "",
            "cpu_latency_ms_per_flow_std": "",
            "run_status": "formal_complete",
            "source_file": "results/m1_test_summary/summary.json",
            "notes": "Continuous-score AP/AUC and empirical ROC points use full Test; classification uses frozen Validation threshold.",
        },
        {
            "method": "1D-CNN",
            "role": "main_supervised_deep",
            "dataset_version": "original",
            "protocol": "stratified",
            "variant": "main",
            "evaluation_split": "test",
            "seed_count": 5,
            "configuration": "delivered fixed CNN1D configuration",
            "threshold_policy": "validation-selected 1% FPR threshold",
            "average_precision_mean": m2["summary"]["average_precision"]["mean"],
            "average_precision_std": m2["summary"]["average_precision"]["std"],
            "roc_auc_mean": m2["summary"]["roc_auc"]["mean"],
            "roc_auc_std": m2["summary"]["roc_auc"]["std"],
            "tpr_at_empirical_fpr_1pct_mean": tpr["1D-CNN"][0.01][0],
            "tpr_at_empirical_fpr_1pct_std": tpr["1D-CNN"][0.01][1],
            "tpr_at_empirical_fpr_5pct_mean": tpr["1D-CNN"][0.05][0],
            "tpr_at_empirical_fpr_5pct_std": tpr["1D-CNN"][0.05][1],
            "fixed_threshold_test_fpr_mean": "",
            "fixed_threshold_test_fpr_std": "",
            "fixed_threshold_test_tpr_mean": "",
            "fixed_threshold_test_tpr_std": "",
            "accuracy_mean": m2["summary"]["accuracy"]["mean"],
            "accuracy_std": m2["summary"]["accuracy"]["std"],
            "macro_f1_mean": m2["summary"]["macro_f1"]["mean"],
            "macro_f1_std": m2["summary"]["macro_f1"]["std"],
            "training_seconds_mean": m2["summary"]["training_seconds"]["mean"],
            "training_seconds_std": m2["summary"]["training_seconds"]["std"],
            "cpu_latency_ms_per_flow_mean": m2["summary"]["cpu_latency_ms_per_flow"]["mean"],
            "cpu_latency_ms_per_flow_std": m2["summary"]["cpu_latency_ms_per_flow"]["std"],
            "run_status": "formal_complete",
            "source_file": "results/m2.json",
            "notes": "M2 operating points follow its delivered evaluator; P6 recorded unresolved P3 tie-handling differences.",
        },
        {
            "method": "Autoencoder A06",
            "role": "main_unsupervised",
            "dataset_version": "original",
            "protocol": "stratified",
            "variant": "main",
            "evaluation_split": "test",
            "seed_count": 5,
            "configuration": "A06",
            "threshold_policy": "99th percentile benign Validation reconstruction error",
            "average_precision_mean": m3_summary["average_precision"]["mean"],
            "average_precision_std": m3_summary["average_precision"]["sample_std"],
            "roc_auc_mean": m3_summary["roc_auc"]["mean"],
            "roc_auc_std": m3_summary["roc_auc"]["sample_std"],
            "tpr_at_empirical_fpr_1pct_mean": tpr["Autoencoder A06"][0.01][0],
            "tpr_at_empirical_fpr_1pct_std": tpr["Autoencoder A06"][0.01][1],
            "tpr_at_empirical_fpr_5pct_mean": tpr["Autoencoder A06"][0.05][0],
            "tpr_at_empirical_fpr_5pct_std": tpr["Autoencoder A06"][0.05][1],
            "fixed_threshold_test_fpr_mean": m3_summary["fpr"]["mean"],
            "fixed_threshold_test_fpr_std": m3_summary["fpr"]["sample_std"],
            "fixed_threshold_test_tpr_mean": m3_summary["tpr"]["mean"],
            "fixed_threshold_test_tpr_std": m3_summary["tpr"]["sample_std"],
            "accuracy_mean": m3_summary["accuracy"]["mean"],
            "accuracy_std": m3_summary["accuracy"]["sample_std"],
            "macro_f1_mean": m3_summary["macro_f1"]["mean"],
            "macro_f1_std": m3_summary["macro_f1"]["sample_std"],
            "training_seconds_mean": m3_training[0],
            "training_seconds_std": m3_training[1],
            "cpu_latency_ms_per_flow_mean": m3_latency[0],
            "cpu_latency_ms_per_flow_std": m3_latency[1],
            "run_status": "formal_complete",
            "source_file": "results/m3_summary/summary.json",
            "notes": "AP/AUC and empirical ROC points use continuous scores; classification columns use q99 frozen Validation threshold.",
        },
    ]

    corrected_sources = {
        "GBDT G05": "results/p7_corrected/m1/smoke/20261114T040513-420a234bebdf4cc287083b0b631d17a9/metrics.json",
        "1D-CNN": "results/p7_corrected/m2_smoke/summary.json",
        "Autoencoder A06": "results/p7_corrected/m3_smoke/search.json",
    }
    corrected_rows = []
    for original in original_rows:
        row = {key: "" for key in original}
        row.update(
            {
                "method": original["method"],
                "role": original["role"],
                "dataset_version": "corrected",
                "protocol": "stratified",
                "variant": "main",
                "evaluation_split": "test",
                "seed_count": 5,
                "configuration": original["configuration"],
                "threshold_policy": original["threshold_policy"],
                "run_status": "reported_full_success_external_metrics_not_synced",
                "source_file": corrected_sources[original["method"]],
                "notes": "User reported successful full rerun on another computer. This checkout contains smoke evidence only; metric fields intentionally blank pending artifact sync.",
            }
        )
        corrected_rows.append(row)

    rows = original_rows + corrected_rows
    fields = list(rows[0])
    with (RESULTS / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    methods = [row["method"] for row in original_rows]
    colors = ["#2878B5", "#F28E2B", "#59A14F"]

    # Figure 1: PR-oriented aggregate summary. Raw curve arrays are not present
    # for every method, so AP is plotted without inventing curves.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, mean_key, std_key, title in [
        (axes[0], "average_precision_mean", "average_precision_std", "Average precision (PR-AUC)"),
        (axes[1], "roc_auc_mean", "roc_auc_std", "ROC-AUC"),
    ]:
        means = [float(row[mean_key]) for row in original_rows]
        stds = [float(row[std_key]) for row in original_rows]
        bars = ax.bar(methods, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0.65, 1.02)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Original stratified Test performance (five-seed mean ± sample SD)", fontsize=13)
    fig.text(0.5, -0.02, "AP summarizes the precision–recall curve; raw PR arrays were not available for all three methods.", ha="center", fontsize=8, color="#555555")
    save_figure(fig, "fig1_test_pr_summary")

    # Figure 2: aligned empirical ROC operating points.
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.34
    for offset, target, label, color in [(-width / 2, 0.01, "FPR 1%", "#4E79A7"), (width / 2, 0.05, "FPR 5%", "#E15759")]:
        means = [tpr[method][target][0] for method in methods]
        stds = [tpr[method][target][1] for method in methods]
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=label, color=color, edgecolor="black", linewidth=0.4)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("True-positive rate")
    ax.set_title("Test TPR at empirical ROC false-positive-rate targets")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig2_tpr_at_fpr")

    # Figure 3: mean family recall using approximately 1% Validation-calibrated
    # operating rules (M1 1%, M2 1%, M3 q99).
    per_method = defaultdict(lambda: defaultdict(list))
    for record in m1_records:
        for family, recall in class_map(record, ["metrics", "per_class"]).items():
            per_method["GBDT G05"][family].append(recall)
    for record in m2["runs"]:
        for family, recall in class_map(record, ["test", "per_class_recall"]).items():
            per_method["1D-CNN"][family].append(recall)
    for record in m3_records:
        for family, recall in class_map(record, ["metrics", "per_class"]).items():
            per_method["Autoencoder A06"][family].append(recall)
    supports = m1["family_coverage"]["supports"]
    families = list(supports)
    matrix = np.array([[statistics.mean(per_method[method][family]) for family in families] for method in methods])
    labels = [f"{family}\n(n={supports[family]})" for family in families]
    fig, ax = plt.subplots(figsize=(16, 4.3), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(methods)), methods)
    ax.set_xticks(np.arange(len(families)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_title("Attack-family recall on Test (five-seed mean; low-FPR operating rules)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < 0.45 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(image, ax=ax, label="Recall", shrink=0.82)
    save_figure(fig, "fig3_per_class_recall")

    # Figure 4: evidence matrix. It intentionally does not pretend that absent
    # corrected metrics are quantitative version deltas.
    status = np.array([[2, 2, 2], [2, 2, 2], [0, 0, 0]])
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#D9D9D9", "#F2CF5B", "#59A14F"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(status, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(3), methods)
    ax.set_yticks(
        range(3),
        ["Original formal Test metrics", "Corrected local smoke evidence", "Corrected full metrics synced locally"],
    )
    for i in range(3):
        for j in range(3):
            label = "Available" if status[i, j] == 2 else "Pending sync"
            ax.text(j, i, label, ha="center", va="center", color="white" if status[i, j] == 2 else "#333333", fontweight="bold")
    ax.set_title("Dataset-version evidence available for P7 integration")
    fig.text(0.5, -0.02, "The corrected full rerun was reported successful on another computer; numerical deltas require its exported metric files.", ha="center", fontsize=8, color="#555555")
    save_figure(fig, "fig4_version_rerun_status")

    figure_data = {
        "scope": "original/stratified/main Test, five seeds",
        "methods": original_rows,
        "tpr_at_empirical_fpr": tpr,
        "per_class_recall_mean": {
            method: {family: statistics.mean(values) for family, values in families_map.items()}
            for method, families_map in per_method.items()
        },
        "corrected_full_rerun": {
            "status": "reported_success_external_metrics_not_synced",
            "local_smoke_evidence": corrected_sources,
        },
    }
    (RESULTS / "p7_figure_data.json").write_text(
        json.dumps(figure_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
