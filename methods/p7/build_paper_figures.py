"""Create compact P7 paper figures in the visual style of the team deliverables."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {
    "GBDT G05": "#0072B2",
    "1D-CNN": "#D55E00",
    "Autoencoder A06": "#009E73",
}
METHODS = list(COLORS)

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.labelsize": 12,
    "axes.titlesize": 12.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "pdf.compression": 9,
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values):
    values = [float(value) for value in values]
    return statistics.mean(values), statistics.stdev(values)


def empirical_tpr(record, target):
    points = record["metrics"]["operating_points"]["operating_points"]
    return next(float(point["tpr"]) for point in points if float(point["target_fpr"]) == target)


def save(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata={"Author": "P7", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def clean_axes(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#D9DDE3", linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)


def main():
    m1 = load(RESULTS / "m1_test_summary" / "summary.json")
    m2 = load(RESULTS / "m2.json")
    m3 = load(RESULTS / "m3_summary" / "summary.json")

    m1_sum = next(x for x in m1["five_seed_summaries"] if x["method"] == "M1_GBDT" and x["target_validation_fpr"] == 0.01)
    m1_records = [x for x in m1["per_seed_threshold_records"] if x["method"] == "M1_GBDT" and x["config"]["target_validation_fpr"] == 0.01]
    m3_sum = next(x for x in m3["five_seed_summary"] if x["quantile"] == 0.99)
    m3_records = [x for x in m3["records"] if x["method"] == "Autoencoder" and x["config"]["benign_quantile"] == 0.99]

    overall = {
        "GBDT G05": {
            "Average precision": (m1_sum["ap"]["mean"], m1_sum["ap"]["sample_std"]),
            "ROC-AUC": (m1_sum["roc_auc"]["mean"], m1_sum["roc_auc"]["sample_std"]),
        },
        "1D-CNN": {
            "Average precision": (m2["summary"]["average_precision"]["mean"], m2["summary"]["average_precision"]["std"]),
            "ROC-AUC": (m2["summary"]["roc_auc"]["mean"], m2["summary"]["roc_auc"]["std"]),
        },
        "Autoencoder A06": {
            "Average precision": (m3_sum["average_precision"]["mean"], m3_sum["average_precision"]["sample_std"]),
            "ROC-AUC": (m3_sum["roc_auc"]["mean"], m3_sum["roc_auc"]["sample_std"]),
        },
    }

    low_fpr = {
        "GBDT G05": {
            "TPR at FPR 1%": mean_std(empirical_tpr(x, 0.01) for x in m1_records),
            "TPR at FPR 5%": mean_std(empirical_tpr(x, 0.05) for x in m1_records),
        },
        "1D-CNN": {
            "TPR at FPR 1%": (m2["summary"]["tpr_at_fpr_1pct"]["mean"], m2["summary"]["tpr_at_fpr_1pct"]["std"]),
            "TPR at FPR 5%": (m2["summary"]["tpr_at_fpr_5pct"]["mean"], m2["summary"]["tpr_at_fpr_5pct"]["std"]),
        },
        "Autoencoder A06": {
            "TPR at FPR 1%": mean_std(empirical_tpr(x, 0.01) for x in m3_records),
            "TPR at FPR 5%": mean_std(empirical_tpr(x, 0.05) for x in m3_records),
        },
    }

    thresholded = {
        "GBDT G05": {
            "Accuracy": mean_std(x["metrics"]["classification"]["accuracy"] for x in m1_records),
            "Macro-F1": (m1_sum["macro_f1"]["mean"], m1_sum["macro_f1"]["sample_std"]),
        },
        "1D-CNN": {
            "Accuracy": (m2["summary"]["accuracy"]["mean"], m2["summary"]["accuracy"]["std"]),
            "Macro-F1": (m2["summary"]["macro_f1"]["mean"], m2["summary"]["macro_f1"]["std"]),
        },
        "Autoencoder A06": {
            "Accuracy": (m3_sum["accuracy"]["mean"], m3_sum["accuracy"]["sample_std"]),
            "Macro-F1": (m3_sum["macro_f1"]["mean"], m3_sum["macro_f1"]["sample_std"]),
        },
    }

    per_family = defaultdict(lambda: defaultdict(list))
    for record in m1_records:
        for item in record["metrics"]["per_class"]["classes"]:
            per_family["GBDT G05"][item["attack_family"]].append(float(item["recall"]))
    for record in m2["runs"]:
        for item in record["test"]["per_class_recall"]["classes"]:
            per_family["1D-CNN"][item["attack_family"]].append(float(item["recall"]))
    for record in m3_records:
        for item in record["metrics"]["per_class"]["classes"]:
            per_family["Autoencoder A06"][item["attack_family"]].append(float(item["recall"]))

    supports = m1["family_coverage"]["supports"]
    families = list(supports)
    family_labels = {
        "botnet": "Botnet",
        "ddos": "DDoS",
        "dos goldeneye": "DoS GoldenEye",
        "dos hulk": "DoS Hulk",
        "dos slowhttptest": "DoS SlowHTTPTest",
        "dos slowloris": "DoS Slowloris",
        "ftp-patator": "FTP-Patator",
        "heartbleed": "Heartbleed",
        "infiltration": "Infiltration",
        "portscan": "PortScan",
        "ssh-patator": "SSH-Patator",
        "web attack - brute force": "Web Brute Force",
        "web attack - sql injection": "Web SQL Injection",
        "web attack - xss": "Web XSS",
    }
    family_matrix = np.array([[statistics.mean(per_family[method][family]) for method in METHODS] for family in families])

    rows = []
    for figure, values in [("Figure 1", overall), ("Figure 2", low_fpr), ("Figure 4", thresholded)]:
        for method, metrics in values.items():
            for metric, (mean, std) in metrics.items():
                rows.append({"figure": figure, "method": method, "metric": metric, "attack_family": "", "support": "", "mean": mean, "sample_std": std, "scope": "original/stratified/main Test; seeds 42-46"})
    for family in families:
        for method in METHODS:
            values = per_family[method][family]
            rows.append({"figure": "Figure 3", "method": method, "metric": "Attack-family recall", "attack_family": family_labels[family], "support": supports[family], "mean": statistics.mean(values), "sample_std": statistics.stdev(values), "scope": "low-FPR rule: GBDT 1%; CNN 1%; AE q99"})
    with (RESULTS / "p7_plot_values.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with plt.rc_context(RC):
        # Figure 1: one clean dot plot instead of two unequal bar panels.
        fig, ax = plt.subplots(figsize=(8, 5.8))
        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.28, top=0.90)
        y = np.arange(len(METHODS))
        offsets = {"Average precision": -0.10, "ROC-AUC": 0.10}
        markers = {"Average precision": "o", "ROC-AUC": "s"}
        series_colors = {"Average precision": "#0072B2", "ROC-AUC": "#D55E00"}
        for metric in ["Average precision", "ROC-AUC"]:
            values = [overall[m][metric][0] for m in METHODS]
            errors = [overall[m][metric][1] for m in METHODS]
            ax.errorbar(values, y + offsets[metric], xerr=errors, fmt=markers[metric], markersize=5.5,
                        capsize=3, linewidth=1.2, color=series_colors[metric],
                        markerfacecolor=series_colors[metric], markeredgewidth=0.8, label=metric)
            for value, error, yy in zip(values, errors, y + offsets[metric]):
                ax.text(value + error + 0.008, yy, f"{value:.4f}", va="center", fontsize=9,
                        color=series_colors[metric])
        ax.set_yticks(y, METHODS)
        ax.invert_yaxis()
        ax.set_xlim(0.70, 1.025)
        ax.set_xlabel("Test score")
        ax.set_title("Cross-method ranking performance")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
                  columnspacing=2.2, handlelength=2.8, borderaxespad=0)
        clean_axes(ax)
        fig.text(0.5, 0.025, "Original stratified Test; five-seed mean ± sample SD.", ha="center", fontsize=9, color="#444444")
        save(fig, "fig_p7_overall_performance")

        # Figure 2: same geometry and line/marker vocabulary.
        fig, ax = plt.subplots(figsize=(8, 5.8))
        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.28, top=0.90)
        offsets = {"TPR at FPR 1%": -0.10, "TPR at FPR 5%": 0.10}
        markers = {"TPR at FPR 1%": "o", "TPR at FPR 5%": "s"}
        series_colors = {"TPR at FPR 1%": "#0072B2", "TPR at FPR 5%": "#D55E00"}
        for metric in ["TPR at FPR 1%", "TPR at FPR 5%"]:
            values = [low_fpr[m][metric][0] for m in METHODS]
            errors = [low_fpr[m][metric][1] for m in METHODS]
            ax.errorbar(values, y + offsets[metric], xerr=errors, fmt=markers[metric], markersize=5.5,
                        capsize=3, linewidth=1.2, color=series_colors[metric],
                        markerfacecolor=series_colors[metric], markeredgewidth=0.8, label=metric)
            for value, error, yy in zip(values, errors, y + offsets[metric]):
                if value + error + 0.015 <= 1.02:
                    xx, align = value + error + 0.015, "left"
                else:
                    xx, align = value - error - 0.015, "right"
                ax.text(xx, yy, f"{value:.4f}", va="center", fontsize=9, ha=align,
                        color=series_colors[metric])
        ax.set_yticks(y, METHODS)
        ax.invert_yaxis()
        ax.set_xlim(0.40, 1.03)
        ax.set_xlabel("True-positive rate")
        ax.set_title("Detection at low false-positive rates")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
                  columnspacing=2.2, handlelength=2.8, borderaxespad=0)
        clean_axes(ax)
        fig.text(0.5, 0.025, "Empirical Test ROC operating points; five-seed mean ± sample SD.", ha="center", fontsize=9, color="#444444")
        save(fig, "fig_p7_tpr_at_fpr")

        # Figure 3: methods as columns and families as rows for aligned labels.
        fig, ax = plt.subplots(figsize=(8, 7.2))
        fig.subplots_adjust(left=0.28, right=0.89, bottom=0.10, top=0.91)
        image = ax.imshow(family_matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(METHODS)), METHODS)
        ax.set_yticks(range(len(families)), [f"{family_labels[f]}  (n={supports[f]})" for f in families])
        ax.set_title("Attack-family recall at low-FPR operating rules")
        ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
        ax.tick_params(axis="y", length=0)
        for i in range(len(families)):
            for j in range(len(METHODS)):
                value = family_matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > 0.62 else "#222222", fontsize=8.5)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
        colorbar.set_label("Recall")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(0.5, 0.025, "Cell values are five-seed means; n is Test support.", ha="center", fontsize=9, color="#444444")
        save(fig, "fig_p7_family_recall")

        # Figure 4: replace the non-paper status matrix with numeric metrics.
        fig, ax = plt.subplots(figsize=(8, 5.8))
        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.28, top=0.90)
        offsets = {"Accuracy": -0.10, "Macro-F1": 0.10}
        markers = {"Accuracy": "o", "Macro-F1": "s"}
        series_colors = {"Accuracy": "#0072B2", "Macro-F1": "#D55E00"}
        for metric in ["Accuracy", "Macro-F1"]:
            values = [thresholded[m][metric][0] for m in METHODS]
            errors = [thresholded[m][metric][1] for m in METHODS]
            ax.errorbar(values, y + offsets[metric], xerr=errors, fmt=markers[metric], markersize=5.5,
                        capsize=3, linewidth=1.2, color=series_colors[metric],
                        markerfacecolor=series_colors[metric], markeredgewidth=0.8, label=metric)
            for value, error, yy in zip(values, errors, y + offsets[metric]):
                if value + error + 0.008 <= 1.02:
                    xx, align = value + error + 0.008, "left"
                else:
                    xx, align = value - error - 0.008, "right"
                ax.text(xx, yy, f"{value:.4f}", va="center", fontsize=9, ha=align,
                        color=series_colors[metric])
        ax.set_yticks(y, METHODS)
        ax.invert_yaxis()
        ax.set_xlim(0.77, 1.025)
        ax.set_xlabel("Test score")
        ax.set_title("Threshold-dependent Test performance")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
                  columnspacing=2.2, handlelength=2.8, borderaxespad=0)
        clean_axes(ax)
        fig.text(0.5, 0.025, "Frozen low-FPR rules: GBDT 1%, CNN 1%, autoencoder q99; mean ± sample SD.", ha="center", fontsize=9, color="#444444")
        save(fig, "fig_p7_threshold_metrics")

    print("Saved four paper figures and results/p7_plot_values.csv")


if __name__ == "__main__":
    main()
