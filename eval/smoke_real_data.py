"""Real-data interface checks with a dummy scorer, not experimental results."""

import json
from pathlib import Path

import numpy as np

from src.model_loader import ModelDataset
from .metrics import (
    classification_metrics,
    compute_curves,
    measure_inference,
    paired_bootstrap,
    per_class_recall,
    tpr_at_fpr,
)


ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []

    for version in ("original", "corrected"):
        for protocol in ("stratified", "temporal"):
            for variant in ("main", "no_destination_port"):
                directory = ROOT / "data/split" / version / protocol
                if variant == "no_destination_port":
                    directory = directory / variant

                for split in ("train", "val", "test"):
                    path = directory / f"{split}.parquet"
                    dataset = ModelDataset(path)

                    # Read one batch through the actual downstream interface.
                    iterator = iter(dataset)
                    try:
                        X, y, metadata = next(iterator)
                    finally:
                        iterator.close()

                    expected_features = 140 if variant == "main" else 75
                    if (
                        X.ndim != 2
                        or X.shape[1] != expected_features
                        or len(X) != len(y)
                        or len(y) != len(metadata)
                        or not np.isfinite(X).all()
                    ):
                        raise ValueError(f"Invalid loaded batch: {path}")

                    if not np.array_equal(
                        y, metadata["y_binary"].to_numpy()
                    ):
                        raise ValueError(f"Label alignment failed: {path}")

                    row_ids = metadata["row_id"].tolist()
                    if len(set(row_ids)) != len(row_ids):
                        raise ValueError(f"Duplicate batch row IDs: {path}")

                    item = {
                        "version": version,
                        "protocol": protocol,
                        "variant": variant,
                        "split": split,
                        "batch_rows": int(len(y)),
                        "features": int(X.shape[1]),
                        "passed": True,
                    }

                    if split == "test":
                        # Dummy scores independent of labels.
                        scores = np.zeros(len(y), dtype=float)
                        pred = (scores >= 0.5).astype(int)

                        curves = compute_curves(y, scores)
                        operating = tpr_at_fpr(y, scores)
                        classification = classification_metrics(
                            y, pred,
                            is_attempted=metadata["is_attempted"].to_numpy(),
                        )
                        families = per_class_recall(
                            y, pred, metadata["attack_family"].tolist()
                        )

                        def dummy_predict(batch):
                            return np.zeros(len(batch), dtype=float)

                        timing = measure_inference(
                            dummy_predict, X,
                            batch_size=min(512, len(X)),
                            warmup=1, repeats=3,
                        )

                        comparison = None
                        negative = np.flatnonzero(y == 0)[:128]
                        positive = np.flatnonzero(y == 1)[:128]

                        if len(negative) and len(positive):
                            indices = np.concatenate([negative, positive])
                            selected_ids = [row_ids[i] for i in indices]
                            comparison = paired_bootstrap(
                                y[indices], scores[indices], scores[indices],
                                row_ids_a=selected_ids,
                                row_ids_b=selected_ids,
                                n_bootstrap=20,
                            )
                            if not all(
                                comparison[key] == 0
                                for key in ("difference", "ci_lower", "ci_upper")
                            ):
                                raise ValueError("Identical-score comparison failed")

                        item["dummy_checks"] = {
                            "curve_status": curves["status"],
                            "operating_points": operating,
                            "classification": classification,
                            "per_class": families,
                            "inference": timing,
                            "paired_bootstrap": comparison,
                            "bootstrap_note": (
                                "Small interface-check subset only."
                                if comparison is not None
                                else "Skipped: first batch contains one class."
                            ),
                        }

                    records.append(item)
                    print(version, protocol, variant, split, "PASS", flush=True)

    output = {
        "purpose": "interface_check_only_not_experimental_findings",
        "scope": "first_batch_of_each_delivered_model_file",
        "dummy_scorer": "constant_zero",
        "passed": all(item["passed"] for item in records),
        "records": records,
    }

    target = ROOT / "eval/p2_acceptance/interface_smoke.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("All 24 data-loading combinations passed.")
    print("Saved: eval/p2_acceptance/interface_smoke.json")


if __name__ == "__main__":
    main()
