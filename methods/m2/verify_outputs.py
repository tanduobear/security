
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = ROOT / "results" / "m2.json"

PREDICTION_DIR = (
    ROOT / "results" / "preds_m2_corrected" / "predictions"
)

SEEDS = [42, 43, 44, 45, 46]


def calculate_hash(row_ids):
    content = "\n".join(row_ids.tolist()).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def main():
    print("=" * 60)
    print("M2 OUTPUT VERIFICATION")
    print("=" * 60)

    if not RESULT_PATH.exists():
        raise FileNotFoundError(RESULT_PATH)

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        result = json.load(f)

    runs = result.get("runs", [])

    if len(runs) != len(SEEDS):
        raise ValueError(
            f"Expected 5 runs, found {len(runs)}"
        )

    run_map = {
        int(run["seed"]): run
        for run in runs
    }

    reference_ids = None
    reference_hash = None

    for seed in SEEDS:
        print(f"\nChecking seed {seed}...")

        if seed not in run_map:
            raise ValueError(f"Seed {seed} missing from m2.json")

        path = (
            PREDICTION_DIR
            / f"seed_{seed}_test_predictions.npz"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:

            required = {
                "row_ids",
                "y_true",
                "y_score",
                "y_pred",
                "attack_family",
                "threshold",
                "row_ids_sha256",
            }

            missing = required - set(data.files)

            if missing:
                raise ValueError(
                    f"Seed {seed}: missing fields {missing}"
                )

            row_ids = data["row_ids"].astype(str)
            y_true = data["y_true"]
            y_score = data["y_score"]
            y_pred = data["y_pred"]
            families = data["attack_family"].astype(str)

            n = len(row_ids)

            if not (
                len(y_true)
                == len(y_score)
                == len(y_pred)
                == len(families)
                == n
            ):
                raise ValueError(
                    f"Seed {seed}: array lengths differ"
                )

            if len(np.unique(row_ids)) != n:
                raise ValueError(
                    f"Seed {seed}: duplicate row IDs"
                )

            if not np.isfinite(y_score).all():
                raise ValueError(
                    f"Seed {seed}: non-finite scores"
                )

            if not np.isin(y_true, [0, 1]).all():
                raise ValueError(
                    f"Seed {seed}: invalid true labels"
                )

            if not np.isin(y_pred, [0, 1]).all():
                raise ValueError(
                    f"Seed {seed}: invalid predictions"
                )

            actual_hash = calculate_hash(row_ids)
            stored_hash = str(data["row_ids_sha256"].item())

            if actual_hash != stored_hash:
                raise ValueError(
                    f"Seed {seed}: stored hash mismatch"
                )

            if reference_ids is None:
                reference_ids = row_ids.copy()
                reference_hash = actual_hash
            else:
                if not np.array_equal(reference_ids, row_ids):
                    raise ValueError(
                        f"Seed {seed}: row ID order differs"
                    )

                if actual_hash != reference_hash:
                    raise ValueError(
                        f"Seed {seed}: row ID hash differs"
                    )

            run = run_map[seed]

            if run.get("row_ids_sha256") != actual_hash:
                raise ValueError(
                    f"Seed {seed}: JSON hash mismatch"
                )

            if int(run.get("prediction_samples", -1)) != n:
                raise ValueError(
                    f"Seed {seed}: JSON sample count mismatch"
                )

            print("  Samples:", n)
            print("  SHA256:", actual_hash)
            print("  Prediction file: OK")

    print("\n" + "=" * 60)
    print("ALL FIVE SEEDS PASSED")
    print("=" * 60)
    print("Shared test samples:", len(reference_ids))
    print("Shared row ID hash:", reference_hash)
    print("M2 prediction outputs are internally consistent.")


if __name__ == "__main__":
    main()
