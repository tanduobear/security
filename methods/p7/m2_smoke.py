"""Small corrected-data M2 smoke check. It never opens the test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def collect(path, limit):
    import numpy as np
    from src.model_loader import ModelDataset

    arrays, labels = [], []
    total = 0
    source = ModelDataset(path)
    for X, y, _ in source:
        take = min(len(y), limit - total)
        if take:
            arrays.append(X[:take])
            labels.append(y[:take])
            total += take
        if total >= limit:
            break
    if not arrays:
        raise ValueError(f"No rows in {path}")
    return np.concatenate(arrays), np.concatenate(labels), float(source.scale_pos_weight)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "methods/p7/m2_corrected.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/p7_corrected/m2_smoke")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg["data"] != {"version": "corrected", "protocol": "stratified", "variant": "main"}:
        raise ValueError("P7 M2 smoke requires corrected/stratified/main")
    output = args.output.resolve()
    output.relative_to((ROOT / "results/p7_corrected").resolve())
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError("Smoke summary already exists; choose a new output directory")

    import numpy as np
    import torch
    from methods.m2.model import CNN1D

    base = ROOT / "data/split/corrected/stratified"
    train_X, train_y, weight = collect(base / "train.parquet", 2048)
    val_X, val_y, _ = collect(base / "val.parquet", 512)
    expected = float(cfg["expected_scale_pos_weight"])
    if not np.isclose(weight, expected, rtol=1e-12):
        raise ValueError("Corrected scale_pos_weight changed")

    torch.manual_seed(int(cfg["seeds"][0]))
    model = CNN1D(train_X.shape[1]).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    X = torch.from_numpy(train_X).float()
    y = torch.from_numpy(train_y).float()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(X[:256]).reshape(-1)
    loss = loss_fn(logits, y[:256])
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(val_X).float())).reshape(-1)
    if not torch.isfinite(scores).all():
        raise ValueError("M2 smoke produced non-finite scores")

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "smoke_only",
        "dataset": cfg["data"],
        "test_accessed": False,
        "seed": int(cfg["seeds"][0]),
        "train_rows_loaded": len(train_X),
        "validation_rows_loaded": len(val_X),
        "input_features": int(train_X.shape[1]),
        "scale_pos_weight": weight,
        "single_batch_loss": float(loss.detach()),
        "validation_score_min": float(scores.min()),
        "validation_score_max": float(scores.max()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

