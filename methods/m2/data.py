
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.model_loader import ModelDataset


class FlowDataset(Dataset):
    def __init__(self, model_dataset):
        self.X_list = []
        self.y_list = []
        self.meta_list = []

        print("正在加载数据，请稍候...")

        for X, y, meta in model_dataset:
            self.X_list.append(X)
            self.y_list.append(y)
            self.meta_list.append(meta)

        if not self.X_list:
            raise ValueError("Dataset is empty.")

        self.X = torch.from_numpy(
            np.concatenate(self.X_list, axis=0)
        ).float()

        self.y = torch.from_numpy(
            np.concatenate(self.y_list, axis=0)
        ).float().reshape(-1)

        import pandas as pd
        self.meta = pd.concat(
            self.meta_list,
            ignore_index=True
        )

        # 使用 ModelDataset 提供的真实权重。
        # 不再使用训练样本总数作为 pos_weight。
        self.scale_pos_weight = getattr(
            model_dataset,
            "scale_pos_weight",
            None
        )

        if self.scale_pos_weight is not None:
            self.scale_pos_weight = float(
                self.scale_pos_weight
            )

        if len(self.X) != len(self.y):
            raise ValueError("X and y have different lengths.")

        if len(self.meta) != len(self.y):
            raise ValueError("Metadata and y have different lengths.")

        if not torch.isfinite(self.X).all():
            raise ValueError("X contains NaN or Inf.")

        if not torch.isfinite(self.y).all():
            raise ValueError("y contains NaN or Inf.")

        unique_labels = set(self.y.unique().tolist())

        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"Expected binary labels 0/1, got {unique_labels}"
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


def get_dataset(version, protocol, variant, split):

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")

    base = Path("data/split") / version / protocol

    if variant == "no_destination_port":
        base = base / "no_destination_port"

    path = base / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件: {path}"
        )

    print(f"读取数据: {path}")

    raw_ds = ModelDataset(path)

    return FlowDataset(raw_ds)
