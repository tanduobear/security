from pathlib import Path
import numpy as np
import torch
import yaml
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
            
        self.X = torch.from_numpy(np.concatenate(self.X_list, axis=0)).float()
        self.y = torch.from_numpy(np.concatenate(self.y_list, axis=0)).float()
        import pandas as pd
        self.meta = pd.concat(self.meta_list, ignore_index=True)
        
        # 获取真实的正负样本权重
        self.scale_pos_weight = getattr(model_dataset, "scale_pos_weight", 1.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx

def get_dataset(version, protocol, variant, split):
    base = Path("data/split") / version / protocol
    if variant == "no_destination_port":
        base = base / "no_destination_port"
    
    # 读取配置，看看有没有设置 force_split
    try:
        with open("methods/m2/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "force_split" in cfg and cfg["force_split"]:
            split = cfg["force_split"]
    except Exception:
        pass

    path = base / f"{split}.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}，请确保数据已下载并放在正确位置。")
        
    print(f"读取数据: {path}")
    raw_ds = ModelDataset(path)
    return FlowDataset(raw_ds)