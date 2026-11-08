import yaml
import torch
import os
import torch
torch.set_num_threads(os.cpu_count())
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.m2.data import get_dataset
from methods.m2.model import CNN1D

def train():
    with open("methods/m2/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_ds = get_dataset(cfg["data"]["version"], cfg["data"]["protocol"], cfg["data"]["variant"], "train")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    
    input_dim = train_ds.X.shape[1]
    model = CNN1D(input_dim).to(device)
    
    pos_weight = torch.tensor([train_ds.X.shape[0]], dtype=torch.float32, device=device) # 简单占位，实际应用读取权重的
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    print(f"开始训练 {cfg['epochs']} 轮...")
    history = []
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        
        for X, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束, 平均 Loss: {avg_loss:.4f}")
        history.append({"epoch": epoch+1, "loss": avg_loss})
        
    train_seconds = time.time() - start_time
    
    os.makedirs(cfg["output"]["model_dir"], exist_ok=True)
    model_path = os.path.join(cfg["output"]["model_dir"], f"model_seed{cfg['seed']}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    history_path = os.path.join(cfg["output"]["model_dir"], f"history_seed{cfg['seed']}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f)
        
    print(f"训练完成，耗时 {train_seconds:.2f} 秒。")

if __name__ == "__main__":
    train()