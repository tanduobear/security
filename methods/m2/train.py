
import os
import sys
import json
import time
import random
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.m2.data import get_dataset
from methods.m2.model import CNN1D


# ============================================================
# 2. 随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 保证实验尽可能可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. 单个随机种子的训练
# ============================================================

def train_one_seed(cfg, train_ds, device, seed):

    print("\n" + "=" * 65)
    print(f"开始训练 Seed {seed}")
    print("=" * 65)

    set_seed(seed)

    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    learning_rate = float(cfg["learning_rate"])

    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"

    num_workers = int(cfg.get("num_workers", 0))

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        drop_last=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_dim = train_ds.X.shape[1]

    model = CNN1D(input_dim).to(device)

    # --------------------------------------------------------
    # 正确读取 pos_weight
    # --------------------------------------------------------

    pos_weight_value = float(train_ds.scale_pos_weight)

    if not np.isfinite(pos_weight_value) or pos_weight_value <= 0:
        raise ValueError(
            f"Invalid scale_pos_weight: {pos_weight_value}"
        )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"AMP enabled: {use_amp}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"pos_weight: {pos_weight_value:.12f}")
    print(f"Training samples: {len(train_ds)}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Seed {seed} | Epoch {epoch + 1}/{epochs}"
        )

        for X, y, _ in progress:

            X = X.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            )

            y = y.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            ).view(-1)

            optimizer.zero_grad(set_to_none=True)

            # GPU mixed precision
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                logits = model(X).view(-1)

                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            batch_samples = y.size(0)

            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

        avg_loss = total_loss / total_samples

        history.append({
            "epoch": epoch + 1,
            "loss": float(avg_loss)
        })

        print(
            f"Seed {seed} | Epoch {epoch + 1}/{epochs} "
            f"| Loss: {avg_loss:.6f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_seconds = time.time() - start_time

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    output_dir = Path(cfg["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"model_seed{seed}.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    # --------------------------------------------------------
    # Save training history and metadata
    # --------------------------------------------------------

    history_data = {
        "seed": seed,
        "training_seconds": float(training_seconds),
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "CPU"
        ),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pos_weight": pos_weight_value,
        "use_amp": use_amp,
        "training_samples": len(train_ds),
        "model_path": str(model_path),
        "history": history
    }

    history_path = output_dir / f"history_seed{seed}.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            history_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nSeed {seed} 训练完成")
    print(f"训练时间: {training_seconds:.2f} 秒")
    print(f"模型保存: {model_path}")
    print(f"历史保存: {history_path}")

    return history_data


# ============================================================
# 4. 绘制训练曲线
# ============================================================

def plot_training_curves(all_results, figure_file):

    figure_path = Path(figure_file)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for result in all_results:

        seed = result["seed"]

        losses = [
            item["loss"]
            for item in result["history"]
        ]

        epochs = range(1, len(losses) + 1)

        plt.plot(
            epochs,
            losses,
            label=f"Seed {seed}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("M2 CNN1D Training Loss")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    print(f"\n训练曲线已保存: {figure_path}")


# ============================================================
# 5. 主程序
# ============================================================

def train():

    config_path = Path("methods/m2/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    requested_device = cfg.get("device", "cuda")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "配置要求使用 CUDA，但当前 PyTorch 无法使用 GPU。"
        )

    device = torch.device(requested_device)

    if device.type == "cuda":
        torch.set_num_threads(4)
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        torch.set_num_threads(4)
        print("\n警告：当前使用 CPU。")

    # --------------------------------------------------------
    # Load dataset once
    # --------------------------------------------------------

    print("\n正在加载训练数据...")

    data_cfg = cfg["data"]

    train_ds = get_dataset(
        data_cfg["version"],
        data_cfg["protocol"],
        data_cfg["variant"],
        "train"
    )

    print("\n数据加载完成")
    print(f"训练集大小: {len(train_ds)}")
    print(f"特征数量: {train_ds.X.shape[1]}")

    # --------------------------------------------------------
    # Verify class weights
    # --------------------------------------------------------

    actual_pos_weight = float(train_ds.scale_pos_weight)

    print(
        f"从预处理文件读取的 scale_pos_weight: "
        f"{actual_pos_weight}"
    )

    if not np.isclose(
        actual_pos_weight,
        4.9298640100563995,
        rtol=1e-6
    ):
        raise ValueError(
            "当前读取的 pos_weight 与刚才验证的数值不一致，"
            "请先检查数据版本和预处理文件。"
        )

    # --------------------------------------------------------
    # Train all seeds
    # --------------------------------------------------------

    seeds = cfg.get("seeds", [42])

    all_results = []

    for seed in seeds:

        result = train_one_seed(
            cfg=cfg,
            train_ds=train_ds,
            device=device,
            seed=int(seed)
        )

        all_results.append(result)

    # --------------------------------------------------------
    # Save combined summary
    # --------------------------------------------------------

    output_dir = Path(cfg["output"]["model_dir"])

    summary_path = output_dir / "training_summary.json"

    summary = {
        "experiment_name": cfg["experiment_name"],
        "seeds": [int(s) for s in seeds],
        "pos_weight": actual_pos_weight,
        "device": str(device),
        "results": [
            {
                "seed": result["seed"],
                "training_seconds": result["training_seconds"],
                "model_path": result["model_path"]
            }
            for result in all_results
        ]
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_training_curves(
        all_results,
        cfg["output"]["figure_file"]
    )

    print("\n" + "=" * 65)
    print("全部 M2 训练任务完成！")
    print("=" * 65)

    for result in all_results:
        print(
            f"Seed {result['seed']}: "
            f"{result['training_seconds']:.2f} seconds"
        )

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    train()
