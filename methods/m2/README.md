# M2：一维卷积神经网络（1D-CNN）

## 1. 项目简介

M2 使用一维卷积神经网络（1D-CNN）进行网络流量二分类任务，判断网络流量是否属于恶意流量。

模型以每条网络流量的 140 个特征作为输入，并输出对应的预测分数。

为了评估模型的性能和稳定性，本实验使用五个不同的随机种子进行训练和测试：

- Seed 42
- Seed 43
- Seed 44
- Seed 45
- Seed 46

## 2. 模型结构

本实验使用的 1D-CNN 主要包含以下结构：

1. 输入层：接收 140 个网络流量特征。
2. 第一层一维卷积层：使用 32 个卷积核提取局部特征。
3. 第二层一维卷积层：使用 64 个卷积核进一步提取特征。
4. Batch Normalisation：对特征进行归一化处理。
5. ReLU：使用非线性激活函数。
6. Adaptive Average Pooling：对卷积特征进行自适应平均池化。
7. 全连接输出层：输出二分类预测分数。

## 3. 训练配置

实验采用以下训练参数：

| 参数 | 配置 |
|---|---|
| 模型 | 1D-CNN |
| 输入特征数量 | 140 |
| 训练轮数（Epochs） | 40 |
| 批次大小（Batch Size） | 256 |
| 学习率（Learning Rate） | 0.001 |
| 优化器（Optimizer） | Adam |
| 正样本权重（pos_weight） | 4.929864 |
| 随机种子（Seeds） | 42、43、44、45、46 |
| 混合精度训练（AMP） | 开启 |
| 训练设备 | NVIDIA GeForce RTX 4060 Laptop GPU |

实验使用相同的数据集划分和类别不平衡处理方案，以便与其他模型进行比较。

## 4. 实验环境

本实验使用以下软硬件环境：

- Python：3.14.3
- PyTorch：2.14.1+cu126
- CUDA：12.6
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU

运行实验前，需要安装项目所需的 Python 依赖包。

## 5. 数据集

实验需要以下数据文件：

```text
data/
└── split/
    └── original/
        └── stratified/
            ├── train.parquet
            ├── val.parquet
            └── test.parquet
```

数据集规模如下：

| 数据集 | 样本数量 |
|---|---:|
| Validation | 367,036 |
| Test | 346,544 |
| 输入特征数量 | 140 |

注意：运行训练和评估脚本之前，需要确保数据文件已放置在上述路径中。

## 6. 模型训练

在项目根目录下打开 PowerShell，执行：

```powershell
python methods/m2/train.py
```

训练脚本会根据配置文件中的随机种子依次训练五个模型，并保存训练结果。

训练输出文件保存在：

```text
results/
└── preds_m2_corrected/
    ├── model_seed42.pth
    ├── model_seed43.pth
    ├── model_seed44.pth
    ├── model_seed45.pth
    ├── model_seed46.pth
    ├── history_seed42.json
    ├── history_seed43.json
    ├── history_seed44.json
    ├── history_seed45.json
    ├── history_seed46.json
    └── training_summary.json
```

其中：

- `.pth` 文件：保存每个随机种子训练得到的模型权重。
- `history_seed*.json`：保存对应模型的训练历史。
- `training_summary.json`：汇总五次训练的时间和模型保存路径。

训练曲线图的保存路径由 `methods/m2/config.yaml` 中的配置决定。

## 7. 模型评估

完成模型训练后，在项目根目录下执行：

```powershell
python methods/m2/evaluate.py
```

评估脚本会加载训练好的模型，并在测试集上计算各项性能指标。

评估结果保存在：

```text
results/m2.json
```

本实验使用以下指标评估模型：

- ROC-AUC：衡量模型区分正负样本的能力。
- Average Precision（AP）：衡量模型在不同召回率下的精确率表现。
- TPR @ 1% FPR：当误报率为 1% 时的真正率。
- TPR @ 5% FPR：当误报率为 5% 时的真正率。
- Accuracy：分类准确率。
- Macro-F1：正负类别 F1 分数的宏平均。
- CPU Inference Latency：CPU 平均推理延迟。
- CPU Throughput：CPU 每秒能够处理的流量数量。

模型决策阈值根据验证集进行选择，而不是直接使用测试集选择阈值。

## 8. 实验结果

以下结果为五个随机种子的平均值及标准差：

| 评估指标 | 平均值 ± 标准差 |
|---|---:|
| ROC-AUC | 0.9809 ± 0.0040 |
| Average Precision (AP) | 0.9172 ± 0.0233 |
| TPR @ 1% FPR | 0.6115 ± 0.0700 |
| TPR @ 5% FPR | 0.7951 ± 0.0986 |
| Accuracy | 0.9211 ± 0.0126 |
| Macro-F1 | 0.8478 ± 0.0292 |
| CPU Latency (ms/flow) | 0.0424 ± 0.0025 |
| CPU Throughput (flows/s) | 23,665.66 ± 1,384.94 |

五次训练的平均耗时约为：

- 每个 Seed：1,929.10 秒
- 每个 Seed：约 32.15 分钟

## 9. 实验复现说明

为了方便其他研究者复现实验，项目保留了以下文件：

- `methods/m2/config.yaml`：保存模型训练配置。
- `methods/m2/train.py`：模型训练脚本。
- `methods/m2/evaluate.py`：模型评估脚本。
- `results/preds_m2_corrected/`：保存五个 Seed 的模型权重和训练历史。
- `results/m2.json`：保存模型评估结果。
- `figures/fig_m2_training_corrected.png`：保存训练曲线图。

本实验使用五个随机种子评估模型的稳定性，并分别记录 GPU 训练耗时与 CPU 推理性能。

以上文件共同提供了模型配置、训练记录、模型权重及评估指标，便于检查和复现实验结果。