# 网络安全课程项目 — P1 数据交付

**仓库用途**：8 人线性流水线协作，P1（数据获取→校验→清洗）交付物统一存放在本仓库，供 P2 及后续成员接手。

## 克隆与 LFS（重要）

两个清洗数据 parquet（共约 720 MB）使用 **Git LFS** 存储。克隆前必须先安装并启用 Git LFS：

```bash
# 1. 安装 Git LFS（Windows 可到 https://git-lfs.com 下载，或已随 Git 安装）
git lfs install

# 2. 克隆本仓库（LFS 文件会自动下载）
git clone https://github.com/tanduobear/security.git
cd security
```

> 未启用 Git LFS 时，`data/clean/*.parquet` 只会下载到 134 字节的文本指针，无法直接使用。
> P2 划分数据也使用 Git LFS；数据体积与获取方式见本文末尾。

## 目录结构

```
security/
├── data/clean/            # 清洗后交付数据（Git LFS）
│   ├── original.parquet   # CIC-IDS2017 原版（2,520,798 行）
│   └── corrected.parquet  # CIC-IDS2017 修正版（2,099,971 行）
├── logs/
│   └── clean_log.md       # 清洗日志（含删除行数与处理明细）
├── tables/
│   └── class_distribution.csv  # 类别分布（原版良性 83.11% / 修正版 75.36%）
├── docs/
│   ├── schema_diff.md     # 原版与修正版字段差异
│   └── introduction.md    # Introduction 初稿（P1 写作任务）
├── 交付登记表.docx        # 各棒次交接登记（P1 第 1 行）
├── 进度看板.docx          # 项目进度跟踪
└── 问题记录.docx          # 问题与解决记录
```

## 交接说明（P1 → P2）

- 输入：`data/clean/` 两份 parquet、`logs/clean_log.md`、`docs/schema_diff.md`
- 任务：泄漏剔除与划分（分层 70/15/15 + 按天时序），产出 `data/split/` 与 `docs/feature_dict.md`
- 类别分布详见 `tables/class_distribution.csv`，行数等统计见 `logs/clean_log.md`

## 数据来源

| 数据集 | 版本 | 来源 |
|---|---|---|
| CIC-IDS2017 | 原版 | ModelScope 镜像（官网 UNB） |
| CIC-IDS2017 | 修正版 | Kaggle: improved-cicids2017 |
| UNSW-NB15 | 可选 | UNSW 官网（未交付，按手册可选） |

## 分工与写作任务（P1）

- S1 技术任务：数据获取、校验（SHA256）、四步清洗（列名去空格 → inf 转 NaN 删除 → 全行去重 → 标签小写/去空格/修复 U+FFFD）
- 写作任务：**Introduction（600 词）**，10/29–10/31（D5–D7）期间交 P8，交付位置 `docs/introduction.md`

## P2 技术产物与使用入口

- 字段分类及共同特征：[docs/feature_dict.md](docs/feature_dict.md)
- 标签映射与原始划分：[docs/split_report.md](docs/split_report.md)
- 最终模型数据、有效样本量及训练参数：[docs/preprocess_report.md](docs/preprocess_report.md)
- 全流程复核及已修复问题：[docs/p2_full_review.md](docs/p2_full_review.md)
- 运行时推荐使用`src.model_loader.ModelDataset`，按文件位置自动匹配处理器、权重和特征白名单。

模型数据位于`data/split/{original,corrected}/{stratified,temporal}/`的`train.parquet`、`val.parquet`和`test.parquet`，删除端口版本位于对应目录的`no_destination_port/`。最终数据请使用`model_*_indices.parquet`有效索引；原始`*_indices.parquet`保留作原始划分追溯。转换成float32后的额外重复清除会改变验证/测试数量，最终比例和逐类支持数以预处理报告为准，不能只看原始划分数字。

## P2 交付范围与复现

本次交付包含两版数据的分层及按天时序划分、主版及删除目的端口消融、训练折拟合的预处理器、最终有效索引、配置、验证记录和单元测试。所有方法必须共享相同版本与协议的数据及处理器。

分层目标为70/15/15。转换为float32后，按训练、验证、测试的顺序保留样本并清除跨折重复；原版最终比例为71.20%/14.81%/13.99%，不是严格70/15/15。最终样本量和类别支持数以 docs/preprocess_report.md 为准。Methods §3.1 正文及 P3 本人验收尚需另行完成。

在仓库根目录运行（Python 3.12，依赖版本见 requirements）：

```powershell
python -m pip install -r src/requirements-preprocess.txt
python src/build_split_indices.py --work-dir .tmp/split-build
python src/verify_split_indices.py
python src/build_model_data.py --work-dir .tmp/preprocess-build
python src/finalize_model_data.py --work-dir .tmp/precision-guard
python src/verify_model_data.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

直接使用已交付数据时，无需重新生成。安装 Git LFS 后使用 `git lfs pull` 下载实际 Parquet；data/split 约3.12GB，另需清洗输入数据空间及重建时的临时磁盘空间。Git LFS 是否可下载取决于仓库账户实际存储与带宽额度。

