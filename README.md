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
> LFS 免费额度：1 GB 存储 / 1 GB 带宽（每月）。本项目已使用约 720 MB，请勿再上传大文件。

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
