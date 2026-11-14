# P7 正式复跑结果与交接

2026年11月14日完成 M1、M2、M3 的 corrected 正式复跑及结果文件校验。执行人员为本机用户，承担 P7 复跑。环境为 Windows、Python 3.11.13、PyTorch 2.11.0+cu128、RTX 5090 Laptop GPU。M1 使用 CPU，M2 与 M3 搜索使用 CUDA，M3 最终评估使用 CPU。

## 运行与验收

训练源码固定在提交 `3632afa39a6d07442c5c7a16054dbb6aa0fd0b9b`。M1 v2 审批负载 SHA256 为 `8cea962ee3c6a221bb7307799d53dcf016779440b4dbebdeeea281e399917718`。运行根目录为本机 `p7-run-3632afa/`，原始结果均位于其 `results/p7_corrected/` 下。集成与交付位于 `p7-delivery/` 的 `p7-final-delivery` 分支。

| 阶段 | 本机时间 | 完成内容 |
|---|---|---|
| M1 | 14:42:58–15:35:25 | G05 与 RF_FIXED 各五种子，共10次训练、20条阈值记录 |
| M2 训练 | 16:04:02–18:54:34 | 固定 CNN，种子42–46，每个40轮，CUDA |
| M2 评估 | 18:57:48–18:58:42 | 五种子验证阈值、Test评估与CPU延迟测量 |
| M3 搜索 | 19:01:29–23:02:46 | A01–A08各五种子，共40次，获选A05 |
| M3 评估 | 23:05:09–23:08:35 | A05五种子，每个q95与q99，共10条记录 |

实际阶段收据见 `results/p7_corrected_summary/receipts.json`。M2首次启动缺少tqdm，在训练开始前退出。补装后完成上述正式训练。

验收核对了 M1 审批负载、源码和配置哈希、77项归档文件、10个冻结任务及20份唯一记录。M2核对数据身份、五种子、训练历史和CUDA信息。M3核对40组唯一任务、参数、验证集选优规则、模型及预测哈希、阈值来源、数据版本和源码。三种方法均使用 corrected/stratified/main 的314,577条Test样本。结果中的混淆矩阵与FPR、TPR一致。

M1先冻结全部模型和验证阈值，再创建Test加载器。M3搜索记录 `test_accessed=false`，阈值来自良性验证数据，评估记录与已冻结阈值一致。原P6交付的搜索和阈值记录经文件复核，原版A06结果继续保留。此次未重新拟合original模型。验收覆盖文件、来源与跨字段一致性，具体限制见下文及 `results/p7_acceptance.json`。

## 结果

数值为五种子均值，样本标准差见 `results/summary.csv`。

| 方法与版本 | AP | ROC-AUC | TPR@1% FPR | TPR@5% FPR |
|---|---:|---:|---:|---:|
| M1 original G05 | 0.999495 | 0.999907 | 0.999966 | 0.999997 |
| M1 corrected G05 | 0.99999985 | 0.99999996 | 1.000000 | 1.000000 |
| M2 original | 0.917158 | 0.980888 | 0.611478 | 0.795075 |
| M2 corrected | 0.986249 | 0.997791 | 0.941949 | 0.999861 |
| M3 original A06 | 0.742710 | 0.904003 | 0.527228 | 0.617721 |
| M3 corrected A05 | 0.866415 | 0.979057 | 0.183953 | 0.953509 |

主模型排序在四项指标和两版数据上均为M1、M2、M3。M3的AP、AUC和5%工作点改善，同时1%工作点下降34.3275个百分点。分类别差异、稀有类别支持数和M2种子43波动已写入正文。M1固定原选优G05，M2固定原CNN，M3沿用原八组候选和选优规则并重新搜索，因此M3版本效应包含选模变化。

## 交付文件

| 交付项 | 路径 |
|---|---|
| 汇总表及40条种子级标量结果 | `results/summary.csv`、`results/p7_figure_data.json` |
| corrected紧凑结果 | `results/m1_corrected.json`、`results/m2_corrected.json`、`results/m3_corrected.json` |
| 八配置记录、冻结记录、阶段收据、源文件哈希 | `results/p7_corrected_summary/` |
| 验收与限制 | `results/p7_acceptance.json` |
| 最终文件校验、文档排版检查与交付哈希 | `docs/p7_delivery_checks.json` |
| 四图对应数值 | `results/p7_plot_values.csv` |
| 图1 排名性能 | `figures/fig_p7_overall_performance.png`、同名PDF |
| 图2 低FPR召回 | `figures/fig_p7_tpr_at_fpr.png`、同名PDF |
| 图3 攻击族召回 | `figures/fig_p7_family_recall.png`、同名PDF |
| 图4 版本差异 | `figures/fig_p7_version_delta.png`、同名PDF |
| Results 4.4及Discussion | `docs/p7_text.md` |
| 三份登记文档 | 根目录 `交付登记表.docx`、`进度看板.docx`、`问题记录.docx` |

紧凑JSON保留标量、分类计数、攻击族、操作点及阈值信息。完整曲线、模型、逐样本预测和缓存保存在本机运行目录。`source_manifest.json`提供源文件字节数与SHA256，用于对照完整归档。

P8可直接从公开汇总重绘四图，使用 `python methods/p7/build_paper_figures.py`。从本机原始输出重新验收并生成公开摘要，使用 `python methods/p7/build_deliverables.py --run-root ../p7-run-3632afa`。全文图号由P8统一编排，上表按P7内部图1–4对应。

## 仍需披露和团队审阅的事项

M2缺少八配置搜索证据。本次固定配置复跑已完成，S5全面验收中的搜索预算问题仍开放，后续补做须单独形成验证集选优记录。原M2导出中的P3曲线及并列操作点差异也保留为历史问题。

3632afa的M2评估器未导出逐样本分数，因此本次无法独立复核其逐行预测身份或生成三方配对bootstrap。original与corrected的总体、类别比例及支持数不同，图4报告均值之差。corrected M1未测量推理时延，表中相应字段为空。跨机器耗时仅描述实际运行。

手册原定图1为PR曲线，现有公共摘要缺少完整三方PR数组，因此采用AP与ROC-AUC对比图，并在论文与交接中披露。四图的300 DPI PNG和PDF均由真实汇总生成。P8仍需核对数字溯源、图号与全文衔接，并填写S7验收栏。
