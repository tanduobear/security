# P7 正式复跑与结果接入说明

2026年11月14日正式corrected训练与评估已完成，结果文件验收通过。汇总、四图、4.4、Discussion及三份登记文档已使用实际结果更新。运行收据与交付索引见[结果与交接](p7_results_and_handover.md)，以下命令保留供复现。

## 三个工作目录

| 目录 | 用途 |
|---|---|
| C:\Users\Administrator\Desktop\Security\repo | 保留原main、P6原版实验及现有GPU环境，不在此重跑corrected |
| C:\Users\Administrator\Desktop\Security\p7-run-3632afa | 固定在3632afa的正式训练目录；corrected数据和全部训练输出在这里 |
| C:\Users\Administrator\Desktop\Security\p7-delivery | P7验收、汇总、作图与最终文书的独立分支，不修改冻结训练源码 |

运行环境为Python 3.11.13、PyTorch 2.11.0+cu128，CUDA可用。corrected实际train/val/test文件已下载至固定运行目录并通过预检。P6完整归档也已就绪。首次M2启动缺少tqdm，补装后五种子训练与评估成功。

## 第一步 下载修正版数据并检查

在PowerShell执行：

```powershell
cd C:\Users\Administrator\Desktop\Security\p7-run-3632afa
$p7python = 'C:\Users\Administrator\Desktop\Security\repo\.venv-p6\Scripts\python.exe'
$p7driver = 'C:\Users\Administrator\Desktop\Security\p7-delivery\methods\p7\run_formal.ps1'

git -c safe.directory=C:/Users/Administrator/Desktop/Security/p7-run-3632afa rev-parse --short HEAD
git -c safe.directory=C:/Users/Administrator/Desktop/Security/p7-run-3632afa lfs pull --include="data/split/corrected/stratified/train.parquet,data/split/corrected/stratified/val.parquet,data/split/corrected/stratified/test.parquet" --exclude=""

powershell -NoProfile -ExecutionPolicy Bypass -File $p7driver -Stage Check -Python $p7python
```

HEAD必须是3632afa。Check只核查环境、数据、配置和审批，不训练；通过后写results/p7_corrected/receipts/Check.json。M1正式计划必须为approved_ready，v2 payload SHA256必须为：

```text
8cea962ee3c6a221bb7307799d53dcf016779440b4dbebdeeea281e399917718
```

旧preflight中的“M1 full rerun remains blocked”描述检查的是历史spec标记，不代表v2执行审批无效。实际可否启动以run_m1_full.py plan的approved_ready及issues为空为准。

如需在这台机器重做小样本自检，可在Check通过后单独执行下面命令；它们不是正式实验结果：

```powershell
& $p7python methods/p7/run.py m1-smoke
& $p7python methods/p7/run.py m2-smoke
& $p7python methods/p7/run.py m3-smoke
```

## 第二步 按阶段串行执行正式任务

每条命令成功结束后再执行下一条。不要并行开启多个大型训练进程。

```powershell
# G05和RF_FIXED，各5种子，共10次拟合；所有阈值冻结后才读取Test。
powershell -NoProfile -ExecutionPolicy Bypass -File $p7driver -Stage M1 -Python $p7python

# 1D-CNN：5种子，随后独立评估。
powershell -NoProfile -ExecutionPolicy Bypass -File $p7driver -Stage M2Train -Python $p7python
powershell -NoProfile -ExecutionPolicy Bypass -File $p7driver -Stage M2Evaluate -Python $p7python

# 自编码器：8配置乘5种子，随后评估获选模型的两个预定阈值。
powershell -NoProfile -ExecutionPolicy Bypass -File $p7driver -Stage M3Search -Python $p7python
powershell -NoProfile -ExecutionPolicy Bypass -File $p7driver -Stage M3Evaluate -Python $p7python
```

包装脚本只调用3632afa原有入口，记录Git版本、配置与驱动哈希、时间和日志。每个已完成阶段有receipt；不会自动把失败当成功。已有完整阶段禁止意外重跑，M2已有模型禁止自动覆盖。

M1原入口不支持中途断点续跑；失败后保留FAILED.json并先分析，不要直接重复启动整批训练。M3支持跳过已经完成且哈希匹配的配置/种子，但中断的单次拟合会从头跑。

全部输出位于p7-run-3632afa/results/p7_corrected/。M1应有20份阈值记录（2模型×5种子×2阈值）；M2应有五个模型和m2_corrected.json；M3应有40次搜索、10份获选AE阈值记录及预测文件。

M1运行时保留原训练窗口，另开PowerShell查看每5秒刷新的进度：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Administrator\Desktop\Security\p7-delivery\methods\p7\watch_m1_progress.ps1
```

显示当前日志、已完成训练及验证任务数（共10项）、Test记录数（共20份）和耗时。这里的进度条统计已完成任务，不表示单次GBDT/RF拟合内部的精确百分比；不修改已封存训练参数或审批哈希。Ctrl+C只退出监控窗口，不会停止另一个窗口中的训练。

## 第三步 验收并生成正式汇总

结果齐全后，切换到交付工作区：

```powershell
cd C:\Users\Administrator\Desktop\Security\p7-delivery
& $p7python methods/p7/build_deliverables.py --run-root C:\Users\Administrator\Desktop\Security\p7-run-3632afa
& $p7python methods/p7/build_paper_figures.py
```

如果存在多个完成的M1运行，第一条命令会停止。检查后加--m1-run并指定要验收的formal子目录；不会擅自选择最新目录。验收失败时不会用冒烟数值或空白corrected行替代正式结果。

输出包括results/summary.csv、results/p7_figure_data.json、results/p7_plot_values.csv和results/p7_acceptance.json，并导出results/{m1,m2,m3}_corrected.json紧凑结果及results/p7_corrected_summary/来源证据。四张最终PNG/PDF为fig_p7_overall_performance、fig_p7_tpr_at_fpr、fig_p7_family_recall和fig_p7_version_delta。旧fig_p7_threshold_metrics及旧fig1–fig4图是历史产物，不应自动当作本次最终四图。

汇总以M1的验证1%阈值、M2验证所选阈值和M3 q99作为分类工作点；TPR@FPR另外使用经验Test ROC指标，不混写。M3图例使用实际获选配置，不固定假定corrected仍为A06。RF辅助结果单列，不纳入三种主模型排名。

不存在的攻击族用null/空白表示，不当作召回为0。original和corrected是独立评估总体，图4仅为五种子均值之差，不冒充配对置信区间。自编码器在相同候选网格中重新选模，版本差异也可能包含选模敏感性，不能全归因于标签修正。

## 验收范围与仍需审阅的事项

验收器检查M1 v2审批、版本、manifest、10个冻结任务、20份唯一记录；M2 corrected身份、CUDA、五种子和训练receipt；M3搜索完整性、选优规则、数据与模型哈希、固定阈值和版本来源；并核对样本数与混淆矩阵。

3632afa的M2评估器没有导出逐样本分数。因此本验收不能独立验证M2逐行预测排序，报告会明确保留该限制；若需要跨方法配对bootstrap，必须另外导出并对齐M2分数，不能编造row_id。未测量的corrected M1推理时延保持空白，不伪造计时数据。不同机器的训练耗时也不能直接当算法速度对比。

## 最后定稿与GitHub交付

真实结果已接入§4.4和Discussion，三张Word登记表已填写执行日期与交付位置。M3获选A05，主模型排序保持，1%工作点下降已披露。P8仍需核对数字溯源、图号和全文衔接。

最终提交包含紧凑汇总、验收证据、定稿PNG/PDF、文书和代码。大型模型、缓存、逐样本预测保留在隔离运行目录。发布前检查git diff --cached --stat。

开发用--preview-original只写results/p7_preview/，与正式结果隔离。验收逻辑的9项单元测试已通过，它们不是corrected真实训练验收。
