# P6 自编码器运行说明

从仓库根目录执行以下 PowerShell 命令。代码已准备，训练和复跑由 P6 本人执行。

## 1 打开仓库并检查 Python

本次工作副本：

```powershell
cd C:\Users\Administrator\Desktop\Security\repo
python --version
```

P2 的 ModelDataset 使用 hashlib.file_digest，需要 Python 3.11 及以上。本次已在 Python 3.11.13、PyTorch 2.11.0+cu128 环境完成训练。已有该环境时直接使用，不必重新安装。新环境可按下列步骤建立。

```powershell
py -3.11 -m venv .venv-p6
.\.venv-p6\Scripts\python.exe -m pip install -r methods/m3/requirements-m3.txt
.\.venv-p6\Scripts\python.exe -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
```

上述为本次成功使用的 CUDA 12.8 构建。仍需核对 torch.cuda.is_available()；本流程 device=auto，无可用 CUDA 时会使用 CPU，正式搜索可能较慢。

后文 `$p6python` 指向本人选定的解释器：

```powershell
$p6python = '.\.venv-p6\Scripts\python.exe'
& $p6python -c "import sys,torch; print(sys.version); print(torch.__version__); print('CUDA:',torch.cuda.is_available())"
& $p6python -m pip freeze > docs/p6_environment.txt
```

## 2 生成 P5 静态验收证据

此命令不读取真实数据，不训练模型，也不会写“验收通过”。它会列出当前逐种子 Schema 不兼容项。

```powershell
& $p6python methods/m3/audit_m2.py
```

输出为 docs/p6_acceptance/m2_static_audit.json。共享表填写草稿在 docs/p6_next_steps.md。

P5 README 原始复跑入口是 python methods/m2/train.py 和 python methods/m2/evaluate.py。当前训练入口会重写已交付模型，评估入口会读取 Test 并覆盖 results/m2.json，因此本说明不把它们放在默认执行序列中。需要复跑时，先在独立副本保留原交付，统一测试使用约定，并记录环境和差异。

## 3 只下载 M3 训练与验证数据

```powershell
git lfs version
git lfs pull --include="data/split/original/stratified/train.parquet,data/split/original/stratified/val.parquet" --exclude=""
```

此处只需要约 254 MB 两份 parquet；不需要下载全部 clean 数据、其他划分或修正版。preprocessor.json、preprocess_manifest.json 已随源码交付。

## 4 检查环境和数据

```powershell
& $p6python methods/m3/run.py preflight
```

train 与 val 的 real_parquet 应为 true。search 不要求 test 已下载。Python、依赖、数据不符合要求时会停止并显示原因。

## 5 先做小样本自检

```powershell
& $p6python methods/m3/run.py smoke
```

只取最多 2048 条训练良性样本、512 条验证良性样本；一组配置、一个种子、两个 epoch。输出 results/m3_smoke/search.json，status=smoke_only。这是流程检查，不能写进论文作为正式实验结果，也不能用来接着执行正式 test 评估。

随后复跑 P5 已交付权重的验证集预测，不重训、不覆盖原交付：

```powershell
& $p6python methods/m3/replay_m2.py --split val
```

结果在 docs/p6_acceptance/m2_replay_val/report.json。此步只核对验证阈值与模型可运行性，不代表已验证 Test 指标，更不能代表重训一致性。GPU/CPU 浮点差异可能影响阈值和边界样本，脚本报告实际差异，默认绝对容差 1e-5，不擅自改写原结果。

若团队已明确允许最终 Test 复核，在第 7 步下载 Test 后可单独执行：

```powershell
& $p6python methods/m3/replay_m2.py --split test
```

结果在 docs/p6_acceptance/m2_replay_test/report.json，并保留五种子逐样本分数。这里只复跑已交付 checkpoint，训练复现仍需独立执行和报告。

## 6 正式训练搜索

运行前阅读并固定 config.json。八个候选配置是本次为 P6 提出的默认方案，未伪称为既有团队决定。记录早停规则、候选结构及辅助方法采样上限；改配置后用新的输出目录重新运行，不覆盖原实验。

```powershell
& $p6python methods/m3/run.py search
```

八配置乘五种子，串行运行。只使用 train/val 的 y_binary=0 样本，不读取 Test，不重新拟合 P2 预处理器。沿用 P2 负类映射，其中可能含 attempted 样本，阈值记录会披露。最佳 epoch 按良性验证 MSE 早停；配置按五种子平均良性验证 MSE 最小选择，同值按编号排序。

每个已完成配置/种子保存 checkpoint 和 run.json。同一命令重跑会跳过配置、数据和权重校验均通过的已完成任务；中断的单次训练从头重跑，不会伪称逐 epoch 恢复。

主要输出：

```text
results/m3_runs/plan.json
results/m3_runs/A01/42/model.pt
results/m3_runs/A01/42/run.json
results/m3_runs/search.json
results/m3_runs/m3_threshold.md
```

## 7 独立执行最终测试评估

仅在正式搜索完成且团队测试执行口径明确后执行。阈值已在验证集良性样本上固定为 95% 和 99% 分位数，误差 >= 阈值判为攻击；不根据 Test 重新选配置或阈值。

```powershell
git lfs pull --include="data/split/original/stratified/test.parquet" --exclude=""
& $p6python methods/m3/run.py evaluate
```

生成获选配置五种子的两个阈值版本。每份 records/*.json 调用 P3 Schema 和数值一致性校验。predictions/*.parquet 保留原始 row_id、标签、攻击族、attempted 标志及异常分数，供 P7 配对比较。

results/m3_runs/m3.json 是索引文件，不是单种子 P3 记录；不要直接把它送入单记录校验器。均值和样本标准差、跨方法配对 bootstrap 可由 P7 从逐种子记录与对齐分数汇总，不能把未完成统计写作已交付。

图默认展示获选模型第一个种子的 Test 分布，包含 q95/q99；不表示五种子均值。

## 8 两个辅助对照

```powershell
& $p6python methods/m3/run.py auxiliary
```

Isolation Forest 和 One-Class SVM 各 fit 一次。为了限制核 SVM 成本，两者使用同一固定 seed=42 的最多 10000 条训练良性样本，配置及训练子集索引摘要写入记录；这是辅助实验的明确范围，不声称全量训练或参与主模型五种子统计。使用全部验证良性样本标定两个阈值，再独立读取 Test 评估。异常分数取负 decision_function，分数越大越异常。

## 9 校验和整理交付

```powershell
Get-ChildItem results/m3_runs/records/*.json | ForEach-Object {
    & $p6python -m eval.validate $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "结果校验失败：$($_.Name)" }
}
Copy-Item results/m3_runs/m3.json results/m3.json
Copy-Item results/m3_runs/m3_threshold.md docs/m3_threshold.md
Copy-Item results/m3_runs/fig_m3_error_dist.png figures/fig_m3_error_dist.png
git status --short
```

本地 checkpoint、逐样本预测与搜索目录已加入忽略规则；完整档案保留本机或团队存储，不用 git add . 上传全部。正式交付需同时提供逐种子 records 的可访问位置，不能只交一个索引。

论文 §3.5 与 §4.3 已按实际结果整理在 docs/p6_text.md。S6 在实际完成并交给 P7 后再填完成时间。

## 10 最终只读核查及交付包

以下均不重新训练或执行模型推理。先用已保存分数改进图的可读性，再审计和打包：

```powershell
& $p6python methods/m3/plot_error_distribution.py
& $p6python methods/m3/audit_handoff.py
if ($LASTEXITCODE -ne 0) { throw '最终核查失败，请先检查 final_audit.json' }
& $p6python methods/m3/package_handoff.py
```

改进图保存为 figures/fig_m3_error_dist_v2.png，原图保留。两幅子图分别显示全范围和明确标注的局部范围，采用相同分箱比较两类分布，局部图不重新归一化。

审计输出 docs/p6_acceptance/final_audit.json，包括14份结果的P3校验、逐样本身份和标签、保存分数复算指标、40个模型哈希、选择依据及当前环境版本。它不代替P7独立复跑阈值。

交付包 submission/P6_handoff.zip 保留索引引用的相对路径，包含代码、全部模型、14份记录、预测、正文和阈值记录，不包含大数据集。附带 SHA256 和包内逐文件清单，解决 m3_runs 被 Git 忽略导致交接不完整的问题。若包已存在，脚本要求先重命名旧包，不覆盖旧交付。

本人仍需在共享表登记包的实际共享位置，由P7验收，不在脚本中代签。
