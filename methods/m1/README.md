# M1 树模型工程与 Validation 交付

RF-only辅助对照的五种子正式训练、完整Validation独立AP验收及归档封存已完成。五次AP复算与训练记录严格一致，均值0.9996364875511452、样本标准差0.0000505703708643；原始归档157/157文件SHA256验证通过。详见 [RF结果摘要](../../results/m1_rf_summary/README.md) 和 [RF验收封存审计](../../docs/m1_rf_validation_seal_audit.md)。资源试运行与正式执行均有分别记录的批准，不将RF用于GBDT主模型选优；未读取真实Test、运行实验Bootstrap或标定阈值。工程准备时166项、逐种子检查点调整后174项均为历史测试记录，本次交付回归结果单独记录在审计报告中。

GBDT阶段记录（2026-11-02）：GBDT-only 的八配置 × 五种子共40次拟合、完整Validation独立AP复算、选优及总归档封存已完成，获选配置为G05。该阶段未读取真实Test、训练RF、运行Bootstrap或标定阈值。公开的 [GBDT结果摘要](../../results/m1_gbdt_summary/README.md) 包含小型汇总和原始MANIFEST副本；模型、完整预测和缓存仅保存在本地，GitHub不包含完整归档。封存细节见 [补封存验收](../../docs/m1_manifest_seal_repair.md)。该次完整回归131项通过（59.831秒）。RF交付未修改已封存GBDT结果。

GBDT-only 入口说明：本阶段获批配置独立保存为 `execution.gbdt-search.v1.json`。`python -m methods.m1.search_gbdt plan` 只展示计划；真实执行需要明确启动开关，不能为复现摘要而自动重跑。入口提供 Job 保护、断点恢复和每组人工检查点，详见 [执行计划](../../docs/m1_gbdt_search_plan.md)。下文涉及 `execution.proposed.json` pending 和旧入口45任务的描述适用于后续完整实验草案；本次不将其一并批准，不使用旧 search 入口启动。

资源试运行历史记录：前次中止后，经用户再次确认，修复的整进程树监控下 G01/42 已成功跑满200轮；fit实测308.18秒，工作进程OS峰值RSS3.64 GiB，所有保存与哈希检查通过。该任务不计入正式搜索、没有计算AP或读取test。详见 [成功记录](../../docs/m1_resource_trial_success.md)，前次中止证据仍保留在 [历史记录](../../docs/m1_resource_trial.md)。后续完整评估执行细则仍为pending，GBDT-only搜索细则已独立批准。

协议范围：NIDS Unified Experimental Protocol v1.0已批准冻结。GBDT-only候选、停止规则、AP平分及资源细则已独立获批并执行；RF固定参数、资源限制及逐种子检查点也已另行获批，五次正式训练及独立验收封存完成。原execution.proposed.json和protocol.pending.json仍表示后续完整评估细则pending，不撤销GBDT/RF独立批准，也不授权真实Test、实验Bootstrap或阈值标定。见 [历史正式运行前验收](../../docs/m1_formal_preflight.md)。

正式交付为[results/m1.json](../../results/m1.json)、[PR PNG](../../figures/fig_m1_pr.png)及[PR PDF](../../figures/fig_m1_pr.pdf)。绘图脚本[plot_pr_curve_enlarged.py](plot_pr_curve_enlarged.py)输出统一文件名，使用真实G05五种子Validation预测；来源与SHA256见[PR验收报告](../../docs/m1_pr_curve_enlarged_report.md)。图为Enlarged View，Recall 0.98–1.00、Precision 0.99–1.00，无inset或双面板。

## 环境与短时测试

使用 Python 3.13。依赖入口仍为 `requirements-m1.txt`，新增 `requirements-lock.txt` 记录本机实际安装的完整依赖版本。锁文件是当前环境快照，不表示已在新机器完成重装验收。

全部216项测试分三套运行：核心174项、工具34项、恢复封存8项。必须全部执行；单独discover tests只覆盖174项。统一说明及本次执行结果见[测试说明](../../docs/testing.md)。在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tools/tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe -B -X utf8 docs/test_rf_acceptance_recovery.py
```

`tests/test_m1.py` 和 `tests/test_m1_executor.py` 全部使用合成数据与临时目录。Smoke 集成测试仅训练 40 行合成样本的 3 棵随机森林及 3 轮梯度提升；正式执行器集成测试的 45 个任务均由不训练的替身验证，不执行 40 次真实拟合。正式 Schema 的非 demo 分支只在临时伪数据上测试，其记录不会作为真实实验导出。

历史正式执行器验收为87项测试通过（原有75项加12项执行器测试），资源保护阶段增至92项（2026-11-01提交准备复跑14.083秒）。GBDT-only工程新增20项后为112项，补封存新增19项后为131项；2026-11-02提交准备完整回归131项全部通过，59.831秒，结尾OK。资源测试使用合成输入、临时目录和低内存子进程探针，不读取真实Test或启动正式训练。

## 保留的 Smoke 入口

下面命令仍可运行原版分层数据的 20,000 训练样本、8,000 验证样本自检，但本轮工程验证没有再次执行真实数据训练：

```powershell
.\.venv\Scripts\python.exe methods/m1/run_smoke.py
```

仍使用 RF 与 HistGradientBoostingClassifier，同一批训练/验证样本，P2 预处理和训练权重，固定 0.5 开发阈值；不打开 test。五个 P3 核心指标接口不变。所有结果只属于开发验证子集，不能用于正式论文模型排名。

`config_smoke.json` 中原模型种子保留为 `seed=42`；抽样分别显式固定为 train=42、validation=43，另有独立 bootstrap/timing seed。改变模型 seed 不再改变样本。以上仅是原有开发配置，不是正式五种子方案。

抽样使用一次分配的特征和元数据缓冲区，保留行在原位替换；每批仅拼接优先级，避免反复复制整个保留特征矩阵。完整攻击族列表来自 P2 `docs/label_split_policy.json`，缺失类明确报告 support=0、recall=null，并分别记录训练/验证缺失清单。支持 P2 主版及 no_destination_port 路径，特征维数始终取加载器白名单，不硬编码为 140。

计时批次使用稳定 row ID 哈希选取，两个模型共用相同有序 ID，并记录真实良性/攻击组成；不强行修改类别比例。正式比较可通过 `timing_indices(..., selected_row_ids=...)` 使用团队批准的共同名单。RF 自身 n_jobs 和 P3 线程池限制分别控制，临时推理限制结束后恢复模型原参数。

## 不可覆盖的运行目录

每次 Smoke 运行创建唯一目录：

```text
results/m1_runs/smoke/<UTC时间加UUID>/
  run.json
  config.json
  environment.json
  metrics.json
  predictions.npz
  pr.png
  models/random_forest.joblib
  models/hist_gradient_boosting.joblib
  MANIFEST.json
```

文件使用排他创建，目录冲突或同名文件写入立即拒绝。成功时才生成 MANIFEST，包含每个文件的大小及 SHA256，绑定配置、环境、模型、预测和指标。运行失败记录 FAILED.json，不能被当作完成结果。完成后拒绝通过同一保存器继续写入。

历史 [Smoke JSON](../../results/m1_smoke.json) 和 [Smoke PR 图](../../figures/fig_m1_smoke_pr.png) 随代码保留，不再用作新运行的覆盖输出。`.tmp/m1_smoke/` 为仅本地保留、不随本次提交上传的临时归档。新模型和预测不会只放在临时目录。上述运行目录及 `results/m1_resource_trials/` 均为本地归档，本次不提交其中的模型、预测、缓存或日志；文档中的目录名仅用于本机定位，不是 GitHub 可访问链接。是否上传大模型、如何为 P5/P7 提供下载仍需另行安排。

独立校验一个新运行目录：

```powershell
.\.venv\Scripts\python.exe -B -m methods.m1.storage results/m1_runs/smoke/<run-id>
```

它检查文件集、大小、哈希、配置对应、完整 Smoke 数值一致性，并从保存的标签、攻击族、attempted 标记和分数独立复算。行顺序由摘要约束；不依赖训练进程中残留的 DataFrame。不加载外部模型、不训练或读取原始数据集。

Smoke 有独立验证格式，复用 P3 指标 Schema 片段，并补充混淆矩阵、AP/ROC 曲线、分类别支持数、时延和 bootstrap 的跨字段检查。不会把 validation 改名为 test 来通过 P3 正式 Schema。兼容校验旧 Smoke 记录；新增 v2 记录还强制完整攻击族、种子、模型有效参数、计时 ID、标签组成与行 ID 摘要。

## 正式准备入口和批准门禁

```powershell
.\.venv\Scripts\python.exe -B -m methods.m1.formal --protocol methods/m1/protocol.pending.json
```

该入口保留给旧版工程准备兼容。protocol.pending.json 已填入获批通用条款并链接统一冻结文件；其外层 pending 仅表示执行细则待确认。旧入口不是本轮正式运行入口。批准记录只记录当前会话中的明确授权，不伪造同学姓名或签字。

旧通用执行器的只读检查入口如下，check不打开数据集；GBDT-only和RF-only独立入口已各自获批完成，不使用该旧入口重跑：

```powershell
.\.venv\Scripts\python.exe -B -m methods.m1.executor check
```

该旧入口当前预期为BLOCKED，原因是它读取的execution.proposed.json仍pending。已完成实验分别使用execution.gbdt-search.v1.json、execution.rf-only.proposed.json及对应独立入口，不受此历史草案状态混淆。测试中的模拟批准只存在内存中，不是正式批准文件。

细则确认后使用独立命令 search、select、evaluate。search 串行执行 40 个 GBDT 和 5 个固定 RF 任务，只读 train/val；select 用完整验证集五种子平均 AP 选 GBDT 并冻结 20 个阈值；evaluate 必须引用已经封存的 selection 目录并显式传入 --permit-final-test。它在打开 test 之前重验配置、模型、获胜配置和验证阈值。不要现在执行这些命令。运行方式、具体待批参数和存档结构见正式运行前验收文档。

## 未来全量数据准备接口

`materialize_preparation()` 限定 train/val 文件，通过 P2 加载器核验来源后，一次预分配磁盘映射 X.npy/y.npy，分块写入元数据 JSONL。`DiskDataset.model_input()` 可用作上下文管理器，显式释放 Windows 映射句柄。`assert_disk_isolated()` 用指定 DuckDB 内存上限检查行 ID 唯一性和跨折交集。

该接口最初通过100行合成数据验收，后来已用于获批GBDT/RF完整Train/Validation实验；实测资源记录见各自正式与资源试运行报告。它避免装载层反复拼接，不承诺sklearn内部完全无复制。后续扩展及Test评估仍需独立授权，不能因历史全量缓存存在就自动执行。

## 后续完整评估仍待确认的细则

- GBDT八组候选、固定轮数/early_stopping=false、AP平分容差1e-6及最小配置ID规则，以及RF固定参数均已获批并执行，不再列为待批。
- 若未来更改停止规则或参数，须另行审批和版本化，不能改写现有封存记录。
- 验证 FPR 并列分数边界处理、Bootstrap 次数/置信度/种子及对比范围。
- 后续P3统一计时名单的生成和交接细则；GBDT/RF搜索阶段已批准使用的资源参数保持原记录，不再列为未批准。
- 后续扩展具体排期。已有的主数据、8 个主配置、五个种子、监督 AP 选优、1%/5% 阈值和扩展顺序不再列为未批准。
