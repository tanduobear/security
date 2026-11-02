# M1 树模型工程准备

当前正式结果（2026-11-02）：GBDT-only 的八配置 × 五种子共40次拟合、完整Validation独立AP复算、选优及总归档封存已完成，获选配置为G05。未读取真实Test、训练RF、运行Bootstrap或标定阈值。公开的 [结果摘要](../../results/m1_gbdt_summary/README.md) 包含小型汇总和原始MANIFEST副本；模型、完整预测和缓存仅保存在本地，GitHub不包含完整归档。封存细节见 [补封存验收](../../docs/m1_manifest_seal_repair.md)。本次完整回归131项通过（59.831秒）。

GBDT-only 入口说明：本阶段获批配置独立保存为 `execution.gbdt-search.v1.json`。`python -m methods.m1.search_gbdt plan` 只展示计划；真实执行需要明确启动开关，不能为复现摘要而自动重跑。入口提供 Job 保护、断点恢复和每组人工检查点，详见 [执行计划](../../docs/m1_gbdt_search_plan.md)。下文涉及 `execution.proposed.json` pending 和旧入口45任务的描述适用于后续完整实验草案；本次不将其一并批准，不使用旧 search 入口启动。

资源试运行历史记录：前次中止后，经用户再次确认，修复的整进程树监控下 G01/42 已成功跑满200轮；fit实测308.18秒，工作进程OS峰值RSS3.64 GiB，所有保存与哈希检查通过。该任务不计入正式搜索、没有计算AP或读取test。详见 [成功记录](../../docs/m1_resource_trial_success.md)，前次中止证据仍保留在 [历史记录](../../docs/m1_resource_trial.md)。后续完整评估执行细则仍为pending，GBDT-only搜索细则已独立批准。

协议范围：NIDS Unified Experimental Protocol v1.0 已按用户报告的 P4/P5/P6/P7 集体批准冻结。GBDT-only候选、固定轮数、AP平分和资源细则随后单独获批并执行。RF参数、Bootstrap及后续完整评估的具体草案仍保持pending；旧完整执行器的实现与合成验收不代表已授权真实最终评估。见 [历史正式运行前验收](../../docs/m1_formal_preflight.md)。

## 环境与短时测试

使用 Python 3.13。依赖入口仍为 `requirements-m1.txt`，新增 `requirements-lock.txt` 记录本机实际安装的完整依赖版本。锁文件是当前环境快照，不表示已在新机器完成重装验收。

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py' -v
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

本轮正式入口如下，check 不打开数据集：

```powershell
.\.venv\Scripts\python.exe -B -m methods.m1.executor check
```

当前预期为 BLOCKED，仅缺具体执行细则批准。protocol.unified.v1.json 为通用协议冻结文件；execution.proposed.json 包含八组 GBDT 和固定 RF 的推荐细则，approval 仍为 pending。测试中的模拟批准只存在内存中，不是正式批准文件。

细则确认后使用独立命令 search、select、evaluate。search 串行执行 40 个 GBDT 和 5 个固定 RF 任务，只读 train/val；select 用完整验证集五种子平均 AP 选 GBDT 并冻结 20 个阈值；evaluate 必须引用已经封存的 selection 目录并显式传入 --permit-final-test。它在打开 test 之前重验配置、模型、获胜配置和验证阈值。不要现在执行这些命令。运行方式、具体待批参数和存档结构见正式运行前验收文档。

## 未来全量数据准备接口

`materialize_preparation()` 限定 train/val 文件，通过 P2 加载器核验来源后，一次预分配磁盘映射 X.npy/y.npy，分块写入元数据 JSONL。`DiskDataset.model_input()` 可用作上下文管理器，显式释放 Windows 映射句柄。`assert_disk_isolated()` 用指定 DuckDB 内存上限检查行 ID 唯一性和跨折交集。

这解决装载层反复拼接问题，不承诺 sklearn 拟合不再产生内部复制。算法线程、DuckDB 内存上限、峰值 RSS 和实际全量耗时仍需协议批准后测量。本轮该接口仅由 100 行合成数据测试，不生成真实全量缓存。

## 尚待确认的执行细则

- 八组具体 GBDT 候选和固定 RF 辅助参数；RF 不占主方法选优机会。
- GBDT 停止细则：本版建议显式固定轮数、不使用 Early Stopping；若选择早停，要先明确并实现另一套经过测试的规则，不能只改一个开关。
- AP 平分容差和决定性平分规则。
- 验证 FPR 并列分数边界处理、Bootstrap 次数/置信度/种子及对比范围。
- 共同计时名单的生成和交接细则、训练/推理线程及分块大小。
- 后续扩展具体排期。已有的主数据、8 个主配置、五个种子、监督 AP 选优、1%/5% 阈值和扩展顺序不再列为未批准。
