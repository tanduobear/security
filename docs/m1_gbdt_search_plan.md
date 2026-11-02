# GBDT-only 正式搜索执行计划

状态：搜索细则已由用户明确批准；本轮仅实现、合成测试和生成计划，未启动真实搜索。统一协议维持 approved；`execution.proposed.json` 中后续评估细则仍 pending。旧 `executor.py search` 仍是 45 任务入口，不能用于本次搜索。

## 配置与批准范围

配置：[execution.gbdt-search.v1.json](../methods/m1/execution.gbdt-search.v1.json)。

- 文件 SHA256：`931d2d6e6639a825b62dd7e21f23f4a63e0e30a6a631b58ffd3469afca81b08b`。
- 获批执行载荷 SHA256（不含 approval，按规范 JSON 序列化）：`b1f67db0e2a9dd8fcd0cd29b799cc1082c9f5cc72d40e20448b83eb3963ef1ae`。
- original / stratified / main，完整 train/val，P2 140 维特征、固定预处理器与训练折 sample_weight，不叠加 class_weight。
- HistGradientBoostingClassifier，max_bins=255、early_stopping=false。固定各配置轮数，不提前淘汰或追加配置。
- 五种子完整验证 AP 均值选优，样本标准差 ddof=1。与最大均值相差不超过 1e-6 时取最小配置 ID。
- 不训练 RF、不打开真实 test、不标定检测阈值、不执行 Bootstrap 或最终评估。不将资源试运行计为正式任务。

| 配置 | max_iter | learning_rate | max_leaf_nodes | min_samples_leaf | L2 |
|---|---:|---:|---:|---:|---:|
| G01 | 200 | 0.05 | 15 | 20 | 1 |
| G02 | 400 | 0.05 | 15 | 20 | 1 |
| G03 | 200 | 0.10 | 15 | 20 | 1 |
| G04 | 200 | 0.05 | 31 | 20 | 1 |
| G05 | 400 | 0.05 | 31 | 20 | 1 |
| G06 | 200 | 0.05 | 15 | 5 | 1 |
| G07 | 200 | 0.05 | 15 | 50 | 1 |
| G08 | 200 | 0.05 | 15 | 20 | 10 |

## 完整任务顺序

| 序号 | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| 1–5 | G01_seed_42 | G01_seed_43 | G01_seed_44 | G01_seed_45 | G01_seed_46 |
| 6–10 | G02_seed_42 | G02_seed_43 | G02_seed_44 | G02_seed_45 | G02_seed_46 |
| 11–15 | G03_seed_42 | G03_seed_43 | G03_seed_44 | G03_seed_45 | G03_seed_46 |
| 16–20 | G04_seed_42 | G04_seed_43 | G04_seed_44 | G04_seed_45 | G04_seed_46 |
| 21–25 | G05_seed_42 | G05_seed_43 | G05_seed_44 | G05_seed_45 | G05_seed_46 |
| 26–30 | G06_seed_42 | G06_seed_43 | G06_seed_44 | G06_seed_45 | G06_seed_46 |
| 31–35 | G07_seed_42 | G07_seed_43 | G07_seed_44 | G07_seed_45 | G07_seed_46 |
| 36–40 | G08_seed_42 | G08_seed_43 | G08_seed_44 | G08_seed_45 | G08_seed_46 |

逐行、从左到右执行；每行结束为一个人工检查点。

## 资源与前台观察

独立入口为 `methods.m1.search_gbdt`。前台父进程保持轻量；每组使用一个 Windows Job 保护的子进程，父进程赋予 Job 内存硬限和 CPU 亲和性、核验成功后发送 GO。子进程此前不加载 ML 栈或数据。选择阶段也走同一保护入口。

- Job 进程树提交内存硬限 8 GiB，CPU 亲和性为 4 个逻辑处理器，训练线程 4、预测线程 1、预测块 8192。
- 每组启动前系统可用内存至少 14 GiB；运行中系统可用内存低于 6 GiB 或整树采样 RSS 超过 8 GiB 时终止 Job。
- RSS、私有内存、Job CPU/I/O、系统可用内存每秒采样；本组监控目录体积每 5 秒采样（不是整个搜索归档体积）。提交内存硬限独立于采样即时生效。
- 外层串行；全局 OS 锁防止另一个新入口运行同时训练。退出、Ctrl+C 或异常关闭后，Job 清理子进程，OS 锁自动释放。
- 不自动降参、取消限制或重试。未移植资源试运行的整次 60 分钟超时或 10 GiB 目录上限，以免把它们误当作已批准搜索细则。
- P2 自建 DuckDB 连接仍无内存参数；由外层 Job 兜底。P4 的跨折 ID 检查单独使用 512 MB 内存池，不改 P2。
- 前台显示阶段变化并约每 30 秒报告资源；每组结束逐一列出五个 AP、训练秒数及资源 CSV 路径。不声称提供逐轮进度。

每组结束后程序要求输入 `CONTINUE`。检查五个任务、资源 CSV、保存校验和异常日志，再决定是否继续。其他输入或 EOF 会停止，保留已完成任务；最后一组也必须确认。建议人工记录：任务 ID、开始/结束时间、训练秒数、验证 AP、资源峰值、输出校验状态和备注。不根据中途 AP 修改配置。

## 归档与恢复

新目录：`results/m1_gbdt_search/<UTC时间-UUID>/`，仅本地运行归档，不覆盖已有 Smoke、Resource Trial 或旧正式入口目录。

- config.json：搜索专用配置；binding.json：配置摘要、代码逐文件哈希、环境、train/val/处理器/清单哈希及统一协议文件哈希；sources/ 保留初始源码快照。
- cache/、cache.json：只包含 train/val 映射缓存及逐文件大小、SHA256。
- attempts/<任务-UUID>/：模型、配置、完整验证标签/row ID/连续攻击分数；每次尝试独立，失败残留不删除。
- completed/<任务>.json：仅在模型重载与预测保存一致性校验通过后发布，绑定任务、输入身份及输出哈希。
- monitor/<组-UUID>/：Job 验证记录、阶段事件、worker.log、资源 CSV、工作结果和父监控结果。
- checkpoints/：人工 CONTINUE 确认记录，绑定本次运行身份。
- selection.json：独立验证选优，不要求 RF；最终 MANIFEST.json 覆盖归档文件大小及 SHA256。

显式恢复时先验证配置、代码（含未提交文件实际字节）、环境及真实 train/val 身份；检查完成任务输出文件集、SHA256、任务参数、预测行号并独立复算 AP。已完成任务跳过，不重新训练；不完整尝试只在用户显式恢复后重新执行。损坏记录、缓存或身份不一致直接拒绝，不自动修复。代码或依赖改变后不能继续同一运行。已封存运行不允许恢复写入。

组间恢复会重新核验缓存和已有记录；这有校验开销，但不会重做已完成的模型拟合。不可解析的完成记录或初始化不完整需人工检查，不能自行删除后强制重试。

## 命令与授权边界

现在可执行的只读计划命令（不读取数据、不训练）：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.search_gbdt plan
```

以下命令仅供后续获得启动确认后，在用户可见的 PowerShell 前台运行；本轮没有执行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.search_gbdt run --confirm-start
# 显式恢复，RUN 为本地归档路径：
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.search_gbdt run --confirm-start --resume <RUN>
# 40 个任务及八次人工确认完成后，独立验证选优：
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.search_gbdt select --confirm-start --resume <RUN>
```

批准实验配置不等于批准立即启动，入口额外要求显式启动开关。工程与合成测试通过不代表全部候选在真实数据上已验证资源可行性；首次 G01 资源试运行不是其他配置的耗时或内存保证。

## 本轮验收记录

2026-11-01，完整 unittest 回归 **112 项全部通过，25.739 秒，结尾 OK**。包含原有 92 项及新增 20 项；没有 skip 的报告。测试入口：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p 'test_*.py' -q
```

新增测试覆盖：精确 40 任务、配置与批准摘要拒绝、显式启动门禁、GO 前禁止装载、合成 worker 到五任务保存、40 次替身 fit 到独立选优、五种子均值/标准差与容差平分、中断后只重做未完成任务、输出/缓存/代码/数据身份不一致拒绝、保存预测独立 AP 校验、人工检查点、并发 OS 锁、原生 Job 内存限额/CPU 亲和性、资源中止清理。原生内存测试在 64 MiB Job 下申请 128 MiB，验证共用内核机制；不会申请 8 GiB 内存。

新增 40 任务测试使用每折很小的合成数据和不训练的模型替身；原有 Smoke 测试仅训练极小合成模型。新增 worker 集成测试通过注入合成数据适配器和替身运行，不是对真实 40 次训练的资源验收。测试的 train/val 夹具没有真实 test 文件，不调用最终评估。

统一协议物理文件 SHA256 仍为 `245618b00d9290d765f8cd185da1c1d3281e4310727c6bdd205297b386aae997`；后续执行草案仍 pending，物理文件 SHA256 仍为 `6e53b4115ddd39463d1a6409265e9188d8cec9512267ef3e6c9552ed753a73c8`。P2/P3 公共代码未修改。没有创建真实搜索归档，没有执行真实数据训练，没有提交或推送本轮改动。
