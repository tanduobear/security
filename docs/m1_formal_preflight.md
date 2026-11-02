# P4 正式执行器实现与运行前验收

日期：2026-11-01。统一协议已批准并冻结；执行细则仍待确认。代码实现和合成数据验收已完成，本轮没有读取真实测试集，也没有执行正式超参搜索、全量训练或真实最终评估。

## 协议与配置

- `methods/m1/protocol.unified.v1.json`：用户报告 P4/P5/P6/P7 集体批准的 NIDS Unified Experimental Protocol v1.0；记录授权来源，不伪造个人签名。
- 统一协议 SHA256：`e6e6d8880a5540419f56dbea5b5270b7fd0bf9392b1114b8b56ebf943ef92360`。
- `methods/m1/protocol.pending.json`：填入已经确定的主数据、主预算、种子、选优和测量设置，并链接冻结协议；外层 pending 专指执行细则。
- `methods/m1/execution.proposed.json`：完整、可验证的推荐执行配置，仍为 pending。没有将具体推荐值写成已经获批。

## 已实现的独立阶段

`methods/m1/executor.py` 提供 check、search、select、evaluate 四个入口。check 不打开数据集；其余阶段均校验通用协议与执行细则的摘要及批准记录。

search 使用 P2 ModelDataset，保持原版分层主版 140 维输入和训练折权重。只加载 train/val 一次，写入磁盘映射缓存，外层只串行一个训练任务，分批生成概率分数。每个 GBDT 配置的五个种子和固定 RF 的五个种子均保存模型、完整验证预测、AP、参数、耗时、逐任务日志及 SHA256。RF 不进入 GBDT 选优。

select 只读取封存的验证搜索产物。核对全部 45 个任务与共同验证行号，按八组配置各自的五种子平均完整验证 AP 选优，保留样本标准差。选优后为获胜 GBDT 五个种子和 RF 五个种子分别标定 1%、5% 验证 FPR，封存 20 个阈值，不重新拟合 train+val。

evaluate 必须显式传入 --permit-final-test，且依赖完整封存的 selection/search 目录。配置、数据身份、获胜配置、模型清单、阈值复算与文件哈希检查都发生在构造测试加载器之前。测试分数按验证阈值固定判断，实际测试 FPR 可以超过验证目标，不能再次校准。调用 P3 全部指标和正式 Schema/数值一致性校验；记录每个模型、种子和阈值的 JSON 及逐样本预测。Bootstrap 按同一测试总体/相同顺序行号比较 GBDT 与 RF，不能把它误解释成固定部署阈值的置信区间。

通用协议允许按算法设定停止规则，本版明确实现的是固定最大轮数方案。推荐不使用 Early Stopping，因此没有隐藏训练内验证划分。若团队选择早停而非此方案，应补齐监控、patience、最小改进、最佳轮恢复等规则并扩展验收；本版会拒绝未实现的停止模式。

## 推荐候选 待确认

GBDT 实现为 sklearn HistGradientBoostingClassifier。共同项：max_bins=255、early_stopping=false、class_weight=None，仅传入 P2 sample_weight。

| ID | max_iter | learning_rate | max_leaf_nodes | min_samples_leaf | l2_regularization |
|---|---:|---:|---:|---:|---:|
| G01 | 200 | 0.05 | 15 | 20 | 1 |
| G02 | 400 | 0.05 | 15 | 20 | 1 |
| G03 | 200 | 0.10 | 15 | 20 | 1 |
| G04 | 200 | 0.05 | 31 | 20 | 1 |
| G05 | 400 | 0.05 | 31 | 20 | 1 |
| G06 | 200 | 0.05 | 15 | 5 | 1 |
| G07 | 200 | 0.05 | 15 | 50 | 1 |
| G08 | 200 | 0.05 | 15 | 20 | 10 |

RF_FIXED 推荐：200 棵树，max_depth=20、max_leaf_nodes=4096、min_samples_leaf=2、max_features=sqrt、bootstrap=true、n_jobs=4、class_weight=None，沿用 P2 sample_weight。同五个种子作为辅助，不搜索 RF、不与 GBDT 竞争主模型名额。

其他推荐：AP 平分容差 1e-6，容差内选择最小候选 ID；验证 FPR 阈值严格高于第 k+1 个降序良性分数，k=floor(alpha*n_benign)，使用 nextafter 处理并列，不拆分相同分数。Bootstrap 2000 次、95% 区间、种子 2026，按同种子的获胜 GBDT 减 RF 比较 AP 和经验 TPR@FPR(0.1%、1%、5%)。这些具体细则尚未批准。

计时推荐：在完整验证集按 row ID 的 SHA256 排名、种子 2026 生成共同 512 行，连同总体身份和行号摘要交 P5/P6/P7 复用；不能各自重选不同样本。CPU batch=512、warmup=5、repeats=30 为已批准条款。训练线程 4、推理线程 1、预测块 8192、跨折检查 DuckDB 上限 512 MB 是待确认资源建议。

## 任务数量和执行顺序

1. 配置检查与执行细则确认，不读数据。
2. train/val 全量准备与隔离检查。
3. G01 至 G08，各按 42–46 顺序执行，共 40 次 GBDT 拟合。
4. 固定 RF 按同五种子执行，共 5 次辅助拟合。
5. 八组五种子平均验证 AP 选优，生成 20 个阈值，不额外重训最优模型。
6. 后续独立授权最终评估：复用 10 个模型，输出 20 份正式记录/逐样本预测、5 张逐种子 PR 对照图；按推荐对比计划有 20 个 Bootstrap 计算任务，每任务 2000 次重采样。
7. 主实验结束后，再按批准顺序安排时序/端口消融及 Corrected 复跑。当前主协议校验不允许悄悄把数据切换成扩展数据。

没有编造全量训练耗时。Bootstrap 完整测试集重采样可能耗时较大，不会在本轮执行。

## 运行档案

每阶段使用独立 UTC时间加UUID 目录及排他创建文件：`results/m1_runs/search/`、`selection/`、`evaluation/`。配置、软件环境、源文件快照、输入哈希、模型、预测、指标和日志受同一阶段 MANIFEST 约束；后续阶段记录前序 MANIFEST 的 SHA256 和可迁移的相对引用。失败目录有 FAILED 状态，不能用于后续阶段。缓存放 `.tmp/m1_formal_cache/`，不是唯一结果保存位置。

正式每条 JSON 符合 P3 单模型/单种子/单测试数据记录 Schema；evaluation_index.json 是独立汇总入口，不冒充单记录 Schema。模型保留在搜索档案，评估档案按 SHA256 引用它；交接必须保留整个引用链。

## 自动化验收

全部 87 项测试通过，unittest 报告 15.756 秒：原有 75 项保持通过，新增 12 项正式执行器测试。新增集成测试仅创建临时伪数据文件，45 次 fit 调用全部使用不训练的模型替身，验证编排而非执行 40 次训练。旧 Smoke 测试仍仅训练极小合成模型。

验收覆盖统一冻结条款、执行参数锁定、五种子平均 AP 而非最好种子/辅助 RF 选优、阈值并列及零误报预算、验证阈值不能保证测试 FPR、显式测试访问门禁、封存档案链及篡改拒绝。临时合成最终评估产生 20 个测试夹具记录，均通过 P3 正式 Schema 和数值一致性校验；它们不会导出为真实实验结果。

执行 `python -m methods.m1.executor check` 的当前预期是 BLOCKED：尚未批准具体执行细则。没有打开真实数据文件，没有启动真实搜索/测试评估。P2/P3 公共代码没有修改。

## 内存接口限制

M1 控制外层串行、磁盘映射输入、分块预测和有限线程，避免全特征概率输出/重复数组拼接。P2 加载器内部的 DuckDB 连接未暴露内存配置；当前 512 MB 只控制 M1 的跨折 ID 查询，不是整个进程或 P2 加载器的硬内存上限。本轮没有改 P2 接口，也未测量真实全量峰值 RSS。开始真实搜索前应关闭无关高内存程序，并在首任务观察进程资源。

## 命令参考

当前只执行检查与短时测试：

```powershell
.\.venv\Scripts\python.exe -B -m methods.m1.executor check
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py' -v
```

将来执行细则真实批准并保存到独立文件后，才能按顺序使用：

```powershell
python -m methods.m1.executor search --execution <approved-details.json>
python -m methods.m1.executor select --run <sealed-search-directory>
python -m methods.m1.executor evaluate --run <sealed-selection-directory> --permit-final-test
```

以上三条本轮均未对真实数据运行。最终评估入口实现完成，不代表已经授权现在读取真实测试集。
