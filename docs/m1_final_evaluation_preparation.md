# P4 最终 Test 评估工程准备

后续状态更新：本文保留工程准备时的pending状态记录。用户随后独立批准执行细则，创建execution.final-test.v1.json并完成基础Test评估；最新结果和归档验证见 [最终Test报告](m1_final_test_report.md)。原proposed草案未改为approved。

当前状态：GBDT/RF 训练、Validation 选优和验收已完成；最终 Test 评估尚未执行。本文件记录工程准备，不构成真实数据阈值标定或 Test 访问授权。

## 独立入口与阶段边界

- [final_stage.py](../methods/m1/final_stage.py)：标准库前台控制器、配置门禁、Windows Job 子进程保护及最终清单封存。
- [final_stage_core.py](../methods/m1/final_stage_core.py)：GO 之后运行的归档核验、Validation 阈值封存及冻结模型评估。
- [execution.final-test.proposed.json](../methods/m1/execution.final-test.proposed.json)：新的独立提案，calibrate/evaluate 均为 pending。
- [test_m1_final_stage.py](../tests/test_m1_final_stage.py)：合成文件和模型替身测试，不读取真实 Test，不拟合模型。

没有修改旧 executor、GBDT/RF 执行器、P2/P3 或历史协议。原 execution.proposed.json 仍 pending，统一协议仍 approved。不会重跑旧的联合 45 任务流程或重新选优。

`plan` 仅读取配置及源码身份，不打开数据集、归档预测或模型，也不创建运行目录：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.final_stage plan
```

后续的 calibrate 与 evaluate 分别需要自己的批准来源、日期、执行 payload 哈希和代码哈希；还必须显式指定 --launch。evaluate 额外需要 --permit-final-test、已封存 calibration 目录和其 MANIFEST SHA256。当前提案不能启动任何真实数据阶段。

## 已确定的实验身份

仅使用 G05 与 RF_FIXED，各保留种子 42、43、44、45、46，共十个已有模型，新增训练次数为零。不会在 Test 上挑选最好种子，也不重新拟合 train+val。

数据为 original / stratified / main，140 个 P2 特征。GBDT 与 RF 的原始模型和 Validation 预测仍分别保存在本地封存归档；其路径和清单 SHA256 固定于新配置。公开 results/m1.json 及其来源仍按原哈希保留，不覆盖为 Test 结果。

开始数据阶段前，校验完整归档文件集合和哈希、绑定身份、任务参数、种子、历史源码及当前核心 P2/P3 依赖、原始环境版本、P2 train/val/处理器身份。十份 Validation 预测独立复算 AP 并核对同一行序、标签和样本数。旧归档内的所有实验文件用 Windows FILE_SHARE_READ 保护，阶段结束再次核验。

## 待确认的执行细则

| 事项 | 已批准的上层要求 | 当前推荐实现，尚未批准执行 |
|---|---|---|
| 阈值 | Validation 标定 1% FPR，附加 5% | 每个模型/种子各自标定。k=floor(alpha*n_benign)，阈值取第 k+1 个降序良性分数向正无穷的 nextafter；score >= threshold 判攻击，不拆分相同分数。实际 Validation FPR 可低于目标。 |
| 共同计时名单 | CPU、512 行、warmup=5、repeats=30、共同 row ID | 优先使用团队已有明确名单。若尚无名单，建议由团队确认后从 Validation 用稳定哈希排序、固定种子 2026 生成一次，并交接 P5/P6/P7；本轮没有生成或擅自批准名单。 |
| Bootstrap 顺序 | 后续需要配对 Bootstrap | 将基础 Test 推理及 P3 指标与 Bootstrap 分开归档。本入口不调用 Bootstrap；后续次数、置信水平及采样细则另行批准，不取消统一协议中的统计要求。 |
| 资源 | 沿用已验收资源保护 | Job 提交上限 8 GiB、4 个逻辑 CPU、推理线程 1、块 8192、外层串行；启动空闲内存至少 14 GiB、运行至少 6 GiB、树 RSS 不超过 8 GiB。另提议启动磁盘空闲至少 20 GiB。 |

计时名单文件建议格式如下；row_ids 必须恰好 512 个唯一、非空且存在于同一完整 Validation 总体中。其完整文件 SHA256 还需填入配置。

```json
{
  "source_split": "val",
  "population_row_ids_sha256": "<完整Validation有序行ID摘要>",
  "row_ids": ["<团队共同的512个实际ID，按计时顺序排列>"]
}
```

## 后续运行行为

1. calibrate 读取已经封存的 Validation 分数，冻结 20 个阈值及共同计时张量；不加载模型，不读取 Test。输出单独的 MANIFEST。
2. evaluate 重验并复算冻结阈值，核对计时行 ID 和 512×140 张量，全部通过后才构造 P2 Test 加载器。Test 文件在读取期间禁止并发写入。验证 Test 行 ID 唯一，并与封存 Train/Validation 元数据进行精确隔离检查，DuckDB 内存池限制为 512 MB。
3. 串行重载十个原模型，核对参数、种子和特征数，分块预测；不对 Test 应用训练权重，不按 Test 指标调整阈值或参数。每个模型仅保存一份完整 NPZ，包含 row ID、真实标签、连续攻击分数、攻击族、attempted 标记及两组固定阈值预测。
4. 生成二十份 P3 Schema 记录、四组五种子均值/样本标准差、完整攻击族缺失报告及推理计时记录。Test ROC 上的经验 TPR@FPR 与固定 Validation 阈值在 Test 上的实际 FPR/TPR 分别报告。
5. 原始结果保存在已忽略的本地 `.tmp/m1_final_runs/` 新目录，含源码快照、环境、配置、阈值来源、完整预测、指标、资源日志及 SHA256 清单。未来公开交付需另行导出轻量摘要，不将此本地目录视为 GitHub 备份。

前台父进程显示阶段、RSS、系统空闲内存等状态；原有监控器保存完整进程树 CPU/RSS、Job 峰值提交和退出状态。所有 P2 加载、归档预测读取、模型重载、预测及指标计算都在 GO 之后的受控工作进程内。控制器只在正常完成后封存；异常保留 FAILED.json 和日志，不自动重试、不降参、不取消限制，也不支持未经核验的半成品续跑。

## 验收范围与限制

新增 25 项测试覆盖十任务编排、双归档读取、文件冻结、pending/启动/Test 门禁、payload/代码锁定、缺失计时身份、AP/预测身份、阈值与模型篡改拒绝、行 ID 隔离、资源中止不重试、预测保存、二十份 P3 Schema 记录、MANIFEST 校验及生产核心的合成校准→封存→评估串联。

合成集成测试产生的 `test.parquet` 是系统临时目录内的占位文件，通过 SyntheticP2 和不训练的 FixtureModel 运行，绝不指向真实 Test。资源包装测试使用监控替身；Windows Job 原生限额及亲和性由已有回归测试覆盖。尚未在新入口上运行真实模型或真实最终数据阶段，因此合成验收不能替代真实运行资源观察。

完整回归命令仍为 [docs/testing.md](testing.md) 中的三套命令。此前 216 项是历史基线；加入本模块后预期为 199 + 34 + 8 = 241 项。首次新测试在沙箱系统临时目录访问处失败，随后在获准环境重跑；失败过程没有访问真实数据或修改实验产物。

工程准备完成不代表最终实验协议已经批准；在共同计时名单、阈值边界规则及阶段授权明确前，不启动 calibrate 或 evaluate。

## 本次验收记录

2026-11-09（北京时间），使用原 `.venv` Python 3.13.0 完成三套完整回归：核心 199 项、工具 34 项、恢复封存 8 项，共 241 项，全部 OK，无 FAIL/ERROR 或跳过。日志仅作为本地工程记录保存在 `.tmp/m1_final_preparation_final_{core,tools,recovery}_tests.txt`，不作为公开可访问链接。

GBDT 原始清单 302/302、RF 原始清单 157/157 逐文件 SHA256 通过。results/m1.json、统一协议、旧 execution.proposed.json 及现有 PR PNG/PDF 保持原哈希；根 README 的原始 P1/P2 全文保持完整。P2/P3 公共代码没有修改。本轮未读取真实 Test、未真实标定阈值、未加载真实模型进行预测，也没有训练、实验 Bootstrap、commit 或 push。

当前提案的 SHA256（填入共同计时名单或后续修改提案时必须重新计算，不得沿用）：

| 对象 | SHA256 |
|---|---|
| execution.final-test.proposed.json 文件字节 | `755069d0ac85ccd277999f0e417d0ca2c682d0a6ee2d1ceb3367ca10e84a31bd` |
| 不含 approvals 的执行 payload | `3c06e9cf555dbd09d0e8ebfa901918b81f4a3de73b811b719cead3e9aaf8a524` |
| 当前评估源码身份集合 | `8f87b54af699ea668b404107af497014faef40b6e3257e7f452a1fac2063d1b5` |

calibrate/evaluate 仍 pending。实际启动前还必须重新检查系统空闲内存和磁盘条件；先前只读检查中约 10.9 GiB 的空闲内存不满足 14 GiB 启动要求，不能把该旧快照当作当前资源已满足的证明。
