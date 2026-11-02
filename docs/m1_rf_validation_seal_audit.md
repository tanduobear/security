# RF五种子Validation验收、封存及交付前检查

日期：2026-11-02（北京时间）。RF_FIXED种子42、43、44、45、46均完成正式训练和逐种子人工CONTINUE。前台训练命令退出码0，五个工作进程正常退出，无资源中止。RF为固定参数辅助对照，不参与GBDT主配置选优。完整结果见 [公开摘要](../results/m1_rf_summary/README.md)。

## 身份及结果验证

使用original / stratified / main完整Train/Validation及P2固定140特征、原处理器和训练折sample_weight，class_weight=null，不重新拟合预处理器。五份Validation各367,036行；任务配置、输入文件、标签摘要和行序摘要与封存G05逐种子匹配。完整Validation行序SHA256：

`c6a10ee0cf3ea5ceb71a4b147bb0c762da36aaa5554ab476173b839d16b9631a`

五份原始模型保存/重载预测与内存预测严格相等；预测文件重读一致。独立accept重新读取每份完整Validation预测，使用P3的AP计算逐一严格核对，没有重训、重新预测或重新选优。五个复算AP与训练记录差值均为0；均值0.9996364875511452，样本标准差（ddof=1）0.000050570370864307355，平均拟合923.18909787999秒。

正式配置文件保留原样，SHA256：

`65fa1b47ad866b1d6d6db53105c47a3bb901bcb8ca005e5e86179b4de16d9043`

## 资源

每个工作进程使用Windows Job提交内存硬限8 GiB、4个逻辑处理器亲和性；外层串行，训练线程4，预测线程1，预测块8192。启动要求空闲内存至少14 GiB，运行中空闲内存低于6 GiB或整树RSS超过8 GiB则中止；没有自动重试、降参或解除保护。

| 种子 | 拟合秒数 | Job峰值提交内存GiB | 整树采样峰值RSS GiB |
|---|---:|---:|---:|
| 42 | 982.32 | 2.821 | 2.464 |
| 43 | 921.46 | 1.677 | 1.706 |
| 44 | 848.94 | 1.674 | 1.800 |
| 45 | 936.29 | 1.634 | 1.657 |
| 46 | 926.94 | 1.653 | 1.769 |

此表为整个受保护尝试（可能含缓存/身份检查）的监控峰值；summary.json中的sampled_task_peak_rss_bytes仅按该任务阶段筛选，口径不同。RSS按秒采样，可能错过瞬时峰值；Job峰值提交内存来自OS原生计数。逐种子CPU、系统空闲内存及完整outcome保存在acceptance_receipt.json；这些不是统一推理时延基准。

## 环境失败与独立恢复

首次独立accept误用了系统Python 3.10.8，缺少duckdb，在身份检查阶段退出，没有开始AP复算或生成summary。失败目录ACCEPT-b65ce02fb5374972808862184947ce7c的worker.log、outcome、资源日志全部保留，未删除、改名或伪造。

随后用户明确批准仅修正环境并继续独立验收和封存。使用正式训练原`.venv/Scripts/python.exe`，Python 3.13.0、duckdb 2.0.0、scikit-learn 1.8.0、numpy 2.3.5；完整依赖环境与训练binding严格一致。原环境检查记录保存在本地归档recovery/venv-acceptance-02，公开环境记录去除本机绝对路径并保留原记录哈希。

新验收使用原run_rf.py accept入口与原资源保护，作为新的ACCEPT尝试执行。原封存器要求恰好一个ACCEPT目录，无法容纳保留的失败尝试；因此使用独立恢复封存脚本，不改原训练源码、配置、完成记录或原封存器：

- [rf_acceptance_recovery.py](rf_acceptance_recovery.py)：与归档恢复脚本字节相同。prepare固定144个已有文件的完整哈希，保存环境和脚本；seal保留原封存检查，额外验证旧失败记录原样保留，并且恰好一个新成功验收在原资源保护下绑定summary。
- [test_rf_acceptance_recovery.py](test_rf_acceptance_recovery.py)：8项合成文件测试，覆盖旧失败记录变化拒绝、额外验收拒绝、summary哈希不符拒绝、资源中止/保护缺失拒绝和原始文件保护。

恢复脚本是此次已完成恢复的审计源代码，固定该本地归档路径，不是通用重跑入口。原归档现已封存，禁止再次执行prepare/accept/seal覆盖它。公开环境派生文件的脱敏不改动封存原件。

## 清单和复核

MANIFEST排他发布前获取全局资源锁及运行锁，并以Windows只读共享句柄阻止原有文件被写入或删除；核验原始144文件仍未变化，核验全部任务、缓存、配置、数据、源码快照、G05参考和成功验收对summary的绑定。旧失败尝试也纳入总清单。

原归档157/157文件通过封存后第二遍完整校验，并另以标准库重新计算逐文件SHA256全部通过。本次提交前再验157/157通过。MANIFEST SHA256为：

`21ac71053cb0142de61d10dbcfa90d86e0f7f51f38bcaceacd0ae4efe75f5690`

RUN.lock为运行时排他协调文件，被排除且记录原因；MANIFEST自身不递归纳入清单，固定外部哈希。GitHub上的MANIFEST副本与本地原件字节相同，其157个条目所关联的大型原始文件只存于本地，非GitHub文件集合。持有完整本地归档时可只读复核：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.run_rf verify --resume results/m1_rf_runs/20261102T103826-3bdbabff97db4e2f99ff8a8a6b0b7767 --manifest-sha256 21ac71053cb0142de61d10dbcfa90d86e0f7f51f38bcaceacd0ae4efe75f5690
```

该命令要求本地完整归档。此处本地定位名是说明，不提供无法访问的在线链接。

## 本次交付回归

- 原.venv运行完整tests目录：174/174通过，82.007秒，无FAIL、ERROR或跳过；其中RF专项43项，既有测试131项。
- 文档目录中的恢复测试：8/8通过，0.450秒。合计182项通过，两套命令分别计数，不把174说成已包含额外8项。
- 测试只使用合成输入和模型替身；Windows原生Job限额测试为小额内存探针，未训练真实模型、读取真实Test或运行实验Bootstrap。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe -B -X utf8 docs/test_rf_acceptance_recovery.py
```

本次未修改P2/P3、统一协议、原execution.proposed.json或任何已封存GBDT结果。统一协议继续approved，原execution.proposed.json继续pending。未读取真实Test、重训任何模型、运行实验Bootstrap或标定阈值；本次提交前检查没有暂存、commit或push。
