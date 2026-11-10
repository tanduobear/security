# P4 GBDT / RF 基础 Test Set 评估报告

2026-11-09（北京时间）。G05 与 RF_FIXED 各五个已有模型的完整 Test 推理、P3 指标校验及归档封存已完成。本次训练次数为零，Bootstrap 尚未运行，不对统计显著性作结论。

## 数据与冻结规则

- original / stratified / main，P2 固定处理器及140个特征；Test 实际346,544行，SHA256为 `f29b51ccd3daf8bc799804100a0dda0454573750eb3e60b7beef0f3d3b0b59d0`。
- GBDT 固定此前获选的G05，RF固定辅助配置，种子均为42、43、44、45、46。复用训练折模型，不重新训练，不重新选优，不合并train+val。
- 每个模型/种子仅用完整Validation良性分数标定1%和5% FPR阈值，使用nextafter处理并列，不拆分相同分数；进入Test后阈值不变。
- Validation阈值和共同计时张量先独立封存，随后才打开Test。共同512行名单由Validation稳定哈希排序、seed=2026生成，见 [m1_shared_timing_rows.v1.json](m1_shared_timing_rows.v1.json)。该文件供P5/P6/P7复用，不表示已通过工具向其他成员发送。
- 使用 [execution.final-test.v1.json](../methods/m1/execution.final-test.v1.json) 的独立授权；原execution.proposed.json、execution.final-test.proposed.json及protocol.pending.json保持原状态。统一协议仍approved。

## 五种子结果

以下为均值 ± 样本标准差（ddof=1）。AP和ROC-AUC基于完整Test连续分数，与分类阈值无关。

| 模型 | AP | ROC-AUC |
|---|---:|---:|
| GBDT G05 | 0.999495 ± 0.000073 | 0.999907 ± 0.000011 |
| RF_FIXED | 0.999227 ± 0.000162 | 0.999842 ± 0.000032 |

| 模型 | Validation标定目标FPR | Test Macro-F1 | Test实际FPR | Test实际TPR |
|---|---:|---:|---:|---:|
| GBDT G05 | 1% | 0.987784 ± 0.003178 | 0.916936% ± 0.243557% | 99.996565% ± 0.000698% |
| RF_FIXED | 1% | 0.984652 ± 0.000404 | 1.155020% ± 0.030967% | 99.992817% ± 0.000855% |
| GBDT G05 | 5% | 0.943272 ± 0.002022 | 4.506311% ± 0.171870% | 99.999688% ± 0.000698% |
| RF_FIXED | 5% | 0.944368 ± 0.000563 | 4.412862% ± 0.047851% | 100.000000% ± 0.000000% |

Test实际FPR允许高于Validation目标，未据此重新校准。P3经验ROC TPR@FPR(0.1%、1%、5%)另在逐种子摘要中保留，不能混同为上述固定Validation阈值的部署表现。

| Seed | GBDT G05 Test AP | RF_FIXED Test AP |
|---:|---:|---:|
| 42 | 0.9995211773199413 | 0.9990793012858298 |
| 43 | 0.9995045813989475 | 0.9993558971238685 |
| 44 | 0.9993695477488194 | 0.9993129414604113 |
| 45 | 0.9995614544433720 | 0.9993612548400047 |
| 46 | 0.9995189193477053 | 0.9990232688289290 |

Test包含全部14个预期攻击族，无缺失族。Heartbleed仅2条、Infiltration仅5条、SQL Injection仅3条，其Recall分母很小，需连同support报告，不能据此推断稳定的罕见攻击泛化表现。逐种子攻击族Recall及support保存在公开摘要中。

## 资源与运行情况

全程Job提交硬限8 GiB、亲和性4个逻辑处理器，推理线程1、预测块8192、外层串行。共同CPU计时使用512行，warmup=5、repeats=30，排除数据读取、模型加载及指标计算。

Test工作进程累计耗时611.73秒，包含归档核验、加载、预测、正式指标/Schema校验和保存，不能作为纯模型推理耗时。峰值Job提交约1.47 GiB，采样进程树峰值RSS约0.72 GiB，最低系统空闲内存约17.39 GiB，峰值CPU约1.33个逻辑核心等效。无资源中止、退出码0。完整单批推理时间和30次采样保存在逐种子cost记录中。

资源日志中的目录大小采样仅覆盖monitor子目录；实际Test归档文件合计529,223,583字节，Validation校准归档753,162字节。不能把monitor子目录字节数当成整个实验磁盘占用。

首次辅助模块启动错误的失败证据已保留，见 [启动故障记录](m1_final_startup_failure.md)。修复通过无数据探针后，由用户明确回复“继续”才进行新尝试。本次成功流程未自动重试，也未覆盖首次失败日志。

## 产物与完整性

- [公开Test摘要](../results/m1_test_summary/summary.json)：完整逐种子指标、两套阈值、五种子汇总、攻击族Recall及计时。
- [验证记录](../results/m1_test_summary/verification.json)：20份正式记录、10份完整预测、清单验证及统计交叉核对。
- [MANIFEST副本](../results/m1_test_summary/MANIFEST.local-archive.json)：原始清单字节副本，关联大型文件保存在本地，GitHub不包含完整归档。

本地Test归档：`.tmp/m1_final_runs/evaluate-20261109T122704-2bf90cbb95e04ceea7c40993a6ec5012/`，82/82文件SHA256通过。

本地Validation阈值归档：`.tmp/m1_final_runs/calibrate-20261109T122637-96d03508235942f5a8d3b66d5a11f335/`，47/47文件SHA256通过。这些路径是本地归档标识，不是可在GitHub访问的链接；不得作为普通临时缓存删除。

| 对象 | SHA256 |
|---|---|
| 已批准执行配置 | `c1294e2a30338f359b132156403170c415f590fb71ef0c9889f48f301dbbdf88` |
| 共同计时名单 | `3894b1d222acfe4e62fe4265037ab1ce1a858840854eece3b8f4b778ed959b1d` |
| Validation阈值MANIFEST | `b7428bfd06022f53306369a679b8ecf74d6082a755bb39cdcb0f724461fdd0cd` |
| Test MANIFEST | `34a1503cd829290714d87dc6c84165295d92323dd1d44839ecbf5cbb49430c4f` |
| 公开Test summary.json | `cb2a95db154792d5dc0d3fb5f2710aba3ed6cc04f12a964e958cf26f371c23e3` |

20份原始正式记录在受保护工作进程内通过P3 Schema及一致性校验；完整NPZ保存后重新读取，核对标签、分数、行ID、元数据、两组预测和重新计算的指标。导出阶段仅提取已封存记录，独立交叉核对五种子均值/标准差，不重新预测或选优。公开summary省略完整曲线数组，不是P3的单模型单种子记录。

原GBDT/RF训练归档、selection.json、results/m1.json及Validation PR图保持原哈希。原results/m1.json继续为Validation-only，不覆盖成Test结果。工程准备的241项合成回归为启动前验收记录；本轮新增的证据为上述真实Test执行及完整性检查，没有把合成Test分数当作正式结果。

本轮没有训练、Bootstrap、Test调参、commit或push。基础Test评估完成不代表统一协议中的后续配对统计及扩展实验已经完成。
