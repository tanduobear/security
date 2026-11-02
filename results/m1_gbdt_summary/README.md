# P4 GBDT-only 正式搜索公开摘要

范围：original / stratified / main，P2固定140维特征，完整Train 1,764,265行、Validation 367,036行。G01–G08各运行种子42、43、44、45、46，共40次拟合；仅用完整Validation平均AP选优。

获选配置 **G05**：HistGradientBoostingClassifier，400轮、learning_rate=0.05、max_leaf_nodes=31、min_samples_leaf=20、L2=1、max_bins=255、early_stopping=false；使用P2训练折sample_weight，不重复设置class_weight。

| 配置 | 五种子Validation AP均值 | 样本标准差（ddof=1） |
|---|---:|---:|
| G01 | 0.999433365 | 0.000099958 |
| G02 | 0.999698466 | 0.000034309 |
| G03 | 0.999682692 | 0.000026511 |
| G04 | 0.999646839 | 0.000021198 |
| G05 | 0.999791727 | 0.000015763 |
| G06 | 0.999451116 | 0.000064948 |
| G07 | 0.999461421 | 0.000066649 |
| G08 | 0.999436374 | 0.000090745 |

40份完整Validation预测AP独立复算，与原记录逐一严格相等，最大绝对差0。AP平分容差1e-6，容差内取最小配置ID。本次G05与第二名差值大于容差。

**以上是验证集搜索结果，不是最终Test性能。** 没有读取真实Test、训练RF、运行Bootstrap或标定阈值。G02种子42/44因Modern Standby出现中断；summary.json保留原始训练墙钟耗时并标注，不能直接当作纯计算耗时比较。

## 目录内容

- [summary.json](summary.json)：公开汇总，包含八配置参数、逐种子AP及训练耗时、数据哈希和阶段边界，不含逐样本预测。
- [selection.json](selection.json)：原选优结果的逐字节副本。
- [MANIFEST.local-archive.json](MANIFEST.local-archive.json)：原始本地归档MANIFEST.json的逐字节副本。

**GitHub不包含MANIFEST列出的全部302个文件。** 40个模型、完整预测数组、映射缓存、原始运行日志和源码快照保存在本地原始归档；本目录不是完整归档备份，也不能仅凭GitHub文件完成302项验证。

MANIFEST中的文件名相对于本地原归档根目录，而非本公开摘要目录。本地定位名（不随本次提交上传，非在线链接）：

`results/m1_gbdt_search/20261101T115824-90de9307e77448aba412cc9ec08a5cbd/`

持有完整本地归档时，可按 [补封存报告](../../docs/m1_manifest_seal_repair.md) 的独立verify命令逐项复核。源数据使用P2约定渠道获取；本次提交不重复上传Parquet。

## 固定SHA256

selection.json：

`859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9`

MANIFEST.local-archive.json（与原MANIFEST.json一致）：

`00128ffea5088c939a2375502372be4ba6b95cc13bcdee4f68e6c1638d2b1d6d`

原清单302/302已复核通过；根目录RUN.lock作为运行时锁排除，原因保留在清单中。MANIFEST不递归包含自身，其SHA256在此单独固定。统一协议保持approved；后续完整评估草案execution.proposed.json仍为pending。
