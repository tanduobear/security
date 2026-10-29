# P2 划分索引生成与验证

在仓库根目录运行。Python版本及实际运行依赖见每套索引的manifest；本次使用Python 3.12.4。

```powershell
python -m pip install -r src/requirements-splits.txt
python src/build_split_indices.py --work-dir ../.audit/split-build
python src/verify_split_indices.py
```

当前电脑已在仓库上一级`.audit-deps`安装DuckDB，可沿用现有依赖执行：

```powershell
& 'C:/Program Files/Python312/python.exe' src/build_split_indices.py --work-dir ../.audit/split-build --dependency-dir ../.audit-deps
& 'C:/Program Files/Python312/python.exe' src/verify_split_indices.py --dependency-dir ../.audit-deps
```

生成器读取两份clean数据以及docs内的字段、标签规则；校验源文件SHA-256，不修改clean文件。中间DuckDB数据库和临时排序文件写入work-dir，最终索引写入data/split，生成器汇总写入docs/split_index_results.json，独立验证结果写入docs/split_index_verification.json。

输出是`train_indices.parquet`、`val_indices.parquet`、`test_indices.parquet`及`purged_indices.parquet`，不是包含76个特征的训练数据。`row_index`是源Parquet物理行序号，从0开始，可用DuckDB的`read_parquet(..., file_row_number=true)`连接。不要按标签、日期或排序后的DataFrame新索引取回源行。`row_id`是版本名加该行号；`group_id`是75字段的稳定特征组SHA-256。

每个协议目录的manifest记录：源数据和规则哈希、生成器哈希、库版本、索引文件哈希、样本量、清除量、类别分布和重叠检查结果。正式发布索引时同时保存manifest及对应规则文件。后续如修改规则或clean输入，应重新生成全部相关索引，不能直接沿用旧manifest。

固定随机组顺序为：组大小降序，然后`sha256('42:'+group_id)`升序，最后group_id升序。分配代价、平局顺序见split_report。类别覆盖修复固定按family_id、train/val/test处理，优先移动最小可移动组，组大小相同时按group_id排序；不拆组，不使原折失去原本可覆盖的类别。本次四套结果无需覆盖修复。

精确分组采用统一DOUBLE值的有序JSON对象，键为f00至f74；整数列先检查绝对值不大于2^53−1，正负零统一，禁止四舍五入。重复SHA-256候选的完整JSON最小值与最大值须相同，否则生成器立即报错。该定义保证保守控制主特征及删除端口消融的精确特征重复，不代表已识别所有近重复或同源网络会话。

随机划分组同时含多个真实标签时整组分配、保留标签；时序重复清除只删除后续折与较早折同组的行。清除记录保存原始标签、预定折及原因，支持逐类核对。验证脚本独立读取导出文件，对照源行验证标签和日期，检查每条源行恰好有一个保留/清除去向、两类跨折重叠为零，并核对索引、规则及生成器哈希。

索引生成器本身不提供缩放、编码或类别权重；这些现已由preprocess.py及build_model_data.py完成，详见[预处理报告](../docs/preprocess_report.md)。各类模型必须共享同一版本与协议的索引，且只在其训练折拟合预处理器。
