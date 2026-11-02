# M1公开Validation汇总

[results/m1.json](../results/m1.json)由[独立汇总器](../tools/aggregate_m1.py)从GBDT和RF公开JSON摘要自动提取。它使用[独立Schema](../schemas/m1.schema.json)，不是P3最终评估结果Schema，也不能冒充完成了Test评估。

## 内容和来源

- `gbdt.candidates`：G01–G08八组原参数、逐种子Validation AP、原始拟合耗时、均值和样本标准差。
- `gbdt.selected_candidate`：已选G05的完整副本，五种子值须与八组结果中的G05逐项一致；不重新进行配置搜索或选优。
- `rf.validation_results`：RF五种子结果，原模型/预测的文件哈希、统计与配对身份。
- `gbdt.execution_config`和`rf.execution_config`：原正式执行配置完整副本；`unified_protocol`保留原协议。RF仍为辅助，不能参与GBDT选优。
- `provenance.sources`：12个白名单公开JSON来源的仓库相对路径、字节数和SHA256。另记录汇总器与Schema自身SHA256。
- `dataset.input_file_sha256`：P2输入身份，仅从已有摘要复制哈希，不打开Parquet或重新拟合处理器。
- 历史AP复算、归档验收、RF资源和公开环境证据均保留；与本轮执行行为明确区分。

GBDT来源为`results/m1_gbdt_summary/`，RF来源为`results/m1_rf_summary/`。公开MANIFEST副本仍指向各自本地原始归档，GitHub不含全部302/157文件。本轮验证公开文件哈希及相互引用，未重新扫描本地完整归档、未读取模型或预测数组，也未再次从预测计算AP。GBDT G02种子42/44的Modern Standby耗时警告及原始墙钟时间保持原样，不能将其当作纯计算耗时比较。

## 生成和校验

在仓库根目录使用现有.venv及已安装的jsonschema：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 tools/aggregate_m1.py
.\.venv\Scripts\python.exe -B -X utf8 tools/aggregate_m1.py --check
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tools/tests -p 'test_*.py' -v
```

生成是确定性的，不添加当前时间等可变字段。已有文件字节相同则保持原样，字节不同时拒绝覆盖，须先审查来源/工具变化并显式另行处理。`--check`只读校验现有文件。Schema约束结构、范围、任务数和Validation阶段边界；Python语义校验另外核对来源哈希、逐种子身份、配置、均值和ddof=1样本标准差。语义校验不能仅由JSON Schema替代。

22项独立测试通过，无FAIL、ERROR或跳过，覆盖完整45次训练记录、重复/缺失配置与种子、G05副本、逐种子AP变化、均值/标准差误用、配置/来源哈希变化、配对证据、非法分数、文件访问白名单和禁止新增未完成指标。测试使用公开轻量JSON和内存内篡改副本，不访问真实Test或运行任何模型。

## 阶段边界

`scope.validation_only=true`，其余本轮训练、预测读取、预测级AP复算、Test、Bootstrap、阈值标定和完整本地归档重验标记均为false。只保留已经完成的AP及拟合耗时；未计算ROC-AUC、Macro-F1、TPR@FPR、攻击族Recall、P3统一推理基准或显著性。本轮没有修改封存原件、既有公开摘要、P2/P3、实验配置或统一协议，也没有commit或push。

汇总器、Schema和测试独立放在tools/、schemas/下，不改变已封存训练代码的身份。
