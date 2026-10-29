# P2 标签映射与划分方案

> 完整复核更新：下列索引与数量是原始划分阶段结果。实际模型数据另有float32精度重复清除，最终有效索引、样本量和比例以[预处理报告](preprocess_report.md)为准。

版本v1.0，2026年10月26日。此文件包含已确定规则、实际输入统计以及已完成的划分索引结果。前文“划分前”数字不等于最终交付量，最终样本量和无重叠检查见末尾执行结果。模型特征数据现已生成，结果见[预处理报告](preprocess_report.md)。

## 预测目标与标签规则

主任务采用二分类：正常为0，攻击为1。这样三个方法（包括只学良性分布的自编码器）可以用统一攻击分数比较。分类别召回按真实攻击族分组计算二分类检出率，不要求自编码器输出攻击族。family_id用于分组元数据，不是第二个必须训练的多分类任务。

每行同时保留：label_raw（原始Label）、y_binary、attack_family、is_attempted。它们都是目标/评估元数据，均不得进入特征X。修正版Attempted Category只用于核查标注，不进入X。

- benign保持良性。
- 所有以“ - attempted”结尾的标签映射为良性，is_attempted=true；不删除这些行，也不把attempted当独立模型类。主指标将这些流量计作负类；额外报告该子群的误报率及样本量。
- 原版bot与修正版botnet统一为botnet。
- 新增Day后的原版三个Web标签实际为web attack双空格brute force/sql injection/xss，与P1类别表中的连字符写法不一致；本规则明确映射为web attack - brute force/sql injection/xss，保留label_raw，不改动clean文件。各类计数与P1表相符。
- infiltration - portscan按实际扫描行为归为portscan，原始名称保留，不能误当成纯infiltration；这是本项目的归并选择，不声称它和原版PortScan逐行对应。
- 其他有效攻击标签保留细粒度名称，不把四种DoS或三种Web Attack统统合并，避免淹没稀有攻击。未知标签立即报错，不默认归为良性。
- 两版独立处理，不把同一行序号或同一标签视为跨版本配对流。版本性能差异包含真值与流构造变化，不是逐流因果对照。

修正版发布者明确允许按研究目标处理attempted，并建议不确定时归为良性；2021工作中的基准实验也采取这一方式。本方案采用行为检测口径，而非仅因攻击意图就视为恶意。依据：[CNS2022下载说明](https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html)、[WTMC2021实验说明](https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html)。

## 全部原始标签映射

| 版本 | 原始标签 | 输入行数 | 二分类y | 攻击族/良性组 | attempted |
|---|---|---:|---:|---|---|
| original | benign | 2,095,057 | 0 | benign | false |
| original | bot | 1,948 | 1 | botnet | false |
| original | ddos | 128,014 | 1 | ddos | false |
| original | dos goldeneye | 10,286 | 1 | dos goldeneye | false |
| original | dos hulk | 172,846 | 1 | dos hulk | false |
| original | dos slowhttptest | 5,228 | 1 | dos slowhttptest | false |
| original | dos slowloris | 5,385 | 1 | dos slowloris | false |
| original | ftp-patator | 5,931 | 1 | ftp-patator | false |
| original | heartbleed | 11 | 1 | heartbleed | false |
| original | infiltration | 36 | 1 | infiltration | false |
| original | portscan | 90,694 | 1 | portscan | false |
| original | ssh-patator | 3,219 | 1 | ssh-patator | false |
| original | web attack  brute force | 1,470 | 1 | web attack - brute force | false |
| original | web attack  sql injection | 21 | 1 | web attack - sql injection | false |
| original | web attack  xss | 652 | 1 | web attack - xss | false |
| corrected | benign | 1,582,561 | 0 | benign | false |
| corrected | botnet | 736 | 1 | botnet | false |
| corrected | botnet - attempted | 4,067 | 0 | benign | true |
| corrected | ddos | 95,144 | 1 | ddos | false |
| corrected | dos goldeneye | 7,567 | 1 | dos goldeneye | false |
| corrected | dos goldeneye - attempted | 80 | 0 | benign | true |
| corrected | dos hulk | 158,468 | 1 | dos hulk | false |
| corrected | dos hulk - attempted | 581 | 0 | benign | true |
| corrected | dos slowhttptest | 1,740 | 1 | dos slowhttptest | false |
| corrected | dos slowhttptest - attempted | 3,368 | 0 | benign | true |
| corrected | dos slowloris | 3,859 | 1 | dos slowloris | false |
| corrected | dos slowloris - attempted | 1,847 | 0 | benign | true |
| corrected | ftp-patator | 3,972 | 1 | ftp-patator | false |
| corrected | ftp-patator - attempted | 12 | 0 | benign | true |
| corrected | heartbleed | 11 | 1 | heartbleed | false |
| corrected | infiltration | 36 | 1 | infiltration | false |
| corrected | infiltration - attempted | 45 | 0 | benign | true |
| corrected | infiltration - portscan | 71,767 | 1 | portscan | false |
| corrected | portscan | 159,066 | 1 | portscan | false |
| corrected | ssh-patator | 2,961 | 1 | ssh-patator | false |
| corrected | ssh-patator - attempted | 27 | 0 | benign | true |
| corrected | web attack - brute force | 73 | 1 | web attack - brute force | false |
| corrected | web attack - brute force - attempted | 1,292 | 0 | benign | true |
| corrected | web attack - sql injection | 13 | 1 | web attack - sql injection | false |
| corrected | web attack - sql injection - attempted | 5 | 0 | benign | true |
| corrected | web attack - xss | 18 | 1 | web attack - xss | false |
| corrected | web attack - xss - attempted | 655 | 0 | benign | true |

## 映射后输入统计

| 攻击族/良性组 | original | corrected |
|---|---:|---:|
| benign | 2,095,057 | 1,594,540 |
| botnet | 1,948 | 736 |
| ddos | 128,014 | 95,144 |
| dos goldeneye | 10,286 | 7,567 |
| dos hulk | 172,846 | 158,468 |
| dos slowhttptest | 5,228 | 1,740 |
| dos slowloris | 5,385 | 3,859 |
| ftp-patator | 5,931 | 3,972 |
| heartbleed | 11 | 11 |
| infiltration | 36 | 36 |
| portscan | 90,694 | 230,833 |
| ssh-patator | 3,219 | 2,961 |
| web attack - brute force | 1,470 | 73 |
| web attack - sql injection | 21 | 13 |
| web attack - xss | 652 | 18 |

## 两套划分的共同约束

1. seed固定42；将输入文件SHA-256、实现版本、库版本和最终索引写入manifest。row_id使用版本名+clean文件0基行序号，不能使用修正版id作为全局唯一标识。
2. 主版本76字段与删除端口消融75字段共用同一份索引。因此防重复组在75个共同字段上定义，避免只在端口不同的记录跨折后污染消融。字段按feature_policy.json固定顺序，统一数值类型（不四舍五入、不分桶），序列化后生成稳定SHA-256组键。必须对哈希碰撞候选作精确字段比较；组键和row_id不进入X。
3. 相同特征但不同标签的记录放入同一组，保留并登记冲突，不用全量多数标签改写答案，也不把冲突行分散到不同折。相同75字段只是保守的防跨折重复约束，不声称它们一定是同一物理网络连接。
4. 每版、每种划分由三个模型共享；不对不同模型重新抽样，也不在原版和修正版之间合并数据。每版独立按相同规则生成索引，不能要求行ID跨版本相同。
5. 划分后只在训练折拟合编码、缩放、类别权重；不在划分前SMOTE/采样。不对验证或测试做类别平衡。自编码器从训练折中只取y_binary=0训练，验证良性子集标定阈值，测试标签只用于最终评分。attempted归为良性是公开的真值政策，须在报告中披露。

## 方案一 分组分层随机划分

目标：train/val/test约70%/15%/15%；以映射后的15个attack_family（含benign）分层。不是按原始27个标签训练27类模型。原始标签和attempted子群的分布另行报告。

分配单位为上述特征组。固定seed=42，组按大小降序处理，同大小用固定种子生成的稳定组优先级打破平局；对每个候选折，计算加入该组后全体折的总量及各攻击族计数相对目标的平方归一化偏差，选偏差最小者，同分按train、val、test固定顺序。目标比例作为软约束，组不跨折是硬约束。不得反复换seed直到模型指标更好。

稀有类规则：heartbleed仅11行、修正版有效SQL injection仅13行，不能要求三折数目严格按小数比例。实现前检查每类的独立组数；若至少3组，分配中尽量保障三折均有样本；若不足3组，优先训练与测试覆盖并明确缺失折，不复制或合成验证/测试样本。最终是否满足类别覆盖必须实测，不能仅凭总行数承诺。

分配代价定义：对每折s、每族c，累加((实际行数n_sc−目标行数t_sc)/max(t_sc,1))²，再加各折总行数的同类归一化平方偏差，所有项权重均为1。标签混合组使用完整族计数向量。候选代价按放入该组之后计算。最后在不拆组前提下检查并确定性修复可实现的类别覆盖；若约束不可同时满足则如实报告，不自动降级为逐行随机拆组。

允许因组大小偏离70/15/15，必须报告实际比例和各类计数。若出现某类未分到预期折或整体偏差超过1个百分点，记录原因并暂停宣布分层验收通过，调整应以分组约束为前提并留下规则版本记录，不看模型结果决定。

## 方案二 按天时序划分

| 折 | 抓包日期 | 原版依据 | 修正版依据 |
|---|---|---|---|
| train | 2017-07-03至2017-07-05 | Monday、Tuesday、Wednesday | Timestamp日期 |
| val | 2017-07-06 | Thursday | Timestamp日期 |
| test | 2017-07-07 | Friday | Timestamp日期 |

这是3天/1天/1天的时间顺序协议，不要求样本量为70/15/15，也不保证各折类别比例相同。日期只用于划分，之后从X删除。只用各流开始时间/来源日期决定所属折，没有完整结束时间时不声称排除了所有跨午夜会话。

防重复采用单向清除：训练折优先保留；验证中已出现在训练的75字段组行剔除；测试中已出现在保留训练或保留验证的组行剔除。不得把后面的数据挪进前面来补比例。逐版、逐折、逐标签报告剔除量；测试指标代表清除后留存的较晚流量，不能冒充完整周五总体表现。若后续需要不清除的全日结果，只能单独标明为辅助诊断。

### 按日实际输入量 清除前

| 日期 | original | corrected | 预定折 |
|---|---:|---:|---|
| 2017-07-03 | 502,650 | 371,621 | train |
| 2017-07-04 | 421,576 | 322,078 | train |
| 2017-07-05 | 602,278 | 496,640 | train |
| 2017-07-06 | 398,434 | 362,075 | val |
| 2017-07-07 | 595,860 | 547,557 | test |

| 版本 | train清除前 | val清除前 | test清除前 |
|---|---:|---:|---:|
| original | 1,526,504 | 398,434 | 595,860 |
| corrected | 1,190,339 | 362,075 | 547,557 |

### 日期与攻击类型相关造成的缺类

- original：验证集有而训练无的组：infiltration, web attack - brute force, web attack - sql injection, web attack - xss；测试集有而训练无的组：botnet, ddos, portscan。这是清除前统计，清除后需再核对。
- corrected：验证集有而训练无的组：infiltration, portscan, web attack - brute force, web attack - sql injection, web attack - xss；测试集有而训练无的组：botnet, ddos, portscan。这是清除前统计，清除后需再核对。

因此时序协议是时间迁移/未见攻击的二分类检测检验，不能用“闭集15类分类准确率”解释。训练时不为未见族伪造样本；每个攻击族都报告测试支持数，支持为0时召回记为NA而非0，支持很少时披露不确定性。这里只能称为在训练折未出现的标签族，不能直接宣称真实零日检测。

## 输出目录与下游接口

为避免两版×两套划分相互覆盖，使用：

```text
data/split/original/stratified/{train,val,test}.parquet
data/split/original/temporal/{train,val,test}.parquet
data/split/corrected/stratified/{train,val,test}.parquet
data/split/corrected/temporal/{train,val,test}.parquet
```

每个目录额外保存manifest.json、类分布、行ID索引和清除日志；预处理器按版本和划分分别在相应训练折拟合。P3按同一version+protocol取数据，P4/P5/P6主实验先用original/stratified，时序作为二次检验，P7使用corrected下的对应协议。实际提交文件路径应登记在交接表，不使用模糊的单份train/val/test覆盖四组输入。

P3验收应检查：行ID及防重复组跨折交集为空；train/val/test无遗漏或有明确剔除原因；标签映射覆盖全部输入；分层方案类别支持与比例可解释；时序日期严格前后分离；元数据没有混入X。

## 当前完成状态

已完成标签全覆盖映射、两版逐日/逐类统计、分组随机分配、时序重复清除、索引导出及独立复核，结果见下节。后续已拟合训练折预处理器并导出模型数据，详见[预处理报告](preprocess_report.md)。

机器可读规则及全部逐日攻击族计数见[label_split_policy.json](label_split_policy.json)。

## 划分索引执行结果

四套划分索引已生成并通过独立导出复核。此处给出清除后的最终样本量；每个目录中还保留manifest和清除记录。当前产物为索引，不是预处理后的特征数据。

| 数据版本 | 协议 | train | val | test | 清除行数 | 行号跨折重叠 | 特征组跨折重叠 |
|---|---|---:|---:|---:|---:|---:|---:|
| original | stratified | 1,764,265 | 378,266 | 378,267 | 0 | 0 | 0 |
| original | temporal | 1,526,504 | 373,030 | 525,627 | 95,637 | 0 | 0 |
| corrected | stratified | 1,467,942 | 316,014 | 316,015 | 0 | 0 | 0 |
| corrected | temporal | 1,190,339 | 330,382 | 433,710 | 145,540 | 0 | 0 |

### 随机方案实际比例

| 版本 | train占源数据 | val占源数据 | test占源数据 |
|---|---:|---:|---:|
| original | 69.9884% | 15.0058% | 15.0058% |
| corrected | 69.9030% | 15.0485% | 15.0485% |

随机方案每版15个映射后分组类别在三折均有样本；没有为覆盖类别移动组的修复操作。固定规则先将全部组分配一次，不通过更换seed追求模型表现。实际比例偏差均小于方案规定的1个百分点。

### 时序清除量

| 版本 | train清除 | val清除 | test清除 |
|---|---:|---:|---:|
| original | 0 | 25,404 | 70,233 |
| corrected | 0 | 31,693 | 113,847 |

清除仅作用于较晚折重复出现的75字段组；不代表这些原始流量全都无效。清除前后类别数及原始标签细分在各manifest的class_distribution中保留，purged_indices.parquet保存逐行去向。

### 混合标签特征组

| 版本 | 特征组总数 | 含多个映射后类别的组 | 含多个原始标签的组 |
|---|---:|---:|---:|
| original | 2,232,894 | 270 | 270 |
| corrected | 1,714,098 | 15 | 127 |

混合标签组没有被多数投票改标；随机方案整组分配，时序方案遵守日期和单向清除规则。

### original stratified 最终类别支持数

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,466,743 | 314,157 | 314,157 |
| botnet | 1,364 | 292 | 292 |
| ddos | 89,610 | 19,202 | 19,202 |
| dos goldeneye | 7,200 | 1,543 | 1,543 |
| dos hulk | 120,994 | 25,926 | 25,926 |
| dos slowhttptest | 3,660 | 784 | 784 |
| dos slowloris | 3,769 | 808 | 808 |
| ftp-patator | 4,151 | 890 | 890 |
| heartbleed | 7 | 2 | 2 |
| infiltration | 26 | 5 | 5 |
| portscan | 62,988 | 13,852 | 13,854 |
| ssh-patator | 2,253 | 483 | 483 |
| web attack - brute force | 1,029 | 221 | 220 |
| web attack - sql injection | 15 | 3 | 3 |
| web attack - xss | 456 | 98 | 98 |

### original temporal 最终类别支持数

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,323,598 | 370,851 | 347,924 |
| botnet | 0 | 0 | 1,855 |
| ddos | 0 | 0 | 128,012 |
| dos goldeneye | 10,286 | 0 | 0 |
| dos hulk | 172,846 | 0 | 0 |
| dos slowhttptest | 5,228 | 0 | 0 |
| dos slowloris | 5,385 | 0 | 0 |
| ftp-patator | 5,931 | 0 | 0 |
| heartbleed | 11 | 0 | 0 |
| infiltration | 0 | 36 | 0 |
| portscan | 0 | 0 | 47,836 |
| ssh-patator | 3,219 | 0 | 0 |
| web attack - brute force | 0 | 1,470 | 0 |
| web attack - sql injection | 0 | 21 | 0 |
| web attack - xss | 0 | 652 | 0 |

清除后测试有而训练无的攻击族：botnet, ddos, portscan。测试中支持为0的族不计算召回率，不填0冒充漏检。

### corrected stratified 最终类别支持数

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,117,353 | 238,594 | 238,593 |
| botnet | 516 | 110 | 110 |
| ddos | 66,605 | 14,270 | 14,269 |
| dos goldeneye | 5,297 | 1,135 | 1,135 |
| dos hulk | 110,940 | 23,764 | 23,764 |
| dos slowhttptest | 1,218 | 261 | 261 |
| dos slowloris | 2,701 | 579 | 579 |
| ftp-patator | 2,780 | 596 | 596 |
| heartbleed | 7 | 2 | 2 |
| infiltration | 26 | 5 | 5 |
| portscan | 158,354 | 36,238 | 36,241 |
| ssh-patator | 2,073 | 444 | 444 |
| web attack - brute force | 51 | 11 | 11 |
| web attack - sql injection | 9 | 2 | 2 |
| web attack - xss | 12 | 3 | 3 |

### corrected temporal 最终类别支持数

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,011,761 | 258,480 | 255,093 |
| botnet | 0 | 0 | 736 |
| ddos | 0 | 0 | 95,144 |
| dos goldeneye | 7,567 | 0 | 0 |
| dos hulk | 158,468 | 0 | 0 |
| dos slowhttptest | 1,740 | 0 | 0 |
| dos slowloris | 3,859 | 0 | 0 |
| ftp-patator | 3,972 | 0 | 0 |
| heartbleed | 11 | 0 | 0 |
| infiltration | 0 | 36 | 0 |
| portscan | 0 | 71,762 | 82,737 |
| ssh-patator | 2,961 | 0 | 0 |
| web attack - brute force | 0 | 73 | 0 |
| web attack - sql injection | 0 | 13 | 0 |
| web attack - xss | 0 | 18 | 0 |

清除后测试有而训练无的攻击族：botnet, ddos, portscan。测试中支持为0的族不计算召回率，不填0冒充漏检。

### 可复现实现和独立复核

- [生成器](../src/build_split_indices.py)、[独立验证器](../src/verify_split_indices.py)、[依赖与运行方法](../src/README_splits.md)。
- [四套生成结果汇总](split_index_results.json)、[独立验证结果](split_index_verification.json)。四套均PASS；逐文件哈希、输入哈希、规则哈希和生成器哈希均核对通过。
- 验证器对所有导出行核对源文件标签、日期和行ID；核对每条源行恰好出现于保留或清除清单一次；核对两类重叠为零、随机类别覆盖和时序边界。
- 标签和字段JSON保留生成时规则快照。label_split_policy.json内“方案确定”的status是规则制定阶段的记录，实际执行状态以本节、manifest和验证结果为准；不为更新显示状态而改写运行时输入哈希。

索引文件路径：

```text
data/split/{original,corrected}/{stratified,temporal}/
  train_indices.parquet
  val_indices.parquet
  test_indices.parquet
  purged_indices.parquet
  manifest.json
```

训练折内预处理模块与模型数据产物已完成，见[预处理报告](preprocess_report.md)；P3验收签字、P2写作与登记仍需单独完成。
