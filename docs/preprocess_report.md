# P2 共用预处理模块与数据接口

本阶段将已核验的划分索引转换为模型可读取的数据，模块入口为[src/preprocess.py](../src/preprocess.py)。执行完成状态与最终核验结果在本文末尾记录。每个版本、每种划分各拟合一个处理器，所有模型共享该处理器和同一组数据。

## 已确定的处理规则

- 75个数值字段：只用对应训练集拟合StandardScaler的均值与总体方差，使用z=(x−mean)/scale，常量列scale=1。保留负值、零值及原有异常值，不在验证或测试上估计裁剪阈值，不额外做log变换、插补或特征筛选。输出float32并检查有限性。
- 目的端口：只用训练集统计出现次数，固定保留频率最高的64个端口，频数相同按端口号升序决胜。各自独热编码，其他端口包括训练中未见端口统一归OTHER；不使用目标编码，也不使用标签计算编码。该规则保留常见服务身份但合并稀有服务，不能将它描述为无损保留全部端口信息。
- 主版本：75个标准化数值字段加64个端口类别及OTHER，共140个特征。该数值由本次四套训练词表实际达到64个类别确定，并非所有可能数据都固定140维。
- 删除端口消融：直接投影为相同75个数值特征，沿用同一处理器、划分、行序、标签和统计量，不重新拟合或重新划分。
- 类别权重：仅按训练集二分类计数，w_c=N_train/(2*N_train,c)。同时保存scale_pos_weight=N_benign/N_attack供树模型接口使用；使用者不要将class_weight与scale_pos_weight重复叠加而双重加权。
- 权重只供训练损失使用，不对验证或测试重新计算，不用它们重加权最终评分。不通过重采样或SMOTE平衡验证/测试；转换精度重复清除会删除部分验证/测试行，最终样本量和类别分布见末尾。
- Label原始标签、二分类y、攻击族、attempted标记、日期、行号及group_id全部作为元数据独立保留，任何模型均须按处理器特征白名单读取X。

标准化和类别平衡权重的定义参考：[StandardScaler官方文档](https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html)、[compute_class_weight官方文档](https://scikit-learn.org/1.8/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)。实际安装版本、源数据、索引及程序哈希记录于preprocess_manifest。端口top64预算是在模型训练前固定的工程选择，未根据验证或测试分数调参。

## 产物位置

```text
data/split/{original,corrected}/{stratified,temporal}/
  train.parquet
  val.parquet
  test.parquet
  preprocessor.json
  preprocess_manifest.json
  model_train_indices.parquet
  model_val_indices.parquet
  model_test_indices.parquet
  precision_purged_indices.parquet
  no_destination_port/
    train.parquet
    val.parquet
    test.parquet
```

原来的*_indices.parquet、purged_indices.parquet和manifest.json继续保留，清洗文件未修改。main数据每行包含140个特征及9个元数据字段，消融每行包含75个特征及同样9个元数据字段。元数据列数不计入模型输入维度。

preprocessor.json为可读取的JSON参数，不依赖反序列化任意pickle代码。它记录训练行数和行号摘要、75列mean/variance/scale、端口词表及训练端口计数、二分类权重和输入出处。transform不能修改这些状态；同一对象不允许第二次fit，也拒绝以split=val/test调用拟合。

## 生成与核验

在仓库根目录安装src/requirements-preprocess.txt后运行：

```powershell
python src/build_model_data.py --work-dir ../.audit/preprocess-build
python src/finalize_model_data.py --work-dir ../.audit/precision-guard
python src/verify_model_data.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

当前电脑DuckDB位于工作目录上一级，可使用Python 3.12并给前三条命令增加`--dependency-dir ../.audit-deps`。分块读取和写出，不将两份clean一次载入内存；输入排序和合并的临时文件放在显式work-dir。运行脚本需预留几GB临时磁盘空间。

独立核验重新从训练索引连接原始clean数据，核对均值、总体方差、缩放、端口词表及完整频数、类别权重和训练行号摘要。全量扫描主/消融文件，检查有限值、独热合法性、元数据与索引一致、两版本数值投影完全相同；每折固定种子抽1024行从clean重算转换结果。源数据、索引、处理器、程序及生成文件均核对SHA-256。

单元测试覆盖：验证极端值和未知端口不改变训练参数、类别不平衡权重、常量列、消融一致性、拒绝验证集拟合及重复拟合、拒绝元数据作输入、参数保存读取与未见端口策略。

## 下游使用

```python
from src.model_loader import ModelDataset

dataset = ModelDataset('data/split/original/stratified/train.parquet')
class_weights = dataset.class_weights
for X, y, metadata in dataset:
    # 自动匹配同目录参数、核验哈希和状态，只把特征白名单放入X。
    # 将class_weights传给具体模型的训练损失，不再次fit预处理器。
    pass
```

删除端口版本将文件改为no_destination_port/train.parquet，由ModelDataset自动识别消融，不手动混用处理器。读取验证/测试仅改文件名，继续使用对应训练折的同一preprocessor.json。模型输入列由dataset.feature_names确定，不可“删除一列Label后把其余列全送入模型”。

自编码器只从训练折y_binary=0的行训练，用验证折良性子集标定阈值。共用缩放器是整个训练折拟合的统一预处理，不为自编码器偷偷重新用验证数据拟合；attempted按既定真值政策归为良性，该政策须写入Methods。时序场景的未见攻击族和清除后的评价范围仍按split_report说明解释。

## 本阶段边界

本阶段不训练模型、不计算实验性能、不撰写待提交的Methods正文、不代填P3验收签名。数据与模块可交P3读取检查，P2整棒是否结束还需对照交接表确认写作和登记要求。

## 执行与验收结果

本节已按完整复核后的数据更新，覆盖此前仅按原始特征组检查的结果。四个处理器、12份主数据和12份消融数据均已核验；训练参数和训练行保持不变，验证/测试额外清除了因float32精度合并而重复的输入。

| 版本与协议 | train | val | test | 本轮额外清除 | 主/消融特征数 |
|---|---:|---:|---:|---:|---|
| original/stratified | 1,764,265 | 367,036 | 346,544 | 42,953 | 140 / 75 |
| original/temporal | 1,526,504 | 366,184 | 520,967 | 11,506 | 140 / 75 |
| corrected/stratified | 1,467,942 | 315,344 | 314,577 | 2,108 | 140 / 75 |
| corrected/temporal | 1,190,339 | 330,323 | 433,642 | 127 | 140 / 75 |

### 最终模型数据比例

分组随机70/15/15是原始划分的目标，后续无标签的精度重复清除只缩减验证/测试。最终比例以下表为准，不能声称最终模型输入严格满足70/15/15；原版随机训练占比超过原先的1个百分点基线偏差阈值，该阈值仅在原始划分阶段通过。应将“分组随机目标70/15/15＋转换后重复清除”作为完整实验协议交接P3，不能隐去这项方法学变化。

| 版本与协议 | train占最终保留量 | val占最终保留量 | test占最终保留量 |
|---|---:|---:|---:|
| original/stratified | 71.2016% | 14.8127% | 13.9857% |
| original/temporal | 63.2445% | 15.1713% | 21.5842% |
| corrected/stratified | 69.9732% | 15.0317% | 14.9951% |
| corrected/temporal | 60.9086% | 16.9023% | 22.1891% |

### 转换精度重复清除

原始75字段不重叠，并不保证标准化后float32仍可区分。完整复核发现跨折相同模型输入组；现统一在75个数值输入上生成精确分组（正负零统一），按train、val、test顺序保留较早折，清除后折。此保守规则同时覆盖140维主输入和75维消融，不利用标签或模型分数选择删除行。

| 版本与协议 | 修正前跨折相同模型输入组 | 修正后 | val额外清除 | test额外清除 |
|---|---:|---:|---:|---:|
| original/stratified | 15,211 | 0 | 11,230 | 31,723 |
| original/temporal | 10,329 | 0 | 6,846 | 4,660 |
| corrected/stratified | 199 | 0 | 670 | 1,438 |
| corrected/temporal | 91 | 0 | 59 | 68 |

原始*_indices.parquet与manifest.json保留原始划分事实；模型数据以model_*_indices.parquet为有效索引。precision_purged_indices.parquet逐行记录本轮排除项和原因，原来的purged_indices.parquet记录时序原始特征重复清除。两轮排除不可混为一次，也不可按原始测试量填写最终指标样本数。

### 仅训练集计算的二分类权重

| 版本与协议 | 训练良性数 | 训练攻击数 | 良性权重 | 攻击权重 | scale_pos_weight |
|---|---:|---:|---:|---:|---:|
| original/stratified | 1,466,743 | 297,522 | 0.601423 | 2.964932 | 4.929864 |
| original/temporal | 1,323,598 | 202,906 | 0.576649 | 3.761604 | 6.523208 |
| corrected/stratified | 1,117,353 | 350,589 | 0.656884 | 2.093537 | 3.187074 |
| corrected/temporal | 1,011,761 | 178,578 | 0.588251 | 3.332827 | 5.665653 |

### 最终验证与测试类别支持数

#### original/stratified

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,466,743 | 302,963 | 282,506 |
| botnet | 1,364 | 272 | 252 |
| ddos | 89,610 | 19,201 | 19,202 |
| dos goldeneye | 7,200 | 1,541 | 1,542 |
| dos hulk | 120,994 | 25,914 | 25,901 |
| dos slowhttptest | 3,660 | 784 | 784 |
| dos slowloris | 3,769 | 807 | 802 |
| ftp-patator | 4,151 | 890 | 890 |
| heartbleed | 7 | 2 | 2 |
| infiltration | 26 | 5 | 5 |
| portscan | 62,988 | 13,852 | 13,854 |
| ssh-patator | 2,253 | 483 | 483 |
| web attack - brute force | 1,029 | 221 | 220 |
| web attack - sql injection | 15 | 3 | 3 |
| web attack - xss | 456 | 98 | 98 |

#### original/temporal

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,323,598 | 364,005 | 343,345 |
| botnet | 0 | 0 | 1,774 |
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

#### corrected/stratified

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,117,353 | 237,924 | 237,155 |
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

#### corrected/temporal

| 类别组 | train | val | test |
|---|---:|---:|---:|
| benign | 1,011,761 | 258,421 | 255,025 |
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

### 核验结论

- 171个实际字段位置及42个原始标签映射覆盖检查通过；两版源数据SHA-256与规则一致。
- 原始划分索引行号、原始特征组重叠为零，全部源行去向可追溯。
- 四套均值、方差、缩放、端口词表与频数、类别权重及训练行号摘要与训练折独立重算一致。
- 实际float32数值输入的跨折重叠为零，而不是仅信任原始group_id；两版随机方案最终仍覆盖全部15组类别的三折。
- 有效模型行与model_*_indices的9个元数据字段逐行一致，全部特征有限、独热合法，主/消融数值投影完全一致。
- 每折1024行从clean重新计算转换值，与导出数据完全一致。
- 原始保留行恰好对应有效模型行或precision_purged行之一，无遗失或重复去向。
- 10项测试通过，包括参数隔离、状态保存读取、拒绝未核验数据、损坏文件及错误版本处理器。

详细证据：[预处理生成结果](preprocess_results.json)、[独立验证结果](preprocess_verification.json)、[原始划分验证](split_index_verification.json)、[8个真实加载组合验证](model_loader_verification.json)、[全流程复核](p2_full_review.md)。原始划分比例和最终模型比例分别报告；精度清除后的分数仅代表留存样本，不能当作未经清除的完整原始数据性能。
