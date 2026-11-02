# P4 RF 辅助对照公开摘要

RF_FIXED使用RandomForestClassifier，200棵树、max_depth=20、max_leaf_nodes=4096、min_samples_leaf=2、max_features="sqrt"、bootstrap=true、n_jobs=4、class_weight=null；仅使用P2训练折sample_weight。数据为original / stratified / main，Train 1,764,265行、Validation 367,036行、固定140个特征及原P2预处理器。

| 种子 | 完整Validation AP（独立复算） | 拟合秒数 |
|---|---:|---:|
| 42 | 0.9995840304657881 | 982.3194612999796 |
| 43 | 0.9996585733245633 | 921.4580291999737 |
| 44 | 0.9996925938130498 | 848.9390259999782 |
| 45 | 0.9996655179639135 | 936.2892107000225 |
| 46 | 0.9995817221884112 | 926.939762199996 |

五份完整Validation预测的独立AP均与训练记录严格相等，最大绝对差0。平均AP为0.9996364875511452，样本标准差（ddof=1）为0.000050570370864307355；平均拟合923.18909787999秒，样本标准差47.95349332916437秒。RF是固定参数辅助对照，不参与GBDT选优，不选择最佳RF种子。

这些是Validation结果；未读取真实Test、运行实验Bootstrap或标定阈值，没有最终测试性能或显著性结论。资源日志不构成P3统一推理计时基准。G05参考仅作数据/行序身份核验，未重训GBDT。

## 文件和原始归档范围

- [summary.json](summary.json)：本地原始summary.json的逐字节副本。
- [MANIFEST.local-archive.json](MANIFEST.local-archive.json)：本地原始MANIFEST.json的逐字节副本。
- [acceptance_receipt.json](acceptance_receipt.json)：本地独立验收收据副本，包含逐种子复算一致性和资源结果。
- [environment.public.json](environment.public.json)：原环境记录的公开派生副本，仅将Python可执行文件绝对路径替换为仓库相对路径，保留原记录SHA256。
- [verification.json](verification.json)：本次交付前的哈希验证及合成回归测试摘要。

**GitHub不包含清单列出的全部157个原始文件。** 模型、完整逐样本预测、磁盘映射缓存、原始日志和源码快照只保存在本地；本目录不是完整归档，不能仅凭这里的文件完成157项验证。

本地归档定位名（非在线链接，不随本次提交上传）：

`results/m1_rf_runs/20261102T103826-3bdbabff97db4e2f99ff8a8a6b0b7767/`

MANIFEST内的路径相对于该本地原始归档根目录，非本公开摘要目录。RUN.lock作为运行时锁被排除，原因记录于MANIFEST；MANIFEST不递归包含自身，其SHA256在此单独固定。

MANIFEST SHA256：

`21ac71053cb0142de61d10dbcfa90d86e0f7f51f38bcaceacd0ae4efe75f5690`

summary.json SHA256：

`4ef10efb5de9ed8f4f2bfaa8bd4d24dcfc1a323f5b819b0161c7de79ac9b7d87`

详细审计、失败验收记录的保留及原.venv恢复过程见 [RF验收封存审计](../../docs/m1_rf_validation_seal_audit.md)。统一协议保持approved，原execution.proposed.json继续pending，RF独立批准不扩展为后续评估批准。
