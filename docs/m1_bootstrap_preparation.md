# P4 Bootstrap 交付准备：待确认

历史记录说明：本文保留早期未批准提案及旧交付命名。用户随后以“Formal Paired Bootstrap — End-to-End Execution”明确批准工程、试跑、20项正式计算、验收及新交付结构，当前配置为 [execution.bootstrap.v1.json](../methods/m1/execution.bootstrap.v1.json)。旧proposed配置仍pending；不得把本文旧目录或未批准试跑状态混同为当前正式执行状态。

本记录只定义交付结构和候选细则，不表示批准试跑或正式Bootstrap。配置草案见 [execution.bootstrap.proposed.json](../methods/m1/execution.bootstrap.proposed.json)；不修改此前已封存实验、P2/P3或原协议。

正式成功后按用户指定结构交付：

```text
results/m1_bootstrap/
  bootstrap_config.json
  bootstrap_results.json
  confidence_intervals.json
  bootstrap_samples.csv
  MANIFEST.json
docs/m1_bootstrap_report.md
figures/fig_m1_bootstrap_comparison.png
```

这些目标文件目前不创建占位版本。试跑结果只保存在独立本地试跑目录，不发布为正式结果。正式交付拒绝覆盖已有文件；MANIFEST应明确路径根、源码/环境/输入预测的身份，以及JSON、CSV、报告、图的SHA256。清单自身不纳入自己的SHA256，另在验收记录保存清单哈希。

## 文件内容约定

| 文件 | 内容 |
|---|---|
| bootstrap_config.json | 实际获批配置、原预测归档和P3源码哈希、模型配对、指标、种子、重复次数、资源及CSV模式 |
| bootstrap_results.json | 五种子×四指标的20项比较；保留原始指标值、G05−RF差值及重采样设置 |
| confidence_intervals.json | 20项逐种子95%区间及解释；不直接平均区间端点，不冒充包含训练波动的总体区间 |
| bootstrap_samples.csv | 内容模式待用户确认，见下文；不保存海量逐行抽样索引 |
| MANIFEST.json | 已完成产物的文件身份、字节数和SHA256，排除运行时锁文件并记录原因 |
| m1_bootstrap_report.md | 方法、资源、结果、统计局限、输入来源、完整性和正式/试跑边界 |
| fig_m1_bootstrap_comparison.png | 按指标展示五种子差值及区间，标注G05−RF和零差参考线；仅从正式结果绘制 |

## CSV接口待决事项

P3 `eval.metrics.paired_bootstrap` 当前只返回原始指标、差值、区间和采样参数。函数内的逐次differences数组没有公开返回，不能直接导出真实逐次记录。不能把区间端点伪装成样本、从区间反推样本，或创建空CSV应付交付。

可选方案：

1. **逐次指标差值**：CSV每行保存model_seed、metric、target_fpr、replicate、value_a、value_b、difference。正式预计40,000行。需要P4侧适配，并先用合成数据严格核对P3的采样顺序、点估计、并列处理、区间和结果字段。不修改P3，不使用反射或运行时替换P3内部函数获取数组；具体适配方案需在实现时审查。
2. **比较结果汇总**：CSV保存20项比较的区间摘要，继续直接调用P3。报告必须注明该文件是比较摘要，不是逐次bootstrap样本。

尚未批准CSV内容模式，也未据此开发或运行真实Bootstrap。

## 已有草案候选与计算边界

沿用旧提案的候选值：G05与RF同种子配对，model_seed=42–46；AP为主要指标，经验TPR@FPR(0.1%、1%、5%)为次要指标；每项2,000次，95% percentile区间，Bootstrap seed=2026，按良性/攻击分层有放回抽样，两个模型使用相同抽样位置。

正式共有20项比较、40,000个配对重采样实例。实际需要对两模型分别计算指标，不能据此承诺很快完成。建议在执行器及合成验收通过后，先对seed42/AP做20次真实预测资源计时试跑；本轮没有运行该试跑，也不启动完整正式阶段。

仅使用已封存的346,544行Test预测，核验原MANIFEST、逐文件SHA256、标签、行ID和顺序；不打开Test Parquet、不加载模型、不训练、不重标阈值，不根据Bootstrap结果调整参数、阈值或搜索空间。沿用8 GiB Job、4逻辑CPU、外层串行和既定内存启动/中止门槛。

P3区间条件于当前拟合模型和观测类别比例，不包含模型训练随机性，也不校正流量组内相关性。经验TPR@FPR会在每个重采样内部重新计算ROC操作点，属于描述性统计，不是重新标定部署阈值。不得将20项边际区间解读为整体多重比较显著性，也不把五个种子预测拼接成五倍独立样本。
