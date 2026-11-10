# P4 最终 Test 评估摘要

本目录记录 GBDT G05 与固定 RF 的五种子基础 Test 评估；模型未重训，阈值仅用 Validation 标定。该阶段不包含 Bootstrap，verification.json 中的 bootstrap_run=false 描述的是基础 Test 阶段。后续独立批准的正式 Bootstrap 已完成并另行封存，见 [Bootstrap 摘要](../m1_bootstrap/bootstrap_summary.json) 和 [报告](../../docs/m1_bootstrap_report.md)；本目录原始指标与验收记录保持不变。

summary.json 为省略完整曲线数组的派生汇总，不是 P3 单模型记录。20 份原始 P3 Schema 记录、10 份完整 Test 预测、缓存、环境、源码快照和监控日志保存在本地 `.tmp/m1_final_runs/evaluate-20261109T122704-2bf90cbb95e04ceea7c40993a6ec5012`。该路径是本地封存归档，GitHub 不包含这些大型文件。

MANIFEST.local-archive.json 是原始清单的字节副本；其相对路径以本地原始归档为根，不能用 GitHub 文件缺失判断归档损坏。

原 results/m1.json 与 figures/fig_m1_pr.png/.pdf 仍属于 Validation 阶段，保持原哈希。计时名单见 docs/m1_shared_timing_rows.v1.json；P5/P6/P7 应复用同一有序名单及总体身份。
