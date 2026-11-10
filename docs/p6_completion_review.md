# P6 完成度复核

结论：P6主要实验、英文正文、最终校验和GitHub交付已完成。成果提交9682b45已发布到main。交付登记表、进度看板和问题记录已按较新副本更新P6信息，S6的P7验收栏保留空白，待P7独立验收；S5全面验收仍未闭环。

## GitHub发布

代码、结果摘要、图及正文已发布，正文采用P5的标题与Methods/Results章节格式。完整模型、14份原始P3记录和7份逐样本预测通过submission/P6_handoff.zip（Git LFS）交付，同目录提供SHA256。下载和解压方式见results/m3_summary/README.md。

本次为普通快进推送，未使用force。共享README只追加P6导航，Git规则只增加P6文件范围，未修改队友的方法、数据或已有实验结果。三张Word表的更新保留较新副本中其他成员的全部原有单元格内容，补充P6交付、进度、发现的问题和S5阶段核查；不代签P7验收。

## 已完成并可从文件核对

| 任务 | 当前证据 |
|---|---|
| 八配置五种子主模型搜索 | results/m3_runs/search.json 中40条记录，获选A06 |
| 良性训练与验证选模 | run.py 使用 benign_only；搜索记录保存训练数、验证数和阈值来源 |
| q95、q99两个阈值 | docs/m3_threshold.md，五个种子各一组 |
| 主模型测试结果 | 10份 Autoencoder 单记录及5份逐样本预测 |
| 同族对照 | Isolation Forest、One Class SVM各一次拟合，4份结果及2份预测 |
| 结果索引 | results/m3.json 和 results/m3_runs/auxiliary.json |
| 误差图 | figures/fig_m3_error_dist.png，已查看；极端值使主要分布拥挤，建议使用新绘图脚本 |
| 英文正文 | docs/p6_text.md，两节均有结果来源注释，已纠正早停结束轮数与最佳checkpoint的措辞 |
| P5验证集权重复跑 | docs/p6_acceptance/m2_replay_val/report.json，五种子已检查阈值均在容差内 |

## 收尾状态

11月10日复核补充：P5在m2-delivery（d27d501）补交五种子记录及预测，并删除正文跨评估集比较；结构检查通过，但P3完整校验仍失败，八配置证据未补。详见[最新复核报告](p6_review_20261110.md)。三张Word表已据此以e7d13b1更新main，未合并P5分支。

1. 已完成：docs/p6_acceptance/final_audit.json中status=passed、failures为空，覆盖14份P3记录及保存分数的数值复核；未重新训练或执行模型推理。
2. 已完成：核查报告保存Python、Torch、NumPy、sklearn、DuckDB、pyarrow等当前环境版本；这是审计环境记录，不倒推证明所有历史环境完全相同。
3. 已完成：完整档案已打包并上传Git LFS，ZIP CRC检查通过，附带SHA256和逐文件清单；结果索引保留原始相对路径并新增归档说明。
4. 已完成登记：仓库内三张Word表已填写S6状态、交付物路径、11月9日完成时间，以及问题处理记录。S5验收明确为阶段核查，未宣称全面通过；S6验收结果及验收时间由P7填写。如团队另有在线共享文档，应同步这三份文件中的记录。
5. S5全面验收仍待处理。已交付权重的验证集预测通过，不等于从头重训一致或Test指标复核完成。P5八配置搜索记录、P3逐种子格式和正文评估集对齐仍未见解决证据。

## 需要明确交给P7的事项

P7独立复跑获选模型、确认阈值只由验证负类标定。当前核查只能检查保存记录和代码路径一致性，不能取代独立重算标定分位数。

跨方法配对bootstrap尚未执行，当前comparisons为空。应在预测行、数据版本、评估集和分数一致对齐后，由汇总阶段完成；目前不能写“显著优于”或已给出跨方法置信区间。

修正版复跑和三方总表属于P7后续任务；不要因为P6主线完成就代替P7宣布全部实验完成。按天时序和去端口消融属于扩展顺序，需要团队确认执行人及是否纳入本次范围，当前P6结果仅覆盖original/stratified/main。

辅助方法使用固定10000条负类子样本，自编码器使用全量负类训练，这一范围已在正文披露；辅助方法不能当作同数据量五种子公平比较。当前没有专门的未知攻击族留出实验，因此不宣称已验证未知攻击检测能力。

## P7取得交付包

```powershell
git lfs pull --include="submission/P6_handoff.zip" --exclude=""
Get-FileHash submission/P6_handoff.zip -Algorithm SHA256
Get-Content submission/P6_handoff.sha256.txt
Expand-Archive -LiteralPath submission/P6_handoff.zip -DestinationPath .tmp/p6-review
```

解压到新目录后，按README取得匹配的P2原始数据并独立复跑。若以后修改代码、正文或记录并重新打包，应重新审计并保留旧包版本。
git
