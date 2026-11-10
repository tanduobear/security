# 完整回归测试与交付检查

在仓库根目录使用正式实验原`.venv/Scripts/python.exe`（Python 3.13）。依赖版本见[requirements-lock.txt](../methods/m1/requirements-lock.txt)。历史87/92/131/166/174/182项记录各对应当时范围，不等于当前全部测试数量。

## 三套必跑测试

| 套件 | 命令范围 | 预期数量 |
|---|---|---:|
| 核心代码、P2/P3接口、GBDT/RF、资源、封存、最终评估及Bootstrap | tests/ | 235 |
| 统一汇总、PR数据核验和Enlarged View交付 | tools/tests/ | 34 |
| 保留失败尝试的RF恢复验收 | docs/test_rf_acceptance_recovery.py | 8 |
| 合计 | 三套全部执行 | 277 |

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Core tests failed' }

.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tools/tests -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Tool tests failed' }

.\.venv\Scripts\python.exe -B -X utf8 docs/test_rf_acceptance_recovery.py
if ($LASTEXITCODE -ne 0) { throw 'Recovery tests failed' }
```

单独运行discover tests不会发现另外42项。原216项为历史基线；新增最终评估工程准备25项后为241项，新增Bootstrap工程36项后为277项。保留测试目录结构及跨文件合成夹具引用；不要为减少文件数量删除测试。Enlarged View现有布局测试同时验证真实入口将文件写到统一的figures/fig_m1_pr.png/.pdf，防止再次出现文件名不一致。

所有测试使用合成输入、临时目录及替身。Smoke集成只拟合极小合成模型；不是实际数据训练。Bootstrap单测仅针对合成输入，不是实验Bootstrap。Windows原生资源测试会创建受限小额内存探针及临时日志，不执行真实40次/五种子训练，不读取真实Test。

## 只读汇总检查

```powershell
.\.venv\Scripts\python.exe -B -X utf8 tools/aggregate_m1.py --check
```

这只核对公开JSON、Schema、来源SHA256及统计，不打开原始Parquet、模型或完整预测，不覆盖results/m1.json。

## 正式图复现与资源

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.plot_pr_curve_enlarged
```

正式图统一为[PNG](../figures/fig_m1_pr.png)、[PDF](../figures/fig_m1_pr.pdf)，报告见[Enlarged View验收](m1_pr_curve_enlarged_report.md)。该命令需要本地完整G05归档，只核验封存文件并解码五份Validation预测，独立复算AP后绘图；不训练、不读取Test。它会创建字体缓存，不能与“只读检查命令”混淆。

历史Smoke图从Git原版本恢复，不运行真实Smoke来重建。P2全数据重建/验证命令会涉及全部划分，不属于本次无Test交付检查。

## 本次实际执行结果

2026-11-02（北京时间），本次获批清理与交付修复后使用原.venv执行：

| 套件 | 实际数量 | 结果 | 耗时 |
|---|---:|---|---:|
| tests/ | 174 | OK | 78.168秒 |
| tools/tests/ | 34 | OK | 2.835秒 |
| 恢复封存测试 | 8 | OK | 0.508秒 |
| 合计 | 216 | 全部通过，无FAIL/ERROR或跳过 | 81.511秒 |

原始测试输出仅本地保存在`.tmp/repository_cleanup_{core,tool,recovery}_tests.txt`，不随Git提交。另完成了公开汇总只读校验、G05五种子真实Validation AP严格复算，以及原归档SHA256验证；正式PNG保持原字节，缺失PDF从真实曲线重新导出。Smoke图片从Git HEAD原始对象恢复，没有重训。

## 最终评估工程准备回归

2026-11-09（北京时间），新增25项合成验收后再次使用上述三条命令执行完整回归：tests/ 为199项、tools/tests/ 为34项、恢复封存为8项，共241项，全部OK，无FAIL/ERROR或跳过。此前216项记录保持为历史证据。

日志仅在本地 `.tmp/m1_final_preparation_final_{core,tools,recovery}_tests.txt` 保存。新增集成测试中的Test仅为临时合成夹具；没有读取真实Test、拟合真实模型或执行实验Bootstrap。阶段门禁及仍待批准的细则见[最终评估工程准备](m1_final_evaluation_preparation.md)。

## Bootstrap工程完整回归

2026-11-09（北京时间），新增36项合成测试后完整执行上述三套命令：核心235项（110.321秒）、工具34项（2.718秒）、恢复封存8项（0.512秒），共277项，全部OK。覆盖与P3严格一致性、并列/稀有类别、共享索引、可复现RNG及恢复、哈希篡改拒绝、20项编排、独立CI/SE/RNG验收、三张真实PNG及发布清单校验。

测试只使用合成输入和临时目录；合成Bootstrap不计入正式实验。原始日志仅本地保存在 `.tmp/m1_bootstrap_{core,tools,recovery}_tests.txt`，验收记录为 `.tmp/m1_bootstrap_engineering_acceptance.json`，均不是GitHub可访问链接。
