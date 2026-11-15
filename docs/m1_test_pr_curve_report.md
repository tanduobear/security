# GBDT G05 Test PR Curve 验收报告

输出：[PNG](../figures/fig_m1_test_pr.png)、[PDF](../figures/fig_m1_test_pr.pdf)。仅描述 G05 Test，不作方法比较。

## 数据与完整性

- 来源：[正式 Test 摘要](../results/m1_test_summary/summary.json)。原始封存目录 `.tmp/m1_final_runs/evaluate-20261109T122704-2bf90cbb95e04ceea7c40993a6ec5012/` 仅为本地定位名，GitHub 不包含完整归档。
- Test 归档 82/82 个文件在计算前后通过大小和 SHA256 校验；Windows FILE_SHARE_READ 句柄阻止已有文件并发写入/删除。
- 另核对 GBDT 原 MANIFEST 固定哈希，以及 18 个 G05 相关配置、输入绑定、selection、任务、模型文件的哈希；未宣称重新校验全部 302 个训练归档文件。模型仅字节哈希，不反序列化。
- 五个种子各 346,544 行，良性 282,506 行，攻击 64,038 行；行 ID 唯一且顺序、标签逐项一致。
- original / stratified / main；P2 固定 140 特征、训练参数、模型种子、数据及原始 P3 记录身份全部匹配。
- 仅解码五份 G05 Test NPZ 的 row_ids / y_true / y_score，使用 sklearn average_precision_score 和 roc_auc_score 独立复算；与两套冻结阈值下的原记录严格一致。
- Test 行 ID 摘要：`33455428ecd6ebe255c60819f77958321e643a2f9104a8fcc2cc2ae9685422c3`。
- 标签 uint8 SHA256：`7096835d7bca2b55d6dc5743f68046466cea0e6153a36eb20d8836b9bf0286c8`。
- 原 Test Parquet 身份（未打开）：`f29b51ccd3daf8bc799804100a0dda0454573750eb3e60b7beef0f3d3b0b59d0`。

| Seed | 独立 AP（与封存值差 0） | 独立 ROC-AUC（差 0） | Test NPZ SHA256 |
|---|---:|---:|---|
| 42 | 0.9995211773199413 | 0.9999097292168927 | `59a90ac207890f52a61f858530b6e604f2d5e214e5f565747e95f67720d2ac1a` |
| 43 | 0.9995045813989475 | 0.9999063805572969 | `33497f6383219dcaf08d0696f45ee9f43ee7f610dd38307d6d39467f3c8d9dd4` |
| 44 | 0.9993695477488194 | 0.9998888423665415 | `77a4db2803c62d8a09f49e25feec42eb212a245ecd9d29e5fab0b0cd8ecb3969` |
| 45 | 0.999561454443372 | 0.9999184175405955 | `31c8b0ebf6a36d2b74d1e98cb10c101d5f00c70e1107cae0ee84ae43f685d914` |
| 46 | 0.9995189193477053 | 0.9999093761707423 | `a362b2fe8bf93d0dd886aee9065cf854f3337f4e6e493ae2ceed6a88c11b6647` |

预测位置均为本地原始归档内 `predictions/G05_seed_<seed>.npz`。

## 绘图与论文口径

- sklearn precision_recall_curve(y_true, y_score, pos_label=1) 完整分数计算五条独立曲线；不平滑、不平均，不从 AP 反推曲线。
- 单面板 Enlarged View：Recall 0.98–1.00（间隔 0.005），Precision 0.99–1.00（间隔 0.002）；图注明确局部范围，无 inset 或双面板。
- 五种颜色与线型，图例六位 AP、位于图外；AP 是完整 Test AP，不是局部面积。白底轻网格，PNG 2400×1740、300 DPI，PDF 同图矢量导出并嵌入字体。
- 原 figures/fig_m1_pr.png/.pdf 及其报告仍为 Validation；此次新增文件名明确包含 test，不覆盖旧图。
- 论文图表使用 P4-1 临时编号，最终由 P8 全文统一编号。指标表仅列 G05；不提前排名、不使用 Bootstrap 比较，§4.4 另述。

| Test metric | Mean ± sample SD | Operating rule |
|---|---:|---|
| AP | 0.999495 ± 0.000073 | Threshold-independent |
| ROC-AUC | 0.999907 ± 0.000011 | Threshold-independent |
| Macro-F1 | 0.987784 ± 0.003178 | Frozen Validation 1% FPR threshold |
| TPR | 0.999966 ± 0.000007 | Frozen Validation 1% FPR threshold |
| Actual FPR | 0.009169 ± 0.002436 | Frozen Validation 1% FPR threshold |

以上为五种子均值 ± 样本标准差（ddof=1），所有数值以 0–1 比例给出；AP/ROC-AUC 不依赖阈值，分类指标固定使用仅在 Validation 标定的 1% FPR 阈值。

## SHA256 与复现

- PNG：`c76baf8019442df47b3904ccb33ef85227eeef55f392320885591488f088b417`。
- PDF：`d661df44dc3acf62c770802c53166b93a4351e2c33850b1fa4e11a8faad191f4`。
- 绘图脚本 tools/plot_m1_test_pr.py：`f275a6dc21125d895b5341c37ecdcf1a8c29754431b83410678ca8cb5aa2bee8`。
- 复用数据校验模块 methods/m1/pr_curve_data.py：`23c4c3442a0a6aaff351b396f5c651dd20fb4a0ebde242315824a766a7ac0a76`。
- test_summary_sha256：`cb2a95db154792d5dc0d3fb5f2710aba3ed6cc04f12a964e958cf26f371c23e3`。
- test_manifest_sha256：`34a1503cd829290714d87dc6c84165295d92323dd1d44839ecbf5cbb49430c4f`。
- training_manifest_sha256：`00128ffea5088c939a2375502372be4ba6b95cc13bcdee4f68e6c1638d2b1d6d`。
- selection_sha256：`859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9`。

环境：Python 原 .venv，NumPy 2.3.5、sklearn 1.8.0、Matplotlib 3.10.7。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m tools.plot_m1_test_pr
```

需要本地完整 Test 归档及 G05 训练元数据/模型文件。所有输出不同即拒绝覆盖；原始档案只读。没有读取 Parquet、加载训练矩阵、加载或重训模型，没有新选优、阈值标定或 Bootstrap，没有修改 P2/P3、配置、原始预测与封存结果。
