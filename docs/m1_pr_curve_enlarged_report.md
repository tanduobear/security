# GBDT G05 Validation PR Curve — Enlarged View验收

输出：[PNG](../figures/fig_m1_pr.png)、[PDF](../figures/fig_m1_pr.pdf)。仅为Validation，不代表Test评估。

## 图形规范

- 标题：GBDT G05 — Validation PR Curve (Enlarged View)。
- 仅一个Enlarged View面板，无Full-range View、无inset、无双面板。
- Recall 0.98–1.00，刻度间隔0.005；Precision 0.99–1.00，刻度间隔0.002；两轴三位小数。
- 图内图注明确声明局部坐标范围：Axes show local ranges: Recall 0.98–1.00; Precision 0.99–1.00.
- 五个种子独立曲线，不同颜色和线型；图例位于图外下方，AP保留六位小数。AP来自完整Validation预测，不是局部图范围内积分。
- sklearn precision_recall_curve(y_true,y_score,pos_label=1)，完整阈值点，step(where="post")；不平滑、不平均、不从AP反推曲线，不简化路径。
- 白底、DejaVu Sans、轻网格；PNG 2400×1740、300 DPI；PDF为同图矢量导出，嵌入字体。

## 原始数据和独立验证

- 本地封存归档定位名：`results/m1_gbdt_search/20261101T115824-90de9307e77448aba412cc9ec08a5cbd/`，仅为本地定位说明，非在线链接。
- m1.json、原selection及任务确认G05；original / stratified / main、P2固定140特征，配置及数据绑定一致。
- 归档302/302文件在绘图前后通过SHA256；Windows只读共享句柄阻止归档并发写入或删除。
- 每种子367,036行，良性302,963行、攻击64,073行；行ID唯一、五种子行序和标签逐项一致。
- 行ID摘要：`c6a10ee0cf3ea5ceb71a4b147bb0c762da36aaa5554ab476173b839d16b9631a`。
- 标签uint8摘要：`834caed87cb3586e005cd995f6502a3c4439acec53b45999a0edf70587f9170c`。
- 独立调用average_precision_score，从真实y_true及连续攻击概率y_score复算，五次AP与封存任务及m1.json严格相等，差值均为0。

| Seed | 独立复算AP | 本地归档内预测路径 | 预测SHA256 |
|---|---:|---|---|
| 42 | 0.9997869026152735 | `attempts/G05_seed_42-19993af0a29546c8863c91ed80ee14c4/validation.npz` | `f1185bbfcf0b4f392fb867e9390c80262882aa5eb629b27eabde9041ead72a6d` |
| 43 | 0.9998024922803087 | `attempts/G05_seed_43-a6574317b6aa43b486dfa50c2dfbe351/validation.npz` | `07b5ec9ba5b79bdccc2a772fd3b70676e3e16277fae00ae972da03faee35e4c6` |
| 44 | 0.9997679802089401 | `attempts/G05_seed_44-172f6717c90143a79877f5f705cf295d/validation.npz` | `bc0ca0667e34410d0c98e87ca478c2c669a60eef798021679cbeb76c37d0fb11` |
| 45 | 0.9998087637225794 | `attempts/G05_seed_45-7ae5dd960f72422dbb20fb86129e6026/validation.npz` | `b2bfc150e04f00900065b4e3312f5637770de05969c6008782db863ee058c658` |
| 46 | 0.9997924946636689 | `attempts/G05_seed_46-8790c2465ab34445b3ea9ec6e7cd62e2/validation.npz` | `cb780dd84b1ed96b8231e37d9351a052e67fcc9feac82f11fa3570ac541ed666` |

参数：`{"class_weight": null, "early_stopping": false, "l2_regularization": 1.0, "learning_rate": 0.05, "max_bins": 255, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 20}`。

P2输入身份（只核对封存binding，不打开Parquet）：

- `train.parquet`：`a300844eadac44f3d2f4a43084871e96033a60440ff3a43878b9c674def1a77c`。
- `val.parquet`：`1e1159b80cbd147c9ce16d60953394300c1c518b69afbb704c9b075b0293d560`。
- `preprocessor.json`：`d629632aca37b210915b7258231de07955509d81aa1a2921a451b52425b123a6`。
- `preprocess_manifest.json`：`db1e45a300056e5af0cd56fffdbed3c508dbc41d4660cb11c6daa9fa799fcf0a`。

## SHA256

- PNG：`67008041e33ca988ced8af7e24453e687899ef046d9ca0710857ae0153154580`。
- PDF：`c55dcd542bbd1d529a87721930c38d4015882c1642139c99ddb40ca78c6d69f4`。
- 绘图脚本：`9c3bebb52c8d9ad5ee6e184e75f0e48b4f9cdd42fc4669995e57e5eed14ec40e`。
- 数据核验脚本pr_curve_data.py：`23c4c3442a0a6aaff351b396f5c651dd20fb4a0ebde242315824a766a7ac0a76`。
- `manifest_sha256`：`00128ffea5088c939a2375502372be4ba6b95cc13bcdee4f68e6c1638d2b1d6d`。
- `selection_sha256`：`859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9`。
- `config_sha256`：`002d10d4ed1ec7cc06f368001bfe058f5ceaac67cc69af5ec4575de6f8998f4d`。
- `m1_sha256`：`a4b06bc1de785aefba7ab542697d04de21990eb85de30e903ebbc6afb3f2539f`。

## 复现与边界

环境：sklearn 1.8.0、NumPy 2.3.5、Matplotlib 3.10.7。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.plot_pr_curve_enlarged
```

需要本地完整归档。仅解码五个G05 Validation预测，其他原归档文件只做字节哈希；不读取Test、不加载模型、不训练、不重新预测、不运行Bootstrap或阈值标定。原模型、预测、封存归档及P2/P3保持不变。旧绘图版本按用户要求已清理；数据核验逻辑独立保存在pr_curve_data.py。已有PNG/PDF字节相同则保持原样，不同则拒绝覆盖；--refresh-report仅允许在校验通过且图片不变时更新脚本依赖哈希报告。
