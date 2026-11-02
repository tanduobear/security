# RF seed42 真实资源试运行验收

历史阶段记录：下文“正式执行仍pending”指资源试运行结束时的状态。此后另获授权完成五种子正式训练，见 [正式验收审计](m1_rf_validation_seal_audit.md)。本试运行仍独立保留，不计入正式五次结果。

日期：2026-11-02（北京时间）。用户明确授权恰好一次完整数据RF资源试运行，以及仅本次运行的临时防空闲待机请求。结论：成功，未触发资源护栏；仅拟合RF_FIXED seed=42一次。未启动正式五种子、未计算Validation AP或其他正式评估、未读取真实Test、未训练GBDT、未运行Bootstrap或阈值标定，未commit或push。

## 启动前验收

- 原RF工程配置SHA256为 `b7f722561f4df57d414705f57c0a3be3cf5cecfc3025f07c6def4745ebc181fc`，严格匹配。
- 只将trial_approval记录为approved，execution_approval仍pending；模型、数据及资源载荷均未变。
- 批准后的源配置SHA256：`5ef1510b767601fcc7b04361f3f97226788e5ce0fa6a842ff4b92a9b26d9a4f1`。
- 执行载荷SHA256：`481fbd6ea3e33f1b80d375aed56d7ae2a7ad63ec3fe0996c2c76baad198b2865`。
- 独立trial入口已通过验收；补充完整父入口到worker的单拟合、禁止AP合成测试，并确保测试批准夹具不继承真实授权。完整回归167项通过，83.794秒。
- 启动前无其他正式任务，全局资源互斥锁可取得；系统可用内存17.20 GiB，磁盘可用约593.80 GiB。P2 Train/Validation及处理器哈希与封存GBDT输入一致。

## 实际配置与隔离

RandomForestClassifier：n_estimators=200，max_depth=20，max_leaf_nodes=4096，min_samples_leaf=2，max_features="sqrt"，bootstrap=true，n_jobs=4，class_weight=null，random_state=42。使用P2训练折sample_weight；不重新拟合预处理器。

original/stratified/main；Train 1,764,265行、Validation 367,036行、140特征。正式五种子批准未改变。试运行目录为本地归档，未上传，路径不是在线链接：

`results/m1_rf_trials/20261102T094712-04341705641a4b328dde720cf05b59b2/`

只调用 `python -m methods.m1.run_rf trial --confirm-start`，没有调用正式run入口。Job提交内存硬限8 GiB，CPU亲和性mask=15（4逻辑处理器），GO前验证成功。外层串行1，拟合线程4、预测线程1、预测块8192、P4跨折DuckDB池512 MB。运行中系统可用内存低于6 GiB或采样进程树RSS超过8 GiB时中止；没有自动重试、降参或取消限制。

数据与ML导入、P2加载、模型拟合、保存、释放、重载和预测均处于Job保护内。模型先保存并释放原对象，再重载，避免双份森林同时驻留。临时防待机请求于北京时间17:48:03启用、18:04:50自动释放，不修改电源计划；无超过5秒的采样空档。

## A–H 实测结果

| 项目 | 结果 |
|---|---|
| A. 实际model.fit耗时 | 976.42秒，约16分16秒 |
| B. Job最高提交内存 | 2.834 GiB，OS Job原生峰值计数 |
| C. 整个进程树采样峰值RSS | 2.463 GiB，1 Hz采样，可能错过瞬时峰值 |
| 单个实际Python worker的OS峰值RSS | 2.489 GiB；与整树采样峰值不是同一口径 |
| D. 最低系统空闲内存 | 14.509 GiB，按秒采样 |
| E. CPU使用 | 全流程平均约1.64个逻辑核等效；拟合阶段含输入检查约1.70；采样峰值约2.96 |
| F. 阶段完整性 | 完整加载、拟合、模型保存、原模型释放、重载、全Validation预测及保存一致性全部完成 |
| G. 限制和异常 | 工作进程与前台命令退出码均0，abort_reason=null；无资源护栏触发、无自动重试 |
| H. 正式五种子建议 | 可建议沿用同一8 GiB/4逻辑处理器限额串行执行；仍需用户单独批准，未自动启动 |

受保护worker流程约1057.89秒（17分38秒），不含父进程最后归档哈希扫描。以下为阶段事件间隔，不冒充单独训练时间：

| 阶段 | 秒 |
|---|---:|
| 代码、输入及G05参考核验 | 11.29 |
| P2 train/val加载、映射与隔离检查合计 | 52.01 |
| 拟合阶段含输入检查 | 976.75 |
| 第一次完整Validation预测 | 7.09 |
| 模型保存及原对象释放 | 0.31 |
| 模型重载与第二次完整预测校验 | 7.35 |
| 预测保存、重读及完成记录 | 1.49 |

CPU核等效由累积CPU时间除阶段墙钟时间计算；不是单请求时延或标准P3推理基准。资源数值是本次实测，不据此保证其他种子的峰值或耗时完全相同。建议正式阶段仍保留相同护栏、串行执行、人工验收和失败停止策略，且不能自动复用本试运行模型。

## 校验与证据

完成记录只有 `RF_FIXED_seed_42.json`；purpose=resource_trial_only，validation_ap=null，reload_predictions_equal=true，saved_predictions_equal=true。没有独立AP阶段事件。model.n_features_in_=140和200个estimators的运行时检查通过。源代码、P2输入身份及批准配置均保持启动时状态。

资源归档81/81文件SHA256复核通过。MANIFEST排除RUN.lock，记录运行时排除原因，不覆盖任何历史产物。

试运行MANIFEST SHA256：

`0c36fb469e3f31be0da752ec652da0e732e5cacb0f40ea069b579187745e1494`

批准源配置与归档config.json仅相差一行末尾空行；JSON和规范载荷完全一致。只读复核脚本最初错误要求二者字节哈希相等，已改为分别核对各自基准并比较JSON与绑定摘要，未修改实验文件。归档配置有其独立、已封存的SHA256：

`0d807e7a8f40e6be5b3a510daf0932c09fe51b013b8e4458da49da4290a28cc9`

RF试运行证据接口require_trial也以固定MANIFEST哈希通过校验。正式RF目录尚不存在，正式批准仍pending。GBDT原始归档试运行前后均302/302校验通过，原MANIFEST和selection哈希分别仍为：

- `00128ffea5088c939a2375502372be4ba6b95cc13bcdee4f68e6c1638d2b1d6d`
- `859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9`

原始资源CSV、阶段事件、GO前资源证明、worker日志、完成记录、模型、预测和resource_report.json均在该本地试运行目录。临时启动/前台/防待机日志及只读复核摘要位于本地`.tmp/`，不随Git上传。没有把本次试运行记作正式RF实验结果。
