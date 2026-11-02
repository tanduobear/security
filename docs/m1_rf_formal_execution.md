# RF五种子正式执行：逐种子人工检查点

当前交付状态：下文保留训练前的计划及授权说明；实际五种子及逐种子CONTINUE已完成，独立验收和157文件封存通过。详见 [验收封存审计](m1_rf_validation_seal_audit.md) 和 [公开摘要](../results/m1_rf_summary/README.md)。旧系统Python验收失败记录被保留，后续获准使用原.venv进行新验收；封存采用独立恢复脚本，不修改此处冻结的训练代码及配置。

2026-11-02（北京时间）。用户明确批准RF_FIXED种子42、43、44、45、46的正式训练，要求每个种子后必须CONTINUE，不允许自动进入下一种子。模型、P2数据、权重和资源限额保持既定规格。

## 批准配置和原试运行证据

- 配置：`methods/m1/execution.rf-only.proposed.json`，execution_approval现为approved，仅覆盖RF-only。
- 正式配置SHA256：`65fa1b47ad866b1d6d6db53105c47a3bb901bcb8ca005e5e86179b4de16d9043`。
- 正式载荷SHA256：`acc13e8d900d998f4537ed80079117665e9000ac25d1f5f57b0dc0e11449f4e5`。
- 原资源试运行MANIFEST SHA256仍为 `0c36fb469e3f31be0da752ec652da0e732e5cacb0f40ea069b579187745e1494`，原件不改写、不额外重跑、不复用为正式模型。
- 唯一控制策略变化：checkpoint由五个种子后暂停改为每个种子后暂停。
- 兼容性记录：`methods/m1/rf_per_seed_compatibility.v1.json`，SHA256 `90a7f94a35e5d6c0f6aa91c61c2adac2bcf0de059cc7e53f08e03ea936cb9c1f`。

试运行的严格代码身份检查没有被删除：只有run_rf.py和其测试文件的已固定新旧SHA差异可按本记录接受；其他必要代码、数据、参数、依赖环境变化仍拒绝。模型适配器和Windows Job监控器字节未变。对原快照与当前代码执行AST检查，拟合、权重、预测、保存释放重载、行序比较、资源保护和汇总核心均一致；execute_tasks只允许增加精确的target_seed过滤与对应关键词参数。原试运行的批准记录、源快照和所有结果哈希保留。

## 执行和检查点

入口仍为 `python -m methods.m1.run_rf run --confirm-start`，显式引用成功资源试运行及固定MANIFEST SHA256。父进程逐次启动run_42到run_46的独立受保护worker；每个worker只允许拟合指定种子。P2加载、缓存核验、RF拟合、模型保存、释放重载、Validation预测和AP保存校验均处于8 GiB Job及4逻辑处理器亲和性保护下。

每个worker完成后释放训练进程，父进程显示该种子的AP、fit秒数、Job峰值提交内存、采样RSS和CPU摘要，停在前台CONTINUE提示。确认记录绑定task_id、原运行身份及完成记录SHA256，保存在checkpoints/RF_FIXED_seed_N.json。任何其他输入均安全停止，未收到确认不能启动下一种子；异常或资源中止抛出并保留日志，没有自动重试。

显式恢复时，已确认且哈希匹配的种子跳过；已完成但未确认的种子先在保护下验证并跳过拟合，再等待人工确认；不重复训练。五次确认后才生成最终checkpoint.json并结束训练命令。之后独立accept复算五份Validation AP、汇总均值/样本标准差和资源，再独立seal/verify归档，不训练、不读取真实Test。

全程使用固定200树、depth20、max_leaf_nodes4096、min_samples_leaf2、sqrt、bootstrap=true、n_jobs4、class_weight=null，P2训练折sample_weight。外层串行；预测线程1、预测块8192；启动前至少14 GiB系统空闲内存，运行中低于6 GiB或整树RSS超过8 GiB即中止。不降参、不取消限制、不重训GBDT，不运行Bootstrap或阈值标定，不自动commit/push。

用户另行授权临时防待机：只在种子实际运行时请求SYSTEM_REQUIRED，进入人工检查点时释放，控制进程退出时清理；不修改电源计划。辅助日志仅本地保存在`.tmp/`，不会作为实验模型或正式结果。

本文件为执行计划与授权范围记录，不代表种子已经训练完成。实际运行目录、结果和逐种子校验另行记录；所有大型RF产物仅存于本地results/m1_rf_runs/，不覆盖GBDT或试运行归档。
