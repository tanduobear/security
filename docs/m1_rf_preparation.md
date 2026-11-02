# P4 RF-only 辅助对照工程准备

历史阶段记录：下文的pending、未实测和166项测试描述仅对应最初工程准备阶段，不代表当前状态。此后资源试运行、正式五种子及独立验收封存均已完成，当前状态见 [正式验收审计](m1_rf_validation_seal_audit.md)。下文旧配置哈希保留作为历史证据，不用于当前启动。

日期：2026-11-02（北京时间）。仅完成代码、配置草案和合成验收；未启动真实RF训练、资源试运行或后续评估，未提交或推送。本轮固定规格已按用户指令记录，资源试运行和正式执行的批准记录均为pending，不能把工程完成当作启动授权。

## 配置与任务

独立配置：[execution.rf-only.proposed.json](../methods/m1/execution.rf-only.proposed.json)。文件SHA256：

`b7f722561f4df57d414705f57c0a3be3cf5cecfc3025f07c6def4745ebc181fc`

拟定执行载荷SHA256（不含两个批准对象）：

`481fbd6ea3e33f1b80d375aed56d7ae2a7ad63ec3fe0996c2c76baad198b2865`

模型固定为RandomForestClassifier：n_estimators=200、max_depth=20、max_leaf_nodes=4096、min_samples_leaf=2、max_features="sqrt"、bootstrap=true、n_jobs=4、class_weight=null。P2完整original/stratified/main Train/Validation，固定140特征、原处理器和训练折sample_weight，不重复类别加权。RF为辅助，不参加GBDT配置选优，不挑选单个最佳RF种子。

正式任务严格为：

1. RF_FIXED_seed_42
2. RF_FIXED_seed_43
3. RF_FIXED_seed_44
4. RF_FIXED_seed_45
5. RF_FIXED_seed_46

本阶段不生成GBDT任务，不读取真实Test，不运行真实模型Bootstrap或阈值标定。

## 复用边界和保护

新增入口：[run_rf.py](../methods/m1/run_rf.py)，默认plan只读配置、不打开数据。旧search_gbdt.py、seal_gbdt.py、resource_trial.py、models.py以及P2/P3模块均未修改。

安全复用的组件：

- search_gbdt：唯一目录、排他写入、身份/环境摘要、磁盘映射缓存、跨折行ID隔离、全局及运行OS锁。
- resource_trial：Windows Job、GO前限额与CPU亲和性验证、整个进程树RSS/CPU/I/O监控及清理。
- models：RF构造、random_state、P2 sample_weight、正类连续概率分数、预测线程临时限制和恢复。
- executor：分块预测和验证元数据数组；不调用其45任务编排或最终Test入口。
- seal_gbdt：中性的Windows只读共享句柄、原始文件清单和运行锁排除机制；不调用其硬编码40任务/G05的补封存验证器。

GBDT的execute_group、select和专用封存器不直接用于RF，因为算法、任务集合、选优语义及状态均不同。RF只新增自身编排/验收/封存，避免修改已验收的历史路径。

外层串行1；Job提交内存硬限8 GiB、CPU亲和性4个逻辑处理器、训练线程4。启动前系统空闲内存至少14 GiB；运行中低于6 GiB或采样整树RSS超过8 GiB时安全终止。Job提交限额即时生效，RSS阈值按秒采样。没有自动降参、放宽限额或自动重试。

父进程在轻量状态校验批准和启动开关；子进程收到受保护GO后才导入ML和加载数据。P2装载、G05验证预测身份读取、缓存核验、RF拟合、模型保存、释放、重载、分块预测和独立AP验收全部发生在Job内。封存入口仅核验身份和文件哈希，不加载模型或预测数组。

RF保存后先释放原始森林并gc.collect，再加载保存模型进行一致性检查，不同时保留两份200树森林。推理线程暂设1，退出后恢复n_jobs=4，保存的模型参数不被预测线程设置改变。预测块8192避免生成整个Validation的双列概率矩阵。

## 结果、恢复和人工检查

本地正式目录：`results/m1_rf_runs/<UTC时间-UUID>/`；资源试运行：`results/m1_rf_trials/<UTC时间-UUID>/`。均已加入.gitignore；是本地归档路径，非GitHub在线链接，不覆盖GBDT、Smoke或Resource Trial历史目录。

每个任务保留模型、参数、完整Validation的row ID、y_true、连续y_score、攻击族和attempted标记。模型重载分数必须逐项相等；保存预测重读后独立复算AP，与内存原始AP严格比较。完成记录绑定配置、代码、环境、P2输入哈希和逐文件SHA256，失败尝试保留。

显式恢复时先核验所有已完成任务、配置和输入身份，以及保存预测的AP；已完成任务跳过，仅重新执行未完成任务。代码、环境、数据、参考G05或任何输出哈希不一致时拒绝。已完成任务来自被中止的监控尝试时，只要任务本身通过独立校验即可保留，验收摘要明确标注该监控尝试中断，不伪造正常退出。

五个种子完成后前台展示AP、训练时间、Job峰值提交内存和采样RSS，然后要求人工输入CONTINUE。之后用独立accept入口复算五份预测AP并汇总均值、样本标准差、训练耗时和每任务采样峰值；不做模型选优。最后seal入口冻结原文件，排除RUN.lock并记录原因，排他原子发布MANIFEST；verify入口用固定MANIFEST哈希重验文件集。

## RF资源可行性：尚未实测

当前安装环境sklearn Tree.NODE_DTYPE为64字节；二分类节点值数组每节点另计16字节。按200棵树、每树至多4096叶估算，活跃节点上限为200×8191=1,638,200，节点和值数组约124.98 MiB。这只是节点有效载荷估算，不是模型对象大小或进程峰值内存保证。

Train float32特征矩阵约0.92 GiB，Validation约0.19 GiB。还需考虑sklearn要求的布局副本、四个并发树构建工作区、bootstrap索引和权重、序列化、预测以及P2加载的中间缓冲；GBDT试运行成功不能证明RF安全。P2加载器内部DuckDB仍无可传入的内存上限，外层Job负责硬保护；P4跨折检查使用512 MB池，不修改P2。

建议先用原样200树参数、种子42、完整Train/Validation，在同样护栏下独立试运行一次，覆盖加载、拟合、保存、释放、重载及预测。试运行只测资源与一致性、不计算AP、不挑参数、不计入正式五次。它成功封存且用户检查峰值/余量后，才考虑正式五种子。试运行失败保留诊断，不能自动降参或重试。没有真实RF训练耗时预测。

正式run入口要求提供这次RF试运行的目录和固定MANIFEST SHA256，验证算法参数、数据、代码、依赖环境、完整数据规模、资源护栏、保存重载证明及清单。GBDT资源报告不能代替RF证据。仅批准文件的状态变化可与试运行后的正式批准分开记录，不因此改写试运行；模型/数据/必要代码变化会使证据失效。

## G05后续配对身份

原GBDT归档302/302再次验证通过，selection.json和MANIFEST保持原哈希。G05五个种子均为367,036条验证记录，行序摘要一致。RF运行时对封存参考执行只读共享保护并核验全部哈希，不反序列化GBDT模型；读取G05五份Validation文件的实际row ID和标签摘要，与RF验证数据逐种子比较后才允许拟合或接受结果。

RF的binding/paired_reference/summary保留相同种子、同数据版本、同预处理器、同Validation行序和标签身份，支持后续配对比较。本轮没有进行Bootstrap、置信区间或检测阈值计算；也没有声称已完成P3统一推理基准测试。

## 尚需确认和授权

用户已明确给定RF模型、五个种子、P2数据、权重及硬资源限额，本轮不重复索要这些规格的批准。仍需确认：

1. 单次完整RF资源试运行的真实执行授权；建议种子42、全部200树，不计入正式五次、也不复用为正式结果，因此会额外产生一次资源性质的拟合。
2. 草案执行细则：外层串行1、预测线程1、预测块8192、P4隔离检查512 MB，以及人工CONTINUE的操作约定。这些工程默认值已实现，但执行批准记录仍pending。
3. RF试运行成功并人工检查后，另行明确批准正式五种子执行，填写来源、日期及完整载荷摘要；仅试运行获批不能启动正式run。
4. 若需临时防自动待机，须另行授权；默认不修改电源行为。已知GBDT阶段待机会污染训练墙钟时间，RF应保持机器唤醒。

统一协议保持approved；原execution.proposed.json仍pending。新文件中的specification_approval仅记录本轮固定规格及工程准备授权，trial_approval与execution_approval都pending。Bootstrap、阈值和完整最终评估细则不随本阶段一并批准。

## 本轮实际验收

- RF专项35项合成测试通过；最终完整回归166项全部通过，77.730秒，结尾OK，无跳过。
- 覆盖五RF任务、pending/启动开关门禁、P2权重与随机种子、断点恢复、坏哈希拒绝、模型释放后重载、独立AP、人工确认门禁、G05参考校验、试运行证据、验收封存和原参考文件不变。
- Windows原生测试在64 MiB Job下申请128 MiB验证限额，并验证4逻辑处理器亲和性和资源中止清理；不申请8 GiB内存。
- 五个森林使用不训练的替身，数据仅为临时80/640行合成输入。试运行证据正向测试仅模拟完整规模的元数据，不构成真实RF资源证据。
- 发现的日志编码问题已在RF父入口修正，子进程继承UTF-8，未改动历史监控器。原GBDT归档、P2/P3及三个既有协议/配置文件哈希不变。
- 真实RF目录没有创建，真实RF/GBDT未拟合；既有Bootstrap回归单元测试仅使用合成数据，不是RF/G05实验。

## 入口

当前允许的只读计划命令：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.run_rf plan
```

后续阶段为trial → run（显式引用成功RF试运行及其固定MANIFEST SHA256）→ 人工CONTINUE → accept → seal → verify。除只读plan/verify外，入口要求对应批准及--confirm-start；当前pending配置会在输出和数据访问前阻止真实执行。本轮没有运行这些真实阶段。

结论：RF-only工程基础和合成验收完成；仍不具备直接启动正式五种子训练的条件。下一步是批准执行细则并授权单次RF资源试运行，检查实测结果后再决定正式启动。
