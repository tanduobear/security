# G01 正式搜索人工检查点记录

本次属于已批准的 GBDT-only 正式搜索，区别于 Smoke 和资源试运行。G01 五次拟合完成，工作进程退出码 0；记录时前台父进程等待人工 CONTINUE，G02 尚未启动。

本地归档（不随 Git 提交，目录名仅供本机定位）：`results/m1_gbdt_search/20261101T115824-90de9307e77448aba412cc9ec08a5cbd`。

配置：`methods/m1/execution.gbdt-search.v1.json`；文件 SHA256：`931d2d6e6639a825b62dd7e21f23f4a63e0e30a6a631b58ffd3469afca81b08b`。

完整训练 1,764,265 行、验证 367,036 行，140 个 P2 特征；使用训练折 sample_weight，未重复设置 class_weight。G01 固定 200 轮、学习率 0.05、最大叶子 15、最小叶样本 20、L2=1、max_bins=255、early_stopping=false。

| 种子 | 完整验证 AP | fit 秒 | 任务采样峰值 RSS/GiB | 任务平均 CPU/逻辑核等效 | 文件校验 |
|---|---:|---:|---:|---:|---|
| 42 | 0.999326104 | 274.32 | 3.53 | 1.68 | 通过 |
| 43 | 0.999496543 | 284.65 | 3.39 | 1.63 | 通过 |
| 44 | 0.999322288 | 272.86 | 3.39 | 1.69 | 通过 |
| 45 | 0.999517824 | 278.89 | 3.39 | 1.65 | 通过 |
| 46 | 0.999504067 | 279.58 | 3.39 | 1.64 | 通过 |

五种子验证 AP 均值 0.999433365，样本标准差（ddof=1）0.000099958。当前只有 G01 完成，不据此选择全局最优配置。

RSS 和 CPU 按 1 Hz 采样的完整任务阶段汇总，包含训练、验证预测及保存校验；不是单独 fit 阶段的 CPU，也不是 OS 精确峰值。fit 秒为模型拟合调用实测时间。

受保护工作流程用时 1657.17 秒；全流程采样峰值 RSS 3.53 GiB；Job 峰值提交内存 3.47 GiB；最低系统空闲内存 13.44 GiB。Job 提交内存硬限 8 GiB，CPU 亲和性 4 个逻辑处理器，护栏未触发。

复核：五个模型、五份验证预测、五份任务配置共 15 个文件的文件集、大小、SHA256 全部通过；五份完成记录的配置和身份绑定一致。工作进程已完成模型重载与保存预测一致性校验。缓存全部文件、初始源码快照、当前源码和 train/val/处理器/清单哈希再次核验通过。

原始证据：本地归档内 completed/、attempts/、binding.json、cache.json、sources/ 及 monitor/G01-*/ 下的 resource_samples.csv、events.jsonl、guard_verified.json、worker_result.json、outcome.json。CSV 的目录体积列仅是本组监控目录体积，不代表整个搜索归档。

没有读取真实 test，没有训练 RF，没有运行 Bootstrap 或最终评估，没有自动重试、降参或取消限制，没有覆盖 Smoke 或 Resource Trial。统一协议保持 approved，后续 execution.proposed.json 仍为 pending。

检查完成后由用户在前台终端输入 CONTINUE，才允许进入 G02；本报告不代替该确认。
