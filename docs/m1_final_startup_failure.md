# 最终评估首次启动记录：辅助模块参数错误

2026-11-09（北京时间），用户已批准 Validation 阈值标定、最终 Test 评估和共同计时名单生成。启动前系统空闲内存约18.47 GiB、磁盘约641.99 GiB，均达到既定门槛。

首次流程在计时名单辅助进程启动时退出，尚未读取真实数据。原因是控制脚本把脚本绝对路径作为 `python -m runpy` 的模块名参数传入，runpy 按模块名解析而失败。该错误不涉及模型、预测、数据身份或资源越限。

- 工作进程退出码1，资源监控 abort_reason=null。
- Job提交硬限8 GiB，亲和性4个逻辑处理器，GO前校验通过。
- 峰值Job提交约7.23 MiB；没有模型加载、训练、阈值标定或Test访问。
- 独立已批准配置和共同计时名单均尚未发布；原pending草案保持原哈希。
- 失败证据仅保存在本地 `.tmp/m1_final_pipeline/authorized-20261109T122327-e9168ce5a8454b58bc7111777ee67b94/`，包含FAILED.json、worker.log、outcome.json及guard_verified.json。此路径是本地归档，不是GitHub可访问链接；不覆盖或删除。

已将辅助入口修正为 `python -m docs.m1_final_timing_worker`。真实数据逻辑没有执行重试；仅用 `--guard-probe` 验证模块可启动、等待GO并正常退出。探针在8 GiB Job/4逻辑CPU保护下通过，退出码0，不导入ML模块、不读取真实数据。

探针证据仅本地保存在 `.tmp/m1_final_launcher_probes/61c549dfef8d40d59bbd2c51f3e842b9/`。修正后的辅助源码在 [m1_final_timing_worker.py](m1_final_timing_worker.py)，真实执行时仍会把源码副本保存到本地运行目录。

按 `automatic_retries=0`，首次异常后已停止。新一次真实尝试需用户确认；后续仍沿用已批准模型、阈值规则和资源限制，不重训、不运行Bootstrap。
