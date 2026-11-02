# GBDT-only 原归档补封存验收

日期：2026-11-02（北京时间）。结论：MANIFEST 成功生成，302/302 文件 SHA256 验证通过；原 selection.json 保持指定哈希。P4 GBDT-only 搜索、Validation 选优与结果归档已正式封存完成，不代表最终 Test 评估完成。

本地归档（不随 Git 上传，以下路径仅供本机定位）：

`results/m1_gbdt_search/20261101T115824-90de9307e77448aba412cc9ec08a5cbd/`

## 故障和修复范围

原 `methods/m1/search_gbdt.py` 的 run_selection 在持有 RUN.lock 字节锁时，将其交给 inventory/sha 读取，Windows 因锁冲突抛出 PermissionError。该源码保留原样，原 binding.json 和 sources/ 也未改写。

新增独立、仅使用标准库的补封存入口 `methods/m1/seal_gbdt.py`。它不加载模型或预测数组，不计算 AP、不重新选优、不训练、不读取真实 Test，不运行 RF、Bootstrap 或阈值标定。MANIFEST 的 repair_provenance 独立记录补封存工具 SHA256、封存前快照 SHA256及原代码身份保留声明。

## 完整性和并发保护

- 查询活动搜索、选优和封存进程；排除本工具自身及其启动祖先，没有发现其他活动正式任务。
- 非阻塞取得全局资源锁和原目录 RUN.lock；活动锁持有者会导致立即拒绝，不删除锁、不绕过互斥。
- 对302个原归档文件以及原源码、输入身份文件、封存前基线、补封存工具取得 Windows FILE_SHARE_READ 句柄。现有写入者导致拒绝；封存期间阻止原文件被写入或删除。
- 原始文件集与封存前快照严格匹配；发布前再次核对活动进程、文件集和全部哈希。出现任何变化即停止。
- 模型、预测和配置的120个文件对照40个完成记录验证；8个缓存文件、51个源码快照、输入身份及八组与SELECT监控完成记录通过。八次人工确认均绑定原运行身份。
- 清单先在归档外的临时文件完整写入并 fsync，再通过排他硬链接发布为 MANIFEST.json；若目标已存在则拒绝，绝不覆盖。
- 发布后保持文件共享保护，对全部条目重新计算 SHA256。随后退出补封存进程，再由独立 verify 命令以固定 MANIFEST SHA256 完整复核一次，均为302/302通过。

唯一排除的运行时文件为根目录 `RUN.lock`：它只用于OS互斥，不含实验内容，封存期间仍持有并保留在原位置；原因写入 MANIFEST.excluded_runtime_files。不存在宽泛的 *.lock 排除。MANIFEST.json不递归包含自身，其SHA256在本报告中单独固定。

## SHA256

selection.json（修复前后严格一致）：

`859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9`

MANIFEST.json：

`00128ffea5088c939a2375502372be4ba6b95cc13bcdee4f68e6c1638d2b1d6d`

封存前302文件快照（仅本地 `.tmp/gbdt_preseal_original_files.json`，不随Git上传）：

`96a4cd5045d7b365adc9dea4d6b16dd65d3ec93171416635618539fc4f63875c`

302是MANIFEST.files条目数，不含根RUN.lock和MANIFEST自身。原归档只新增MANIFEST.json，没有覆盖原模型、预测、selection.json、配置或日志。

## 合成文件测试

`tests/test_m1_seal_gbdt.py`：19项全部通过，33.097秒。测试仅使用独立临时目录的合成字节文件，覆盖锁持有时清单构建、活动进程与锁拒绝、现有写入者拒绝、并发线程写入/删除拒绝、模型/缓存/源码/selection/基线变化拒绝、缺失文件拒绝、文件集并发新增拒绝、清单不可覆盖及封存后篡改检测。

本轮没有重新运行实验或重新计算AP。此前已通过的40份Validation预测独立AP复算及G05选优结果保持原样。G02种子42/44的历史待机计时影响标注保留，不改原始耗时。

## 独立只读复核命令

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m methods.m1.seal_gbdt verify --run results/m1_gbdt_search/20261101T115824-90de9307e77448aba412cc9ec08a5cbd --manifest-sha256 00128ffea5088c939a2375502372be4ba6b95cc13bcdee4f68e6c1638d2b1d6d
```

验收结果：无缺失、无额外未记录实验文件、无SHA256不匹配。原40模型和预测均保留；没有修改P2/P3公共模块、统一协议或后续评估草案，也未提交或推送本轮修复。
