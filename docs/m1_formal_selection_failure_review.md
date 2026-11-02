# Formal Selection 复核：选优通过，归档封存失败

2026-11-02（北京时间）。启动前确认搜索命令退出码0，G08已确认，40任务、120个模型/预测/任务配置文件及缓存和源码快照SHA256通过。

独立select工作进程在8 GiB Job硬限、4逻辑处理器亲和性下正常完成；40个任务的完整Validation预测AP均经P3 compute_curves独立复算，与记录严格相等（最大绝对差0）。按既定五种子平均AP及1e-6平分规则选出G05。未重新训练、读取Test、运行RF/Bootstrap或阈值标定。

| 配置 | 验证AP均值 | AP样本标准差 | 原始fit秒：均值 | 原始fit秒：样本标准差 |
|---|---:|---:|---:|---:|
| G01 | 0.999433365 | 0.000099958 | 278.06 | 4.67 |
| G02 | 0.999698466 | 0.000034309 | 725.08 | 348.62 |
| G03 | 0.999682692 | 0.000026511 | 231.62 | 13.12 |
| G04 | 0.999646839 | 0.000021198 | 257.19 | 27.77 |
| G05 | 0.999791727 | 0.000015763 | 500.10 | 15.02 |
| G06 | 0.999451116 | 0.000064948 | 224.94 | 10.71 |
| G07 | 0.999461421 | 0.000066649 | 242.18 | 19.97 |
| G08 | 0.999436374 | 0.000090745 | 235.40 | 23.85 |

G02种子42/44含已确认的Modern Standby中断，原始训练墙钟耗时保留，不可直接作为纯计算耗时比较；未擅自扣除或重跑。

封存故障：父进程在持有RUN.lock字节锁时调用inventory(run)，将该锁文件也当作结果文件读取并哈希，Windows拒绝读取被锁字节，抛出PermissionError。已在独立临时目录复现同样错误，解锁后可正常哈希；没有修改实际归档来复现。位置：methods/m1/search_gbdt.py 的run_selection、inventory、sha及run_lock。

selection.json已排他生成，SHA256为 `859a883fe6ecc12246fb2f08c5ab3e2707530a6906f52c1c2863d9b12ba0d2c9`。MANIFEST.json尚未生成；前台select命令退出码1。不能将本次状态标为完整正式封存成功。按照用户要求，发现问题后停止，未重试、未覆盖selection.json、未修改正式源码或模型预测。

本地归档（不随Git上传，非在线链接）：`results/m1_gbdt_search/20261101T115824-90de9307e77448aba412cc9ec08a5cbd`。故障证据：本地`.tmp/gbdt_selection_foreground.txt`和归档内monitor/SELECT-*/的outcome.json、worker_result.json、guard_verified.json。逐任务复算通过记录汇总在本地`.tmp/formal_selection_failure_review.json`。

建议后续经确认后修复封存器：将活动锁明确排除出结果清单，保留互斥锁保护，并设计已有selection.json状态下的安全补封存流程和Windows回归测试。不得删除锁绕过互斥或覆盖已生成selection.json，也不得悄然改写原运行的代码身份。此建议本轮未执行。
