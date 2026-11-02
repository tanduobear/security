# P4 对 P3 的本机验收记录

验收日期：2026-11-01。验收分支：`p4-tree-models`。
上游提交：`85bbcfff96265c6f3ac4e5a03f79169c94cafa53`。

本机接口验收通过：47 项单元测试全部通过，合成示例成功调用五个核心指标接口，生成的结果通过 JSON schema 与数值一致性校验。此结论仅覆盖本机环境与合成数据接口，不代表真实 Parquet 数据已验收，也不是 M1 实验结果。

## 环境与检查

Python 3.13.0，使用仓库内 `.venv`。依赖按 `eval/requirements-eval.txt` 及其引用文件安装；完整版本见 `environment.txt`。

| 检查 | 结果 | 证据 |
|---|---|---|
| pip check | 无依赖冲突 | dependency_check.txt |
| 单元测试 | 47 项通过 | tests.txt |
| 五个核心指标接口及辅助分类指标 | 合成示例运行成功 | demo.txt、result_demo.json |
| schema 与数值一致性校验 | 通过 | validation.txt |

首次沙箱测试因临时目录权限失败，日志仅在本地保留于 `docs/p4_acceptance/sandbox_test_failure.txt`（相对于仓库根目录，本次不提交，非在线链接）；在沙箱外复跑后 47 项全部通过。再次确认时，`tests.txt` 已更新为成功运行的完整日志：47 项测试，1.869 秒，结尾为 OK，无 FAIL 或 ERROR。机器可读首次验收状态见 `checks.json`，再次确认结果见 `recheck.json`。

## P3 PR 范围与冲突复核

2026-11-01 通过 GitHub API 检查 [PR #1](https://github.com/tanduobear/security/pull/1)：状态 open，merged=false，mergeable=true，mergeable_state=clean。当前无合并冲突，尚未合并。

刷新远程记录后，main 为 `9df4ccec95c66c4237494b855caa7f1770fd80da`，P3 为 `85bbcfff96265c6f3ac4e5a03f79169c94cafa53`。main 是 P3 的祖先。

对两棵 Git 文件树逐项比较：main 的所有既有文件均保留相同路径、对象内容和文件模式，没有修改或删除。PR 仅新增 17 个文件，其中 eval/ 下 15 个，tests/ 下 2 个。P1、P2 已有代码、数据、文档、配置及 LFS 指针均未改变。文件清单见 `pr_files.txt`。该冲突结论适用于本次检查的提交，后续 main 或 P3 更新后应重新确认。

示例输出保存于本目录，上游 `eval/examples/` 文件已恢复运行前内容。接口验收阶段未下载真实数据；后续真实数据准备结果见下节。上游代码与评估实现未修改，未训练模型，未提交或推送 GitHub。

## P4 主实验数据准备

2026-11-01，仅下载原版分层划分主版本的 train/val/test 与三个 model_*_indices 文件，共 460.95 MB。官方 Git LFS 客户端传输停滞后，使用同一官方 LFS batch API 下载，逐文件核对内容寻址 SHA256，并写入标准 LFS 缓存后通过 git lfs checkout 检出。缓存与工作目录各保存一份，合计约 922 MB。未下载其他协议、消融、修正版及清洗数据。

运行 `python docs/p4_acceptance/verify_primary_data.py`，六个文件与 P2 清单哈希全部一致，ModelDataset 的清单与预处理器校验通过，完整遍历三份数据后结果如下。

| 数据 | 行数 | 良性 | 攻击 | 特征数 |
|---|---:|---:|---:|---:|
| train | 1,764,265 | 1,466,743 | 297,522 | 140 |
| val | 367,036 | 302,963 | 64,073 | 140 |
| test | 346,544 | 282,506 | 64,038 | 140 |

特征矩阵维度正确且无非有限数，标签为 0/1，元数据行数匹配。各数据文件行 ID 唯一，与对应有效索引完全一致；训练、验证、测试行 ID 两两无重叠。没有重新划分数据或重新拟合预处理器。

结果见 `primary_data_check.json`，复核脚本为 `verify_primary_data.py`。这属于文件完整性与加载检查，不替代模型实验或全部上游数据生成流程复现。下一步是建立 M1 的单配置、单种子开发流程。

## 交付登记表 S3 可填写内容

验收人：P4。

验收结果：本机复跑 47 项单元测试全部通过；合成数据跑通五个核心指标函数，生成结果通过 schema 与数值一致性校验。验收版本 85bbcfff96265c6f3ac4e5a03f79169c94cafa53，Python 3.13.0。证据路径 docs/p4_acceptance/。本次仅为接口验收，不包含真实数据验收。

验收时间：2026-11-01。

以上登记文本已准备；外部 Word 交付登记表尚未改写。

## 复跑命令

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe -m eval.demo
.\.venv\Scripts\python.exe -m eval.validate eval/examples/result_demo.json
```

最后两条命令会重新生成上游示例文件，耗时和时间戳可能变化。其输出仅用于接口检查。
