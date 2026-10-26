# CIC-IDS2017 数据清洗日志（P1）

生成时间：2026-10-24 11:58:39
清洗规则：列名去空格 → inf/-inf 转 NaN 后删除 → 全行去重 → 标签小写去空格

## original 清洗日志（2026-10-24 11:58:39）
- 文件 Monday-WorkingHours.pcap_ISCX.csv：读入 529,918 行
- 文件 Tuesday-WorkingHours.pcap_ISCX.csv：读入 445,909 行
- 文件 Wednesday-workingHours.pcap_ISCX.csv：读入 692,703 行
- 文件 Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv：读入 170,366 行
- 文件 Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv：读入 288,602 行
- 文件 Friday-WorkingHours-Morning.pcap_ISCX.csv：读入 191,033 行
- 文件 Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv：读入 286,467 行
- 文件 Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv：读入 225,745 行
- 合并后总行数：2,830,743（11:58:50，耗时 10s）
### 第 1 步 列名去空格：进 2,830,743 行，删 0 行，剩 2,830,743 行
### 第 2 步 inf/-inf→NaN 后删除：进 2,830,743 行，含 inf 行 2,867，删 2,867 行（含 NaN 行），剩 2,827,876 行
### 第 3 步 全行去重：进 2,827,876 行，删 307,078 行，剩 2,520,798 行
### 第 4 步 标签小写+去空格：进 2,520,798 行，删 0 行，剩 2,520,798 行（列 Label→Label；U+FFFD 损坏字符修复 2,143 行）
- 已保存：C:\Users\HUAWEI\Desktop\tyl\大学\大三上\网络安全\data\clean\original.parquet（行数 2,520,798，耗时 62s）

## corrected 清洗日志（2026-10-24 11:59:41）
- 文件 monday.csv：读入 371,624 行
- 文件 tuesday.csv：读入 322,078 行
- 文件 wednesday.csv：读入 496,641 行
- 文件 thursday.csv：读入 362,076 行
- 文件 friday.csv：读入 547,557 行
- 合并后总行数：2,099,976（11:59:50，耗时 9s）
### 第 1 步 列名去空格：进 2,099,976 行，删 0 行，剩 2,099,976 行
### 第 2 步 inf/-inf→NaN 后删除：进 2,099,976 行，含 inf 行 5，删 5 行（含 NaN 行），剩 2,099,971 行
### 第 3 步 全行去重：进 2,099,971 行，删 0 行，剩 2,099,971 行
### 第 4 步 标签小写+去空格：进 2,099,971 行，删 0 行，剩 2,099,971 行（列 Label→Label；U+FFFD 损坏字符修复 0 行）
- 已保存：C:\Users\HUAWEI\Desktop\tyl\大学\大三上\网络安全\data\clean\corrected.parquet（行数 2,099,971，耗时 47s）
- 类别分布表已保存：C:\Users\HUAWEI\Desktop\tyl\大学\大三上\网络安全\tables\class_distribution.csv
- 字段差异表已保存：C:\Users\HUAWEI\Desktop\tyl\大学\大三上\网络安全\docs\schema_diff.md

---

## 补充（2026-10-26，按 P2 要求）：original 新增 Day 列

原版 CIC-IDS2017 原始数据无 Timestamp 列，日期信息按文件名拆分（周一至周五 8 个 CSV）。P1 合并时将来源文件的日期信息丢失，现重新合并清洗流程，在合并阶段从文件名解析 `Day` 列后追加：

- Day 取值：`Monday` / `Tuesday` / `Wednesday` / `Thursday` / `Friday`
- 合并顺序：Monday → Tuesday → Wednesday → Thursday-AM(WebAttacks) → Thursday-PM(Infiltration) → Friday-AM(Morning) → Friday-PM(PortScan) → Friday-PM(DDos)
- 去重逻辑：按原 79 列去重（`Day` 不参与去重判断），跨天重复行保留第一次出现的日期
- 中间步骤数字完全复现：合并 2,830,743 → 删 inf 2,867 → 删重复 307,078 → 最终 2,520,798 行（与原日志一致）
- Day 分布：Wednesday 602,278 | Friday 595,860 | Monday 502,650 | Tuesday 421,576 | Thursday 398,434
- 验证：无 NaN、无 inf、抽样 1000 行无重复；Label 分布与原数据完全一致（benign 2,095,057 等）
- corrected.parquet 无需改动（修正版自带 Timestamp 列）

## P2 下游抽样验收记录（2026-10-26）

- 验收时间：2026-10-26 16:28:17（北京时间，UTC+08:00）。
- 验收人签署：P2
- 验收范围：两版清洗后数据分别随机抽取 1000 行，检查全行重复及数值列正负 inf，并核对实际总行数与本日志第 4 步最终行数。
- 抽样方法：不放回 reservoir 随机抽样，固定 seed=42；每版独立抽样。重复检查按样本全部字段精确去重，inf 检查覆盖样本全部数值列。

| 数据文件 | 日志最终行数 | 实际总行数 | 抽样行数 | 样本重复行数 | 样本含 inf/-inf 行数 | 验收结果 |
|---|---:|---:|---:|---:|---:|---|
| data/clean/original.parquet | 2,520,798 | 2,520,798 | 1000 | 0 | 0 | 通过 |
| data/clean/corrected.parquet | 2,099,971 | 2,099,971 | 1000 | 0 | 0 | 通过 |

验收结论：两版数据各随机抽查 1000 行，均未发现全行重复或正负 inf；实际总行数与清洗日志一致。本次“随机抽样与行数核对”下游验收通过。

本记录仅确认上述验收动作，不替代对日期来源、数据来源证明及其他交付事项的核查。抽样通过不等同于对全量数据无重复、无 inf 的单独证明。
