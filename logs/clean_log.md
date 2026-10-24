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