# P2 特征字典与字段处理规则

版本：v1.0。核查日期：2026年10月26日。范围：P1已交付的original（80列，含Day）和corrected（91列）。本文件确定字段角色及共同模型输入，不执行划分、不修改clean数据，不训练模型。

两版统一使用76个原始输入字段X，Label单独保留为原始监督标注，再按标签协议生成二分类y_binary。编码后的矩阵维度由训练折拟合的编码器确定，不一定为76维。原版Day和修正版Timestamp仅用于划分。目的端口保留，另设删除目的端口的75字段消融。删除字段分为标识/标签泄漏、重复、跨版本不可对齐三种理由；并非所有被删除字段都存在标签泄漏。

## 字段数量与覆盖范围

| 版本 | 保留作X | 保留作y | 删除 | 仅用于划分 | 合计 |
|---|---:|---:|---:|---:|---:|
| original | 76 | 1 | 2 | 1 | 80 |
| corrected | 76 | 1 | 13 | 1 | 91 |

保留类别包含76个X字段及1个Label，但Label不得混入模型矩阵。两版的特征名和顺序统一采用下表原版列顺序；修正版按映射重命名，不能按原始位置直接拼接。原版79列前版本不在本规则适用范围。

## 必须遵守的处理顺序

1. 保留原始clean文件只读，另建立稳定行ID（版本名+clean行序号），行ID只用于追溯与重叠核对，不进入X。修正版id不是全局唯一ID。
2. 将Label分离为y，将Day或Timestamp分离为元数据；先完成分层或时序划分，再从模型矩阵删除日期。Label用于分层不等于允许把Label当输入。
3. 时序映射：Monday=2017-07-03，Tuesday=2017-07-04，Wednesday=2017-07-05，Thursday=2017-07-06，Friday=2017-07-07。必须按该顺序而非英文字符串排序；Timestamp按其已有日期解析，时区未核实前不擅自转换。具体训练/验证/测试日期边界将在split_report中确定。
4. 按下列白名单分别取列并映射为同一76字段顺序，显式拒绝缺列、多列和重复映射。删除仅指排除X，原文件和审计元数据保留。
5. 完成泄漏剔除后，再检查所选X上的重复和跨折重叠；精确重复分组/去重策略在划分方案中另定，不擅自混用不同标签的相同X。
6. 缩放、编码、类别权重及任何常量/冗余筛选仅在训练折fit，再对验证和测试transform。不得用全量标签相关性决定删列，也不按全量分布筛选特征。
7. 两版目的端口均通过0–65535整数范围检查，实际存储类型见逐字段表。它是服务类别标识，不能把端口号的大小差当作天然相似度；编码方式在预处理模块中明确，词表只从训练折学习。端口0先保留为取值，不视为缺失。
8. 其他流统计先保持数值语义，不在本步骤统一将负数改0或删行；-1可能表示无适用TCP窗口。异常值、缩放、log变换规则须有字段级依据并统一应用三类方法。保留可能包含常量/相关特征，不代表承诺每列有独立信息。
9. 消融只从同一划分、同一预处理流程中排除Destination Port，得到75个原始输入字段；编码后维度另计，不重新划分，不改变其他模型预算。
10. Label是原始标注来源，最终训练目标y_binary及攻击族映射详见[标签与划分方案](split_report.md)和label_split_policy.json；原始15类与修正版27类的名称差异不能靠字段重命名自动消除。禁止为了标签对齐静默丢弃样本。

## 两版字段名称对齐

共有57个同名模型字段，加上下列19组逻辑含义对应字段，合计76个。映射依据字段定义及提取器的流统计命名，是概念映射，不保证两个版本算法、单位、方向判定或数值分布完全一致。修正版重新提取带来的变化本来就是敏感性实验的一部分；在Methods中披露这一限制。

| 原版与统一名称 | 修正版来源名称 |
|---|---|
| `Destination Port` | `Dst Port` |
| `Total Fwd Packets` | `Total Fwd Packet` |
| `Total Backward Packets` | `Total Bwd packets` |
| `Total Length of Fwd Packets` | `Total Length of Fwd Packet` |
| `Total Length of Bwd Packets` | `Total Length of Bwd Packet` |
| `Min Packet Length` | `Packet Length Min` |
| `Max Packet Length` | `Packet Length Max` |
| `Avg Fwd Segment Size` | `Fwd Segment Size Avg` |
| `Avg Bwd Segment Size` | `Bwd Segment Size Avg` |
| `Fwd Avg Bytes/Bulk` | `Fwd Bytes/Bulk Avg` |
| `Fwd Avg Packets/Bulk` | `Fwd Packet/Bulk Avg` |
| `Fwd Avg Bulk Rate` | `Fwd Bulk Rate Avg` |
| `Bwd Avg Bytes/Bulk` | `Bwd Bytes/Bulk Avg` |
| `Bwd Avg Packets/Bulk` | `Bwd Packet/Bulk Avg` |
| `Bwd Avg Bulk Rate` | `Bwd Bulk Rate Avg` |
| `Init_Win_bytes_forward` | `FWD Init Win Bytes` |
| `Init_Win_bytes_backward` | `Bwd Init Win Bytes` |
| `act_data_pkt_fwd` | `Fwd Act Data Pkts` |
| `min_seg_size_forward` | `Fwd Seg Size Min` |

CWE Flag Count与CWR Flag Count暂不映射，分别删除。Fwd Header Length.1不映射；原版全量比较与Fwd Header Length完全相同。Protocol、方向性RST、ICMP字段和Total TCP Flow Time虽然可能有用，但原版没有对应输入，不用补零或从Label猜测的方式补齐。

## original 全部字段处理表

| 序号 | 原始字段 | 类型 | 分类 | 输出角色或统一名 | 理由 |
|---:|---|---|---|---|---|
| 1 | `Destination Port` | BIGINT | 保留 | Destination Port | 保留目的服务端口语义（手册要求）；实验端口有场景捷径风险，另设删除此列的消融，不能称为无风险。 |
| 2 | `Flow Duration` | BIGINT | 保留 | Flow Duration | 流持续时间，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 3 | `Total Fwd Packets` | BIGINT | 保留 | Total Fwd Packets | 前向包数量，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 4 | `Total Backward Packets` | BIGINT | 保留 | Total Backward Packets | 后向包数量，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 5 | `Total Length of Fwd Packets` | BIGINT | 保留 | Total Length of Fwd Packets | 前向包累计长度，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 6 | `Total Length of Bwd Packets` | BIGINT | 保留 | Total Length of Bwd Packets | 后向包累计长度，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 7 | `Fwd Packet Length Max` | BIGINT | 保留 | Fwd Packet Length Max | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 8 | `Fwd Packet Length Min` | BIGINT | 保留 | Fwd Packet Length Min | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 9 | `Fwd Packet Length Mean` | DOUBLE | 保留 | Fwd Packet Length Mean | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 10 | `Fwd Packet Length Std` | DOUBLE | 保留 | Fwd Packet Length Std | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 11 | `Bwd Packet Length Max` | BIGINT | 保留 | Bwd Packet Length Max | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 12 | `Bwd Packet Length Min` | BIGINT | 保留 | Bwd Packet Length Min | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 13 | `Bwd Packet Length Mean` | DOUBLE | 保留 | Bwd Packet Length Mean | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 14 | `Bwd Packet Length Std` | DOUBLE | 保留 | Bwd Packet Length Std | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 15 | `Flow Bytes/s` | DOUBLE | 保留 | Flow Bytes/s | 每秒流字节数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 16 | `Flow Packets/s` | DOUBLE | 保留 | Flow Packets/s | 每秒流包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 17 | `Flow IAT Mean` | DOUBLE | 保留 | Flow IAT Mean | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 18 | `Flow IAT Std` | DOUBLE | 保留 | Flow IAT Std | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 19 | `Flow IAT Max` | BIGINT | 保留 | Flow IAT Max | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 20 | `Flow IAT Min` | BIGINT | 保留 | Flow IAT Min | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 21 | `Fwd IAT Total` | BIGINT | 保留 | Fwd IAT Total | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 22 | `Fwd IAT Mean` | DOUBLE | 保留 | Fwd IAT Mean | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 23 | `Fwd IAT Std` | DOUBLE | 保留 | Fwd IAT Std | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 24 | `Fwd IAT Max` | BIGINT | 保留 | Fwd IAT Max | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 25 | `Fwd IAT Min` | BIGINT | 保留 | Fwd IAT Min | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 26 | `Bwd IAT Total` | BIGINT | 保留 | Bwd IAT Total | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 27 | `Bwd IAT Mean` | DOUBLE | 保留 | Bwd IAT Mean | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 28 | `Bwd IAT Std` | DOUBLE | 保留 | Bwd IAT Std | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 29 | `Bwd IAT Max` | BIGINT | 保留 | Bwd IAT Max | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 30 | `Bwd IAT Min` | BIGINT | 保留 | Bwd IAT Min | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 31 | `Fwd PSH Flags` | BIGINT | 保留 | Fwd PSH Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 32 | `Bwd PSH Flags` | BIGINT | 保留 | Bwd PSH Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 33 | `Fwd URG Flags` | BIGINT | 保留 | Fwd URG Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 34 | `Bwd URG Flags` | BIGINT | 保留 | Bwd URG Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 35 | `Fwd Header Length` | BIGINT | 保留 | Fwd Header Length | 方向性报头长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 36 | `Bwd Header Length` | BIGINT | 保留 | Bwd Header Length | 方向性报头长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 37 | `Fwd Packets/s` | DOUBLE | 保留 | Fwd Packets/s | 方向性每秒包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 38 | `Bwd Packets/s` | DOUBLE | 保留 | Bwd Packets/s | 方向性每秒包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 39 | `Min Packet Length` | BIGINT | 保留 | Min Packet Length | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 40 | `Max Packet Length` | BIGINT | 保留 | Max Packet Length | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 41 | `Packet Length Mean` | DOUBLE | 保留 | Packet Length Mean | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 42 | `Packet Length Std` | DOUBLE | 保留 | Packet Length Std | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 43 | `Packet Length Variance` | DOUBLE | 保留 | Packet Length Variance | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 44 | `FIN Flag Count` | BIGINT | 保留 | FIN Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 45 | `SYN Flag Count` | BIGINT | 保留 | SYN Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 46 | `RST Flag Count` | BIGINT | 保留 | RST Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 47 | `PSH Flag Count` | BIGINT | 保留 | PSH Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 48 | `ACK Flag Count` | BIGINT | 保留 | ACK Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 49 | `URG Flag Count` | BIGINT | 保留 | URG Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 50 | `CWE Flag Count` | BIGINT | 删除 | 不进入X | 原版名称为CWE，修正版为CWR；当前未核实原版提取器中该列确切含义，两版对称排除，避免把拼写相似当作语义证明。 |
| 51 | `ECE Flag Count` | BIGINT | 保留 | ECE Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 52 | `Down/Up Ratio` | BIGINT | 保留 | Down/Up Ratio | 上下行包数比例，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 53 | `Average Packet Size` | DOUBLE | 保留 | Average Packet Size | 平均包大小，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 54 | `Avg Fwd Segment Size` | DOUBLE | 保留 | Avg Fwd Segment Size | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 55 | `Avg Bwd Segment Size` | DOUBLE | 保留 | Avg Bwd Segment Size | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 56 | `Fwd Header Length.1` | BIGINT | 删除 | 不进入X | 与Fwd Header Length重复；全量逐行比较后删除冗余副本。 |
| 57 | `Fwd Avg Bytes/Bulk` | BIGINT | 保留 | Fwd Avg Bytes/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 58 | `Fwd Avg Packets/Bulk` | BIGINT | 保留 | Fwd Avg Packets/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 59 | `Fwd Avg Bulk Rate` | BIGINT | 保留 | Fwd Avg Bulk Rate | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 60 | `Bwd Avg Bytes/Bulk` | BIGINT | 保留 | Bwd Avg Bytes/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 61 | `Bwd Avg Packets/Bulk` | BIGINT | 保留 | Bwd Avg Packets/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 62 | `Bwd Avg Bulk Rate` | BIGINT | 保留 | Bwd Avg Bulk Rate | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 63 | `Subflow Fwd Packets` | BIGINT | 保留 | Subflow Fwd Packets | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 64 | `Subflow Fwd Bytes` | BIGINT | 保留 | Subflow Fwd Bytes | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 65 | `Subflow Bwd Packets` | BIGINT | 保留 | Subflow Bwd Packets | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 66 | `Subflow Bwd Bytes` | BIGINT | 保留 | Subflow Bwd Bytes | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 67 | `Init_Win_bytes_forward` | BIGINT | 保留 | Init_Win_bytes_forward | 前向初始TCP窗口，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 68 | `Init_Win_bytes_backward` | BIGINT | 保留 | Init_Win_bytes_backward | 后向初始TCP窗口，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 69 | `act_data_pkt_fwd` | BIGINT | 保留 | act_data_pkt_fwd | 前向携带数据的包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 70 | `min_seg_size_forward` | BIGINT | 保留 | min_seg_size_forward | 前向最小段大小，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 71 | `Active Mean` | DOUBLE | 保留 | Active Mean | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 72 | `Active Std` | DOUBLE | 保留 | Active Std | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 73 | `Active Max` | BIGINT | 保留 | Active Max | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 74 | `Active Min` | BIGINT | 保留 | Active Min | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 75 | `Idle Mean` | DOUBLE | 保留 | Idle Mean | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 76 | `Idle Std` | DOUBLE | 保留 | Idle Std | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 77 | `Idle Max` | BIGINT | 保留 | Idle Max | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 78 | `Idle Min` | BIGINT | 保留 | Idle Min | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 79 | `Label` | VARCHAR | 保留 | y，禁止进入X | 原始监督标注，单独保留并按标签协议生成y_binary及攻击族；绝不进入X。 |
| 80 | `Day` | VARCHAR | 仅用于划分 | 划分元数据 | 抓包时间/来源日用于按天分组与审计，生成划分后不进入X；禁止作为数值或类别特征编码。 |

## corrected 全部字段处理表

| 序号 | 原始字段 | 类型 | 分类 | 输出角色或统一名 | 理由 |
|---:|---|---|---|---|---|
| 1 | `id` | DOUBLE | 删除 | 不进入X | 文件内记录编号，可编码抓包顺序；不作为模型输入，也不直接作为全局唯一行ID。 |
| 2 | `Flow ID` | VARCHAR | 删除 | 不进入X | 连接标识可能拼接IP、端口和协议，可记忆主机或会话身份。 |
| 3 | `Src IP` | VARCHAR | 删除 | 不进入X | 源主机身份会编码实验网络和攻击主机，影响泛化。 |
| 4 | `Src Port` | DOUBLE | 删除 | 不进入X | 原版无对应字段，主比较两版必须同输入；源端口本身不等于已证明的标签泄漏。 |
| 5 | `Dst IP` | VARCHAR | 删除 | 不进入X | 目的主机身份会编码实验网络和受害主机，影响泛化。 |
| 6 | `Dst Port` | DOUBLE | 保留 | Destination Port | 保留目的服务端口语义（手册要求）；实验端口有场景捷径风险，另设删除此列的消融，不能称为无风险。 |
| 7 | `Protocol` | DOUBLE | 删除 | 不进入X | 有真实协议语义，但原版没有交付此字段，为统一两版特征集合排除；不伪造或按标签补值。 |
| 8 | `Timestamp` | VARCHAR | 仅用于划分 | 划分元数据 | 抓包时间/来源日用于按天分组与审计，生成划分后不进入X；禁止作为数值或类别特征编码。 |
| 9 | `Flow Duration` | DOUBLE | 保留 | Flow Duration | 流持续时间，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 10 | `Total Fwd Packet` | DOUBLE | 保留 | Total Fwd Packets | 前向包数量，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 11 | `Total Bwd packets` | DOUBLE | 保留 | Total Backward Packets | 后向包数量，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 12 | `Total Length of Fwd Packet` | DOUBLE | 保留 | Total Length of Fwd Packets | 前向包累计长度，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 13 | `Total Length of Bwd Packet` | DOUBLE | 保留 | Total Length of Bwd Packets | 后向包累计长度，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 14 | `Fwd Packet Length Max` | DOUBLE | 保留 | Fwd Packet Length Max | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 15 | `Fwd Packet Length Min` | DOUBLE | 保留 | Fwd Packet Length Min | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 16 | `Fwd Packet Length Mean` | DOUBLE | 保留 | Fwd Packet Length Mean | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 17 | `Fwd Packet Length Std` | DOUBLE | 保留 | Fwd Packet Length Std | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 18 | `Bwd Packet Length Max` | DOUBLE | 保留 | Bwd Packet Length Max | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 19 | `Bwd Packet Length Min` | DOUBLE | 保留 | Bwd Packet Length Min | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 20 | `Bwd Packet Length Mean` | DOUBLE | 保留 | Bwd Packet Length Mean | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 21 | `Bwd Packet Length Std` | DOUBLE | 保留 | Bwd Packet Length Std | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 22 | `Flow Bytes/s` | DOUBLE | 保留 | Flow Bytes/s | 每秒流字节数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 23 | `Flow Packets/s` | DOUBLE | 保留 | Flow Packets/s | 每秒流包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 24 | `Flow IAT Mean` | DOUBLE | 保留 | Flow IAT Mean | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 25 | `Flow IAT Std` | DOUBLE | 保留 | Flow IAT Std | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 26 | `Flow IAT Max` | DOUBLE | 保留 | Flow IAT Max | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 27 | `Flow IAT Min` | DOUBLE | 保留 | Flow IAT Min | 包到达间隔的全流统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 28 | `Fwd IAT Total` | DOUBLE | 保留 | Fwd IAT Total | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 29 | `Fwd IAT Mean` | DOUBLE | 保留 | Fwd IAT Mean | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 30 | `Fwd IAT Std` | DOUBLE | 保留 | Fwd IAT Std | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 31 | `Fwd IAT Max` | DOUBLE | 保留 | Fwd IAT Max | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 32 | `Fwd IAT Min` | DOUBLE | 保留 | Fwd IAT Min | 包到达间隔的前向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 33 | `Bwd IAT Total` | DOUBLE | 保留 | Bwd IAT Total | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 34 | `Bwd IAT Mean` | DOUBLE | 保留 | Bwd IAT Mean | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 35 | `Bwd IAT Std` | DOUBLE | 保留 | Bwd IAT Std | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 36 | `Bwd IAT Max` | DOUBLE | 保留 | Bwd IAT Max | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 37 | `Bwd IAT Min` | DOUBLE | 保留 | Bwd IAT Min | 包到达间隔的后向统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 38 | `Fwd PSH Flags` | DOUBLE | 保留 | Fwd PSH Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 39 | `Bwd PSH Flags` | DOUBLE | 保留 | Bwd PSH Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 40 | `Fwd URG Flags` | DOUBLE | 保留 | Fwd URG Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 41 | `Bwd URG Flags` | DOUBLE | 保留 | Bwd URG Flags | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 42 | `Fwd RST Flags` | DOUBLE | 删除 | 不进入X | 原版没有方向性RST统计；不能用双向RST Flag Count替代。 |
| 43 | `Bwd RST Flags` | DOUBLE | 删除 | 不进入X | 原版没有方向性RST统计；不能用双向RST Flag Count替代。 |
| 44 | `Fwd Header Length` | DOUBLE | 保留 | Fwd Header Length | 方向性报头长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 45 | `Bwd Header Length` | DOUBLE | 保留 | Bwd Header Length | 方向性报头长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 46 | `Fwd Packets/s` | DOUBLE | 保留 | Fwd Packets/s | 方向性每秒包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 47 | `Bwd Packets/s` | DOUBLE | 保留 | Bwd Packets/s | 方向性每秒包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 48 | `Packet Length Min` | DOUBLE | 保留 | Min Packet Length | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 49 | `Packet Length Max` | DOUBLE | 保留 | Max Packet Length | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 50 | `Packet Length Mean` | DOUBLE | 保留 | Packet Length Mean | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 51 | `Packet Length Std` | DOUBLE | 保留 | Packet Length Std | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 52 | `Packet Length Variance` | DOUBLE | 保留 | Packet Length Variance | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 53 | `FIN Flag Count` | DOUBLE | 保留 | FIN Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 54 | `SYN Flag Count` | DOUBLE | 保留 | SYN Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 55 | `RST Flag Count` | DOUBLE | 保留 | RST Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 56 | `PSH Flag Count` | DOUBLE | 保留 | PSH Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 57 | `ACK Flag Count` | DOUBLE | 保留 | ACK Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 58 | `URG Flag Count` | DOUBLE | 保留 | URG Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 59 | `CWR Flag Count` | DOUBLE | 删除 | 不进入X | 虽是有效TCP标志统计，但原版CWE列映射尚未证实；与原版对应候选一起从共同输入排除。 |
| 60 | `ECE Flag Count` | DOUBLE | 保留 | ECE Flag Count | TCP控制标志统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 61 | `Down/Up Ratio` | DOUBLE | 保留 | Down/Up Ratio | 上下行包数比例，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 62 | `Average Packet Size` | DOUBLE | 保留 | Average Packet Size | 平均包大小，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 63 | `Fwd Segment Size Avg` | DOUBLE | 保留 | Avg Fwd Segment Size | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 64 | `Bwd Segment Size Avg` | DOUBLE | 保留 | Avg Bwd Segment Size | 包或段长度统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 65 | `Fwd Bytes/Bulk Avg` | DOUBLE | 保留 | Fwd Avg Bytes/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 66 | `Fwd Packet/Bulk Avg` | DOUBLE | 保留 | Fwd Avg Packets/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 67 | `Fwd Bulk Rate Avg` | DOUBLE | 保留 | Fwd Avg Bulk Rate | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 68 | `Bwd Bytes/Bulk Avg` | DOUBLE | 保留 | Bwd Avg Bytes/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 69 | `Bwd Packet/Bulk Avg` | DOUBLE | 保留 | Bwd Avg Packets/Bulk | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 70 | `Bwd Bulk Rate Avg` | DOUBLE | 保留 | Bwd Avg Bulk Rate | 方向性批量传输统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 71 | `Subflow Fwd Packets` | DOUBLE | 保留 | Subflow Fwd Packets | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 72 | `Subflow Fwd Bytes` | DOUBLE | 保留 | Subflow Fwd Bytes | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 73 | `Subflow Bwd Packets` | DOUBLE | 保留 | Subflow Bwd Packets | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 74 | `Subflow Bwd Bytes` | DOUBLE | 保留 | Subflow Bwd Bytes | 子流包数或字节数统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 75 | `FWD Init Win Bytes` | DOUBLE | 保留 | Init_Win_bytes_forward | 前向初始TCP窗口，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 76 | `Bwd Init Win Bytes` | DOUBLE | 保留 | Init_Win_bytes_backward | 后向初始TCP窗口，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 77 | `Fwd Act Data Pkts` | DOUBLE | 保留 | act_data_pkt_fwd | 前向携带数据的包数，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 78 | `Fwd Seg Size Min` | DOUBLE | 保留 | min_seg_size_forward | 前向最小段大小，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 79 | `Active Mean` | DOUBLE | 保留 | Active Mean | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 80 | `Active Std` | DOUBLE | 保留 | Active Std | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 81 | `Active Max` | DOUBLE | 保留 | Active Max | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 82 | `Active Min` | DOUBLE | 保留 | Active Min | 流活跃时段统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 83 | `Idle Mean` | DOUBLE | 保留 | Idle Mean | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 84 | `Idle Std` | DOUBLE | 保留 | Idle Std | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 85 | `Idle Max` | DOUBLE | 保留 | Idle Max | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 86 | `Idle Min` | DOUBLE | 保留 | Idle Min | 流空闲间隔统计，两版可按相同概念对齐；作为离线流量行为特征保留。 |
| 87 | `ICMP Code` | DOUBLE | 删除 | 不进入X | 有协议语义但原版无对应字段，主比较统一排除。 |
| 88 | `ICMP Type` | DOUBLE | 删除 | 不进入X | 有协议语义但原版无对应字段，主比较统一排除。 |
| 89 | `Total TCP Flow Time` | DOUBLE | 删除 | 不进入X | 原版无可确认的对应字段；不直接映射为Flow Duration，避免定义和单位不一致。 |
| 90 | `Label` | VARCHAR | 保留 | y，禁止进入X | 原始监督标注，单独保留并按标签协议生成y_binary及攻击族；绝不进入X。 |
| 91 | `Attempted Category` | DOUBLE | 删除 | 不进入X | 标注衍生字段；全量核查其非负值恰对应attempted标签，不能作为预测该标签的输入。 |

## 本地核查证据

- 原版Fwd Header Length两列的逐行差异数：0。删除重复列不依赖标签。
- 修正版Attempted Category非负值行数：11,979；其非负性与Label是否以“ - attempted”结尾的不一致行数：0。这是直接的标注信息，不纳入X。
- 两版目的端口空值、非有限数、非整数或超出0–65535范围的行数均为0。
- 此处检查用于解释预先确定的字段角色，不依据测试集模型效果选择特征。全表定义覆盖两版全部171个实际列位置，无遗漏或一列多类。

## 依据与可追溯资料

- 组内八人分工手册P2任务卡：删除身份与时间信息；保留目的端口并做消融；训练折内拟合预处理。
- [P1字段差异表](schema_diff.md)、[P1清洗日志](../logs/clean_log.md)，以及当前两份Parquet的实际schema与全量必要核查。
- [CICFlowMeter官方FlowFeature字段定义](https://github.com/ahlashkari/CICFlowMeter/blob/master/src/main/java/cic/cs/unb/ca/jnetpcap/FlowFeature.java)：包数、字节数、IAT、窗口、Bulk、Active/Idle等字段族与标识字段的定义。该文件不是本地原始CSV所用提取器的精确版本证明。
- [修正版发布者说明](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html)：修正版背景及attempted标签；具体Attempted Category泄漏判断同时由本地全量交叉统计支持。
- [机器可读规则](feature_policy.json)：白名单、列映射、删除理由、消融列和本地校验信息。后续预处理应直接读取同一规则，避免各模型自行选列。

当前输入文件SHA-256：

- original：`f12c82092c414603405de0a7173d7c07f125e0784ff7486103c3c56112933861`
- corrected：`a9e1df46df7234be6fe90fed7101be3a083feab05ca466f2ec35f97e3f43a329`
