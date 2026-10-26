# 原版与修正版字段差异表（P1）

生成时间：2026-10-24 12:00:28（2026-10-26 更新：原版新增 Day 列）

- 原版列数：80（含新增 Day 列），修正版列数：91
- 共有列数：58
- 仅原版有：22 列（含新增 Day 列）；仅修正版有：33 列

| 列名                          | 原版   | 修正版   | 说明     |
|:----------------------------|:-----|:------|:-------|
| Day                          | 有    | 无     | 2026-10-26 新增：Monday/Tuesday/Wednesday/Thursday/Friday，从原始文件名解析，用于时序划分 |
| Avg Bwd Segment Size        | 有    | 无     | 仅原版存在  |
| Avg Fwd Segment Size        | 有    | 无     | 仅原版存在  |
| Bwd Avg Bulk Rate           | 有    | 无     | 仅原版存在  |
| Bwd Avg Bytes/Bulk          | 有    | 无     | 仅原版存在  |
| Bwd Avg Packets/Bulk        | 有    | 无     | 仅原版存在  |
| CWE Flag Count              | 有    | 无     | 仅原版存在  |
| Destination Port            | 有    | 无     | 仅原版存在  |
| Fwd Avg Bulk Rate           | 有    | 无     | 仅原版存在  |
| Fwd Avg Bytes/Bulk          | 有    | 无     | 仅原版存在  |
| Fwd Avg Packets/Bulk        | 有    | 无     | 仅原版存在  |
| Fwd Header Length.1         | 有    | 无     | 仅原版存在  |
| Init_Win_bytes_backward     | 有    | 无     | 仅原版存在  |
| Init_Win_bytes_forward      | 有    | 无     | 仅原版存在  |
| Max Packet Length           | 有    | 无     | 仅原版存在  |
| Min Packet Length           | 有    | 无     | 仅原版存在  |
| Total Backward Packets      | 有    | 无     | 仅原版存在  |
| Total Fwd Packets           | 有    | 无     | 仅原版存在  |
| Total Length of Bwd Packets | 有    | 无     | 仅原版存在  |
| Total Length of Fwd Packets | 有    | 无     | 仅原版存在  |
| act_data_pkt_fwd            | 有    | 无     | 仅原版存在  |
| min_seg_size_forward        | 有    | 无     | 仅原版存在  |
| Attempted Category          | 无    | 有     | 仅修正版存在 |
| Bwd Bulk Rate Avg           | 无    | 有     | 仅修正版存在 |
| Bwd Bytes/Bulk Avg          | 无    | 有     | 仅修正版存在 |
| Bwd Init Win Bytes          | 无    | 有     | 仅修正版存在 |
| Bwd Packet/Bulk Avg         | 无    | 有     | 仅修正版存在 |
| Bwd RST Flags               | 无    | 有     | 仅修正版存在 |
| Bwd Segment Size Avg        | 无    | 有     | 仅修正版存在 |
| CWR Flag Count              | 无    | 有     | 仅修正版存在 |
| Dst IP                      | 无    | 有     | 仅修正版存在 |
| Dst Port                    | 无    | 有     | 仅修正版存在 |
| FWD Init Win Bytes          | 无    | 有     | 仅修正版存在 |
| Flow ID                     | 无    | 有     | 仅修正版存在 |
| Fwd Act Data Pkts           | 无    | 有     | 仅修正版存在 |
| Fwd Bulk Rate Avg           | 无    | 有     | 仅修正版存在 |
| Fwd Bytes/Bulk Avg          | 无    | 有     | 仅修正版存在 |
| Fwd Packet/Bulk Avg         | 无    | 有     | 仅修正版存在 |
| Fwd RST Flags               | 无    | 有     | 仅修正版存在 |
| Fwd Seg Size Min            | 无    | 有     | 仅修正版存在 |
| Fwd Segment Size Avg        | 无    | 有     | 仅修正版存在 |
| ICMP Code                   | 无    | 有     | 仅修正版存在 |
| ICMP Type                   | 无    | 有     | 仅修正版存在 |
| Packet Length Max           | 无    | 有     | 仅修正版存在 |
| Packet Length Min           | 无    | 有     | 仅修正版存在 |
| Protocol                    | 无    | 有     | 仅修正版存在 |
| Src IP                      | 无    | 有     | 仅修正版存在 |
| Src Port                    | 无    | 有     | 仅修正版存在 |
| Timestamp                   | 无    | 有     | 仅修正版存在 |
| Total Bwd packets           | 无    | 有     | 仅修正版存在 |
| Total Fwd Packet            | 无    | 有     | 仅修正版存在 |
| Total Length of Bwd Packet  | 无    | 有     | 仅修正版存在 |
| Total Length of Fwd Packet  | 无    | 有     | 仅修正版存在 |
| Total TCP Flow Time         | 无    | 有     | 仅修正版存在 |
| id                          | 无    | 有     | 仅修正版存在 |