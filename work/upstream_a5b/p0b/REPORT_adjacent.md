# P0-b ④ 相邻层迷你网格结果（Split / Crop 全网格 + Slice / Concat / ReLU 探针）

日期：2026-10-03。脚本 `work/upstream_a5b/micro/repro_adjacent_pack4.py`，
日志 `repro_adjacent_patch_dev1.log`（+trufor1 补丁轮）、`repro_adjacent_stock_dev1.log`
（归档 stock 轮，作灵敏度对照）。设备 P4（index 1）。

## 结果

| 变体 | 补丁轮 | stock（known-bad） |
|---|---|---|
| Split（Input→Split 回读） | **0 / 232 失配** | 96 / 232 失配 |
| Crop（Input→Crop 恒等回读） | **0 / 232 失配** | 56 / 232 失配 |
| Slice / Concat / ReLU（子进程探针 15 例） | **15/15 SIGSEGV(139)** | 15/15 SIGSEGV(139) |

- Split/Crop 网格 = c∈1..8 × 13 小尺寸 + c∈{2,3,4} × 4 边界尺寸
  （含 967×714 hw%4=2、967×712 hw%4=0、715×967 hw%4=1、714×968 hw%4=0）。
- stock 轮的失配**严格**落在 `c≥2 且 (h·w)%4≠0`，`(h·w)%4==0` 全 ok
  → 该网格能检出已知上传缺陷，**灵敏度成立**；补丁轮同网格零失配 → ④（Split/Crop）通过。

## Slice / Concat 被第二缺陷阻断（未覆盖，不计入通过）

[已验证事实] 迷你网中抽取 Slice / Concat / ReLU 的 GPU 输出**必 SIGSEGV(139)**：
- 与 `(h·w)%4` 无关（2×2=hw%4=0 也崩）；
- 与 blob 命名（数字/字母）无关；
- 与输入 Mat 是否零拷贝无关（plain / clone / create / create_like 全崩）；
- 在 Crop 之后或之前仍崩（`Input→Crop→Slice`、`Input→Slice→Crop` 均崩）；
- **stock 轮同样崩** → 属已知"第二缺陷"（`work/STATE.md:280-283`、C6/C7），非本补丁引入。

因此 Slice / Concat 的迷你网格**无法执行**，按判据不冒充通过。替代覆盖：② 的全网逐 blob
rel_L2 已抽取真实 TruFor 全网 2651 blob（含 Slice / Concat / ReLU 输出）在 4 个尺寸上的
GPU-vs-CPU 结果，补丁轮无超包络、无平移形态（见 `REPORT_relL2.md`）。Crop 由本网格直接覆盖。

## 判据与证伪

④ 用户判据为"网格零损坏"。Split/Crop 网格补丁轮零失配 → 通过；Slice/Concat 因第二缺陷
无法取得网格证据 → 该项记"未覆盖（被独立缺陷阻断）"，由用户裁决是否接受 ② 的全网替代证据。
