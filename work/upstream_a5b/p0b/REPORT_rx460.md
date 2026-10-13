# P0-b RX460（AMD，device 0）补测结果

日期：2026-10-03。判据沿用已固定版本（① |ΔAUC|≤2e-3 且翻转<0.1%；
② 包络=3×同设备对照 714×968 max rel_L2，逐 blob ≤包络且 shift_flag=0）。
环境：`+trufor1` 补丁轮，`/root/ncnn_pypi_env`，AMD Radeon RX460 2GiB (RADV POLARIS11)。

## ① 任务级（≤1MP 子集，9 样本）

RP460 max_mp=1.0，故取 713×965/714×967/713×967（b3 的 ~1.7MP 裁剪超出档位，未跑）。

| 指标 | 实测 | 门 |
|---|---|---|
| \|ΔAUC\| max | **4.4e-4** | ≤2e-3 ✅ |
| prob 翻转@thr99 max | 4.4e-5 | <1e-3 ✅ |
| prob 翻转@0.5 max | 3.9e-5 | <1e-3 ✅ |
| conf max_abs | 2.45e-2~2.91e-2 | 记录项 |

`A5B_GATE=PASS`。证据：`a5b_mod4_rx460.jsonl` / `a5b_mod4_metrics_rx460.log`。

## ② 全网逐 blob rel_L2（4 尺寸，RX460）

| 尺寸 | rem | max rel_L2 | ≤包络(1.440e-2) | shift_flag | skip |
|---|---|---|---|---|---|
| 714×968 对照 | 0 | 4.801e-3 | — | 0 | 46 |
| 714×967 | 2 | 4.819e-3 | 是 | 0 | 46 |
| 713×965 | 1 | 4.837e-3 | 是 | 0 | 46 |
| 713×967 | 3 | 4.821e-3 | 是 | 0 | 46 |

skip 集合与对照一致（46，同 P4）。配对 test/ctl 比值中位 1.00、p99 1.11~1.20
（个别比值极大源于对照该 blob rel 恰为 0，非异常）。RX460 对照 max 与 P4 对照
max 几乎相同（4.8007e-3 vs 4.8005e-3）。
注：2551-2651 段在 RX460 上整块 OOM(137)，已细分 30/30/30/11 补跑；全部 18+4 块 EXIT=0。

## C8 同设备跨 fresh 进程逐位一致（RX460）

两样本 × 两 fresh 进程，三输出 md5 全部相同：
- `b22_r0.01`（714×967）：prob `582e9bb12c9c`、conf `b7bfdfb3065c`、det `caa30c4a1736` 双跑一致；
- `b22_r0.001_rem1`（713×965）：prob `7dcf2fb884d3`、conf `b60443b1f4d9`、det `b6df0ee5895c` 双跑一致。

## 结论

RX460 上 ① 与 ②（含 C8）**全部通过**，判据未再改动。至此 ncnn 准入的
"机制性根因 / 扩展网格（Split/Crop 网格 + ② 替代 Slice/Concat）/ A5b 原尺寸（P4+RX460）"
三项证据在 NVIDIA(P4) 与 AMD(RX460) 双侧齐备；仅剩 Windows（用户裁决为书面缺项）。
