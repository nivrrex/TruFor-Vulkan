# P0-b ② 补丁轮全网逐 blob 回归（714×967，rem2）

日期：2026-10-02。解释器：`/root/ncnn_pypi_env/bin/python`（ncnn `1.0.20261002+trufor1`，
.so md5 `b0e53111dfe2a1cbbc6a8d56bae49947`）。网络：`work/ncnn_a1/trufor.ts.ncnn.param/.bin`。
规模：`len(n.blobs())=2651`，其中 **2605** 个 blob 可 extract（其余 46 个为索引查名失败，
与修复无关）。**"2605"这一数字的出处 = 两侧 TSV 的实际行数**（此前登记为"未找到出处"，
本次核实推翻）。

## 命令（原文）

```bash
# CPU 侧（单进程，~136s）
work/pnnx_gpu_blob_diff.py --side cpu --size 714x967 --out work/upstream_a5b/p0b/cpu_714x967.tsv
# GPU 侧（P4=device 1，分块 fresh 进程，避免单进程 ~600 blob 处 OOM exit137）
work/pnnx_gpu_blob_diff.py --side gpu --size 714x967 --device 1 --range 1-550    --out .../gpu_714x967_1-550.tsv
work/pnnx_gpu_blob_diff.py --side gpu --size 714x967 --device 1 --range 551-1100 --out .../gpu_714x967_551-1100.tsv
work/pnnx_gpu_blob_diff.py --side gpu --size 714x967 --device 1 --range 1101-1650 --out .../gpu_714x967_1101-1650.tsv
work/pnnx_gpu_blob_diff.py --side gpu --size 714x967 --device 1 --range 1651-2200 --out .../gpu_714x967_1651-2200.tsv
work/pnnx_gpu_blob_diff.py --side gpu --size 714x967 --device 1 --range 2201-2651 --out .../gpu_714x967_2201-2651.tsv
# 对照组（hw%4==0，补丁新分支不触发，即"从未有过缺陷"的输入）
work/pnnx_gpu_blob_diff.py --side cpu --size 714x968 --out .../ctl_cpu_714x968.tsv
work/pnnx_gpu_blob_diff.py --side gpu --size 714x968 --device 1 --range <同上五段> --out .../ctl_gpu_714x968_*.tsv
```

比较脚本 `pnnx_gpu_blob_diff.py --side cmp` 的阈值是 `rel_thr=1e-3`，比较对象为每 blob 的
`mean/std/norm`（float64），相对量以 CPU 侧绝对值为分母。

## 结果（`cmp_summary.log` 原文可复现）

| 数据集 | n | rel>1e-3 | >1e-2 | >1e-1 | >1.0 | max |
|---|---|---|---|---|---|---|
| 对照 714×968（hw%4=0，补丁不适用） | 2605 | **827** | 93 | 8 | 0 | 0.124 |
| 补丁前 714×967（旧 `gpu_714x967.tsv`） | 2605 | 1935 | 883 | 87 | 3 | 1.92 |
| 补丁后 714×967（本次） | 2605 | **822** | 94 | 1 | 0 | 0.137 |

- 补丁后超阈 blob 中，**对照数据集未超阈的"净增"仅 15 个，rel 全部落在 1.00e-3~1.17e-3**；
  补丁前同口径净增 1128 个（worst 0.17）。
- [工程判断，置信高] 该 harness 的相对 mean/std/norm 判据对近零均值 blob 有固有噪声底：
  从来不含缺陷的对齐尺寸对照组本身就有 827/2605 超 1e-3。故字面判据"零失配（rel_thr=1e-3）"
  在此 harness 下不可达；补丁后已回到对照组的噪声水平。
- 与本仓 `work/LOG.md` 自身教训一致：统计型 mean/std 对平面错位型损坏不敏感，定位 onset
  须逐元素 rel_L2。② 的 TSV 统计比较属筛查，不属判定性仪器。

## 判据待决

用户预设 ② 判据为"零失配"。按字面执行则**不通过**（对照组合格线本身即 827）。
本目录只落原始证据，不做判据改写；由用户裁决 ② 的判定口径。
