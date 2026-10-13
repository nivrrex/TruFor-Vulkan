# P2-a 补丁轮确定性与显存复测（+ CUDA 真机检查）结果

日期：2026-10-03。判据沿用已固定版本，未改。补丁轮 `+trufor1`（.so md5 `b0e53111…`）。

## P4：≥2MP 同设备跨 fresh 进程逐位一致

样本 `b5_r0.01`（1264×1926 = 2.43MP），两次 fresh 进程，每次约 37.8s，**无 OOM（EXIT=0）**：

| 输出 | run1 md5[:12] | run2 md5[:12] | 结论 |
|---|---|---|---|
| prob | `7d5b06c810fa` | `7d5b06c810fa` | 逐位一致 |
| conf | `d1e1c972680a` | `d1e1c972680a` | 逐位一致 |
| det | `78be1c43aeae` | `78be1c43aeae` | 逐位一致 |

## RX460：1.0MP 档确认（不 OOM + 确定性）

样本 `b3_r0.01_1mp`（999×999 = 0.998MP，(H·W)%4=1，取自 b3_r0.01 裁剪），两次 fresh 进程，
每次约 20.5s，**无 OOM（EXIT=0）**；运行后 VRAM 占用 429MB：

| 输出 | run1 | run2 | 结论 |
|---|---|---|---|
| prob | `dcaa7078a538` | `dcaa7078a538` | 逐位一致 |
| conf | `dcd41e32e280` | `dcd41e32e280` | 逐位一致 |
| det | `d5e9aab19479` | `d5e9aab19479` | 逐位一致 |

结论：RX460 1.0MP 档在补丁轮下可跑、可复现、不 OOM；2.4MP 档在 P4 上可跑、可复现、不 OOM。
其余（RX460 超 1.0MP、P4 超 2.4MP）仍由生产守卫拒绝，未主动触发 OOM 复测。

## CUDA 真机检查（A 仓 `tests/cuda_numerics_check.py`，`ortgpu_env`）

`LD_LIBRARY_PATH` 指向 venv 内 nvidia/*/lib（否则 CUDA EP 会静默回退 CPU，见 LOG 陷阱条）。

| 项 | 512×512 | 1264×1926（2.43MP） |
|---|---|---|
| out | md5 `5a4772bb4016` 命中，rel_L2 2.457e-05 | rel_L2 2.482e-04 |
| conf | md5 `909891b928c9` 命中，max_abs 3.905e-04（记录） | max_abs 4.475e-03（记录） |
| det | md5 `18757f258295` ∈ 已知模式集，max_abs 2.146e-06 | max_abs 1.404e-03 |

`CUDA_CHECK=PASS`。按用户建议**收紧 det 门**：`max_abs≤2e-2` → **`≤5e-3`**
（依据=本次真机实测 max 1.404e-3，约 3.5× 余量；512² 仍叠加已知模式 md5 硬门），
收紧后复跑仍 PASS。日志：A `doc/p1_logs/cuda_numerics_real.log`（收紧前/后）。
