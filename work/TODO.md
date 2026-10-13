# work/TODO.md

规则：全部 DONE 或 BLOCKED（附证据）之前，不做总结性回复。禁止 `ask_user`，
除非涉及删数据、改生产默认路径、重启机器。同一路径连续失败 3 次换路径并记 LOG。
每项完成后 `git commit` 并更新 `work/STATE.md`。

## 预测表（偏差 >30% 必须解释；无实测印证不得写"根因"或"完成"）

| ID | 目标 | 预测节省 | 状态 |
|---|---|---:|---|
| T0 | 基线与方法 | — | DONE（旧"消融 BLOCKED"部分撤回：`make_noop` 生成 `def` 缩进 4 空格 + `y[...] = 0` Ellipsis 字面量，属自家 bug；`work/ablate_poc.py` 手写合法 no-op 能 compile+run，整图逐类 ablation 尚未做） |
| T1 | depthwise/group 3x3 | ~300 ms | DONE 实测 268.1 ms |
| **N1a** | **max4 标量归约** | ~330 ms | **DONE 实测 342.3 ms（+5%），rel_L2 全 0，整图 1096.3 ms** |
| T2 / N1b/c | decoder 融合块 | ~230 ms（SKIP 实测 **180.3**） | **DONE/已接入（撤回旧 BLOCKED）**：`BlockBuilder.add_func` 可新增**带类型** GlobalVar，"Python 无法创建带类型 GlobalVar"的旧结论是错的。5 个 replacement PrimFunc + typed_call_tir 已跑通，run_trufor 按 PRE→rewrite→POST 接线，每轮 rewrote 2 call sites，验收逐位一致。kernel 全部就绪（stage1 19.30 + stage2 23.50 = 42.8 ms/次，vs 图内 100.2）。 |
| T3 | 其余 layout（GELU 融合） | ~100-150 ms | DONE 实测 191.0 ms |
| T4 | shared-memory transpose | ~100 ms | PARTIAL（VEC=8 已得 ~54 ms；shared 路径未做） |
| T5 | 已有 kernel 继续调优 | ~60-90 ms | DONE 实测 ≈0（整图噪声 21 ms 淹没 3% 孤立差异） |
| T6 | CPU+GPU 并行吞吐脚本 | — | DONE 1.003 tile/s = 2.24× |
| T7 | 收尾（补丁清单、DEPLOY.md、评审包） | — | PARTIAL（报告/DEPLOY/补丁完成；评审包未打） |
| T8 | matmul 簇手写 GEMM | 35-50 ms | **DONE 实测 ~53 ms（906→853），60 处替换，验收 1.29e-06 不变** |
| T9 | P4 8G vs RX460 2G 同 pipeline A/B | — | **DONE，决定不切生产**：P4 stock 1456 ms（比 RX460 stock 快 1.6×），P4 delivered 4948 ms（比 P4 stock 慢 3.4×，比 RX460 delivered 慢 5.8×）。P4 与 RX460 都跳过同样的 11 个候选（4 MEAN + 7 非-GELU transpose），原因是 swap_conv 形状签名过匹配、`_verify_new` 拒得对，非驱动差异。顺带把 `scalar_reduce` 的 `sh[t+st]` 越界读（非 reader 线程 t+st ≥ NT，SPIR-V OpSelect 两边都求值）clamp 到 `T.min(t+st, NT-1)`；RX460 交付重跑 median 866（原 853，噪声内），替换集完全一致。 |
| T10 | 动态尺寸 ONNX（ORT-CPU 回退，方案 c） | — | **DONE**：`scripts/export_onnx_dynamo_dynamic.py` + `scripts/verify_dynamic_ort.py`，产物 `reports/onnx-dynamo/trufor_dynamic_dynamo.onnx`（279 MB 内联单文件，H/W 符号轴、batch=1）。ORT vs truFor-src 在 256²/512²/640×480/1200×900/518×731（含非 32 整除）rel_L2 3–6e-05、corr 1.0，map==input 契约全尺寸成立；512² 处与生产 fixed-512 交叉核对 rel_L2 ~3e-07。仅 ORT 用，Vulkan 仍 512-only。 |

停止条件：整图 ≤1100 ms（理想 ≤900），或全部 DONE/BLOCKED。**已达成：889.4 ms，越过理想线 ≤900 ms。**

**CPU 独立进程基线**（taskset 0-3, 30 次）：median 2129.4 ms，IQR 宽 57.9 ms（混测时 310 ms，
证实同进程是波动主因）。两条比值：GPU 侧同构建路径 **2.661×**，跨 ORT CPU **2.297×**。

**重做后的可得收益表**（当前 32 处替换、926.8 ms 基线，两次包夹基线漂移 1.1 ms）：
可优化项 SKIP 上限合计 **102.7 ms (11.1%)**，其中 matmul 簇 6 个占 86.5 ms。
按 SKIP×(1−1/加速比)、手写 GEMM 1.3–2.0× 估：**可得 24–51 ms**。
另：我方 decoder kernel 自身仍占 67.5 ms（原 180.3 ms 的 2.7×），是目标不是收益。

**复核后的最终数字**（同 session、三配置现场构建）：
stock 0 替换 **2472.7 ms** / 30 替换 1057.2 ms / +decoder 929.6 ms / **交付 889.4 ms** = **2.780×**。
验收 rel_L2（对现场构建的零替换图）3.07e-07 ~ 5.66e-07，map_corr 1.0000000。

****停止线已达成：整图 1096.3 ms ≤ 1100。** 剩余项按 SKIP 表继续，但每一项的绝对值都变小了。

N0c SKIP 排名（漂移校正，基线 1456.86）改变了优先级**：
max4 330.1 / conv2d1 242.9 / decoder 融合 180.3 / conv2d30 75.6 /
reshape27 52.1 / matmul29 49.8 / reshape10 45.8 / transpose24 44.6 …
前三项合计 753 ms。max4 单独一项就大于已入库的全部卷积+decoder+1x1 之和。

当前 1441.4 ms，距 1300 还差 141 ms。T2 是唯一还有较大空间的一项（估 330 ms），
kernel 与调用点都已就位，只差一个 TVM 内部的 pattern 检查需要绕过。

## DONE

- **T0 基线**（同 session ABAB）：CPU 2126.6（IQR 宽 126.5）、GPU 原始 2478.1、
  GPU 手写 1768.2。**"1.37×" 更正为同 session 的 1.20×**。governor=powersave。
  卷积流量算术：depthwise 502.8 MB / dense 296.2 / grouped 108.0。
- **T0 消融（常量 0.5）**：**BLOCKED**，7 次尝试均以 native 崩溃或静默失败告终。
  此前"no-op 因 denormal 变慢"的说法**未经测试，不再作为结论引用**。
  B3 的类别上界（transpose 546 / other 353）**仍未验证**。
- **T1 depthwise**：**DONE**。DLight 3.12-3.20 GB/s → 手写 28.47 GB/s（8.9×）。
  整图 1770.1 → 1502.0 ms，省 268.1 ms（预测 260，偏差 +3%）。
  3 张真实图片 rel_L2 ≤ 5.09e-07，map_corr 1.0000000。

## 备注（易踩的坑）

- `T.alloc_shared` + `T.tvm_storage_sync` 在 TIRX **flat 形式**下编译运行成功但返回全 0。
  不要走 TIRX 裸 API。T4 应改用 `T.sblock` 形式 + s_tir Schedule 原语。
- DLight 修复后的 Transpose 规则在 (1,16384,512) 上也只有 ~5 GB/s，说明单纯分块不够，
  要检查读写是否同时向量化、是否有 bank conflict。
- 消融用 no-op 会踩 denormal：T0 之前 `other` 类消融返回负收益就是这么来的。
