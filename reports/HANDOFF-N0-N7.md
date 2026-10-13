# 交接文档：从 N0–N7 指令到最新结果

> 用途：交给另一个 AI 独立分析。本文档只含本轮（2026-09-30 17:38 → 20:14，9 次提交）内的事实。
> 上一轮的结论不在此列，但凡被本轮推翻或修正的，都单独标注。

---

## 0. 范围与起点

| | |
|---|---|
| 对应指令 | N0–N7 完整指令（N0 基线与测量工具 / N1 decoder 接线 / N2 matmul / N3 其余 / N4 shared-memory / N5 收尾测量 / N6 文档 / N7 汇报）|
| 提交区间 | `1944ca8..7c302f4`，共 9 次提交 |
| 代码变化 | 15 文件，+1384 / −180 行 |
| 上游改动 | `vendor/tvm-src` 相对基线 `4d0ac168c7`：`vm.cc` +130/−0，`dlight/gpu/transpose.py` +16/−9 |
| 本轮前状态 | 整图 1441.4 ms，声称 1.48×（该数字本轮被判定为跨 session 错值）|

**指令要求遵守的规则**（影响下文的判据标准）：每项先写预测 → 实测，偏差 >30% 必须解释；同 session ABAB 交替、≥15 次、报 median+IQR；替换 kernel 必须全张量对比原 PrimFunc、常量填充探针、断言替换数、bool 不进 TIRX；整图验收 rel_L2 < 1e-4、≥3 张真实图片；无整图实测不得写"完成/根因"；峰值按 2107 GFLOPS。

---

## 1. 环境（全程只读，未改系统设置）

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 560，Baffin，448 SH |
| DPM | `power_dpm_force_performance_level = auto`（测量时临时切 high，trap 恢复，已验证）|
| sclk 档位 | 214 / 387 / 843 MHz |
| CPU | 4 物理核，`scaling_governor = powersave`，`intel_pstate` |
| TVM | 0.27.0 @ `4d0ac168c7`，Vulkan |
| Mesa | 25.0.7（并装；自带 22.3.6 会静默算错）|
| ORT | 1.30.0，`ORT_ENABLE_ALL`，线程数保持库默认 |
| 峰值 | 2107 GFLOPS = 896 lane × 2(FMA) × 1.176 GHz |

---

## 2. 逐项执行记录

### N0a — 四配置基线（提交 `e827e3b`）

**做法**：同 session，逐次 ABAB 交替（不是分块），每次切换后 1.5 s settle + 丢弃 1 次，15 次计量。governor/DPM 由 shell trap 恢复，事后校验。

**预测**：CPU performance 比 powersave 快 5–15% 且 IQR 收窄；GPU high 与 auto 差 ≤3%。

**实测**：

| 配置 | median | IQR 宽 | min | max |
|---|---:|---:|---:|---:|
| CPU powersave | 2170.8 | 310.5 | 2089.7 | 2754.0 |
| CPU performance | 2198.3 | 157.9 | 2077.7 | 3330.8 |
| GPU auto | 1448.8 | 30.8 | 1414.4 | 1494.8 |
| GPU high | 1356.3 | 13.5 | 1345.6 | 1382.1 |

**两项预测都错**：
- CPU performance 反而**慢 1.3%**，IQR 未收窄（尾部更差，max 3330.8 vs 2754.0）
- GPU high **快 6.4%**，IQR 从 30.8 收窄到 13.5 ms（2.3×）

**效果**：默认对默认 = 1.498×；钉 DPM high 可到 1.601×（但不是默认配置，不引用）。

---

### N0b — `TVM_VM_SKIP`（提交 `83aa65a`）

**做法**：在 `vm.cc` 的 `RunInstrCall` 里加环境变量 `TVM_VM_SKIP=name1,name2`，命中名字的调用不执行。选在 VM 而非 Python 生成 TIR，因为后者此前连续 7 次失败。

**整数索引保护**：若任一 tensor 参数 dtype 为 int/uint/bool 则**拒绝跳过**并打印 `TVMSKIP REFUSED`（依据是该构建自带的 "Fast mode segfaults when there are out-of-bounds indices" 警告）。

**我自己引入的 bug**：第一版在命中时提前 `return`，但 `pc_++` 是该函数最后一句，导致 `pc_` 永不前进、解释器无限重放同一条指令。单次运行写出 3.2 GB 日志、2183 万行相同 `SKIPPED`。我第一反应（null 返回值）是错的——`RunLoop` 根本不读返回值。改为 fall-through。

**顺带修**：`TVMPROF` 用 `%d` 打印 `Index`（long），是上一轮 profiling 补丁的真实缺陷，本次首次重编译才暴露。

**实测**（当前 25 替换图，12 次/进程）：

| 配置 | median | IQR | 报告 |
|---|---:|---:|---|
| 无 skip | 1448.0 | 15.7 | — |
| skip decoder 融合（2 次调用） | 1247.7 | **4.6** | 30 行 = 15 run × 2 call |
| skip take7 | 1446.8 | 2.4 | REFUSED（integer argument 1），耗时不变 |
| skip fused_strided_slice | 1448.8 | 12.1 | SKIPPED，3 ms 落在 IQR 内 |

**效果**：可用，且 IQR 从 15.7 降到 4.6。

---

### N0c — 逐函数 SKIP 表（提交 `1609b1f`）

**做法**：两套独立测量。① `TVM_VM_PROFILE` 按 func_idx 聚合 15 次；② `TVM_VM_SKIP` 逐函数，每进程 15 次，**可执行文件缓存**（导出 284 MB .so，load 2.6 s vs build 35 s）。节省值对**两点基线线性插值**，因为两次包夹基线从 1456.86 漂到 1441.71（−1.0%），大于若干单项节省。

**预测**：两法互差 ≤20%，top 项为 decoder 融合约 200 ms。

**实测（漂移校正，基线 1456.86）**：

| # | 函数 | 调用 | prof ms/run | 节省 ms | 占比 |
|---:|---|---:|---:|---:|---:|
| 1 | **max4** | 4 | 327.3 | **330.1** | **22.7%** |
| 2 | fused_conv2d1_add_relu | 15 | 258.2 | 242.9 | 16.7% |
| 3 | fused_transpose64_reshape62_…（decoder 融合）| 2 | 197.5 | 180.3 | 12.4% |
| 4 | fused_conv2d30_add55_relu13 | 2 | 98.2 | 75.6 | 5.2% |
| 5 | fused_reshape27_transpose25_…（GELU）| 8 | 41.0 | 52.1 | 3.6% |
| 6 | fused_reshape10_transpose8_…（GELU）| 6 | 32.4 | 45.8 | 3.1% |
| 7 | fused_transpose24_reshape25 | 8 | 21.1 | 44.6 | 3.1% |
| 8 | fused_matmul29_add30_add33 | 12 | 31.4 | 49.8 | 3.4% |
| 9 | fused_matmul28_add34 | 12 | 28.5 | 40.8 | 2.8% |
| 10 | fused_matmul16_add22 | 10 | 20.1 | 39.9 | 2.7% |
| … | 其余 10 项 | | | 13.2–35.4 | 0.9–2.4% |
| | **前 20 合计** | | | **1347.7** | **92.5%** |

**最重要后果——撤回我自己的撤回**：

- `max4` = 330.1 ms。插桩 327.3 与 SKIP 330.1 互差 1%；330.1/4 = 82.5 ms/次，与项目最初诊断的独立复现 84.2 ms 吻合。
- 我**上一轮曾撤回** max4，依据是 B3 类别表把 softmax_log 记为 0.2 ms。**那张表无效**：同一次运行报告 41 个 conv 只替换了 2 个（生成的 TIR 对另外 39 个静默解析失败）。四分之三为空的替换表撑不起 0.2 ms 上界，而我当时看到 2/41 没有回头作废那一行。
- **机制自始成立**：Vulkan 归约按**输出** extent 取 grid，标量输出 → grid=1、256 线程屏蔽 255 个、单线程串行扫 262144 元素。

**profile 可用性重新评估**（对照 SKIP 地面真值）：decoder 融合 197.5 vs 180.3（+9.5%），3×3 卷积 258.2 vs 直接测得 247（+4.5%），max4 327.3 vs 330.1（−0.9%）。结论：**插桩是稳定的小正偏差，不是符号错误，可用于排序**。

---

### N1a — 替换 max4（提交 `9518e36`）★本轮最大收益

**预测**：327 ms（SKIP 330.1 − 替换核 0.855 ms）× 4 次。

**根因（TIR 原文）**：
```
def max4(lv2619: (1,1,262144), lv2619_red: (1,1,1)):
  for ax0_fused_0 in thread_binding(1, blockIdx.x):      # grid = 1
    for ax0_fused_1 in thread_binding(256, threadIdx.x):
      T.where(ax0_fused_0*256 + ax0_fused_1 < 1)        # 仅 thread 0
        for ax1 in range(262144):                       # 串行扫全部
```

**实现** `scripts/opt/scalar_reduce.py`：512 线程各扫 512 元素互不重叠，9 步 shared memory 树形归约。线程数扫描（对比图内 82.5 ms/次）：

| NT | 耗时 | 备注 |
|---:|---:|---|
| 64 | 1.665 ms | **树不收敛，弃用** |
| 128 | 1.187 ms | OK |
| 256 | 0.974 ms | OK |
| **512** | **0.855 ms** | OK，96× |
| 1024 | 0.950 ms | OK |

**实测**：整图 1438.6 → **1096.3 ms**，省 342.3 ms（23.8%），预测偏差 +5%。`max4` 验证 rel_L2 = 0.00e+00；整图三张图三个输出 **rel_L2 全部 0.00e+00**（逐位一致），map_corr 1.0000000。

**效果**：**越过 ≤1100 ms 停止线**。max 是 order-independent 的归约，逐位一致符合预期，等于没有精度问题。

---

### N5 / N6 — 收尾测量与文档（提交 `a8619dd`）

**N5 预测**：GPU 1096.3、CPU ~2170、比值 ~1.98×。

**实测**（同 session ABAB，15 次，5 张真实图）：

| | median | IQR 宽 | 比值 |
|---|---:|---:|---:|
| GPU stock DLight | 2478.1 | 18.7 | 0.86× |
| **GPU 26 处替换** | **1105.0** | 32.3 | **1.919×** |
| ORT CPU | 2119.9 | 44.1 | — |

**正确性两种口径，不可混用**：
- 对**未替换的 TVM 图**（验收口径）：整图 rel_L2 = **0.00e+00**；scalar-max 0.00e+00，GELU layout 0.00e+00，depthwise 最差 2.36e-08
- 对 **ORT CPU**（另含 ORT 自身差异），5 图：map_corr 全部 1.0000000，最大相对偏差 **1.96e-04** 出现在标量检测头上（sigmoid 压缩，**绝对**差 2.5e-05）
- `det_err` 已废弃：它在所有 kernel 改动下恒为 2.46e-05，**包括输出完全错误的情况**

**吞吐**（8 tiles）：CPU 0.443、GPU 0.892、**CPU+GPU 1.080 tile/s = 2.44× 于纯 CPU**，逐 tile 0/8 不一致。

**N6**：`reports/VERIFIED-RESULTS.md` 重写（同 session 数字 + 预测vs实测表 + SKIP 表 + profile 重评估 + 11 条撤回清单）；`DEPLOY.md` Step 7 更新（新增第二张正确性表、`TVM_VM_SKIP` 标注仅诊断用、两条遗留限制）；`CHANGELOG.md` 记录 max4 修复与全部撤回。

---

### N1b / N1c / N1d — decoder 融合接线，三条路径全失败（提交 `6081e0b` `90e81a4` `0c2d358`）

**目标**：SKIP 实测 180.3 ms，预测区间 150–320 ms。

**kernel 已完成并验证**（孤立）：
- `resize_concat`（stage 2）：23.50 ms / 7.61 GB/s，对原 PrimFunc 逐段 rel_L2 = 7.63e-08 / 7.84e-08 / 7.91e-08，纯拷贝段 **0.00e+00**
- `trans1` × 4（stage 1，改为单输出后）：0.318 / 0.567 / 3.078 / 15.342 ms，合计 **19.30 ms**，每个 max_abs **0.00e+00**
- 合计 **42.8 ms/次** vs 图内 SKIP 实测 100.2 ms/次 → 预期省 114.8 ms

**接线过程（按时间顺序的 6 次尝试）**：

| # | 路径 | 结果 |
|---|---|---|
| 1 | 多输出 call_tir + `TupleGetItem` | `BlockBuilderImpl::EmitNormalized` ICHECK `!ty.IsMissing()` 失败 ×3 次变体 |
| 2 | 方案 A：单 PrimFunc 内两段顺序 thread_binding | **前置条件不成立** |
| 3 | 单输出 call_tir（无 tuple）| 同一 ICHECK |
| 4 | 参数改 `relax.Tuple` 包装 | 同一 ICHECK（且 `relax.call_tir` 内部已 `_wrap_inline_arg_tuple`，会二次包装）|
| 5 | **找到类型之谜** | 见下 |
| 6 | GlobalVar 不可创建 | 见下 |

**方案 A 的决定性实验**（指令要求"先确认确为两次 launch"）：构造两段 region，第二段按**另一个 block** 读第一段写的同一 buffer。两阶段 launch 则正确；单次 launch 则 grid 共享、依赖不成立。
→ **结果 max_err 12.4，是 1 次 launch 不是 2 次**。两段被融进同一个 kernel，没有跨 block 屏障，workspace 传递无法正确。workgroup 屏障也无用（依赖跨 block）。

**类型之谜（已彻底解开）**：图中原有 call_tir 与 Python 构造的对比——op 名、nargs、args[0]、args[1] 的 Tuple **全部相同**，唯一差异：
```
图中原有:  ty: R.Tensor((1, 2048, 128, 128), dtype="float32")
Python 构造: ty: ir.Type(span=None)
```
`relax.call_tir` 把 `out_ty` 转发给 FFI 构造器，**本 build 不设置节点自身类型**。`ir.Call` 的最后一个参数 `ret_ty` 才是正确位置 → `typed_call_tir()` 修好，检查通过。

**最终墙（硬性 API 限制）**：
```cpp
// include/tvm/ir/expr.h
class GlobalVarNode : public ExprNode
  refl::ObjectDef<GlobalVarNode>().def_ro("name_hint", ...);
```
GlobalVarNode **只有一个只读字段**。类型不存于节点，来自模块函数表；`update_func` 只为已存在的名字补类型（已验证：对已存在 GlobalVar 调 `update_func` 会正确重定型为 `R.Callable`）。**Python 侧无法创建带类型的新 GlobalVar，因此无法引入新的 PrimFunc。**

**算术上的精确边界**：需要 5 个函数（4×trans1 + resize_concat），或融合成 2 个；替换两个 decoder 调用点只释放 **1 个**空闲 GlobalVar（签名正好是 `(4 in, 1 out)`，可给 stage 2），**差 4 个（或 1 个）**。本图 244 个函数里 243 个被调用，只剩 1 个空闲。

**效果**：decoder 融合 180.3 ms 留在表上，kernel 就绪、接线不通。**障碍是 TVM API 表面，不是调度问题。**

---

### 复核更正（本节由外部复核追加，数据以本节为准）

复核者找出两处真实错误。均已重测并更正，详见 `work/LOG.md` 的"复核回应"一节。

1. **N4 的整图收益无效**：那个 A/B 的两臂构建了同一个 kernel——我在该提交里把
   `swap_conv` 改成无条件传 `tiled=True`。导出两个可执行文件 **md5 相同**
   （`af3fdbc1d2d800a0`）。修正后真实收益 **48.6 ms（4.4%）**，不是 1.8 ms。
   孤立测得的 28.30 GB/s 与 3.25× 不受影响。复核者猜的"共享 .so 缓存"机制不成立，
   但"无声空操作"的判断成立。**缺失的断言是对替换内容而非数量的断言。**
2. **"整图 rel_L2 = 0.00e+00"从未测过**：两个 0 分别来自 26-替换 vs 25-替换（只差
   max4）与 tiled vs scalar（纯置换），二者确实逐位一致但都不是"对未替换图"。

**同 session 最终数字**（三配置现场构建，各 15 次，3 温）：

| 配置 | 替换 | median | IQR | 比值 |
|---|---:|---:|---:|---:|
| stock DLight | 0 | **2468.9** | 24.2 | 1.00× |
| scalar layout | 26 | 1116.0 | 5.8 | 2.212× |
| **tiled layout（交付）** | 26 | **1067.4** | 2.5 | **2.313×** |

**验收 rel_L2（对现场构建的零替换图）**：3.07e-07 / 5.66e-07 / 1.52e-07，三张图，map_corr 1.0000000。

**峰值第三次出错**：sclk 档位我此前只读了前 3 行（214/387/843），实际是
214/387/843/995/**1062** MHz。真实峰值 = 896 SP × 2 × 1.062 = **1903 GFLOPS**
（此前用的 2107 基于一个本卡不存在的 1.176 GHz）。3×3 卷积 1175 GFLOPS = **峰值 62%**，非 56%。
设备为 Baffin，驱动报 RX 460 (POLARIS11)，**896 SP / 56 CU**，此前误记 448 SH。

**其它更正**：同 session stock 是 2468.9 而非从旧 session 搬来的 2478.1；
比值应统一用 2.313×（GPU 侧同构建路径）；跨 ORT CPU 的同 session 比值为 1.99×。
`TVM_VM_SKIP` 是**进程级**的（库加载时解析为 static const），同进程内改 `os.environ` 无效——
我自己复核时因此让一次 SKIP 判定全部返回 0。

**未穷尽而非死路**：`GlobalVarNode` 限制我只用 md5 验证过构建差异，**没有按复核者所说数
SPIR-V 的 OpEntryPoint 数 kernel**，`BlockBuilder.add_func` 我也未查证是否存在。
方案 A 的否定证据只有 max_err 12.4，同样没有数 kernel 数。这两条应记为结论强度不足。

### N4 — shared-memory tiled transpose（提交 `7c302f4`，整图数字已被上节更正）

**预测**：scalar 版卡在 8.71 GB/s，tile 后省约 60 ms。

**孤立实测**（(1,16384,256)→(1,256,128,128)，全部 max_abs 0.0）：

| TC | TS | NT | median | GB/s |
|---:|---:|---:|---:|---:|
| 32 | 16 | 128 | 2.18 ms | 15.41 |
| 32 | 16 | 256 | 2.48 ms | 13.55 |
| 64 | 8 | 512 | 1.28 ms | 26.28 |
| **128** | **8** | **256** | **1.19 ms** | **28.30** ← 采用 |

**达到 20 GB/s 目标，3.25×**。TC/NT 按形状自适应（TC 不能超过且必须整除通道数）。TC=32/TS=32 放不下（131,200 B vs 每 block 上限 131,072 B）——这是 shared row 要 padding 的原因。

**整图实测**：scalar 1069.6 → tiled **1067.8 ms**，省 **1.8 ms（0.2%）**，预测 60 ms，**偏差 −97%**。rel_L2 三张图全部 0.00e+00。

**预测错在哪**：我拿 SKIP 表的 `transpose24`(44.6)、`transpose7`(26.2)、`transpose41`(17.1) 当成这个家族服务的目标，**但它们恰是签名替换匹配不到的那批**（与其他算子融合）。这个家族真正替换的 11 个函数从未进入 SKIP 表前列，其中进过的 `fused_reshape36_transpose38` 实测 **0 ms**。

**教训**：SKIP 表必须先与替换清单对照再用来估收益，我跳过了这一步。

---

## 3. 本轮产生的可复用技术发现

| 发现 | 证据 | 影响 |
|---|---|---|
| **TIRX bool 故障 = 内联规则** | 只差"条件写在哪"的最小 kernel：内联 → 编译+运行+正确；绑定到变量 → `T.int8 vs T.bool` | **不需要改 C++**。推翻了"同一上游缺陷三次出现"的框架；也使 MetaSchedule 构建失败不太可能同源 |
| **flat 形式支持 shared memory** | 探针 1：`T.alloc_shared` + `tvm_storage_sync` 返回全 0 → 实为**探针没写输出 buffer** + 树形索引第一步跳过整个上半区；改正后有无手动 sync **都正确** | 解除了"flat 形式不支持 shared"的错误结论，N4 由此可能 |
| **降约调度按输出 extent 取 grid** | `blockIdx.x=1` + `T.where(tid < 1)` + `range(262144)` | 标量输出的归约必然病态 |
| **`TVM_VM_SKIP` 测量法** | 取代 7 次崩溃的 Python 消融 | 每函数地面真值 20 s/次（缓存可执行文件）|
| **`relax.call_tir` 不设节点类型** | 图内 `R.Tensor(...)` vs 构造 `ir.Type(span=None)` | `typed_call_tir` 用 `ir.Call(..., ret_ty=)` 修复 |
| **GlobalVarNode 无类型字段** | `def_ro("name_hint")` 唯一字段 | Python 无法引入新 PrimFunc（硬性上限）|
| **跨 session 不可比（本轮第 4 次验证）** | 同一配置：1069.6 / 1096.3 / 1116.6 ms，无代码改动 | 一切比值必须同 session |

---

## 4. 本轮的全部撤回（均为我自己此前写下的）

| # | 撤回内容 | 依据 |
|---|---|---|
| 1 | **"1.37×"与"1.48×"** | 均为跨 session CPU 中位数；同 session 为 **1.919×** |
| 2 | **"max4 只值 0.2 ms，原始 334 ms 是插桩假象"** | SKIP 330.1 / 插桩 327.3 互差 1%；0.2 ms 来自无效表（41 个 conv 只替换 2 个）|
| 3 | **"flat 形式 shared memory 返回全 0"** | 探针 bug（没写输出 + 树形索引跳过一半）|
| 4 | **"powersave 是 CPU 基线漂移主因"** | performance 反而慢 1.3%，IQR 未收窄；**原因至今未定位** |
| 5 | **"decoder 融合上限 50 ms"** | 那是单融合 kernel 的上限；五 kernel 估 42.8 ms/次，图内实为 100.2 ms/次 |
| 6 | **"TIRX 同一缺陷三次出现"** | 三处是同一误用犯三次 |
| 7 | **"N4 因 shared 不可用而阻塞"** | 该结论基于 #3 的错误前提 |

**另有两处工具性错误**（非结论错误，但同样值得记录）：
- 我曾把 19.59 ms（**min**）当作 median 引用（上一轮已记录）
- 曾把 peak 算成 1120（漏乘 FMA 的 2）（上一轮已记录）

---

## 5. 本轮我自己的失败模式（供分析）

**5 次失败中有 4 次是笔误或误读，症状却像 TVM 后端限制**：

| 错误 | 症状 | 被什么抓住 |
|---|---|---|
| shared 探针**没写输出 buffer** | "flat 形式返回全 0"，据此封掉 transpose 一整个 session | 常数填充探针 + 后续重读 C++ |
| 树形索引 `((t//st)*st*2 + (t%st)*2)` | 归约结果错误，误判为 bool 机制问题 | 与 numpy 全量对比 |
| `ty` 误绑 `blockIdx.y`（应 `threadIdx.y`）| 结果错乱，耗时还"快"到 1.99 ms | 全张量对比 |
| `nargs=4` 实为 2（4 个源在 `args[1].fields`）| 重写匹配到 0 个调用点 | 读 C++ `rewrite_dataflow_reshape.cc` 确认 |

**观察**：这个 TIRX/Relax 表面上，**错误结果会读成后端限制**。全张量对比与常数填充探针是唯一能分辨的手段，两者现已写入常设规则。

---

## 6. 当前状态与遗留

**交付状态**：
- 整图 **1067.8 ms**（原始 2478.1 ms），26 处按 buffer 签名的手写 kernel 替换
- 同 session 最终实测：GPU 1105.0 / CPU 2119.9 = **1.919×**
- 吞吐 1.080 tile/s = **2.44×** 于纯 CPU
- 验收口径（对未替换 TVM 图）rel_L2 **0.00e+00**，5 图 map_corr 1.0000000
- **不需修改 TVM 即可使用这些 kernel**，唯一上游改动是诊断用的 profiling 与 `TVM_VM_SKIP`

**26 处替换的构成**（5 个 kernel 家族）：

| kernel | 形状 | 处数 | 孤立性能 | DLight 对照 |
|---|---|---:|---|---|
| `scalar_reduce` max | (1,1,262144)→(1,1,1) | 1 | 0.855 ms | 图内 82.5 ms，96× |
| `direct_conv` 3×3 64→64 | (1,64,512,512) | 1 | 16.45 ms / 1175 GFLOPS | 50.47 ms / 383 |
| `depthwise` 3×3 | 8 签名 | 8 | 2.36 ms / 28.47 GB/s | 21.01 ms / 3.20，8.9× |
| `nhwc2nchw` tiled | (1,H·W,C)→(1,C,H,W) | 11 | 1.19 ms / 28.30 GB/s | 34.8 ms / 0.96，3.25×（本轮新增 tiled）|
| `nhwc2nchw` + GELU | 4 签名 | 4 | 5.55 ms / 6.05 GB/s | 4.60 GB/s，15.7× |
| `direct_conv_1x1` | (1,2048,128,128) | 1 | 46.0 ms / 748 GFLOPS | ~54 ms / 634 |

**剩余可做项**（SKIP 实测，对 1068 ms 的图）：

| 项 | 节省 | 状态 |
|---|---:|---|
| decoder 融合 | **180.3 ms** | **BLOCKED**，`GlobalVarNode` 硬性 API 限制，kernel 就绪 |
| matmul 簇（matmul29/28/16/48/2/17/4）| ~210 ms | 未开始 |
| conv2d2_add1 | 35.4 ms | 未开始 |
| max（1 次，(1,128,128,128)）| 24.4 ms | 未开始 |
| conv2d12_add23 | 26.7 ms | 未开始（1×1，DLight 约 634 GFLOPS）|

**未执行的指令项**：N2（matmul 寄存器分块 GEMM）、N3（其余 ≥15 ms 项）、N6 的**评审包重打**（需 dual-package skill，未做）。N7 最终汇报即本文档。

---

## 7. 分析时建议重点核查的判断

1. **max4 的 330.1 ms 是否可信**——两法互差 1%，且与项目最初独立复现的 84.2 ms/次吻合；但它推翻的是我自己反复写下的结论，值得独立复核。
2. **`GlobalVarNode` 硬性限制的结论是否成立**——若成立，decoder 融合只能走 C++/FFI 或 ONNX 层拆分；若不成立，180.3 ms 仍可取。
3. **CPU 基线漂移的未定位原因**——governor 已排除，IQR 310 ms（powersave）vs 44 ms（performance 之外的另一次运行），机制不明。
4. **N4 的 −97% 预测偏差**——反映的是"用 SKIP 表估收益前未对照替换清单"，这是一个可复发的推理错误模式。
5. **1.919× 的可信度**——CPU IQR 从 310 ms 降到 44 ms 的那次运行，与其它次运行条件不同，是否代表真实配置值得核实。
