# TruFor-Vulkan

TruFor 图像篡改检测在 **Vulkan（AMD/通用 GPU）** 上的落地工程与研究记录仓库。

本仓库聚焦**推理后端本身**：把 TruFor 跑在 GPU 上、证明数值正确、定位并修复上游缺陷、
给出可复现的验证证据。产品/管线侧（"下游应用"）代码**不在本仓库**，另存他处。

---

## 1. 一句话结论

TruFor 全网在 ncnn-Vulkan 上原尺寸推理可用，但上游 ncnn 的 Vulkan **上传路径**存在
确定性缺陷（`c≥2 且 (H·W)%4≠0` 时输入平面平移损坏）。已定位到单行机制、本地 fork 修复
（单文件补丁），并在 NVIDIA(P4) 与 AMD(RX460) 双设备上完成任务级、全网逐 blob、
确定性与相邻层网格验证。

## 2. 仓库内容

```
.
├── README.md                 本文件
├── DEVELOPMENT.md            开发/接手说明（环境、命令、基线、证据索引、TODO）
├── AGENTS.md                 给后续开发者的工程红线
├── LICENSE-NOTICE.md         许可与分发说明
├── GITHUB-UPLOAD.md          上传说明
├── SOURCES.txt               本仓库来源与生成方式
├── docs/
│   └── TVM-VULKAN-README.md  早期 TVM-Vulkan 阶段的 README（历史）
├── CHANGELOG.md / DEPLOY.md / PACKAGE-NOTE.md   早期 TVM 阶段文档（历史）
├── work/
│   ├── STATE.md              现行结论台账（★先读）
│   ├── LOG.md                决策/推翻/失败尝试长记录（★证据链）
│   ├── MIGRATION_SUMMARY.md  阶段一核查 C1–C13、证据索引、下一步计划
│   ├── upstream_a5b/
│   │   ├── FORK_BUILD.md     ncnn 补丁轮构建与分发链
│   │   ├── 0001-*.patch      单文件补丁（47 行）
│   │   ├── micro/            迷你复现（Split / 相邻层 / ReLU / 验证层）
│   │   └── p0b/              P0-b/P1/P2 验证报告与原始数据
│   ├── pnnx_*.py             阶段 A/B 各闸门实验脚本
│   ├── resize_ablation.py    C2（禁止缩图）消融
│   ├── pnnx_blob_relL2.py    逐 blob 逐元素 rel_L2 仪器
│   └── ablation_results/     归档结果（jsonl/log）
├── reports/                  早期 TVM-Vulkan 报告
├── upstream/ patches/ scripts/  早期资产
```

**未入库**：权重、`vendor/`（TVM 源码树）、`work/ncnn_a1/*.param/.bin`、样本与原始图、
`a5b_dumps*/`、`work/upstream_a5b/wheels/`（wheel 不入库，按 `FORK_BUILD.md` 自建）。

## 3. 核心工程结果

### 3.1 根因（已验证事实，`work/LOG.md` fork 节）

`VkCompute::record_upload`（`src/command.cpp`）按 `src.total()*elemsize`（=紧凑 `cstep*c`）
一次性 memcpy 到 16B 对齐 cstep 的 staging；pyncnn 零拷贝包 numpy 时把 `Mat.cstep` 设为紧凑
`H·W` → `c≥2 且 (H·W)%4≠0` 时平面 q 落 `q·hw` 而消费读 `q·cstep`，末平面尾字节从未写入。
内容无关、与算子/winograd/sgemm 无关。

### 3.2 修复

- fork：`ncnn_upstream/ncnn@eab8352`（基于上游 master `9f9d4ec`），单文件 `src/command.cpp`（+16/−1）。
- 补丁：`work/upstream_a5b/0001-fix-vulkan-record_upload-for-compact-layout-external.patch`。
- 构建：`work/upstream_a5b/FORK_BUILD.md`；子模块 `glslang@780b6a2`、`pybind11@d03662f`；
  产物 `.so` md5 `b0e53111dfe2a1cbbc6a8d56bae49947`。
- 官方轮（≤`1.0.20260526`）带此缺陷，**不得用于 Vulkan 后端**。

### 3.3 验证摘要

| 项 | 结果 | 报告 |
|---|---|---|
| 迷你网 Split 网格 | 补丁轮零损坏；stock 轮按 `c≥2 且 hw%4≠0` 损坏 | `work/upstream_a5b/micro/repro_split_pack4.py` |
| 相邻层网格 | Split/Crop 补丁轮 0/232 失配（stock 152/232）；Slice/Concat 被第二缺陷阻断 | `p0b/REPORT_adjacent.md` |
| 任务级（原尺寸） | P4 16/16 + mod4(1/2/3) 12/12；RX460 ≤1MP 9/9 | `p0b/a5b_*.jsonl` |
| 全网逐 blob rel_L2 | P4/RX460 各 4 尺寸 0 超包络、0 平移；stock 正对照 2286/2605 超包络 | `p0b/REPORT_relL2.md` |
| H2（download 对称风险） | validation layer 静默 + 源码判读无对称缺陷 | `p0b/REPORT_h2.md` |
| 确定性 | 同设备跨 fresh 进程逐位一致（P4 2.43MP / RX460 0.998MP） | `p0b/REPORT_p2a.md` |

### 3.4 性能（本机实测，单机不外推）

| 输入 | CPU(ORT) | ncnn-Vulkan P4 | ncnn-Vulkan RX460 | CUDA P4 |
|---|---|---|---|---|
| 512² | 2.65 s | 1.05 s | 1.97 s | 0.224 s |
| 714×967 | 7.4–8.3 s | 4.6–4.9 s | 9.3–11.5 s | — |
| 1264×1926 | 31–33 s | 35–37 s | 88 s | 2.8–3.8 s |

ncnn-Vulkan 只在 ≤0.7MP 优于 CPU，大图持平或更慢；ncnn-CPU 实测**慢于** ONNX Runtime CPU
（512² 5.40 s、0.69MP 11.4 s、2.43MP OOM），故不采用（`work/LOG.md` 决策条）。

### 3.5 后续优化阶段（2026-10-13，本机 RX560 4GiB；实现落在 下游应用，本仓只记录）

- 阶段0：关 winograd + 对 `softmax_361/362` 施加通用层参数 `31=16`（`src/net.cpp:115-132`，
  bit4 = 关闭该层 Vulkan，回退 CPU）。实测 1000×1721 25.47→23.15 s（−9.1%）、
  1264×1926 54.11→40.56 s（−25.0%），GTT 增量 4835→2131 MiB；out/conf rel_L2 相对原配置 ≤2.44e-05，
  跨 fresh 逐位一致。**但仍慢于同机 ORT-CPU（28.67 s）**。
- 闸门逐项：G1 通过（V0/a/b 三变体 × 192×192、384×512 逐 blob：0 越界、0 平移，skip 恒 46）、
  G2 通过、G3 通过、G5 通过（仅 GPU 侧权重×1.001 → 795/2651 blob 越界）、G6 通过；
  **G4 未执行**（按裁决延后到最终配置跑一次）、**G7（P4）待用户实测**。
- 阶段1 spike：**无需 rebase**（基线 `9f9d4ec` 已含 SDPA 及其 Vulkan 实现）；**0001 补丁仍需要**
  （上游对 `src/command.cpp` 的最新改动 `201145bd` 与之无关）；本机在 fp16 关闭时 SDPA 只能走
  `qk_cross/qkv_cross` 两遍路径（`src/layer/vulkan/sdpa_vulkan.cpp:143`），能否达目标线存疑。
- 详见 下游应用 `perf_opt/{GATES,STAGES}.md` 与 `perf_opt/raw/` 原始数据。

## 4. 复现

```bash
# 生成 ncnn 补丁轮（Linux x86_64 / CPython 3.11），见 FORK_BUILD.md
git apply work/upstream_a5b/0001-*.patch
cd <ncnn源码>/python && pip install .
# 产物 .so md5 应为 b0e53111dfe2a1cbbc6a8d56bae49947

# 迷你复现（1=Tesla P4, 0=RX460）
python work/upstream_a5b/micro/repro_split_pack4.py 1
python work/upstream_a5b/micro/repro_adjacent_pack4.py 1

# 逐 blob 逐元素 rel_L2（分块，每块 ~150 blob）
python work/pnnx_blob_relL2.py --size 714x967 --device 1 --range 1-150 --out X.tsv
```
详细步骤、解释器与环境见 `DEVELOPMENT.md`。

## 5. 已知限制（原样保留）

- **第二缺陷（未解释）**：迷你网中抽取 Slice/Concat/ReLU 的 GPU 输出必 SIGSEGV(139)，
  stock 轮同样崩、与 `(H·W)%4` 无关、机制未定位；真实 TruFor 全网 2651 blob × 4 尺寸抽取从不崩。
  **可能表现为进程崩溃，未排除。**
- **验证范围**：Tesla P4（NVIDIA 535.261.03）、Radeon RX460 2GiB（Mesa/RADV 22.3.6）；
  其他卡/驱动/Mesa 版本无数据。Mesa 22.3.6 正是早期 TVM 阶段出过静默错值的版本。
- 逐 blob rel_L2 的对照包络为**事后修订**判据，证明力低于事先固定判据。
- 早期 TVM-Vulkan 方案的失败与结论见 `docs/TVM-VULKAN-README.md`、`reports/`、`CHANGELOG.md`。

## 6. 许可

不对外分发。ncnn 本体 BSD-3（本仓库对其打补丁但不改许可）；TruFor 上游权重为
**非营利**许可且**不入库**（本仓库不含任何权重）。详见 `LICENSE-NOTICE.md`。

## 7. 继续阅读

- `work/STATE.md` / `work/LOG.md` —— 现行结论与完整决策链。
- `work/MIGRATION_SUMMARY.md` —— 阶段一核查逐条回答与证据索引。
- `work/upstream_a5b/FORK_BUILD.md` —— 补丁轮构建/分发。
- `work/upstream_a5b/p0b/REPORT_*.md` —— 各阶段验证报告。
- `DEVELOPMENT.md` —— 接手必读。


## 决策记录新增（2026-10-13，来自 perf_opt 长流程）

- **精度判据放宽**：链级输出 rel_L2 阈值由 1e-5 放宽为 1e-4（记为设定过严）；端到端为准。
- **SDPA 改写**：`scripts/param_rewrite_sdpa.py` 命中 130 条链；产物 `优化版模型资产`
  （sha256 `df7006cc…`）与 `模型权重` 兼容；SDPA 输出需 `Reshape 0=<embed> 1=-1` 降回 dims=2。
- **按尺寸路由**：生产阈值定为 **2.2 MP**（A5b 实测 2.3MP 起 GPU 掉慢稳态）⇒ `NCNN_MAX_MP_HIGH_AVAIL` 2.5 → 2.2。
- **W 裁决**：NetA 采纳 winograd=1、NetB 保持 0（A3 端到端通过；blob 级 1e-4 代理**无预测力**）。
- **双稳态与漂移**：M5@2.43MP 存在快（34.5s/gtt 1750MiB）与慢（39–40.5s/gtt ~3000MiB）两态，
  时钟/温度/负载均不是主因（机制**待验证**）；因此**不得**把任一单值当稳定事实。
- **N0 归档**：中间 blob 注入后 `extract` 崩溃原因未定位；可改处均在 ncnn 侧 ⇒ 低价值归档。
- **最终配置**：双 Net、**fp32 + SDPA**（fp16 已放弃）；NetA w-on → 取 blob 19 → 销毁 → NetB 注入（M5）。


## 决策记录（2026-10-13，块 E 补记；该仓无 STATE/LOG 文件，记为 README 附录）

1. **config 键白名单事故**：`perf_opt` 曾新增自定义键 `trufor.配置键`，导致生产入口
   `生产入口脚本` 报"未知的校准键"**直接拒绝启动**。修复：改用白名单内既有键
   `trufor.ncnn_max_megapixels`（置 0 等效禁用）。
   **教训：验收必须走真实 CLI 入口；绕过封装直接 import 无法发现此类问题。**
2. **阈值复核结果（E1）**：单图 CLI 端到端口径下 2.1/2.2MP 的 GPU 中位数（27.06 / 30.10 s）均慢于
   CPU 中位数×0.95（23.79 / 25.03 s）⇒ 阈值由 2.2MP **下调至 2.0MP**（config 与
   `NCNN_MAX_MP_HIGH_AVAIL` 同步）；warm 口径下 GPU 仍明显占优（15.96 / 17.41 s），两口径差异已入部署文档。
