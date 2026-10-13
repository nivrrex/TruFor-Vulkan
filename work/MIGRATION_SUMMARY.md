# 新对话迁移摘要（TruFor Vulkan 后端链 · A 仓 下游应用 + B 仓 TruFor_Vulkan_dev）

生成日期：2026-10-02。

| 仓库 | HEAD | 工作树 |
|---|---|---|
| B（本仓，TruFor_Vulkan_dev） | `25da713a38d399a28179ddab23e99be9ed417bd4` | 干净（`git status --short` 无输出，本轮实测） |
| A（下游应用） | `9b6627842487cd7fa4a42141f69702ed2a529519` | 干净（本轮实测） |
| ncnn fork 工作树（`ncnn_upstream/ncnn`，两仓之外） | `eab8352`（基于上游 master `9f9d4ec`，未 push） | 干净（本轮实测） |

本文档路径一律写作"相对各自仓库根"，A/B 前缀指明所属仓库。所有技术结论按
[已验证事实] / [工程判断] / [待验证假设] 三级标注；无证据的内容集中在第10节。

阶段一核查（C1–C13）的逐条回答见**附A**，证据索引见**附B**。

---

## 0. 项目强调要求

- **信息分级强制**：每条结论标 [已验证事实]（附 `文件:行号`/commit/命令输出）、
  [工程判断]（附置信度与依据类型）、或 [待验证假设]（不得写成确定性结论）。
  禁止编造来源、数值、commit、行号；查不到写"未找到"。信息不足时不给结论。
- **回答格式**（全局规范）：以"N先生，"开头，最后一行固定"以上回答已遵循所有设定
  规则。"；多步执行任务每轮首行重述当前状态。
- **改动纪律**：只改必须改的部分，不顺手重构、不格式化无关代码；本次修改产生的
  孤立 import/变量须清理，修改前已存在的死代码只指出不删；不擅自扩大任务边界；
  发现更优方案只指出，实施前需用户确认。
- **工具选型**：开源、成熟、低依赖、可本地编译优先；不重复实现已有成熟能力。
- **取证场景红线（C2）**：任何后端一律**原尺寸推理**——resize 到 512 会使篡改信号
  消失（[已验证事实] B `work/STATE.md:66-69`：16 样本×4 路径消融，大图 resize 到
  768/512 后 recall@FPR1% 全部归零）；不得静默降级；不得静默产出可能损坏的结果；
  后端实际使用情况与降级原因必须落报告（`backend_used`/`backend_reason`）。
- **准入红线**：ncnn 进入 auto 的条件为"机制性根因 + 扩展网格零失配 + A5b 原尺寸
  通过 + Windows 抽查"。**前三项有证据（第4节、附A/C3/C4），第四项无**；且本轮核查
  发现既有 A5b 重跑的 16 样本里只有 1 个尺寸（714×967，hw%4=2）真正走补丁新分支
  （附A/C3）。**[2026-10-03 更新] 用户已书面确认保留 ncnn 在 auto；P0-b ①–④ 已在
  P4+RX460 双设备补证完成（详见 B `work/upstream_a5b/p0b/REPORT_P0B.md`、
  `REPORT_rx460.md`、`REPORT_relL2.md`、`REPORT_adjacent.md`、`REPORT_h2.md`）。
  原文“待用户决策”状态结束；auto 链保持不变（cuda→ncnn→cpu）。**
- **B 仓自包含**：B 仓保持零"下游应用"代码依赖，资产只单向拷贝到 A 仓（B `work/STATE.md`
  的多处引用是文档性提及，非代码依赖）。
- **A 仓每步独立提交可回退**；改核心推理文件（`src/trufor_inprocess.py` 等）前先跑 R0
  golden（`src/tests/check_golden.py`）。
- **记录规范**：只记决策及否决替代、接口/数据结构/依赖/调用方式变更、未解决问题与
  已知限制、推翻此前结论的新信息、失败尝试及原因；不记过程性动作。
- **上游缺陷处理**：P1 及以上且无可接受绕过时本地 fork（本次即按此执行），补丁与上游
  代码分开提交，便于 rebase 后剔除。

## 1. 项目目标

**问题**：下游应用的 TruFor 推理原先只有 ORT-CPU 路径，需要 GPU 加速，覆盖
AMD/NVIDIA × Linux/Windows，且**不得**以 resize 到固定 512 为代价（取证语义红线）。

**验收标准**（数值口径为演进后的现行口径）：
- out：rel_L2 ≤ 1e-3（A2 门）。[已验证事实] 本轮 ncnn 实测 out rel_L2=5.053e-04。
- det：max_abs ≤ 2e-2（logit 尺度）且须落入已知离散模式（CUDA 侧 det 双模式，见附A/C10）。
- conf：max_abs 降为**记录项**，改用任务级口径 AUC/召回/阈值翻转评判（A5 评审决定）。
- A5b 任务级门：|ΔAUC| ≤ 2e-3、prob/conf 阈值翻转 < 0.1%。
- 同设备跨 fresh 进程输出逐位一致（确定性）；跨设备允许算术级差异。
- 显存不 OOM（分档守卫，见第7节）；性能判据：GPU 路径应显著优于 CPU（现状见第2节）。

## 2. 运行环境与约束

开发机：Devuan 5 + LXDE，sysvinit，**禁 systemd**；主机内存 16GB（全分辨率瓶颈，见
[已验证事实] B `work/STATE.md:145-146`：3072² 主机 OOM killer exit=137，非显存瓶颈）。

| 项 | 仓库实测值（本轮） |
|---|---|
| Python | 3.11.2 |
| ORT-CPU | onnxruntime 1.29.0（venv `ncnn_pypi_env`，与 A `src/requirements-cpu.txt` 钉版一致） |
| ORT-GPU | onnxruntime-gpu 1.23.2，`get_available_providers()` 返回 Tensorrt/CUDA/CPU（venv `ortgpu_env`）；cuDNN 9.5.1.17（A `src/requirements-gpu.txt`） |
| ncnn | stock pypi `1.0.20260526`（known-bad 参照，B `work/upstream_a5b/wheels/`）/ master `9f9d4ec` 构建 `1.0.20261002` / 补丁轮 `1.0.20261002+trufor1` |
| GPU | Tesla P4 8GB（Vulkan index 1）、Radeon RX460 2GiB（index 0）、llvmpipe（软件渲染，代码中排除） |
| 分发 | Linux 为主（目标机 NVIDIA+AMD）；Windows PyInstaller **未实证**；ncnn 本体 BSD-3 保留 |

[已验证事实] **版本串不可作判据**：本轮 `ncnn_pypi_env/bin/python -c "import ncnn;
print(ncnn.__version__)"` 打印 `1.0.20261002`（无 `+trufor1` 段），而 pip 元数据为
`1.0.20261002+trufor1`；故补丁与否只能靠行为判据或 pip 元数据区分
（同记于 B `work/upstream_a5b/FORK_BUILD.md:35-36`）。
[已验证事实] 当前 `ncnn_pypi_env` 内 `ncnn.cpython-311-*.so` md5 =
`b0e53111dfe2a1cbbc6a8d56bae49947` = fork 构建目录产物 = `FORK_BUILD.md:43` 记录值
（本轮实测三者一致）。

性能现状（[已验证事实]，均本机）：CUDA EP 512² 224ms（CPU ~2700ms）；1264×1926 native
2.8–3.8s（CPU native 33s）；ncnn-Vulkan 512² P4 1.04s / RX460 1.97s；ncnn 全分辨率上限
≈2.4–3MP，4MP 双设备主机 OOM。（数字出处：B `work/STATE.md:73,139-146`）

## 3. 当前结构

### A 仓（下游应用）

| 路径 | 行数 | 职责 |
|---|---|---|
| `src/trufor_inprocess.py` | 378 | 推理主入口 + 后端路由（cpu/cuda/ncnn/auto）+ session 缓存 + `_load_input_tensor`/`_postprocess` |
| `src/trufor_cuda.py` | 65 | CUDA EP 会话构建，providers[0] 假回退断言 |
| `src/封装层` | 127 | ncnn-Vulkan 封装层（私有副本、尺寸守卫、extract 错误码） |
| `src/trufor_backends.py` | 345 | 探测：`ensure_nvidia_libs`/`probe_vulkan`/`probe_ncnn_upload`/`probe_cuda`/`probe_report`、显存分档常量 |
| `src/生产入口脚本` | 1529 | CLI 与叠加管线，`run_trufor_inference` 落 `backend_used`/`backend_reason`、C2 缩图告警 |
| `src/另一入口脚本` | 1687 | CLI 参数校验（`--trufor-device` 白名单 :399） |
| `src/score_comparison.py` | 846 | 批量对比工具（同参数透传） |
| `src/config.json` | 31 | `trufor.device="auto"`、`trufor.ncnn_max_megapixels=null`、内存自适应参数 |
| `src/requirements-{cpu,gpu,ncnn}.txt` | 7/19/22 | 三清单；ncnn 清单强制自建 `+trufor1` 轮 |
| `src/tests/` | — | `test_trufor_routing.py`(636)、`test_trufor_cuda.py`(202)、`test_封装层`(161)、`check_golden.py`(52)、`cuda_numerics_check.py`(141)、`golden/{golden_cpu.json,ncnn_vulkan_192.json}` |
| `src/weights/` | — | `trufor.onnx`(279MB)、`模型资产`(159KB)、`模型权重`(275MB)；ncnn 两件由 B 侧资产拷入；`src/weights/` 整目录被 `.gitignore:3` 排除，三件均不入库（部署时外部放置） |
| `doc/部署方案摘要_v2.md` | — | 部署文档（〇.2 节=后端链现状） |

### B 仓（TruFor_Vulkan_dev）

| 路径 | 职责 |
|---|---|
| `work/STATE.md` | 现行结论台账（阶段A闸门、resize 消融、A5b、fork 修复、S6 落地） |
| `work/LOG.md` | 决策/推翻/失败尝试的长记录（fork 定案节在 :2250 起） |
| `work/TODO.md`、`work/PACKAGING-NOTES.md` | 待办与打包注记 |
| `work/pnnx_a{1..5}*.py`、`pnnx_gpu_blob_diff.py`、`resize_ablation.py`、`ort_cuda_crossproc.py` | 阶段 A/B 各闸门脚本（可原样复跑） |
| `work/ablation_results/` | 结果 jsonl/log/txt（**入库 18 件 / 144KB**，含
  `a5b_patched_native{,_dev0}.jsonl`、`runs.jsonl`——第 C3 节的证据可随仓追溯）；逐样本
  npz 与 `a5b_dumps/`（本地 564MB）由 `.gitignore:23` 排除，可脚本再生 |
| `work/upstream_a5b/FORK_BUILD.md` | fork 构建与分发链（含验证表、残留问题） |
| `work/upstream_a5b/0001-*.patch` | 单文件补丁导出（47 行） |
| `work/upstream_a5b/micro/` | 迷你网复现（`repro_split_pack4.py`、`relu_probe.py`、`conv_discrim.py`、各 `.param`、`ISSUE_draft.md`） |
| `work/upstream_a5b/wheels/` | stock + `+trufor1` 两个 wheel（**gitignore，未入库**，`.gitignore:25`） |
| `work/ncnn_a1/` | pnnx 产物 param/bin（1.2GB，gitignore） |

[已验证事实] **GitHub 上传阻碍（本轮新发现）**：B 仓 git 跟踪的 `reports/` 下有 5 个
275–280MB 的 onnx/mnn 文件超过 GitHub 单文件 100MB 硬限，`.git` 体积 1.3GB。B 仓拟
单独上传 GitHub 前必须先处理（本轮只登记，不改动）。

## 4. 已完成内容

### A 仓（阶段 B，每步一提交）

| 步 | commit | 改了什么 / 为什么 | 验证 |
|---|---|---|---|
| S0 | `bf496d5`、`022ff4c` | 基线快照 + ncnn 资产入 weights + 3 图 CPU golden md5 冻结 | 基线可追溯 |
| S1 | `dce87b4` | 纯搬移拆出 `_load_input_tensor`/`_postprocess` | R0 三图逐位一致 |
| S2 | `ba24515` | `trufor_backends.py` 探测模块 + `--trufor-probe` | 本机枚举=P4/RX460/llvmpipe |
| S3 | `3c8e34b` | CUDA 后端 + providers[0] 断言（防静默回退）+ 缓存重设计 | 假回退单测；512²/2.4MP vs CPU |
| S4 | `cd0b825` | `封装层` 封装层（UAF：结构性自持私有副本）+ 3 unittest | 布局对 ORT 达 A2 口径 |
| S5 | `4bdd4f4` | auto 路由（当时 cuda→cpu）+ backend_info 透传 + 显存分档 | 单测 |
| （旧终态判定，已被推翻） | `02b1360` | "A5b 判 FAIL → ncnn 不入 auto" | 被 fork 修复后重验推翻 |
| S7 | `42ad33b` | CLI 白名单 + 默认 auto + `ncnn_max_megapixels` 可空键 + C2 缩图告警 | 42 单测 + R0 |
| S8 | `bec82b8` | 三清单钉版 + onedir spec/build.sh（nvidia 收集开关）+ 冻结态预加载 | 两态 e2e 逐位（**当时用 stock 轮资产链路**） |
| **S6（现行）** | `935a4cb`、`5c9587f`、`d8fe152`、`9b66278` | auto 链改 **cuda→ncnn→cpu**（v0.5.4）+ 三关门控 + `probe_ncnn_upload` 行为判据 + 文档/CLI/依赖措辞 + 12 新单测 | 43 用例 OK、R0 GOLDEN=PASS（**本轮复跑**）、双环境判据 FAIL/PASS |

现行路由代码：`src/trufor_inprocess.py:306-346`（附A/C1）。

### B 仓（阶段 A 闸门 + fork 修复 F1–F5）

- 阶段 A：A1 转换、A2 CPU 数值参考、A3 Vulkan 数值、A4 性能/显存/主机内存、A5 任务级
  指标（B `2560a9d`、`4fc351a` 等）；resize 消融锁死 C2（`work/STATE.md:66-69`）。
- **F1 机制定案**（`2c36a22`，LOG :2250 起）：`VkCompute::record_upload`
  （上游 `src/command.cpp:414`）按 `src.total()*elemsize`（=紧凑 `cstep*c`）一次性
  memcpy 进 16B 对齐 cstep 的 staging；pyncnn 零拷贝包 numpy 时刻意设紧凑 cstep
  （`python/src/main.cpp:371-372`）→ `c≥2 且 (H·W)%4≠0` 时平面 q 落 `q·hw` 而消费读
  `q·cstep`，末平面尾字节从未写入。[已验证事实]
- **F2 补丁**：fork `eab8352`（单文件 +16/−1），导出 `work/upstream_a5b/0001-*.patch`
  （47 行）；修法=步长不等时**逐平面拷贝**（非取址修正、非改 Split）。
- **F3/F4 重验**：`work/ablation_results/a5b_patched_native.jsonl`（P4）与
  `_dev0.jsonl`（RX460）各 16 样本，均 16/16 过 A5b 门（附A/C3）；迷你网网格补丁轮下
  零损坏（本轮复跑确认，附A/C4）。
- **F5 发版链**：`+trufor1` wheel（10.5MB）产出与安装步骤、验证表、残留问题
  （`work/upstream_a5b/FORK_BUILD.md`）；stock 轮存档作 known-bad 参照。

## 5. 关键决策与理由（含"考虑过 Y，因 Z 放弃"）

1. **GPU 后端选 ncnn-Vulkan + CUDA EP 双路，放弃 TVM 内嵌（M2）**——考虑过 TVM
   （已有手写 kernel 资产），放弃原因：产物体积、仅 512 固定形状、Mesa 22.3.6 静默错值
   （源仓库记录、本机未复现）、冷启动缓存、P4 上可能 ≤1× 收益的外推。
   [工程判断，置信中高，依据=评审材料 + B `work/STATE.md:57-59`]
2. **一律原尺寸推理，放弃 resize/fixed-512 方案**——考虑过缩图（工程上最省事），
   放弃原因：C2 消融证明缩图后篡改信号消失（recall@FPR1% 归零）。[已验证事实]
3. **分块推理（tiling）搁置**——会改变取证语义与拼接边界，且 C2 教训表明尺寸操作有
   语义代价。[工程判断]
4. **conf 门限由 max_abs 改为 AUC/召回/阈值翻转口径**——理由：conf 的 max_abs 残差
   （1.2–4.3e-2）在任务级不构成判定偏移，逐元素门会误杀可用后端。[已验证事实：A5 评审
   + 本轮 conf rel_L2=1.077e-03 记录项]
5. **上游缺陷走本地 fork，不等待上游**——考虑过提交 issue 等修复，放弃原因：P1 级、
   阻断核心功能且无接受绕过（全局规范"上游缺陷处理"条）。
6. **规避路径全部否决**：`use_packing_layout=False`（实测 9/12 仍损坏，同样本同模式，
   [已验证事实] B `work/STATE.md:289-294`）、改 param（Split→Slice/Crop，需先证明无损）、
   padding 输入（引入 C2 同类边界风险）。
7. **补丁轮识别用行为判据而非版本号**——[已验证事实] 运行时版本串不含 local 段（第2节）。
8. **UAF 不改 ncnn 绑定层，改由封装层自持私有副本**——考虑过提 PR 改绑定层生命周期，
   放弃原因：绑定层改动影响面大、上游未必接受，而私有副本能结构性切断调用方生命周期
   义务。[已验证事实：`src/封装层:9-91,106,118`]
9. **explicit 后端不静默降级**——取证可审计性优先；explicit 风险自担且契约不变
   （代价见第8节"explicit 可能静默出错图"）。
10. **默认后端切 auto**——用户批准（阶段 B 决策 2）；**当前争议**：ncnn 是否应留在
    auto 链内尚未定（准入第四项 Windows 抽查缺失 + A5b 补丁分支实际覆盖仅 1 尺寸），
    见第9节 P0 闸门。
11. **CUDA det 采用"落已知模式 + max_abs≤2e-2"双条件门**，不追求逐位——见附A/C10。

## 6. 已尝试但失败的方案

| 尝试 | 失败原因 | 证据 |
|---|---|---|
| TVM/Vulkan 内嵌（M2） | C2/C3/C5 无低成本解，仅解 C4 不改变决策 | B `work/STATE.md:57` |
| pnnx 默认 fp16 权重 | 约 13% 误差，不可用于取证数值口径 | B 阶段 A 记录 |
| "GEMM 是分歧起点"归因 | 误报，后被推翻 | B LOG 归因撤回条 |
| "winograd 致跨进程非确定"归因 | 测试脚本协议混杂；真根因是 python 绑定零拷贝 UAF | B `work/STATE.md:111-119` |
| "Swin encoder 某算子/23×31 latent 窗口分支"归因 | 真根因是上传步长截断，与算子无关 | B `work/STATE.md:241-243`（旧）、`:249`起（改判） |
| "conf 互差 0.92" 双模式归因 | 全套件证据下未再现 | B `work/STATE.md:176` |
| 关 packing 规避 | 9/12 仍损坏（同模式） | B `work/STATE.md:289-294` |
| Net 存活时 `destroy_gpu_instance` | teardown SIGSEGV(139) | B `work/STATE.md:151` |
| S8 首版无条件 glob 收 nvidia 库 | 把系统 python 的 cu13 残留 2.7GB 打进 CPU 包（产物 2.9GB），改为环境检测+开关 | B `work/LOG.md:2227-2228` |
| 缩图告警走 `_err()` | `_err` 抛 `ForensicsError` 会中断推理（A `src/生产入口脚本:385-392`），改直接 stderr | A `src/生产入口脚本:842` |
| 上游 PR #6954 补丁 | 不适用本机问题，已备份回退 | B `work/LOG.md:2166` |

## 7. 接口定义（以代码实签名为准）

```python
# A src/trufor_inprocess.py:238
def infer_trufor(image_path: str, weights_path: str, device: str = "cpu",
                 debug: bool = False, ncnn_max_megapixels=None) -> dict
# 返回 dict: {"map", "conf", "score", "backend_info"}
# backend_info: backend_used("cpu"/"cuda"/"ncnn") / gpu_index / requested / fallback_reason
```

- `device` 取值：`cpu`（永不探测，回归红线）、`cuda`/`cuda:N`（探测不过抛
  RuntimeError，不静默降级）、`ncnn`/`ncnn:N`（**不跑上传判据**，仅 probe_vulkan +
  尺寸守卫，`src/trufor_inprocess.py:165-177,298-305`）、`auto`（cuda→ncnn→cpu，
  `:306-346`）。
- **auto 三关**：①`probe_vulkan()` ok；②`probe_ncnn_upload()` ok；③本图
  `H*W ≤ min(显存分档, ncnn_max_megapixels override)`。ncnn 入选时
  `fallback_reason=None`（不记为降级）；仅落 CPU 才拼
  `"CUDA不可用: <r>; ncnn跳过: <r>"` 写 `report.backend_reason`。
- report/overlay 字段：`result["backend_used"]`、`result["backend_reason"]`、
  `result["trufor_input"]["forensic_warning"]`（C2 缩图告警，
  `src/生产入口脚本:833-841`）；降级原因键 `trufor_backend_fallback` 由调用侧落报。
- config：`trufor.device`（默认 `"auto"`）、`trufor.ncnn_max_megapixels`（`null`=按分档，
  数值只收紧不放宽）；CLI `--trufor-device` 白名单 `("cpu","auto","cuda","ncnn")`
  （`src/另一入口脚本:399`），`--trufor-ncnn-max-mp`。
- 探测函数：`probe_vulkan()` → `{ok,reason,devices:[{index,name,vram_gib,max_megapixels}],
  selected_index}`；`probe_ncnn_upload()` → `{ok,reason,device_index}`；
  `probe_cuda(weights_onnx)` → `{ok,reason,ort_version,providers}`；
  `probe_report(weights_onnx)` → `{cuda,vulkan,upload}`。结果进程内缓存，非可重入
  `_lock`，公共包装 + `_unlocked` 内核。
- 显存分档常量：`NCNN_MAX_MP_HIGHVRAM=2.4`、`NCNN_MAX_MP_LOWPVRAM=1.0`、
  `HIGHVRAM_GIB=6.0`（`src/trufor_backends.py:37-39,199-200`）；llvmpipe/swiftshader 排除。
- ncnn 封装层契约：`run(x)` 内 `blob = np.ascontiguousarray(...)` → `ncnn.Mat(blob)` →
  全部 extract 完成后才 `del ex, mat, blob`（`src/封装层:87-118`）；
  超 max_mp 与 extract 非零返回码抛 RuntimeError；**禁止**调 `destroy_gpu_instance`。
- 缓存键：`(kind, weights_abs, gpu_index, ncnn_max_mp)`，上限 2 实例；
  weights 命名 `weights/trufor.onnx`（cpu/cuda）、`weights/模型资产/.bin`（ncnn）。

## 8. 待解决问题 / 已知边界

按严重度排序（含附A 中所有"未做/未定/未找到"）：

1. **准入证据缺口（P0）——[2026-10-03 已闭合，见第0节更新]**：
   - A5b 补丁轮 16 样本中仅 714×967（hw%4=2，3 个比例）真正走补丁新分支，其余 15
     样本 hw%4==0 走原路径（附A/C3）。
   - 补丁轮下**全网逐 blob 回归**（2651 blob）未做；mini 网格已零失配但未覆盖
     Slice/Concat/Crop 相邻层（附A/C4）。
   - download 路径对称风险（审阅方称 H2）**未验证**：无 `allocsize/cstep/elempack`
     日志、无 Vulkan validation layer 运行记录（附A/C5）。
   - Windows 抽查未做。
2. **explicit `device=ncnn` 不跑上传判据** → stock 轮下 explicit 可静默产出损坏图
   （代码事实：`src/trufor_inprocess.py:165-177`）。文档已写明风险，但无拦截。
3. **probe_ncnn_upload 覆盖范围窄**：单点 c=2/3×3、仅 Split、仅 `selected_index`
   设备、无权重路径（附A/C9）。
4. **迷你网第二缺陷**：Input+ReLU 与 Input+Split+Conv 抽实算层输出 SIGSEGV，补丁轮下
   本轮复跑仍崩（`relu_probe.py`：in0 与 Split 别名正常，`extract("3")` exit 139）。
   与上传缺陷**现象独立**（修复后仍崩），**机制是否同源未定**（附A/C6）。
5. **未解释差异**：全网同平面形状 2651 blob 抽取不崩，迷你网必崩——无解释（附A/C7）。
6. **CUDA 侧**：部署文档未写 det 双模式口径；`src/trufor_cuda.py:7-8` docstring 仍
   声称三输出逐位一致（与更正矛盾）；CUDA ≥2MP 跨进程逐位未测（数据仅 512² 11 样本）
   （附A/C10）。
7. **补丁轮下的确定性与显存**：≥2MP 同设备跨 fresh 进程逐位一致的**证据文件未定位**；
   显存分档未在补丁轮下重测（沿用 A4 + "补丁不改分配尺寸"论断）（附A/C8）。
8. **conf 残差 1.2–4.3e-2**（backbone 末段 latent 2529 起）——已降级为记录项，未定位。
9. **冻结包/Windows 未实证**：S8 两态 e2e 基于 stock 轮资产链路；`+trufor1` 轮下
   冻结包 e2e 未做（B `work/STATE.md:337-338`）。
10. **RX460 2GiB 上限**：同进程连跑 3×2.4MP 后第 4 样本 `vkQueueSubmit -4`
    （amdgpu CS OOM），生产守卫本就拒该组合（B `work/STATE.md:308-310`）。
11. **文档旧表述清单（附A/C11）**：A `src/trufor_cuda.py:7-8`（真矛盾，未改写）、
    B `work/STATE.md:244-247`"auto=cuda→cpu 为终态"与 `:241-243` 边界归因（下游推翻，
    未就地标注）。
12. **ISSUE_draft 待清理**：`:114` "Local notes (not for posting)" 含本地路径与 fork
    commit，提交前须删；`:76` 明确 image/flatten 同类假设未验证；`:106` 的 Input+ReLU
    本轮已实证。是否提交上游由用户决定。
13. **auto max-size 缩图问题**：内存自适应在小内存机上可把输入缩到 600–1000，落入
    C2"静默退化"区间，现以 `forensic_warning` 显式告警而非禁用。
14. **B 仓 GitHub 上传阻碍**：5 个跟踪文件 >100MB、`.git` 1.3GB（第3节）。
15. **License**：上游权重 GRIP-UNINA "仅限非营利用途" 阻塞对外分发（不阻塞开发），
    待法务（B `work/STATE.md:61`；A `doc/部署方案摘要_v2.md:56-57,294`）。

## 9. 下一步计划（接手 AI 用；每项=目标/动作/验收/止损/是否需用户确认）

**P0-a 先对齐事实**
- 目标：确认本摘要与仓库实况一致。动作：核对两仓 HEAD（第0表）、复跑
  `src/tests` 三套 + `check_golden.py`。验收：43 OK + GOLDEN=PASS。
  止损：任一不一致 → 只报告不改文档。不需用户确认。

**P0-b auto 链去留闸门（最高优先，阻塞 ncnn 生产放量）**
- 目标：判定"准入四条件"是否成立。动作（均为已有脚本原样复跑或最小扩展）：
  ①`work/pnnx_a5b_native.py` 以补丁轮对**hw%4≠0 尺寸**（至少补 2–3 个非方形尺寸，
  如 967×714、715×967、3×5 系列放大）跑任务级；②全网逐 blob 回归
  `work/pnnx_gpu_blob_diff.py` 在补丁轮下重跑；③download/上传路径的
  `allocsize/cstep/elempack` 元数据日志或 VK validation layer 检查；④Slice/Concat/Crop
  相邻层的迷你网格扩展。
  验收：|ΔAUC|≤2e-3 且翻转<0.1%；网格与逐 blob 零失配；无越界/验证层报错。
  止损：任一不过 → 报告并**建议 ncnn 退回 explicit**（[工程判断] 置信中）。
  **[2026-10-03 结果] ①–④ 全部通过（② 为判据事后修订；④ Slice/Concat 被第二缺陷
  阻断，经用户裁决以 ② 全网替代；Windows 为书面缺项）。用户确认保留 auto。**

**P1-a explicit 先跑判据**
- 目标：堵住"explicit 静默出错图"。动作：`_get_session(kind="ncnn")` 内在构造 runner
  前调 `probe_ncnn_upload()`，FAIL 抛 RuntimeError。验收：新增单测（stock 轮 mock FAIL
  → 抛错；补丁轮 mock PASS → 正常）。止损：判据本身抛错时按 ok=False 处理（auto 契约）。
  **需用户确认**（属行为契约变更：explicit 从"风险自担"变为"有门"）。

**P1-b 路由回归单测补一条**
- 目标：把当前 auto 决策冻结成回归用例（mock 三探测全成功 → 断言选中后端与决策一致，
  无论 ncnn 在不在链内）。验收：`src/tests/test_trufor_routing.py` 全绿。不需确认。

**P1-c Input+ReLU 同源判定**
- 目标：判定第二缺陷与上传缺陷是否同源。动作：以 `micro/relu_probe.py`（本轮已复跑，
  `extract("3")` 崩）为基础做最小变量实验（去掉 Split / 换 pack1 / 换 CPU 路径对比）。
  验收：给出机制解释或明确"独立缺陷"。止损：3 轮无进展即记入第8节挂起。不需确认。

**P1-d 文档去矛盾 + ISSUE 草稿清理**
- 目标：消除附A/C11 列出的旧表述。动作：改写 A `src/trufor_cuda.py:7-8`（det 双模式
  口径）、在 B `work/STATE.md:241-247` 就地加作废指向；删除 `ISSUE_draft.md:114` 本地注记
  行，提交前先搜上游是否已有同类 issue。验收：grep 不再出现未标注的"三输出逐位一致/
  终态"表述。需用户确认（是否提交上游由用户决定）。

**P2-a 补丁轮确定性与显存重测**：RX460/P4 各 ≥2MP 双 fresh 进程 md5 比对 + 分档实测；
验收：逐位一致 + OOM 边界与分档一致。
**P2-b CUDA det 口径入文档**：部署文档写明"逐位比对只比 out/conf，或对 score 舍入"。
**P2-c Windows 与冻结包 e2e**：等目标机清单。

**暂缓（各有明确前置）**：dual-package 出包（待 auto 链去留定）；CUDA 4GB 捆库策略
（待目标机清单）；DirectML（仅当 Windows 非 NVIDIA 占比可观）；ncnn 上游提交（用户决定）；
License 法务（阻塞分发不阻塞开发）。

**需用户提供**：目标机 OS/GPU 分布、日处理量/SLA、ncnn 在 auto 的最终决定、上游提交时点。

## 10. 未核实信息（无来源或本轮未复跑）

- **H1 cstep 对齐假设 / H2 download 缓冲越界假设**：两标签在 A/B 两仓文档中
  grep 无命中（"H1"/"H2" 未登记），来源为对话与外部审阅 → 全文未核实；本轮亦未做
  validation layer 检查。
- cuBLAS 算法启发式导致 CUDA det 双模式——B `work/STATE.md:172-173` 自身即标
  【待验证假设】。
- pnnx 生成图/ncnn 上游是否已知本缺陷（上游 issue 检索**未做**）。
- Windows 侧一切行为（构建、PyInstaller、驱动、Mesa/RADV 对应物）——未实证。
- Conv 1×1 判别实验结论：仓库有 `micro/conv_discrim.py`/`.param`/`.bin`，但**无运行日志
  与结论记录**，本轮未复跑 → 未核实。
- 各性能数字（224ms/1.04s/1.97s/2.8–3.8s 等）跨机器外推——本机实测，外推未核实。
- "补丁轮下同设备跨 fresh 进程逐位一致"（`FORK_BUILD.md` 验证表声明"抽测"）：本轮
  未找到对应证据文件（附A/C8）→ 暂记为未核实。
- Mesa 22.3.6 静默错值：源仓库记录、本机未复现（B `work/STATE.md:59`）。
- 迷你网 SIGSEGV 的根因：完全未定位。
- "全网 2605 个 blob"：仓库实际记录 2651 total，2605 = 两侧 TSV 实际行数（2026-10-03 更正，附A/C7）。

---

## 附A：阶段一核查 C1–C13 逐条回答

**C1 auto 链现状冲突**——[已验证事实] 当前 HEAD 的链路是 **cuda→ncnn→cpu**，代码
A `src/trufor_inprocess.py:306-346`（本轮直接读取确认）。改变它的提交：A `935a4cb`
（v0.5.4，阶段 B-S6）。被取代的旧表述"auto=cuda→cpu、ncnn 不入 auto"出自 A
`4bdd4f4`/`02b1360`，其依据（A5b 判 FAIL）已被 B 侧 fork 修复 + 补丁轮 A5b 重验
16/16 PASS 推翻（B `work/STATE.md:297-315,316-338`、`work/ablation_results/a5b_patched_native*.jsonl`）。
但"准入门已满足"这一 B 侧结论本身存在覆盖缺口（见 C3/C4/C5 与第9节 P0-b）。

**C2 F3–F5 全文**——机制性根因见第4节 F1（[已验证事实]，B `work/LOG.md:2250` 起、
`work/upstream_a5b/FORK_BUILD.md:7-14`、`micro/ISSUE_draft.md:37-51`）。补丁
= `work/upstream_a5b/0001-fix-vulkan-record_upload-for-compact-layout-external.patch`，
47 行，改 `src/command.cpp` 一处，+16/−1（本轮 `wc -l` + `git show --stat eab8352` 实测）。
所选修法：**步长不等时逐平面拷贝**（不是取址修正，不是改 Split 行为）。
放弃的其他修法记录：仓库只登记了**规避层**替代方案（关 packing 实测证伪、改 param
Split→Slice/Crop 需证无损、padding 引入 C2 风险否决）；**补丁层**的备选（改绑定层
cstep、上传前先做对齐副本、只提文档警告）无专门取舍记录 → 未找到。

**C3 补丁轮下 A5b 原尺寸重跑**——[已验证事实] 两份结果各 16 样本
（`ablation_results/a5b_patched_native.jsonl` = P4、`_dev0.jsonl` = RX460；汇总
`a5b_summary_patched_dev1.txt`/`_dev0.txt`）：

| 指标 | P4(dev1) | RX460(dev0) | 门 |
|---|---|---|---|
| \|ΔAUC\| max | 5.6e-4 | 6.6e-4 | ≤2e-3 ✅ |
| prob 翻转@0.5 max | 3.0e-4 | 3.1e-4 | <1e-3 ✅ |
| prob 翻转@thr99 max | 1.0e-4 | 9e-5 | <1e-3 ✅ |
| conf 翻转@0.5 max | 2.8e-4 | 2.9e-4 | 记录项 |
| Δdet score max\|.\| | 6.3e-3 | 6.3e-3 | 记录项 |
| conf max_abs | 1.37e-2~4.26e-2 | 1.40e-2~4.30e-2 | 记录项 |

尺寸与 hw%4（本轮由 jsonl 的 size 字段实算）：1264×1926(0)、1190×1684(0)、1000×1721(0)、
714×967(**2**)、380×540(0)、512×512(0)；16 样本=6 尺寸×比例。**关键缺口**：只有
714×967 的 3 个样本真正触发补丁新分支，其余 15 个 hw%4==0 走原路径，故任务级证据对
"补丁是否修对所有非对齐尺寸"的实际覆盖 = 1 个尺寸。

**C4 扩展尺寸网格**——[已验证事实] 迷你网网格 c=1..8 × 17 尺寸（含奇数 1×1/1×3/3×1/
3×3/3×5/5×3，含 967×714(hw%4=2) 与 967×712(hw%4=0) 对照），本轮在 `+trufor1` 轮下
原样复跑 `work/upstream_a5b/micro/repro_split_pack4.py 1`（P4）：有序值 c=2 3×3 读回
= 1..18 干净；77 个"预期损坏"组合**全部不损坏**（脚本以 RULE-MISMATCH 计数，mism=77
即零损坏，因脚本判据写于修复前）；967×714 max_abs=0；CPU 路径净；exit 0。
**未覆盖**：仅 `Input→Split` 一种层结构，**Slice/Concat/Crop 相邻层未测**；补丁轮下
全网逐 blob（nb=2651）回归**未做**（既有 2651 记录是修复前定位用，`upstream_a5b/gpu_chunks.log`）。

**C5 H2 验证**——**原记“未做”，2026-10-03 P0-b③ 已补做**。[更正] 原文“全仓 grep
`validation`/`VK_LAYER_KHRONOS` 均无命中”**不准确**：B 仓 `README.md:32,64`、
`CHANGELOG.md:54,64,215,225`、`reports/PHASE8C-DIAGNOSIS.md`、`reports/FINAL-REPORT.md`、
`upstream/ISSUE-take-select.md` 均有命中，但属 **TVM 阶段**（Mesa 22.3.6/25.0.7 A/B，
结论 “silent”），与 ncnn 无关；`allocsize` 确无命中。**“ncnn 补丁轮未跑验证层”**这一
实质结论在原文时点成立。P0-b③ 已补：`VK_LAYER_KHRONOS_validation` 经 `VK_LOADER_DEBUG`
证实加载，对 buffer 上传/下载路径静默（0 VUID）；并源码判读 download 无对称缺陷
（见 B `work/upstream_a5b/p0b/REPORT_h2.md`；**未做** fork 插桩元数据日志）。
H1/H2 标签本身仍不在仓库文档中（属对话与外部审阅来源）。

**C6 Input+ReLU 迷你网 SIGSEGV**——[已验证事实] 本轮在补丁轮下复跑
`micro/relu_probe.py`（脚本硬编码 device 1=P4）：`extract("in0")` 与 `extract("1")`
（Split 别名）正常返回 1..18，`extract("3")`（ReLU 输出）→ **exit 139 SIGSEGV**。
是否同源：上传缺陷在本轮网格已证明被修（C4），而崩溃仍在 → **现象独立**；机制是否
同源**未定**（无定位证据）。STATE 原文记录的补丁后重测对象是 Input+Split+Conv
（`work/STATE.md:281-283`），Input+ReLU 的补丁后重测**此前无存档日志**，本轮补齐。

**C7 "全网不崩、迷你网崩"**——[已验证事实] 全网逐 blob 抽取数量在仓库中记录为
**nb=2651**（`upstream_a5b/gpu_chunks.log:2,6,10,14`）。[2026-10-03 更正] “2605”的出处
已找到 = 两侧 TSV（`upstream_a5b/cpu_714x967.tsv`、`gpu_714x967.tsv`）的**实际行数
2605**（total blob=2651，46 个索引查名失败未记录），原文“未找到 2605”作废。对该差异仍**无任何
解释**：`work/STATE.md:283`、`work/LOG.md:2265`、`ISSUE_draft.md:106` 均记为"未定位/
挂起"→ 登记为未解释差异。

**C8 补丁轮的确定性与显存**——确定性：`FORK_BUILD.md` 验证表声明"同设备跨 fresh 进程
逐位 PASS（P4/RX460 抽测均逐位一致）"，但本轮在 a5b 日志/jsonl 中**未找到对应 md5 证据
字段**；两份 jsonl 是**跨设备**两份（dev1 头注明 Tesla P4，dev0 由日志枚举与耗时推断为
RX460），不是同设备双跑 → ≥2MP 同设备跨进程逐位：证据未定位（待补）。可比照的**修复前**
基线是 A4 记录（B `work/STATE.md:148-149`：P4/RX460 各 2.4MP fresh 进程 conf_md5
698ec932f420 / 43f83a024fc9 逐位一致），补丁轮下未重跑该指纹比对。跨设备非逐位一致
（prob rel_L2 3e-5~3e-4、conf max_abs ≤9.5e-3）为 [已验证事实]。显存：补丁不改分配尺寸、
A4 数值沿用 [工程判断，置信高，依据=补丁只改上传阶段的 memcpy 拆分，不动 create_like
尺寸计算]；分档表
（<3GiB→1.0MP、≥6GiB→2.4MP、>2.4 一律拒）**未在补丁轮下重测**，RX460 2GiB 仅有
"连跑 3×2.4MP 后第 4 样本 `vkQueueSubmit -4`"记录（`work/STATE.md:308-310`）。

**C9 probe_ncnn_upload 判据范围**——[已验证事实] 判据 = `load_param_mem` 3 行
`Input→Split` + c=2、3×3 有序值 + `extract("1")` 逐位比较（A
`src/trufor_backends.py:225-296`）。**不覆盖**：c≥3 的其它通道数（仅由 B 侧网格实验
间接覆盖，判据本身只测 1 点）、其它尺寸、任何实算层（Conv/Gemm/ReLU/Concat/Slice/Crop）、
`image` 上传路径（`command.cpp:614`）、flatten 路径（2653/2692）、download 路径、
多 GPU 时的**非 selected 设备**（`:267` 只测 `selected_index`）、带真实权重的路径。
**explicit 是否跑判据**：[已验证事实] **不跑**——`src/trufor_inprocess.py:165-177`
的 `kind=="ncnn"` 分支只调 `probe_vulkan` 与尺寸守卫，与汇报一致。

**C10 CUDA 侧**——[已验证事实] det 双模式（`94b5d5099968`×8、`18757f258295`×3，
Δlogit≈1.1e-7，Δscore≈3e-8）已在 B `work/STATE.md:170-174` 登记，并已在 A
`src/tests/cuda_numerics_check.py:33-36,50-58` 落成"det max_abs≤2e-2 + 落已知模式"门；
**未写进部署文档**（`doc/部署方案摘要_v2.md` grep "双模式" 无命中）。跨进程一致性数据
范围：仅 512²（`work/ort_cuda_crossproc.py`，原 2 + 新 9 = 11 样本，out/conf 逐位、
det 双模式）；≥2MP 的 CUDA 跨进程逐位**未测**。另 A `src/trufor_cuda.py:7-8` docstring
仍写"三输出跨 fresh 进程逐位一致（含 det）"——与更正矛盾。

**C11 文档一致性审计**（只列不改）：

| 位置 | 旧表述 | 被什么取代 | 是否已改写 |
|---|---|---|---|
| A `src/trufor_cuda.py:7-8` | CUDA 三输出跨进程逐位一致（det 94b5d5099968） | B `STATE:170-174` det 双模式 | **否（真矛盾）** |
| B `work/STATE.md:244-247` | "A5b 不过 → ncnn 不进 auto；下游应用 auto=cuda→cpu 为终态" | `STATE:311-315` + A v0.5.4 | 否（下游推翻，无就地标注） |
| B `work/STATE.md:241-243` | 边界"疑与 23×31 latent 窗口/对齐分支相关" | hw%4 规则（`STATE:268-272`） | 否（仍挂待验证假设） |
| B `work/STATE.md:247` | 最小复现=param+bin+b22 | `STATE:287-288` 作废（3 行 param + 随机张量） | 是（已就地作废） |
| B `work/STATE.md:289-295` | 关 packing 规避可行 → "准入维持 explicit" | `STATE:294-295` 括注作废（补丁即根治） | 是 |
| B `work/LOG.md:2265` / `ISSUE_draft.md:106` | Input+ReLU 崩溃"补丁后依旧" | 本轮复跑实证（C6） | 原文即待验证 → 现可升级为已验证 |
| A `doc/部署方案摘要_v2.md:83`、A `src/生产入口脚本:26,253` | "ncnn 不入 auto" | v0.10.2/v0.5.4 | 是（已标"当时/历史注记"） |
| "版本串同为 1.0.20260526" | 未找到该表述 | 实况：stock pip 元数据 1.0.20260526，补丁轮运行时打印 1.0.20261002（本轮实测） | N/A |
| "崩溃需反序抽取" | 现表述为"先取 Split 再取 in0"（`STATE:280-281`） | 本轮 `relu_probe.py` 顺序抽取即崩（C6） | 部分（草稿已改，STATE 未改） |
| "互差 0.92 归因"/"winograd 归因"/"Swin 归因" | 均已撤回 | `STATE:111-119/249`、`:176` | 是 |

矛盾条目合计 **3 条未改写**（前两行 + STATE:241-243），其余为已标注历史。

**C12 测试与冻结包**——[已验证事实] 本轮复跑（`ncnn_pypi_env`，`+trufor1` 轮）：
`Ran 43 tests ... OK`（14.2s），ncnn 数值记录 out rel_L2=5.053e-04 / max_abs=1.458e-03、
conf rel_L2=1.077e-03 / max_abs=1.438e-02（记录）、det rel_L2=2.284e-04 /
max_abs=3.151e-04；`tests/check_golden.py` → ctrl512/b22/b5 三图 **GOLDEN=PASS**。
冻结包 e2e 在补丁轮 ncnn 下**未做**（`work/STATE.md:337-338` 明示；S8 两态当时基于
stock 轮资产链路）。

**C13 License / 合规**——[已验证事实] ncnn 本体 BSD-3 未改（补丁不动 LICENSE，
`FORK_BUILD.md:59`；A `src/requirements-ncnn.txt:13` 同记）。上游 TruFor 权重
"仅限非营利用途"：B `work/STATE.md:61`（优先级高于一切性能决策，待法务）、A
`doc/部署方案摘要_v2.md:56-57,294`。`ISSUE_draft.md` grep：无本机绝对路径（用户主目录
前缀无命中）、
无权重/内部图片数据（仅尺寸与统计数字）、`:96` 点名 TruFor（公开模型名）；`:114` 为
"Local notes (not for posting)"，含本地相对路径与 fork commit hash，**提交前须删**。
另 B `work/STATE.md:57` 正文含一条本机绝对路径（指向仓外评审材料），属内部台账，
不影响分发。

## 附B：证据索引表

| 结论 | 证据 |
|---|---|
| auto=cuda→ncnn→cpu（现行） | A `src/trufor_inprocess.py:306-346`；commit `935a4cb`；本轮读取 |
| explicit ncnn 不跑上传判据 | A `src/trufor_inprocess.py:165-177`（`kind=="ncnn"` 分支） |
| 判据实现与只测 selected 设备 | A `src/trufor_backends.py:233-296`，`:267` |
| 显存分档常量 | A `src/trufor_backends.py:37-39,199-200` |
| UAF 私有副本契约 | A `src/封装层:9,87-106,118` |
| 43 用例 OK + R0 PASS | 本轮 `python -m unittest tests.test_trufor_routing tests.test_trufor_cuda tests.test_trufor_ncnn` / `python tests/check_golden.py` |
| 版本串不可作判据 | 本轮 `import ncnn; ncnn.__version__` → `1.0.20261002`（+trufor1 轮）；B `FORK_BUILD.md:35-36` |
| 已装 .so = fork 构建 | 本轮 md5 `b0e53111dfe2a1cbbc6a8d56bae49947`（两处一致）；B `FORK_BUILD.md:43` |
| 补丁 47 行 / +16−1 | `wc -l work/upstream_a5b/0001-*.patch`；`git show --stat eab8352` |
| 根因机制 | B `work/LOG.md:2250` 起；`work/STATE.md:268-272`；`micro/ISSUE_draft.md:37-51` |
| A5b 补丁轮 16/16 | `work/ablation_results/a5b_patched_native{,_dev0}.jsonl`（各 16 行）、`a5b_summary_patched_dev{0,1}.txt` |
| 16 样本 hw%4 分布 | 本轮由 jsonl `size` 字段实算（714×967=2，其余=0） |
| 迷你网格补丁轮零失配 | 本轮复跑 `micro/repro_split_pack4.py 1`（mism=77=全净，exit 0） |
| Input+ReLU 补丁轮仍崩 | 本轮复跑 `micro/relu_probe.py`（`extract("3")` exit 139） |
| 全网逐 blob 数=2651 | `work/upstream_a5b/gpu_chunks.log:2,6,10,14` |
| resize 毁检出（C2） | `work/STATE.md:66-69`（`work/ablation_results/runs.jsonl`） |
| det 双模式与门 | `work/STATE.md:170-174`；A `src/tests/cuda_numerics_check.py:33-36,50-58` |
| RX460 2GiB CS OOM | `work/STATE.md:308-310` |
| S8 cu13 残留 2.7GB 失败 | `work/LOG.md:2227-2228` |
| `_err` 抛异常不可用于告警 | A `src/生产入口脚本:385-392,842` |
| B 仓 >100MB 跟踪文件 | 本轮 `git ls-files` + stat（reports/ 下 5 个 275–280MB） |
| wheels 不入库 | `.gitignore:25`；`git ls-files work/upstream_a5b/wheels` → 0 |
| 权重非营利 License | `work/STATE.md:61`；A `doc/部署方案摘要_v2.md:56-57,294` |
| A 仓硬编码 B 仓样本路径（仅登记，P1-d 未改） | A `src/tests/check_golden.py:21`（有 `TRUFOR_TEST_SAMPLES` 兜底）、A `src/tests/test_trufor_cuda.py:179` |
