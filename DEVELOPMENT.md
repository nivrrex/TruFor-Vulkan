# 开发与接手说明（面向后续开发者 / AI）

本仓库是 **TruFor-Vulkan 推理后端的工程与研究记录**。产品/管线侧（"下游应用"）代码不在此仓库，
本仓库只回答"后端如何跑对、缺陷是什么、修了什么、证据在哪"。

先读 `README.md`（结论摘要）与 `AGENTS.md`（红线），再按本文件操作。

---

## 1. 环境

Python 3.11.2，Devuan 5（sysvinit，无 systemd）。本机使用的 venv（换机需重建）：

| venv | 内容 | 用途 |
|---|---|---|
| `/root/ncnn_pypi_env` | onnxruntime 1.29.0(CPU) + 补丁轮 ncnn + numpy 2.4.6 + opencv | ncnn 实验、逐 blob 仪器、A5b 任务级 |
| `/root/ortgpu_env` | onnxruntime-gpu 1.23.2 + nvidia-*-cu12（cuDNN 9.5.1.17） | 与 CUDA 对照 / CUDA 侧复检 |
| `/root/stock_env` | 归档 stock 轮 ncnn（known-bad，`.so` md5 `869ce77910292c65f76cff16fa199c44`） | 仪器灵敏度正对照 |

关键点：
- ncnn 补丁轮 `.so` md5 应为 `b0e53111dfe2a1cbbc6a8d56bae49947`；运行时 `ncnn.__version__`
  **不体现补丁**（仍打印上游串），只能用 `.so` md5、pip 元数据 `+trufor1` 或行为判据区分。
- 用 ortgpu 环境跑 CUDA 时须把 venv 内 `nvidia/*/lib` 加入 `LD_LIBRARY_PATH`，否则 CUDA EP
  会**静默回退 CPU**（假象），判据必须看 `get_providers()[0]`。

---

## 2. 仓库布局

```
work/STATE.md               现行结论台账（阶段A闸门、resize消融、A5b、fork修复）
work/LOG.md                 决策/推翻/失败尝试长记录（fork 定案在文件后半，搜 "fork 修复"）
work/MIGRATION_SUMMARY.md   阶段一核查 C1–C13 + 证据索引 + 下一步计划
work/TODO.md, work/PACKAGING-NOTES.md
work/pnnx_a{1..5}*.py       阶段 A/B 各闸门（可原样复跑）
work/resize_ablation.py     C2 消融（禁止缩图）
work/ort_cuda_crossproc.py  CUDA 跨进程确定性
work/pnnx_gpu_blob_diff.py  逐 blob 统计筛查（mean/std/norm，弱仪器）
work/pnnx_blob_relL2.py     逐 blob 逐元素 rel_L2（判定性仪器）+ 平移形态检测
work/ncnn_a1/               仅 param/bin 转换日志（大文件不入库）
work/ncnn_a2/               参考 npz（ref_512x512 等，已入库）
work/upstream_a5b/          fork 补丁、FORK_BUILD.md、micro 复现、p0b 验证
reports/ upstream/ scripts/ patches/   早期 TVM 阶段资产
docs/TVM-VULKAN-README.md   早期 README（历史）
```

## 3. 常用命令

```bash
# 迷你复现（1=Tesla P4, 0=RX460）；需补丁轮 ncnn
python work/upstream_a5b/micro/repro_split_pack4.py 1        # Split 网格：c=1..8 × 17 尺寸
python work/upstream_a5b/micro/repro_adjacent_pack4.py 1     # Split/Crop 全网格 + Slice/Concat/ReLU 探针
python work/upstream_a5b/micro/val_layer_probe.py split      # 验证层加载下的探测

# 逐 blob 逐元素 rel_L2（必须分块，单进程 ~600 blob 处 OOM；RX460 尾段需 ~30 blob/块）
python work/pnnx_blob_relL2.py --size 714x967 --device 1 --range 1-150 --out out.tsv
# RX460(device 0) 上 714×968 控制 + 714×967/713×965/713×967 三测试尺寸见 p0b/REPORT_rx460.md

# 任务级 A5b（原尺寸，ORT-CPU 参照 vs ncnn-Vulkan）
python work/pnnx_a5b_native.py --backend ort   --manifest work/upstream_a5b/p0b/manifest_mod4.json
python work/pnnx_a5b_native.py --backend ncnn  --device 1 --manifest ... --skip-done --dump-dir <dir>
python work/pnnx_a5b_native.py --backend metrics --manifest ... --dump-dir <dir> --out result.jsonl

# 补丁轮构建与 wheel：见 work/upstream_a5b/FORK_BUILD.md
```

## 4. 基线值（回归判据）

| 项 | 值 |
|---|---|
| ncnn 补丁轮 `.so` md5 | `b0e53111dfe2a1cbbc6a8d56bae49947` |
| ncnn 补丁轮 wheel md5 | `9e803b698951762eee691d59797dd20f` |
| stock 轮 `.so` md5 | `869ce77910292c65f76cff16a199c44`（作 known-bad 对照） |
| CUDA det 双模式 md5 | `94b5d5099968` / `18757f258295` |
| 逐 blob 包络（P4/RX460，714×968 对照） | max rel_L2 4.800e-3 → envelope 1.440e-2 |
| A5b 门 | \|ΔAUC\|≤2e-3、prob 翻转<0.1%、conf 记录 |
| fork 基线 | 上游 master `9f9d4ec`，fork commit `eab8352` |

## 5. 证据索引（按需读，不要全量通读）

- 结论/决策：`work/STATE.md`、`work/LOG.md`。
- 阶段一核查：`work/MIGRATION_SUMMARY.md`（C1–C13 逐条 + 附B 证据表）。
- 准入补证（P0-b）：`work/upstream_a5b/p0b/`
  - `REPORT_P0B.md`（汇总）、`REPORT_relL2.md`、`REPORT_adjacent.md`、`REPORT_h2.md`、
    `REPORT_rx460.md`、`REPORT_p2a.md`。
  - 原始数据：`rel_*.tsv`、`a5b_mod4_*.jsonl`、`manifest_mod4*.json`、`p0b_CRITERIA.md`
    （判据，含"TSV 判据不可达→事后修订"的登记）。
- 补丁/上游：`work/upstream_a5b/FORK_BUILD.md`、`0001-*.patch`、`micro/ISSUE_draft.md`。

## 6. 未完成项 / TODO

1. **第二缺陷根因**：迷你网 Slice/Concat/ReLU 输出 SIGSEGV，机制未定位；可选缓解=子进程隔离。
2. **上游提交**：`micro/ISSUE_draft.md` 是否提交上游由维护者决定（提交前复核其内容）。
3. 其他平台：Windows 上的 ncnn 补丁轮构建（MSVC + Vulkan SDK）未做。
4. 性能：ncnn-Vulkan 在 >0.7MP 无优势；如需继续优化，先确认目标卡的实测收益。

## 7. 接班规则（摘要；完整见 `AGENTS.md`）

- 事实优先、不编造来源/数值；结论标注 `[已验证事实]`/`[工程判断]`/`[待验证假设]`。
- 单机实测不得跨机外推，需标注"未核实"。
- 实验结论必须来自真实命令输出；分块跑以免 OOM（`pnnx_blob_relL2.py` 每块 ~150 blob）。
- 只改必要范围；记录规范见 `AGENTS.md`（只记决策与推翻、接口/依赖变更、未解决问题）。
- 改动 auto 链/产品接口不在本仓库范围内（产品侧另仓）。

## 8. 仓库历史说明

本仓库为**重新初始化的快照**，不含原 `TruFor_Vulkan_dev` 的完整 git 历史（其 `.git` 约 1.3GB
且含 5 个 >100MB 对象）。原 HEAD 与 commit 列表见 `SOURCES.txt` 与 `work/COMMITS.txt`
（若原仓库已随本仓库生成）。如需完整历史，请从原目录取。
