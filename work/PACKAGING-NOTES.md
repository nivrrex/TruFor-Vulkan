# 打包说明与引用归类

本文件记录本次打包（部署包 / 开发包）的取舍，以及引用完整性扫描每一处命中的
人工归类。扫描脚本 `check_refs.sh` 命中是线索不是判定，此处逐类说明，不做沉默通过。

## 包内容取舍

| 类别 | 处理 | 理由 |
|---|---|---|
| `vendor/`（约 16 GB：TVM 源码树、Mesa 构建、Python venv）| 两个包均不含 | DEPLOY.md 第 0-3 步在现场构建；含进来会让部署包失去意义 |
| `reports/` 下的模型文件 | 部署包只放生产模型（267 MB）；开发包放代码/文档实际引用的 4 个（约 1.06 GB）| `reports/` 共约 8.2 GB，其中约 30 个 ONNX 副本与约 2.6 GB MNN 模型仅被历史实验提及，代码无引用 |
| `truFor-src/` 的训练数据与预训练权重（约 190 MB）| 剥离 | 导出 ONNX 只需源码与 `weights/` 下的 checkpoint；三份 COCO 数据清单与 SegFormer 权重无任何引用 |
| `truFor-src/.git`（约 128 MB）| 剥离 | 上游仓库的版本历史，非本项目内容 |
| `__pycache__` / `*.pyc` | 剥离 | 构建产物 |
| 绝对路径 | 改写为相对路径 | `artifacts/baseline_cpu/baseline.json` 的记录值、测量辅助脚本的 ROOT 常量 |

## 引用扫描命中归类

### 一类：运行期引用，文件已在包内

无。扫描在部署包上为 0。开发包中 README、PACKAGE-NOTE、FINAL-REPORT 指向
`reports/` 各子目录的文档已随目录结构一并保留——首次复制时把它们压平了，
是打包脚本的缺陷，已修正。

### 二类：开发期引用，指向 TVM 源码树内部的文件

TVM 源码树不在包内（按 DEPLOY.md 现场构建），因此下列形式的路径在本包中不成立，
而在按 DEPLOY.md 重建了 TVM 源码树的开发机上成立：

    src/...   include/...   python/tvm/...   tests/...
    backends/...   docs/source/...   tvm_ffi/...
    lib/libSPIRV-Tools*.{a,so}   vendor/1.4.341.1
    onnxruntime_ep_webgpu/...

命中位置：`DEPLOY.md`、`patches/README.md`，以及 `reports/` 与 `work/LOG.md` 中
记录 TVM 内部实现的段落。这类引用**保持原样**：它们是技术论述的组成部分，
改写会削弱文档的可核查性。部署包中对应的 `patches/README.md` 已改为不含裸路径
的措辞，因为该文件是部署操作指引，必须逐字可执行。

### 三类：扫描器的误命中

`work/LOG.md` 中 `1/2.34`、`22.6/21.7/13.9`、`44.4/38.5/36.3` 三处被当作路径，
实为数值的比值与列表。属扫描器启发式的误判，不修改日志原文。

### 四类：指向已删除文件的历史陈述

- `work/LOG.md` 提到 `scripts/opt/expand_log_softmax.py`：该日志条目原文即
  "写了又删除"，属**准确的历史陈述**，非死链。
- `reports/OPTIMIZATION-REPORT.md` 第 276 行给出的 `scripts/opt/profile_graph.py`
  命令：该脚本**确已不存在**。逐指令 profiling 后来改为由 TVM 源码树的 `vm.cc`
  补丁实现（即 `patches/` 中那份），报告里的命令是早期形态。这是**报告中的
  过期命令**，如实记录于此；该命令不应被照抄，现行做法见 `DEPLOY.md` 第 7 步。
