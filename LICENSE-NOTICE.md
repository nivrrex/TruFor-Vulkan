# 许可与分发说明（LICENSE-NOTICE）

本仓库**不对外分发**，用途为内部工程记录与研究。

## 各组成部分

| 组件 | 许可 | 说明 |
|---|---|---|
| TruFor 模型与上游代码（GRIP-UNINA） | **非营利/研究用途** | 上游明确限非营利用途；本仓库**不含**权重文件，也不得用于商业或对外分发 |
| 本仓库自有脚本与文档 | 随上游口径，非营利研究使用 | 仓库级 LICENSE 文本待维护者确认后补入 |
| ncnn | BSD-3-Clause | 本仓库对其 `src/command.cpp` 打单文件补丁，**不改动许可**；补丁见 `work/upstream_a5b/0001-*.patch` |
| ONNX Runtime | MIT | 仅在实验环境使用，不随仓库分发 |
| NVIDIA CUDA / cuDNN | NVIDIA 运行时分发条款 | 仅出现在实验环境，不入本仓库 |
| TVM / Mesa 等前期资产 | 各随其许可 | 早期 TVM-Vulkan 研究记录 |

## 未入库内容（有意）

- TruFor 权重（`trufor.onnx`、`模型资产/.bin`）——许可 + 体积。
- ncnn 自建 wheel 与构建产物、TVM vendor 源码树、样本图片、各类 dump。
- `reports/` 下 5 个 >100MB 的 onnx/mnn 模型文件（GitHub 单文件限制）；可由相应导出脚本再生。

## 分发前须知

- 若后续需要对外分发：先解决上游 TruFor 权重的非营利许可问题（法务），并补入仓库级 `LICENSE`。
