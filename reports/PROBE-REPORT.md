# PROBE-REPORT.md — 低成本路线可行性探针（Phase 3 前置）

三项探针的实测结果。目的是在投入 Vulkan SDK 构建之前，先把便宜的判别做完。
**结论：两条低成本路线都不可用，唯一剩下的路是源码构建 ExecuTorch Vulkan delegate。**

## 探针 1 — RX 560 的 Vulkan 设备是否真实可见

先补装了 `vulkan-tools`（此前 `vulkaninfo` 缺失）。

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json vulkaninfo --summary
```

| 字段 | GPU0 | GPU1 |
|---|---|---|
| deviceName | **AMD Radeon RX 460 Graphics (RADV POLARIS11)** | llvmpipe (LLVM 15.0.6, 256 bits) |
| deviceType | `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` | `PHYSICAL_DEVICE_TYPE_CPU` |
| driverName / driverInfo | radv / Mesa 22.3.6 | llvmpipe / Mesa 22.3.6 |
| apiVersion | 1.3.230 | — |
| driverID | `DRIVER_ID_MESA_RADV` | — |

**通过。** Baffin 芯片由固件按 RX 460 枚举，Vulkan 走 RADV 真实硬件路径。`llvmpipe` 以
`deviceType=CPU` 单独枚举为 GPU1，所以任何"选错软件渲染器"的风险可以用
`deviceType == PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` 显式排除。
注意：不设 `VK_ICD_FILENAMES` 时两个 ICD 都会被枚举，程序必须自行按 deviceType 过滤。

## 探针 2 — ONNX Runtime 的 WebGPU EP（Dawn→Vulkan）能否激活

`onnxruntime-ep-webgpu` 0.4.0 装入成功，wheel 内含
`onnxruntime_ep_webgpu/libonnxruntime_providers_webgpu.so`（16.4 MB，动态链接
`libgtk3-nocsd.so.0` / `libstdc++` / `librt`，无缺失依赖）。

`scripts/probe_webgpu.py` 用一个 `Conv(3→8, k=3)` 的最小 ONNX 图做纯运行时存活测试。
四种 provider 组合全部试过：环境变量 `ORT_WEBGPU_DEVICE_TYPE=gpu`、
`provider_options={'deviceType':'gpu'}`、`{'deviceType':'gpu','dawnBackendType':'vulkan'}`、
`{'dawnBackendType':'vulkan'}`。另外把 `.so` 直接拷进 `onnxruntime/capi/` 试过 drop-in 加载。

结果：

```
ort version           : 1.24.4  /  1.30.0
all providers         : [..., 'WebGpuExecutionProvider', 'XnnpackExecutionProvider', ...]
register_execution_provider_library: ok (no exception)
[CPU]    ok  used=['CPUExecutionProvider']  checksum=213542.126584
[WebGPU] ok  used=['CPUExecutionProvider']  checksum=213542.126584   <-- 静默回退
max abs diff CPU vs WebGPU: 0.000e+00                                  <-- 同一份 CPU 结果
```

**失败。** 三条独立证据：

1. `sess.get_providers()` 始终返回 `['CPUExecutionProvider']`，WebGPU 节点一个都没接管。
2. 把日志级别开到 VERBOSE 后，输出中**没有任何** WebGPU / Dawn / adapter 关键字，
   说明 Dawn 的初始化代码根本没被调用，而不是初始化后失败。
3. `ctypes.CDLL` 直接加载该 `.so` 时 `undefined symbol: OrtGetEpFactory`，
   主库未把符号导出给独立 dlopen 的作用域。

这不是参数没调对的问题。`onnxruntime-ep-webgpu` 0.4.0 是 plugin EP（以
`OrtEpFactory` 形式注册），而本机 `onnxruntime` 的 Python wheel 走的是静态链接的
provider 列表，`register_execution_provider_library` 在 1.24.4 和 1.30.0 上都没有把 factory
真正接入 session 创建流程。

**探针 2 结论：WebGPU EP 在本机不可用。** 叠加 `OPERATOR-INVENTORY.md` §1.3 已证实的
"ORT 官方 Vulkan EP 已移除"，**ONNX Runtime 路线整体关闭。**

## 探针 3 — TruFor 能否导出为 ONNX

```bash
python3 scripts/probe_onnx_export.py --model weights/trufor.pth.tar --outdir reports/onnx-probe
```

| exporter | 结果 |
|---|---|
| `dynamo=True`（torch 2.7 默认，torch.export based） | **失败** `ConversionError`，step 3/3 ONNX conversion。缺 `onnxscript` 已补装，问题在模型本身 |
| `dynamo=False`（legacy TorchScript tracer，opset 17） | **成功** 8.4 s，5133 nodes，275.7 MB |

**数值验证**（`scripts/verify_onnx.py`，输入 `tampered2.png` 缩放到 512×512）：

| 输出 | PyTorch | ONNX Runtime CPU | 误差 |
|---|---:|---:|---|
| integrity score | 0.985877 | logit 4.245150 → sigmoid 0.985869 | 7.5e-06 |
| localization map | — | — | max_abs 8.9e-05, mean_abs 8.0e-06, rmse 1.0e-05, **corr 1.00000000** |
| confidence map | — | — | max_abs 7.4e-05, mean_abs 2.0e-05, rmse 2.3e-05, **corr 0.99999998** |

**通过。** 三个输出全部数值等价。ONNX Runtime CPU 2.39 s vs PyTorch CPU 3.26 s（同机同图）。

ONNX 算子分布（`reports/onnx-probe/onnx-export.json`）：

| op | count | op | count |
|---|---:|---|---:|
| Constant | 1608 | Conv | 111 |
| Unsqueeze | 624 | Slice | 78 |
| Transpose | 324 | Div | 74 |
| Add | 318 | Cast | 73 |
| Concat | 316 | Softmax | 40 |
| Reshape | 301 | Relu | 39 |
| Gather | 291 | Erf | 32 |
| Shape | 272 | Squeeze | 16 |
| MatMul | 272 | Gemm | 10 |
| Mul | 156 | Sigmoid | 9 |
| LayerNormalization | 114 | Split / Resize | 8 / 8 |

detection head 完整在内：`ReduceLogSumExp` 4、`LogSoftmax` 2、`Log` 1、
`GlobalAveragePool` 4、`GlobalMaxPool` 4、`ReduceSum` 4。

**两个必须知道的限制：**

1. **不支持动态尺寸。** `onnx input : [1, 3, 512, 512]` 是固定 shape。1608 个 `Constant`
   + 272 个 `Shape` + 291 个 `Gather` 说明 legacy tracer 把中间张量形状当常量烘死了。
   原方案"先固定 512×512"的要求在 ONNX 侧是**硬约束**，不是临时简化。
2. **文件虚胖。** 275.7 MB 中大部分是这些 shape 常量；权重本身只有 68.7M 参数 ≈ 275 MB，
   所以其实没浪费多少，但 `onnx-simplifier` 之类的常量折叠仍有收益。

## 汇总：现在还剩什么

| 路线 | 状态 | 依据 |
|---|---|---|
| CUDA | 不可用 | RX 560 无 CUDA，`torch.version.cuda is None` |
| ROCm | 不可用 | Baffin / Polaris11 不在 ROCm 支持矩阵内（未在本轮实测，属既有工程判断） |
| PyTorch 原生 Vulkan backend | 需自行构建 | 官方标注 no longer maintained |
| **ORT Vulkan EP** | **已移除** | 83 个 release 无 vulkan asset，linux-x64-gpu 包实测只含 cuda/tensorrt |
| **ORT WebGPU EP** | **探针 2 失败** | ORT 1.24.4 与 1.30.0 下均静默回退 CPU，无 Dawn 初始化日志 |
| ExecuTorch Vulkan delegate | 需源码构建 | 唯一未被排除的 Vulkan 路径 |
| OpenCL / 自行写 kernel | 未评估 | 超出本次范围 |

**唯一剩下的 Vulkan 路线是：从源码构建 ExecuTorch Vulkan delegate**（Vulkan SDK
1.4.341.1+ 提供 `glslc` → `cmake -DEXECUTORCH_BUILD_VULKAN=ON` → `vulkan_backend` target），
而官方对该后端在桌面平台的定性是 experimental、shader 面向移动端 GPU 调优。

已知的、构建后必须立刻回答的问题仍然是 `OPERATOR-INVENTORY.md` §5 里的那一个：
`aten::mul`（78 个节点，遍布 backbone）没在 112-op registry 里。这是 Vulkan 部分能否
吞下整个 backbone 的决定因素，构建完成后 10 分钟内可由 `VulkanPartitioner` 给出答案。

## 产物

- `scripts/probe_webgpu.py` — WebGPU EP 存活测试（可复跑）
- `scripts/probe_onnx_export.py` — 双 exporter ONNX 导出探测
- `scripts/verify_onnx.py` — PyTorch vs ORT 数值等价验证
- `reports/onnx-probe/trufor_512x512_legacy.onnx` — 可用的 ONNX 模型（固定 512×512）
- `reports/onnx-probe/verify-512.json` — 数值等价报告

环境变更：`apt install vulkan-tools`；`pip install onnxruntime-ep-webgpu onnx onnxscript`；
`onnxruntime` 从 1.24.4 升到 1.30.0。
