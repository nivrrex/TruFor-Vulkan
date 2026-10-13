# OPERATOR-INVENTORY.md — TruFor → Vulkan feasibility (Phase 2)

Reproduce:

```bash
python3 scripts/operator_inventory.py --model weights/trufor.pth.tar \
  --image truFor-src/test_docker/images/tampered2.png --outdir reports/operator-inventory/run1
python3 scripts/classify_export.py \
  --graph reports/operator-inventory/run1/export-graph.txt \
  --registry reports/operator-inventory/vulkan-op-support-table.csv \
  --outdir reports/operator-inventory/run1
```

## 1. Two facts that invalidate parts of the original plan

### 1.1 The ExecuTorch Vulkan delegate has **no prebuilt Linux x86_64 binary**

Verified against PyPI + GitHub + the official docs:

- `executorch` manylinux_2_28_x86_64 wheels exist for 1.2.0 – 1.5.1, but the wheel ships only
  the Vulkan **Python** layer (`partitioner`, `quantizer`, `op_registry`, `patterns`, `passes`,
  49 `.py` files) plus C++ **sources**. Of the 17 `.so` files in the wheel there is
  `libexecutorch_backend_xnnpack.so`, `..._openvino.so`, `..._qnn.so` — and **no
  `libexecutorch_backend_vulkan.so`**.
- GitHub Releases ship only ARM CMSIS `.pack` / `.pdsc`. No Linux asset of any kind.
- Official build instructions: Vulkan SDK **1.4.341.1 or newer** for `glslc`, then
  `python install_vulkan.py`, or `cmake -DEXECUTORCH_BUILD_VULKAN=ON` targeting the
  `vulkan_backend` CMake target. Android additionally needs NDK r28c.
- `docs/source/backends/vulkan/vulkan-overview.md` states desktop-platform support is
  **experimental** and that the shaders are tuned for mobile GPUs, not desktop/server GPUs.

Consequence: Phase 4 in the original plan is not "export → .pte", it is
"install a ~1.5 GB SDK → build ExecuTorch from source → only then export a `.pte`".

### 1.2 There is **no runtime CPU fallback for a lowered-but-unsupported operator**

Once the `VulkanPartitioner` lowers a node to the Vulkan backend, it must execute on the GPU;
if the runtime cannot execute it, the process crashes. "Partial delegation" therefore only means
*nodes the partitioner chose not to lower*, decided statically — not automatic rescue.

### 1.3 ONNX Runtime's Vulkan EP is gone too

Checked all 83 `onnxruntime` release tags: no asset name contains `vulkan`. The
`onnxruntime-linux-x64-gpu-1.24.4.tgz` (205 MB) was downloaded and unpacked — its `lib/` contains
only `libonnxruntime_providers_cuda.so`, `..._shared.so`, `..._tensorrt.so`. `grep -i vulkan` on
the listing: no hits. The provider source and docs are 404 on `main` and on the `gh-pages` branch,
and `onnxruntime-vulkan` does not exist on PyPI.

The only ORT path to Vulkan on Linux is now the **WebGPU EP** (`pip install onnxruntime-ep-webgpu`,
`manylinux_2_28_x86_64` wheel, Dawn's Linux backend is Vulkan). Coverage of that EP for a
SegFormer-style graph is unverified.

## 2. Graph extraction

`torch.export.export(model, (1,3,512,512), strict=False)` **succeeds**. No export blocker.

- graph nodes (incl. placeholders): **2826**
- executable `torch.ops.*` nodes: **1830**
- distinct aten ops: **41**

Runtime cross-check with `TorchDispatchMode` on `tampered2.png` (256x256): 2920 aten calls,
41 distinct ops — the same 41, confirming the export graph is representative.

## 3. Classification against the Vulkan operator registry

Registry = `docs/source/backends/vulkan/vulkan-op-support-table.csv`, 112 operators, generated
from `backends/vulkan/op_registry.py`. Downloaded to
`reports/operator-inventory/vulkan-op-support-table.csv`.

Verdict levels:

- **L0** exact aten name in the registry
- **L1** not listed under that name, but ExecuTorch's lowering passes rewrite it to a registered
  form (dim-order normalisation, decomposition). The landing registry entry is named per row.
- **REMOVED** eliminated before lowering (eval-mode no-ops, meta ops)
- **L2** no registered form and no available decomposition

| verdict | ops | graph nodes | share |
|---|---:|---:|---:|
| L0 | 14 | 588 | 32.1% |
| L1 | 21 | 1026 | 56.1% |
| REMOVED | 3 | 133 | 7.3% |
| **L2** | **3** | **83** | **4.5%** |

Machine-readable: `reports/operator-inventory/run1/export-classification.json`.

### The three L2 operators

| op | nodes | where | why it blocks |
|---|---:|---|---|
| `aten::mul` | 78 | whole backbone | Not in the registry, and there is no `add`/`sub`/`div`/`neg`/`pow` decomposition of a general elementwise multiply. Sources: SegFormer attention scaling `(q @ k.transpose(-2,-1)) * self.scale` (`lib/models/cmx/encoders/dual_segformer.py:128`) and CMX fusion gating. |
| `aten::logsumexp` | 4 | detection head only | Not registered. From `weighted_statistics_pooling` (`lib/models/cmx/layer_utils.py:21`), on `[B,C,H,W]` → `[B,C,2]` tensors. |
| `aten::log_sigmoid` | 1 | detection head only | Not registered; `_log_softmax` is present but is not a special case of it. Same function. |

### Two lower-risk items

- `aten::sum` (4 nodes) **is** registered, but with the caveat *"keepdim=True required; max 2D
  reductions"*. The detection head sums with `keepdim=False`. That is a caller-side fix, not a
  missing kernel.
- `aten::conv2d` maps to the registry's `aten::convolution`, whose caveat is *"batch=1 for 2D
  conv"*. TruFor inference already runs `batch_size=1` (`test.py` comment: *"1 to allow arbitrary
  input sizes"*), so the constraint is satisfied — but it is a real constraint.

### Ops that are registered but were not exercised at the default path

`aten::_weight_int8pack_mm`, `aten::embedding`, `aten::flip`, `aten::max_pool2d_with_indices`,
`aten::round`, `aten::arange`, `aten::full`/`full_like` — quantization- and utility-flavoured;
their presence does not help FP32 inference.

## 4. Verdict against the plan's four options

The plan offered A/B/C/D. The measured answer is **C + D, and the C part is mild**:

- **A** (go straight to ExecuTorch Vulkan) — **no**, because there is nothing to install; the
  delegate must be built from source first. That is a hard prerequisite, not a detail.
- **B** (export is blocked) — **no**, export succeeds cleanly.
- **C** (Vulkan lacks key operators) — **partly yes**: 3 operators, 4.5% of nodes.
- **D** (local model rewrite needed) — **yes, but small**. Two of the three L2 ops
  (`logsumexp`, `log_sigmoid`, 5 nodes total) live entirely in the detection head, which operates
  on a `[B,8] → [B,128] → [B,1]` MLP. That head is trivially partitioned to CPU and would not
  stop the rest of the network from running on Vulkan. The real question is `aten::mul` (78
  nodes across the backbone).

## 5. Open question that only a real build can answer

The absence of `aten::mul` from a Vulkan operator registry is surprising — elementwise multiply
is about as basic as a shader gets, and the registry does contain `add`, `sub`, `div`, `neg`,
`pow`, `exp`, `log`, `rsqrt`, `sqrt`, `tanh`, `sigmoid`, `gelu`, `abs`, `round`, `clamp`. Three
plausible readings, not yet distinguishable without running the partitioner:

1. `op_registry.py` simply omits a declaration that the backend can already execute;
2. a lowering/fusion pass rewrites the specific `mul` patterns TruFor produces (e.g. scalar-scaled
   matmul folds into `addmm`/`linear` with a scale) into a registered op;
3. it is genuinely unimplemented, and every elementwise multiply must stay on the CPU partition.

**This is the single fact that decides whether the project is cheap or expensive, and it is not
determinable from documentation.** It is a 10-minute experiment once the delegate is built:
export TruFor, run the `VulkanPartitioner`, and read which nodes it refuses to lower.

## 6. Recommended next step (one command, then re-decide)

Build the ExecuTorch Vulkan delegate and run the partitioner **without** changing any TruFor
code. Cost: Vulkan SDK download + `glslc` + a source build. Nothing downstream of that decision
should be started until it returns.

```bash
# 1. Vulkan SDK 1.4.341.1+ (glslc) -- official tarball, no root install required
# 2. export ExecuTorch, build the vulkan_backend target
# 3. scripts/partition_probe.py: export TruFor @512x512, run VulkanPartitioner,
#    report lowered-vs-not node counts and the top reasons
```

Stop conditions, unchanged from the plan:

- partitioner cannot lower a large fraction of the graph → stop, ONNX/WebGPU route or stop;
- graph partitions but the resulting `.pte` crashes on RX 560 → stop, report the failing op;
- partitions and runs → proceed to Phase 5/6 (RX 560 timing + CPU-reference equivalence).
