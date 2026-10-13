# TVM-VULKAN-REPORT.md — Phase 6D/6E/6F/6G

**Summary: TVM's Vulkan backend compiles the whole TruFor graph and runs it on the RX 560, and every individual operator computes correctly — but the assembled network returns wrong numbers. The divergence is localised to the SegFormer backbone, already at its first stage output.**

## 6D — Vulkan compile: PASS

Source build of TVM 0.27.0 with `USE_VULKAN=ON` (see `TVM-ENV.md`), 633/633 targets,
0 errors. `libtvm_runtime_vulkan.so` is built and loaded.

Device probe with `{"kind":"vulkan","from_device":0}` — capabilities read from the
device, not assumed:

| property | value |
|---|---|
| device_name | **AMD Radeon RX 460 Graphics (RADV POLARIS11)** |
| device_type | `discrete` |
| driver | `radv`, 22.3.6, Vulkan API 341 (1.3.41) |
| `supports_int64` | true |
| `supports_float16` | **false** |
| `supports_float64` | true |
| `supports_int16` / `int8` / `int32` | true / true / true |
| `thread_warp_size` | 1 |
| `max_threads_per_block` | 256 |
| `max_block_size_x/y/z` | 1024 / 1024 / 1024 |
| `max_shared_memory_per_block` | 65536 |
| `max_per_stage_descriptor_storage_buffer` | 8388606 |
| `max_spirv_version` | 66816 |
| `max_storage_buffer_range` | 4294967295 |

`relax.build(ir, target=vulkan)` on the full 1918-node model: **28.3 s, 0 failures,
no unsupported operators, no shader compile errors, no host-fallback nodes.**

A note: `tvm.support.libinfo()["USE_VULKAN"]` still reports `OFF` on the source
build even though the runtime is present and working — do not trust that field,
test `tvm.vulkan().exist` and the loaded libraries instead. (The first plain
`{"kind":"vulkan"}` build fails with `Vulkan target does not support Int64
capability`; `from_device: 0` fixes it, which is the documented remedy in the
error text.)

## 6E — operator smoke test: PASS

Small graph (conv → relu → conv → relu), eager vs TVM-llvm vs TVM-vulkan:

| backend | build | steady | max abs err | corr |
|---|---:|---:|---:|---:|
| tvm-llvm | 0.2 s | 0.0039 s | 1.192e-07 | 1.0000000000 |
| **tvm-vulkan** | 0.4 s | — | 1.192e-07 | **1.0000000000** |

The Vulkan execution path is sound: correct device, correct dispatch, correct
results.

## 6F — full TruFor on Vulkan: WRONG NUMBERS

```
build ok in 28.4s
VULKAN RAN. steady = 0.002 0.001 0.001 0.001 0.001 s
score  ref=0.985869  vk=0.619905   err=3.66e-01
map    max_abs=9.823e-01  mean=6.797e-01  rmse=7.875e-01  corr=-0.2901420146  IoU=0.000000  Dice=0.000000
conf   max_abs=2.929e-01  mean=8.340e-02  corr= 0.0204556956
```

The graph **runs to completion and does not crash** — a clear improvement over
the ExecuTorch path, which aborted in GPU allocation. But the result is
uncorrelated with the reference (map correlation **-0.29**).

Two things were ruled out before concluding that:

- **not a race / not unsynchronised**: three consecutive runs of the same input
  produce bit-identical output (`sum = 80235.43016380072` three times);
- **not "any individual kernel is wrong"**: see below.

**The 0.001 s latency is not a speedup** — it is the time to dispatch work whose
result is wrong, and it is far too small for a 68.7 M-parameter 512×512 forward
pass. It must not be quoted as performance.

## 6E′ — per-operator isolation: 20/23 correct

Every operator kind TruFor uses, built as its own tiny ONNX and compared against
eager PyTorch on the RX 560:

| operator | max abs err | corr | verdict |
|---|---:|---:|---|
| conv2d (pad 1) | 2.384e-07 | 1.00000000 | OK |
| conv2d (no pad) | 2.384e-07 | 1.00000000 | OK |
| gelu | 3.576e-07 | 1.00000000 | OK |
| linear | 4.172e-07 | 1.00000000 | OK |
| softmax | 7.451e-09 | 1.00000000 | OK |
| log_softmax | 7.153e-07 | 1.00000000 | OK |
| layer_norm (4-D) | 9.537e-07 | 1.00000000 | OK |
| layer_norm (2-D) | 3.576e-07 | 1.00000000 | OK |
| sigmoid | 1.192e-07 | 1.00000000 | OK |
| resize2d bilinear | 0.000e+00 | 1.00000000 | OK |
| matmul (4-D) | 9.766e-04 | 1.00000000 | OK |
| reduce_logsumexp | 0.000e+00 | 1.00000000 | OK |
| erf→gelu chain | 7.451e-09 | 1.00000000 | OK |
| mean global | 0.000e+00 | 1.00000000 | OK |
| amax global | 0.000e+00 | 1.00000000 | OK |
| **BIG conv 64ch @512×512** | 1.252e-06 | 1.00000000 | OK |
| **BIG linear 16384→1280** | 1.371e-06 | 1.00000000 | OK |
| **BIG matmul 2048×128×2048** | 1.162e-07 | 1.00000000 | OK |
| **BIG layernorm 4096** | 2.384e-07 | 1.00000000 | OK |
| **BIG resize 128→16** | 5.960e-08 | 1.00000000 | OK |
| attention 4-D | 0.000e+00 | n/a | OK (max_abs = 0) |
| **attention 5-D, CMX layout** | 0.000e+00 | n/a | OK (max_abs = 0) |
| batchnorm | — | — | test-harness bug, not re-run |

The two `corr = n/a` rows have `max_abs = 0.0` — they are exact; correlation is
undefined only because the mean-reduced output has zero variance.

Extra large-scale checks that specifically target the backbone's shapes:

| case | shape | max abs | corr |
|---|---|---:|---:|
| softmax | [1,5,2048,2048] | 1.192e-06 | 1.00000000 |
| softmax | [1,5,1024,1024] | 8.941e-07 | 1.00000000 |
| full attention | [1,5,2048,64] out | 4.545e-07 | 1.00000000 |
| full attention | [1,5,1024,64] out | 4.768e-07 | 1.00000000 |

So the exact quantities the backbone is built from — including the 2048×2048
attention matrices and the 5-D CMX cross-attention layout — are all correct in
isolation.

## 6G — stage isolation: the backbone diverges

`scripts/tvm_stage_isolation.py` splits the unmodified model along its own
structure and tests each stage on Vulkan with the real stage inputs:

| stage | build | max abs | corr | verdict |
|---|---:|---:|---:|---|
| noiseprint++ (DnCNN, 17 conv) | 1.04 s | 1.489e-03 | **0.99999967** | **OK** |
| backbone (SegFormer-B2 dual) | 25.81 s | 2.469e+00 | 0.90293958 | **DIVERGES** |

A 17-layer conv net over the full 512×512 input is correct to 1.5e-3 with
correlation 0.99999967. The backbone is the first stage that breaks, and the run
stops there as planned.

Per-output breakdown of the backbone (all four feature maps, from one run):

| output | shape | max abs | mean abs | corr | ref std | got std |
|---|---|---:|---:|---:|---:|---:|
| f0 | [1,64,128,128] | 1.561e+00 | 2.205e-01 | 0.96872518 | 1.1135 | 1.1134 |
| f1 | [1,128,64,64] | 1.898e+00 | 3.616e-01 | 0.90517311 | 1.0071 | 1.0424 |
| f2 | [1,320,32,32] | 2.434e+00 | 2.327e-01 | 0.96936225 | 1.1239 | 1.2128 |
| f3 | [1,512,16,16] | 2.808e+00 | 5.380e-01 | 0.92972199 | 1.5059 | 1.8113 |

The error is **already present at f0**, i.e. inside stage 1, and grows with depth.
For f0 the relative error is roughly 20 % of the signal's standard deviation, and
`got_std ≈ ref_std`, so the distribution is right but individual elements are
badly off. That is not the signature of float32 rounding.

## What this does and does not establish

Established by measurement:

- The RX 560 + RADV + Mesa 22.3.6 stack executes TVM Vulkan kernels correctly.
- TVM's Vulkan backend implements every operator TruFor needs, including the
  large and 5-D ones, correctly in isolation.
- Assembling them into the SegFormer-B2 dual-modality backbone produces wrong
  values, starting inside its first stage.

Not established:

- **Which component inside stage 1 is at fault.** It is not any single operator
  (all tested correct), and not the attention shapes (tested at 2048 tokens).
  The remaining candidates are TVM's Vulkan **layout/packed-dim transform**,
  kernel **fusion** applied across the stage, or the **accumulation strategy**
  the Vulkan backend picks for wide reductions in this specific graph.
- Whether it is a TVM bug, a driver (RADV 22.3.6) bug, or an
  interaction. Not separable without further bisection.

## Reproduce

```bash
export PYTHONPATH=$PWD/vendor/tvm-src/python
export TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

# 6D compile only
vendor/tvm-venv/bin/python scripts/tvm_vulkan_compile.py \
  --onnx reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx --outdir reports/tvm

# 6E + 6F
vendor/tvm-venv/bin/python scripts/tvm_vulkan_run.py \
  --onnx reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx \
  --image truFor-src/test_docker/images/tampered2.png --outdir reports/tvm

# 6E' per-operator
vendor/tvm-venv/bin/python scripts/tvm_vulkan_op_isolation.py --outdir reports/tvm/op-isolation2

# 6G per-stage
vendor/tvm-venv/bin/python scripts/tvm_stage_isolation.py \
  --model weights/trufor.pth.tar \
  --image truFor-src/test_docker/images/tampered2.png \
  --outdir reports/tvm/stage-isolation
```

Note the scripts need the *source* build on `PYTHONPATH`; the PyPI wheel has
`USE_VULKAN=OFF` and cannot produce a Vulkan executable. Also note the
source build exposes `tvm.__version__` while the wheel only has
`tvm._version.__version__`.
