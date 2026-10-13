# TruFor on RX 560 via Vulkan

Running the TruFor image-forensics model (RGB + Noiseprint++ → SegFormer-B2 fusion → localisation,
confidence, integrity score) on an **AMD Radeon RX 560 (Baffin)** through a Vulkan compute path,
and finding out what actually works.

## Answer in one line

**It runs, it is correct, and the GPU is genuinely fast — except for one kernel.**
With Mesa ≥ 25.0.7 the full 1918-node graph reproduces the reference exactly on the GPU
(`map corr = 1.0000000`, IoU = Dice = 1.0) at **0.94×** the latency of ONNX Runtime on 4 CPU cores.
Measured per class, the GPU is **4-5× faster** on elementwise and softmax, reaches
**383 GFLOPS** on convolution (against a 4-core CPU at ~94 GFLOPS average) and **386 GFLOPS** on
large matmul. The entire deficit comes from one thing: **TVM's Vulkan transpose kernel runs at
~5 GB/s, about 40× slower than a plain copy**, and the SegFormer backbone contains 320 of them.

Two findings define this project. Everything that looked like a compiler or runtime bug was the
installed driver (Mesa 22.3.6 returns silently wrong images). And the measurement study's *first*
version was wrong in the opposite direction — it timed H2D+D2H transfer together with execution and
concluded the GPU was never useful, which is also false.

## The finding worth keeping

```
ExecuTorch Vulkan  ->  aborts in GPU allocation
MNN               ->  fails shape inference on the CPU backend
TVM Vulkan        ->  runs; WRONG results
  ... operator-level suite says conv/layernorm/attention are all fine
  ... stage-level bisect says the SegFormer backbone diverges
  ... gdb points at TVM's VulkanDeviceAPI::FreeWorkspace
  ... a "minimal reproducer" blames TVM's take/select kernel
  ... validation layer: silent
  ... ASan: unusable (binding conflict); gdb: inconclusive
  ... ONE VARIABLE CHANGED: Mesa 22.3.6 -> 25.0.7
Mesa 25.0.7        ->  everything exact
```

A reproducible, isolated, single-operator-looking failure is still allowed to be a driver bug. The
minimal reproducer that survived every other check was the thing that had to be thrown away. This
cost days and is the most transferable lesson in the repository.

## Deployment

See `DEPLOY.md`. Requirements: Mesa/RADV ≥ 25.0.7, TVM 0.27.0 built with `USE_VULKAN=ON`, the
included ONNX model, 512×512 fp32 input.

## Reports

Read in this order.

| document | what it establishes |
|---|---|
| `reports/FINAL-REPORT.md` | **start here.** Final conclusion, the Mesa A/B, every retraction, and the optimisation entry points |
| `DEPLOY.md` | the working configuration, build steps, known-bad environments, regression triage |
| `reports/BASELINE.md` | the PyTorch CPU reference: checkpoint integrity, per-stage timings, reference tensors |
| `reports/PROBE-REPORT.md` | cheap routes eliminated first: CUDA/ROCm absent, ORT Vulkan EP removed, ORT WebGPU EP inert, ONNX export numerics |
| `reports/operator-inventory/OPERATOR-INVENTORY.md` | 1830-node / 41-operator census, graded against the ExecuTorch registry — and a correction to it |
| `reports/TVM-ENV.md` | TVM build prerequisites, the two CMake pitfalls, LLVM/Vulkan/SDK versions |
| `reports/TVM-IMPORT.md` | why the legacy-tracer ONNX cannot be imported, and the two exact export-time rewrites that fix it |
| `reports/TVM-CPU.md` | TVM LLVM correctness gate (pass) and why the generic LLVM path is not the answer |
| `reports/TVM-VULKAN-REPORT.md` | the first Vulkan run: compiles, runs, wrong; per-operator and per-stage isolation |
| `reports/PHASE8-STAGE1-LOCALISATION.md` | the stage-1 bisect and the take/select "bug" — **later shown to be the driver** |
| `reports/PHASE8A-8C-REPORT.md` | the Slice+Squeeze rewrite: exact, but it does not fix the model |
| `reports/PHASE8C-DIAGNOSIS.md` | validation layer clean, ASan blocked, gdb localisation — **later misattributed** |
| `reports/MNN-VULKAN-REPORT.md` | MNN converts but cannot compute shapes; its Vulkan backend is never reached |
| **`reports/capability/CAPABILITY-REPORT.md`** | **where the RX 560 actually stands**: the corrected device-resident operator table, the module split, and the root cause — TVM's Vulkan transpose kernel at ~5 GB/s. Also records what the first version of this report got wrong |

Raw evidence for every claim lives beside the reports: `reports/mesa-ab/` holds the A/B runs and
the production baseline, `reports/bench-production/benchmark.json` holds the per-image timings.

## Upstream

`upstream/README.md` distinguishes what may be filed from what must not:

- `workspace_repro.cpp` — real TVM robustness issue (cross-thread workspace alloc/free is
  unsupported by the Vulkan device API). Reproduces identically on both driver versions, pure C++
  against an ASan build.
- `ISSUE-take-select.md` — **do not submit.** Written before the root cause was known; the behaviour
  it describes is Mesa/RADV 22.3.6, not TVM.

## Repository layout

```
truFor-src/    upstream TruFor, unmodified (commit ae54475)
weights/       official checkpoint (needed only to re-export the ONNX)
reports/       reports, evidence, and the production ONNX model
scripts/       every step of the investigation, each runnable on its own
scripts/bench/  the GPU capability study (Gate 0-7), independent of correctness code
upstream/      material for the TVM project
vendor/        TVM / Vulkan SDK / Mesa unpacked sources and builds (not redistributable)
artifacts/     CPU reference tensors
```

`truFor-src/` was never modified. Everything else is ours.

## Honest summary of where this landed

- **Correct GPU execution on RX 560: yes**, at a fixed 512×512, fp32, batch 1.
- **A dynamic-H/W ONNX exists for the CPU side only**: `scripts/export_onnx_dynamo_dynamic.py`
  re-exports with symbolic H/W so ONNX Runtime accepts any input size and returns the map at the
  input size (verified by `scripts/verify_dynamic_ort.py`). The Vulkan/TVM path cannot consume it and
  stays 512×512.
- **Performance benefit: small.** 0.94× overall — the GPU is 4-5× faster on elementwise and
  softmax and reaches 383 GFLOPS on convolution, but loses on layout transforms.
- **The first blocker was a three-year-old driver**, not the model, not the compiler, not the runtime.
  Mesa 22.3.6 returns silently wrong images.
- **The second is TVM's Vulkan transpose kernel**: ~5 GB/s, 3-10× slower than the CPU for the same
  op, and 320 instances in the backbone. Ruled out by experiment: DPM state, `thread_warp_size`,
  operator fusion, convolution quality, depthwise convolution.
- **Next:** find out why 320 explicit transposes survive `ConvertLayout`; failing that, tile the
  transpose kernel; failing that, A/B another Vulkan backend on the same driver.

If you are reviewing this, the two places where a different conclusion is possible are the
device-resident operator table (it is what reversed the earlier verdict) and the arithmetic that ties
320 transposes to the measured 158 ms backbone gap.
