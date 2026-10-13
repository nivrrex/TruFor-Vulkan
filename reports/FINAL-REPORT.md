# FINAL-REPORT.md — TruFor on RX 560 via Vulkan: solved, and it does not pay off

**Two separate conclusions, both measured.**

1. **It works.** The complete 1918-node TruFor graph runs on the RX 560 through TVM's Vulkan backend
   and reproduces the reference exactly: `map correlation = 1.0000000`, IoU = Dice = 1.0, integrity
   score error 2.4e-06, on every image tested.
2. **It is slower than the CPU.** 0.81–0.82× — the GPU path is ~20% *slower* than ONNX Runtime on
   4 CPU cores. Per the project's own gate: **NO MATERIAL GPU GAIN**.

And a correction that matters more than either: **every "TVM Vulkan is buggy" conclusion in the
previous rounds was wrong.** The root cause was the installed driver, Mesa 22.3.6. A strict A/B with
Mesa 25.0.7 — same TVM, same model, same flags, same GPU, only the driver swapped — turns every
failure into a pass.

---

## 1. The decisive A/B

Only one variable changed: the Mesa/RADV version, loaded side-by-side without touching the system
install (`LD_LIBRARY_PATH` + `VK_DRIVER_FILES` from a backports unpack in `vendor/mesa-ab/`).

| | Mesa 22.3.6 | **Mesa 25.0.7** |
|---|---|---|
| driver reported by the device | 22.3.6, api 1.3.230 | **25.0.7, api 1.4.305** |
| full TruFor, map correlation | **-0.2873656449** | **0.9999999999** |
| full TruFor, map max abs err | 9.815e-01 | **7.293e-05** |
| integrity score error | 3.552e-01 | **2.438e-06** |
| IoU / Dice | 0.000000 / 0.000000 | **1.000000 / 1.000000** |
| `VK_LAYER_KHRONOS_validation` | silent | silent |
| take/select reproducer | **9 of 15 wrong** | **0 of 15 wrong** |

Everything else was held fixed: TVM 0.27.0 built from the same source, the same
`trufor-512x512-slicefix.onnx`, the same `{"kind":"vulkan","from_device":0}` target, the same RX 560
(`AMD Radeon RX 460 Graphics (RADV POLARIS11)`, `deviceType=discrete`).

## 2. What this retroactively invalidates

| earlier claim | status now |
|---|---|
| "TVM Vulkan miscomputes dim-0 take/select when `src[-2] == 1`" (Phase 8, with a minimal reproducer) | **not a TVM bug.** The same reproducer is 15/15 correct on Mesa 25.0.7. It was a driver bug that happened to be shape-selective. **Do not file that issue on apache/tvm** — it would be wrong. `upstream/ISSUE-take-select.md` is retained only as the record of how it was found, and it must not be submitted. |
| "the divergence is a TVM operator/codegen bug, driver interaction only *unlikely*" (Phase 8D) | **wrong.** The driver was the cause. The shape-selectivity of the symptom is what made it look like a codegen bug. |
| "`VulkanDeviceAPI::FreeWorkspace` is the corruption site" (Phase 8C, from the gdb stack) | **misattributed.** The gdb frame is real, but the corruption that produced it came from the driver underneath; a driver-side fault surfaces as a bogus frame in whatever C++ code touches it next. |
| "removing `ApplyDefaultSchedule` triggers heap corruption" (Phase 8B) | consistent with the driver theory — different pipeline, different driver exposure, same underlying fault. |
| "the 5-D x 4-D broadcast / Fusion / layout transforms" | all were downstream noise from the same driver fault. |

What genuinely survived scrutiny, and is a **separate, real TVM issue** (see §3): cross-thread
workspace use is not supported and fails with clean errors.

## 3. What is still a real TVM problem (independent of the driver)

`upstream/workspace_repro.cpp` is a pure-C++ reproducer (no Python, no PyTorch, no model) built
against the ASan-instrumented TVM, driving the workspace C API directly — the exact call path gdb
showed. It behaves identically on both Mesa versions:

| case | result |
|---|---|
| A1 single-thread alloc/free (incl. 64 nested LIFO) | **pass** |
| A2 alloc on thread A, free on thread B | fails: `Attempted to free a vulkan workspace on a CPU-thread that has never allocated a workspace` |
| A3 8 threads x 64 allocs, cross-thread interleaved frees | fails: `trying to free things that has not been allocated` |
| A4 free on a thread that never allocated | fails, same message |
| A5 freeing thread *has its own pool* (so the `ICHECK(pool)` guard passes) | fails one level deeper, inside the wrong pool's bookkeeping |

`VulkanDeviceAPI::FreeWorkspace` uses `pool_per_thread.Get()` (lookup only) while
`AllocWorkspace` uses `GetOrMake`, and `pool_per_thread` is a per-thread `ThreadMap`; the CPU device
uses one thread-local accessor for both. Cross-thread workspace use is therefore unsupported.

**This does not affect the model.** Every failure is a clean `TVM_FFI_ICHECK` exception, never a
silent wrong value and never the heap corruption seen on Mesa 22.3.6. It is worth reporting upstream
as a robustness/clarity issue, with `upstream/workspace_repro.cpp` as the artefact.

## 4. Performance — the honest numbers

512×512, fp32, batch 1, 3 different images, 2 warm-up + 5 steady-state runs, host copy **inside** the
timed region so the Vulkan queue is drained (earlier numbers in this project omitted that and were
meaningless):

Production baseline, using the **original** (unpatched) ONNX on Mesa 25.0.7 — 4 images:

| image | ONNX Runtime CPU | TVM Vulkan | speedup | map correlation | score err |
|---|---:|---:|---:|---:|---:|
| tampered2.png | 2183.1 ms | 2590.9 ms | **0.84×** | 1.0000000 | 2.44e-06 |
| tampered1.png | 2197.5 ms | 2623.5 ms | **0.84×** | 1.0000000 | 2.67e-05 |
| pristine1.jpg | 2199.8 ms | 2622.3 ms | **0.84×** | 1.0000000 | 7.38e-06 |
| pristine2.jpg | 2153.0 ms | 2620.0 ms | **0.82×** | 1.0000000 | 4.55e-06 |

The Slice+Squeeze rewrite was verified unnecessary on this driver: the original and the rewritten
model give byte-identical errors (`7.293e-05` / `2.438e-06` / `corr 0.9999999999` / IoU = Dice = 1.0),
so the rewrite is kept **out** of the production path. See `DEPLOY.md`.

For reference, the same ONNX through TVM's LLVM backend takes **~452 s** per image (186× slower than
ORT CPU) — the generic LLVM path is not tuned, and it is included only to show that "it is TVM" is
not the explanation for the Vulkan result.

### Why the GPU loses here

Measured device capabilities: `supports_float16 = false` (Polaris has no fp16 shader support),
`thread_warp_size = 1`, `max_threads_per_block = 256`. The model is 68.7 M parameters over
512×512 activations of a dual-modality SegFormer-B2 — a large activation-bound workload. The Vulkan
kernels used here are the **default schedules**: no MetaSchedule tuning was applied, and TVM's
Vulkan backend is documented as focused on mobile GPUs. Meanwhile the 4-core CPU baseline is already
a highly-tuned oneDNN/MLAS path that handles this graph in ~2.1 s.

This is a *not-yet-optimised* result, not a ceiling. Tuning is explicitly out of scope per the
project rule "do not modify the model to obtain a speedup" — and tuning the compiler, not the model,
would be the legitimate next move.

## 5. Answers to the project's own questions

1. **Can TruFor run on the RX 560 over Vulkan?** Yes — with TVM 0.27.0 and Mesa ≥ 25.0.7, at 512×512,
   numerics identical to the reference.
2. **How much runs on the GPU?** The whole graph. The single Vulkan delegate carries all 1918 nodes.
3. **What still runs on the CPU?** Only preprocessing (image decode/resize) and the softmax/sigmoid
   post-processing done in numpy afterwards.
4. **Is the GPU faster than the CPU?** **No.** 0.81–0.82×. NO MATERIAL GPU GAIN.
5. **Error vs the reference?** map `corr = 1.0000000`, `max_abs = 7.3e-05`; confidence
   `corr = 0.9999999981`; integrity score error 2.4e-06; IoU = Dice = 1.0.
6. **Worth writing custom Vulkan kernels?** Not on this hardware. The bottleneck is not a missing
   kernel; it is that a Baffin-class GPU with no fp16 support and untuned shaders is slower than a
   tuned 4-core CPU for this graph.
7. **Which layer was failing?** **The driver** (Mesa/RADV 22.3.6), not the model, not the compiler,
   not the runtime, not the hardware. One genuinely separate TVM robustness issue remains (§3).

## 6. Route history, for the record

| route | outcome |
|---|---|
| ExecuTorch Vulkan 0.6.0 | **not viable.** Built from source (Vulkan SDK + buck2 + LLVM 18), partitions TruFor into 88 delegates, then aborts during GPU allocation with `free(): invalid pointer`. Independent of Mesa (a different runtime with a different symptom), never reached model execution. |
| MNN 3.6.1 | **not viable.** Converts cleanly (after canonicalising the ONNX and disabling `--convertMatmulToConv`), but `createSession` returns a Session whose shape inference never completes: `Compute Shape Error for cat_26`, all output shapes empty. Same on 3.4.1 / 3.5.0. Vulkan never exercised. |
| **TVM Vulkan 0.27.0** | **works**, once the driver is new enough. |

## 7. Reproducing the working configuration

```bash
# Mesa 25.0.7 unpacked side by side; nothing in the system install changes
apt-get download -t bookworm-backports mesa-vulkan-drivers=25.0.7-2~bpo12+1
dpkg-deb -x mesa-vulkan-drivers_*.deb vendor/mesa-ab/root

export PYTHONPATH=$PWD/vendor/tvm-src/python
export TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build
export LD_LIBRARY_PATH=$PWD/vendor/mesa-ab/root/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=$PWD/vendor/mesa-ab/root/usr/share/vulkan/icd.d/radeon_icd.json

vulkaninfo --summary          # expect driverVersion 25.0.7, apiVersion 1.4.305

vendor/tvm-venv/bin/python scripts/final_benchmark.py \
  --model weights/trufor.pth.tar \
  --onnx reports/phase8a/trufor-512x512-slicefix.onnx \
  --images truFor-src/test_docker/images/*.png truFor-src/test_docker/images/*.jpg \
  --outdir reports/mesa-ab/bench --size 512 --repeats 5 --only ort-cpu,tvm-vulkan
```

## 8. Where the artefacts live

| | |
|---|---|
| Phase 1 CPU baseline | `reports/BASELINE.md`, `artifacts/baseline_cpu/` |
| Operator inventory | `reports/operator-inventory/OPERATOR-INVENTORY.md` |
| Probes | `reports/PROBE-REPORT.md` |
| TVM environment / import / CPU / Vulkan | `reports/TVM-ENV.md`, `TVM-IMPORT.md`, `TVM-CPU.md`, `TVM-VULKAN-REPORT.md` |
| MNN | `reports/MNN-VULKAN-REPORT.md` |
| Phase 8 localisation and its corrections | `reports/PHASE8-STAGE1-LOCALISATION.md`, `PHASE8A-8C-REPORT.md`, `PHASE8C-DIAGNOSIS.md` |
| **Mesa A/B (the decisive one)** | `reports/mesa-ab/{A-mesa2236,B-mesa2507}/`, `reports/mesa-ab/bench/benchmark.json` |
| Verified ONNX model (512×512) | `reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx` |
| Rewritten ONNX (Gather→Slice+Squeeze) | `reports/phase8a/trufor-512x512-slicefix.onnx` |
| Upstream material | `upstream/workspace_repro.cpp` (usable), `upstream/ISSUE-take-select.md` (**do not submit**), `upstream/tvm-vulkan-take-select-repro.py` (driver-sensitive) |
| Environment changes | see §9 |

`truFor-src/` was never modified: `git -C truFor-src status --porcelain` is empty.

## 9. Environment changes made during this investigation

`apt install vulkan-tools gdb python3-venv`; `pip install apache-tvm onnx onnxscript onnxsim ruamel.yaml
zstd torchao==0.10.0 MNN onnxruntime-ep-webgpu protobuf>=6.31.1`; `onnxruntime` 1.24.4 → 1.30.0;
llvm-18-dev from apt.llvm.org; a `bookworm-backports` apt source (packages downloaded, **not**
installed); `torchgen` symlinked into `/usr/lib/python3/dist-packages`; ExecuTorch 0.6.0 built from
source in `vendor/executorch` with its example runner's static memory pool raised 4 MB → 1.5 GB.

## 10. If someone picks this up next

The correct engineering question is no longer "why doesn't it work" — it works. It is
**"can a tuned Vulkan pipeline beat a tuned 4-core CPU on a Baffin-class GPU for a 68.7 M-parameter
SegFormer?"** Concretely, in order of expected value:

1. MetaSchedule tuning against `VulkanTarget` with the real device (`from_device:0` already gives
   the right limits). The default schedule is the obvious first thing to try.
2. Re-measure at lower input resolution — the workload is activation-bound, and the ratio should move.
3. Do **not** ship the Slice+Squeeze rewrite. On a correct driver it changes nothing (verified
   byte-identical), and it only exists to work around Mesa 22.3.6; `DEPLOY.md` records 22.3.6 as a
   known-bad environment rather than something to accommodate.
4. File the cross-thread workspace issue (`upstream/workspace_repro.cpp`) — real, reproduced
   identically on both driver versions, and unrelated to model correctness.
5. Profile before tuning: at 2.62 s vs 2.18 s the gap is only ~0.44 s, so it matters whether that
   is kernel execution or launch/memory/synchronisation overhead. MetaSchedule only pays off if the
   first part dominates.

And record the practical lesson, because it cost this project several days: **a reproducible,
minimal, single-operator-looking failure is still allowed to be a driver bug.** The operator-level
suite, the stage-level bisect and the gdb stack all pointed at TVM; only the one-variable driver
swap settled it.
