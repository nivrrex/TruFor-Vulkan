# TVM-CPU.md — Phase 6C: TVM LLVM correctness gate

**Result: PASS.** TVM reproduces the verified reference to float32 rounding.

## Build and run

```
tvm 0.27.0  target=llvm
ORT CPU reference: 2.30s  det=0.98586897
import ok
build ok in 8.7s
outputs: [[1, 2, 512, 512], [1, 1, 512, 512], [1, 1]]
```

Input: `truFor-src/test_docker/images/tampered2.png` resized to 512×512, float32
in [0,1] — identical tensor for both runtimes.

## Numerical equivalence (TVM LLVM vs ONNX Runtime CPU, same ONNX file)

| output | metric | value |
|---|---|---|
| integrity score | abs err | **2.631e-06** |
| | ref / tvm | 0.98586897 / 0.98586634 |
| localization map | max abs err | 7.313e-05 |
| | mean abs err | 2.436e-06 |
| | RMSE | 4.968e-06 |
| | **correlation** | **0.9999999999** |
| | IoU @0.5 | **1.000000** |
| | Dice @0.5 | **1.000000** |
| confidence map | max abs err | 3.600e-05 |
| | mean abs err | 3.996e-06 |
| | RMSE | 4.985e-06 |
| | **correlation** | **0.9999999981** |

**CORRECTNESS: PASS** (gate: max abs err < 1e-3, score err < 1e-4).

IoU and Dice are both exactly 1.0 because the binarized localization maps are
identical at threshold 0.5 — the maps are not merely close, they segment the
same pixels.

## Performance: 486.10 s per image

| runtime | latency | ratio |
|---|---:|---:|
| ONNX Runtime CPU | 2.30 s | 1.0× |
| **TVM LLVM CPU** | **486.10 s** | **0.0047×** (211× slower) |
| TruFor PyTorch CPU (same image, from BASELINE.md) | 3.69 s | — |

This does **not** gate the Vulkan path and is not evidence against TVM's Vulkan
backend, but it should not be glossed over either:

- the LLVM backend here is the generic, untuned path — no autotuning, no
  vectorisation hints, one `mean`/`amax` over dims (2,3) lowering to a scalar
  reduction per element, and 324 `Transpose` + 297 `Reshape` in the graph whose
  layout conversions are not folded away;
- the imported Relax IR shows `Globals (#382)` of which the bulk are
  `transpose*`, `matmul*`, `strided_slice*` — i.e. a lot of the time is very
  likely spent on layout shuffling, not on the model's arithmetic;
- TVM's normal answer to this is `relax_pipeline` tuning / layout
  transformations / MetaSchedule (which needs a training-free tuning pipeline
  per target). None of that was applied here because the CPU numbers are not the
  gate — Phase 6D/6E are about the Vulkan target.

I am recording the 211× gap rather than trying to fix it: tuning the LLVM
backend would not make the RX 560 any faster, and it would consume the budget
that Phase 6D needs.

## API notes for TVM 0.27 (all found the hard way)

- `relax.build(ir, target="llvm")` returns a `VMExecutable`. Neither
  `rt_mod["main"](x)` nor `rt_mod(x)` works — the bytecode entry point is not
  exported as a plain module function, and `Executable.jit().get_function("main")`
  fails with `Module has no function 'main'`.
- The working pattern (from `tests/python/relax/test_transform_codegen_pass.py`)
  is:
  ```python
  exe = relax.build(ir, target="llvm")
  vm  = relax.VirtualMachine(exe.jit(), tvm.cpu())
  out = vm["main"](tvm_input)      # returns a tvm_ffi.Array of 3 tensors
  ```
  `tvm.runtime.VirtualMachine` does not exist; it is `relax.VirtualMachine`.
- Output unpacking: `out` is a `tvm_ffi.Array`, not a list/tuple. Index it
  (`out[0]`, `out[1]`, `out[2]`) and call `.numpy()` on each element. `tvm.numpy()`
  was removed in 0.27, and the elements do not support `np.asarray`.
- `relax_pipeline="zero"` (direct codegen, no VM) is **not** a workaround: it
  fails with `CodeGenVM cannot emit this Relax operator directly` on
  `R.call_tir(fused_conv2d_add_relu, ...)` because it needs the lowering passes
  to have run first.
- One `T.topi` warning is emitted ~30 times during build and is benign:
  `Fast mode segfaults when there are out-of-bounds indices`.

## Reproduce

```bash
vendor/tvm-venv/bin/python scripts/tvm_cpu_gate.py \
  --onnx reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx \
  --image truFor-src/test_docker/images/tampered2.png \
  --outdir reports/tvm
```

Artefact: `reports/tvm/tvm-cpu-report.json`.
