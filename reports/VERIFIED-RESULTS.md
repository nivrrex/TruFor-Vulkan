# VERIFIED-RESULTS.md

Every number here was measured, with the method stated. Predictions that did not
hold are in the same table as the rest, marked. Claims made earlier in this
project and later withdrawn are in the last section, because a reader who saw
them needs to know they were wrong.

## Environment

Read only, nothing left changed on the system:

| | |
|---|---|
| GPU | Baffin board, driver reports "AMD Radeon RX 460 Graphics (RADV POLARIS11)", PCI ID covers RX 460/560D/560. 896 SP (56 CU), **not 448**. `power_dpm_force_performance_level = auto` |
| `pp_dpm_sclk` levels | 214 / 387 / 843 / 995 / **1062** MHz (the first three are all of `pp_dpm_sclk`; the earlier "843" was the highest only because three lines had been read) |
| CPU | 4 physical cores, `scaling_governor = powersave`, `intel_pstate` |
| TVM | 0.27.0 at commit `4d0ac168c7`; the kernels need no TVM change |
| Mesa | 25.0.7 side by side; the bundled 22.3.6 produces silently wrong output |
| ONNX Runtime | 1.30.0, `ORT_ENABLE_ALL`, thread count left at the library default |
| Peak used throughout | **1903 GFLOPS** = 896 SP x 2 (FMA) x **1.062 GHz** (highest DPM tier). The 2107 used before this revision came from a 1.176 GHz figure that is not a tier on this card. This is the third peak error in this project. |

## Headline

Single image, 512x512, ABAB interleaved in one session, 15 timed runs each after 3
warmups, 5 real images:

| path | median | IQR width | vs stock |
|---|---:|---:|---:|
| **GPU, stock DLight schedules, 0 substitutions** | **2468.9 ms** | 24.2 ms | 1.00x |
| GPU, 26 substitutions, scalar layout | 1116.0 ms | 5.8 ms | 2.212x |
| **GPU, 26 substitutions, tiled layout (delivered)** | **1067.4 ms** | 2.5 ms | **2.313x** |

All rows built fresh in one session from the same ONNX, 15 timed runs each after
3 warmups, 5 real images. An earlier revision of this table quoted 2478.1 ms for the stock row
and a 1.919x ratio; the 2478.1 came from an older session and is withdrawn, and
the ratio there divided by a CPU median from a different session.

Measured against ONNX Runtime CPU in a separate same-session run: CPU 2119.9 ms
(IQR 44.1), giving 2.313 x x 2119.9 / 2468.9 = 1.988x. The headline uses the
**GPU-side** ratio because it is a like-for-like comparison of two builds of the
same graph; the CPU-crossing ratio is 1.99x and both should be read as ~2x given
the ±2% cross-session drift on the GPU side.

Throughput, `scripts/opt/tile_throughput.py`, 8 tiles:

| configuration | tile/s | vs |
|---|---:|---|
| CPU only, ORT 4 threads | 0.443 | — |
| GPU only | 0.892 | 2.01x |
| **CPU worker + GPU worker over a shared queue** | **1.080** | **2.44x over CPU** |

(Measured before the tiled kernels landed; the GPU lane would now be faster.)

Per-tile outputs agreed between lanes on 8 of 8 tiles in every setting.

## Correctness

Two different comparisons, and they are not interchangeable.

**Per kernel, at substitution time**, each candidate against the PrimFunc it
replaces, full tensor, and rejected if it does not reproduce:

| family | worst rel_L2 | buffers |
|---|---:|---:|
| scalar max (`max4`) | 0.00e+00 | 2 |
| NCHW/NHWC + GELU layout | 0.00e+00 | 2 |
| depthwise 3x3 | 2.36e-08 | 4 |

These zeros are legitimate: a maximum is order-independent and a layout change
is a permutation, so both are exact by construction. An earlier revision of this
document extended that to a "whole graph" row of 0.00e+00, which was **never
measured**; the two zeros it came from were 26-vs-25 substitutions and
tiled-vs-scalar.

**The acceptance criterion** is the whole 26-substitution graph against a
**freshly built zero-substitution graph**, same session. Measured properly, all
three images, all three outputs:

| image | map | conf | det | map corr |
|---|---:|---:|---:|---:|
| tampered2.png | 3.06e-07 | 5.66e-07 | 2.29e-07 | 1.0000000 |
| tampered1.png | 3.06e-07 | 5.66e-07 | 2.29e-07 | 1.0000000 |
| pristine1.jpg | 3.06e-07 | 5.66e-07 | 2.29e-07 | 1.0000000 |
| pristine2.jpg | 3.06e-07 | 5.66e-07 | 2.29e-07 | 1.0000000 |

**Against ONNX Runtime CPU**, which additionally carries ORT's own numerical
differences, on 5 images:

| image | map | conf | det | map corr | score abs diff |
|---|---:|---:|---:|---:|---:|
| tampered2.png | 1.33e-05 | 1.90e-05 | 1.96e-04 | 1.0000000 | 2.47e-05 |
| tampered1.png | 4.33e-05 | 4.04e-05 | 6.75e-05 | 1.0000000 | 6.70e-06 |
| pristine1.jpg | 1.21e-04 | 8.38e-05 | 1.54e-04 | 1.0000000 | 3.33e-05 |
| pristine2.jpg | 1.31e-04 | 1.06e-04 | 5.74e-05 | 1.0000000 | 1.04e-05 |

The worst figure, 1.96e-04, is on the scalar detector output, which is a squashed
sigmoid: a score difference of 2.5e-05 in absolute terms is what produces it.
Map correlation is 1.0000000 on every image.

`det_err` is **not** used as an acceptance criterion. It read 2.46e-05 under every
kernel change in this project, including changes that produced completely wrong
output, so it carries no information.

## The hand-written kernels

All substituted by buffer signature from `scripts/opt/swap_conv.py`, on the
Python side just before `relax.build`. No TVM change is needed to use them. 26
substitutions in total.

| kernel | shape | sites | isolated | DLight equivalent |
|---|---|---:|---:|---:|
| `scalar_reduce` max | (1,1,262144) -> (1,1,1) | 1 | **0.955 ms** | 82.5 ms in-graph, 96x |
| `scalar_reduce` channel max | (1,128,128,128) -> (1,128,1,1) | 4 | **0.936 ms**, 20.6x | 15.476 ms |
| `direct_conv` 3x3 64->64 | (1,64,512,512) | 1 | 16.45 ms, 1175 GFLOPS = 62% of the 1903 peak | 50.47 ms, 383 GFLOPS |
| `depthwise` 3x3 | 8 signatures | 8 | 2.36 ms, 28.47 GB/s | 21.01 ms, 3.20 GB/s, 8.9x |
| `nhwc2nchw` tiled | (1,H*W,C)->(1,C,H,W) | 11 | **1.19 ms, 28.30 GB/s** (3.25x over the scalar 8.71) | 34.8 ms, 0.96 GB/s |
| `nhwc2nchw` + GELU | 4 signatures | 4 | 5.55 ms, 6.05 GB/s | 4.60 GB/s, 15.7x |
| `direct_conv_1x1` | (1,2048,128,128) | 1 | 46.0 ms, 748 GFLOPS | ~54 ms, 634 GFLOPS |
| `decoder_split` stage 1 x4 + stage 2 | 4 transposes + resize/concat, 2 call sites | 2 | 38.1 ms per call | 100.2 ms per call in-graph |
| `resize_concat` vector store, VEC=4 | stage 2 alone, (1,2048,128,128) | 1 | **5.45 ms**, 32.82 GB/s | 24.58 ms, 7.28 GB/s, 4.51x |

### The second largest win, corrected

`nhwc2nchw` at VEC=8 was measured at 8.71 GB/s and later at 28.30 GB/s with a
shared-memory tile, 3.25x in isolation. The first whole-graph measurement of the
tiled version reported 1.8 ms and concluded the family was not a cost worth
chasing. **That measurement was invalid**: swap_conv had been changed to pass
`tiled=True` unconditionally, so the "scalar" arm of the A/B built the tiled
kernel too. Two exported executables were byte-identical, md5
af3fdbc1d2d800a0. With tiled made a parameter the two builds differ and the
gain is **48.6 ms, 4.4 percent**, against a prediction of about 60 ms. The
assertion that was missing is on the substitution's **content**, not its count.

### The largest single win: max4

`max4` alone was **330.1 ms of a 1456.9 ms run, 22.7 percent**, and it was written
off for most of this project. The schedule derives its grid from the OUTPUT
extents; a scalar output means one block, 255 of its 256 threads masked, and one
thread scanning 262144 elements serially. A maximum over 1 MB should take
microseconds at 100 GB/s; it took 82.5 ms per call. Replacing it with 512 threads
over disjoint chunks and a 9-step tree reduction through shared memory gives
0.855 ms, and the substitution is **bit-exact**: rel_L2 0.00e+00 on the whole
graph, all three outputs.

### A latent bug in the shipped scalar max, found and fixed

The tree reduction in `scalar_reduce` started its stride at NT instead of NT/2,
so the first step read `sh[t + NT]`, one past the array; and there was no barrier
between steps, with every thread writing the output, which ThreadSync cannot
insert because the reads sit inside an `if_then_else`. **Both passed two
independent full-tensor verifications on the scalar shape** and shipped. Writing
the per-channel variant, which has 128 blocks instead of 1, failed 12 out of 12 and
exposed them. Now: stride from NT/2, `T.tvm_storage_sync("shared")` at the end of
every step, and only thread 0 writes the output. Re-verified at four thread counts,
12 repeats each, max_abs 0.00e+00. The whole-graph rel_L2 is unchanged, so the bug
was benign in practice, but the kernel is now provably correct rather than
accidentally correct.

**Accumulator count per thread decides the convolution kernels.** For the 3x3
64->64 sweep: acc=64 gives 1206 GFLOPS, acc=128 about 320, acc=256 collapses to
17-20 as the accumulators spill out of the register file.

## Where the time goes, and what is left

Measured with `TVM_VM_SKIP`, one process per function, 15 runs each, savings
against a linearly interpolated baseline because the two bracketing baselines
drifted 1.0 percent over the sweep. Baseline 1456.86 ms, before max4 was fixed:

| function | sites | saving | share |
|---|---:|---:|---:|
| max4 | 1 | 330.1 ms | 22.7% |
| fused_conv2d1_add_relu | 15 | 242.9 ms | 16.7% |
| fused_transpose64_reshape62_... (decoder fusion) | 2 | 180.3 ms | 12.4% |
| fused_conv2d30_add55_relu13 | 2 | 75.6 ms | 5.2% |
| fused_reshape27_transpose25_... (GELU) | 8 | 52.1 ms | 3.6% |
| fused_matmul29_add30_add33 | 12 | 49.8 ms | 3.4% |
| fused_reshape10_transpose8_... (GELU) | 6 | 45.8 ms | 3.1% |
| fused_transpose24_reshape25 | 8 | 44.6 ms | 3.1% |
| fused_matmul28_add34 | 12 | 40.8 ms | 2.8% |
| fused_matmul16_add22 | 10 | 39.9 ms | 2.7% |
| top 20 total | | 1347.7 ms | 92.5% |

max4 is now done, so the remaining headroom, in order, is the decoder fusion at
180.3 ms, the 1x1 convolution at 75.6 ms, then a cluster of 40 to 52 ms items
dominated by matmuls and GELU-fused layout changes.

**The decoder fusion is now wired in.** It was recorded as structurally
impossible for most of this project, on the grounds that `GlobalVarNode` has one
read-only member and so a new PrimFunc cannot be introduced from Python. That
was wrong: `relax.BlockBuilder(mod).add_func(func, name)` returns a typed
GlobalVar, and the rewrite was blocked only because it was not used. Worth
130.4 ms, 12.3 percent, against a prediction of 103.

## The instrumented profile, re-evaluated

`TVM_VM_PROFILE` was previously dismissed as an unreliable ranking tool. Against
SKIP ground truth it is a **consistent small positive bias**, not a sign
inversion, and is usable for ranking:

| function | instrumented | SKIP |
|---|---:|---:|
| decoder fusion | 197.5 ms | 180.3 ms (+9.5%) |
| 3x3 convolution, 15 calls | 258.2 ms | 247 ms measured directly (+4.5%) |
| max4 | 327.3 ms | 330.1 ms (−0.9%) |

The class-level ablation table from an earlier session is a different matter and
is invalid: it reported 2 of 41 substitutions applied, because generated TIR was
silently failing to parse for the rest.

## Withdrawn claims

Each of these was stated in this project and is contradicted by its own later
measurement.

1. **"1.37x, then 1.48x."** Both divided the GPU by a CPU median from a different
   session. The same-session figure is **1.919x**.
2. **"max4 costs 0.2 ms, and the original 334 ms claim was a profiling artefact."**
   Both wrong in the same direction. The original 334 ms was right: SKIP measures
   330.1 ms and the instrumented profile 327.3 ms, agreeing to 1 percent, and
   330.1/4 is 82.5 ms per call against the 84.2 ms standalone reproduction from
   the first diagnosis. The 0.2 ms came from the class ablation table, which was
   invalid because 39 of its 41 substitutions had silently failed. This was
   retracted wrongly once and has now been retracted back.
3. **"Shared memory returns zeros in the flat tirx form."** A bug in my probe, not
   a property of the form: it never wrote its output buffer, and its tree index
   skipped the whole upper half at the first step. With the output written and the
   standard tree, both an explicit `tvm_storage_sync` and leaving the barrier to
   ThreadSync are correct.
4. **"A 1771 GFLOPS convolution."** The sweep divided by `OC // OCB` with OCB=24,
   computing 48 of 64 output channels. The impossible number, above the 2107
   peak, is what exposed it.
5. **"19.59 ms for the 3x3 kernel."** That was a minimum quoted as representative;
   the median over 15 runs was 22.83 ms.
6. **"Peak is 1120 GFLOPS."** 896 lanes without the FMA factor of two. 987 GFLOPS
   is 47% of peak, not 88%.
7. **"The depthwise kernels are under 1 GFLOP each, not worth a kernel."** The
   wrong criterion; they are memory bound, and they were the largest remaining
   block at 293 ms.
8. **"DLight's 1x1 convolution runs at about 909 GFLOPS."** Self-contradictory,
   since that is faster than the kernel that replaced it; about 634.
9. **"powersave is the main cause of the CPU baseline drift."** Measured and false:
   switching to performance is 1.3 percent *slower* and the IQR does not narrow.
   The cause remains unidentified.
10. **"A no-op ablation is slower because of denormal floats."** Never tested; the
    constant-fill rewrite could not be made to run. The SKIP method replaced it
    and is the ground truth in this document.
11. **"Draining a budget: decoder fusion ceiling is 50 ms."** That was the
    single-fused-kernel ceiling, true and irrelevant. The five-kernel split
    estimates 50.8 ms per call, and SKIP puts the real in-graph cost at 100.2 ms
    per call over two calls.

## Reproducing

```sh
vendor/tvm-venv/bin/python scripts/opt/run_trufor.py --bench --reps 15 IMAGE
vendor/tvm-venv/bin/python scripts/opt/tile_throughput.py --tiles 8
```

`run_trufor.py` prints every substitution with its verification result and falls
back to the stock schedules for anything that does not match. See `DEPLOY.md`
Step 7.
