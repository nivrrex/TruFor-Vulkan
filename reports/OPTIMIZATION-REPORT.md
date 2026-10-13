# OPTIMIZATION-REPORT.md — TruFor on RX 560 / TVM Vulkan

**Outcome: the target was not reached. The graph is still 2540 ms against a 2317 ms CPU
baseline (0.91x). What was achieved is a complete diagnosis and a verified-but-not-integrated
fix for the single largest defect.** No claim of speedup is made anywhere below.

## Baseline (re-measured in the final harness, 3 warm-up + 15 runs, un-instrumented)

| | value |
|---|---|
| whole graph, median | **2540.1 ms** (min 2524.5, max 2549.0) |
| ONNX Runtime CPU reference | 2317 ms |
| correctness | map corr 1.0000000, IoU 1.000000, Dice 1.000000, score err 2.44e-06 |
| PrimFuncs | 243 |

Category times below come from the instrumented run and are for **ranking only** — per
instruction syncs inflate small kernels, so the whole-graph number is the only judge.

| category | time | rank |
|---|---:|---|
| conv | 1320.9 ms | 1 |
| layout (transpose/reshape/copy) | 612.6 ms | 2 |
| reduce (incl. max4) | 352.9 ms | 3 |
| matmul | 351.8 ms | 4 |
| other + actfn + resize | 109.9 ms | 5 |

Per-instruction sum 2748.2 ms against 2851.5 ms instrumented wall = **96.4%**, meeting the
>=95% acceptance rule. This also closes the ~830 ms that was previously unaccounted for, and
corrects two of my own underestimates (layout 357 -> 612.6, matmul ~180 -> 351.8).

## Verified: `max4` is a degenerate single-threaded reduction

`max4` reduces `(1,1,262144)` to a scalar, 13.4% of the graph (334 ms, and 334.3 ms profile
== 84.2 ms isolated x 4, closing exactly).

```python
for ax0_fused_0 in T.thread_binding(T.int64(1),    thread="blockIdx.x"):
    for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
        T.where(ax0_fused_0 * 256 + ax0_fused_1 < 1)   # 255 threads masked off
        lv2619_red[0,0,0] = -inf
        for ax1 in range(T.int64(262144)):             # one thread, serial
```

The Vulkan reduction schedule derives its grid from the **output** extents, so a scalar output
means `grid = 1`. 1.0 MB then takes 84 ms = 0.012 GB/s against 112 GB/s peak. The same
operator with a per-channel output is fine (`blockIdx.x = 64`).

## Verified fix (isolated, bit-exact) — 140x

| variant on (1,1,262144) | GPU | value |
|---|---:|---|
| as TVM schedules it | 82.98 ms | 0.99999899 |
| reshape 4-D + one reduce | 83.17 ms | — |
| **reshape (1,1,512,512) + two reduces** | **0.59 ms** | 0.99999899 |

`numpy` reference is 0.99999899. All agree, so the split is bit-exact and worth ~332 ms on
the whole graph.

## Why the fix is not integrated

Two graph-level rewrites were implemented and both did nothing (2540.1 -> 2544.4 ms; TIR still
`grid_x = 1`), because **`max4` does not exist in either graph**:

```
lv2617 = R.nn.log_softmax(x:(1,1,262144), axis=-1)
```

`max4` is the numerically-stabilising `max` that **legalisation of `log_softmax`** emits
(`x - log(sum(exp(x - max(x))))`). It appears for the first time in the TIR, after both the
ONNX graph and the Relax IR are gone. A graph rewrite therefore cannot reach it.

The remaining routes, in order of cost:

1. **Keep `log_softmax` as a single kernel — RULED OUT, not merely unattempted.** This was the
   recommended next step; it is not available. `relax.nn.log_softmax` is registered in
   `src/relax/op/nn/nn.cc:249` with only `FInferType`, no `FTVMLegalize`, and the decomposition is
   therefore inside the generic `LegalizeOps` with no configuration hook. Enumerated both
   namespaces: `tvm.relax.transform` and `tvm.tirx.transform` contain no softmax-related pass
   (`IPCAllReduceRewrite`, `LowerAllocTensor`, `LowerIntrin`, ... only). There is no switch to turn
   the decomposition off in 0.27.
2. Rewrite the emitted TIR after legalisation: split the 262144 axis into two reductions, which
   the isolated test showed is worth 140x on that kernel. Reachable, but the graph-rewrite
   infrastructure for it does not exist yet.
3. Fix the scheduler so a reduction's grid comes from the **reduce** axis extent rather than the
   output extent. This is the general fix and would remove the need for 1 and 2. The intended
   method is to call `dl.gpu.Reduction().apply(func, target, False)` on the unscheduled PrimFunc
   and observe where it declines; a first attempt to extract the PrimFunc at the right pipeline
   stage returned zero functions and needs redoing, so this remains open rather than concluded.

## Stage B — attempted, feasibility probe failed

`static_shape_tuning_pipeline(30, target=vulkan)` runs and accepts the target, but
**28 of 30 generated schedules fail to build** (2 survive). A 500-trial budget would not rescue
a 2/30 survival rate, so per the fallback rule this path was abandoned rather than scaled. The
subsequent abort was a missing `xgboost` in the cost model, which is unrelated and fixable.

Net: no convolution tuning was performed. The largest block (~1200 ms by the instrumented
ranking, and the ranking is independently corroborated here: Noiseprint++'s 17 convolutions take
791 ms on the GPU against 50.47 ms each measured standalone) has **no working automatic tuning
path on this stack**.

The plan's remaining option is ncnn's `benchncnn` on the same convolution as an upper-bound
reference. That would answer whether 383 GFLOPS is near the ceiling; it is a measurement, not a fix.

## Retraction added after the rewrite

The rewrite changed the diagnosis. One `log_softmax` really costs **7.2 ms** in isolation, not the
**83 ms** the per-instruction profile charged to `max4`: the instrumented run syncs on both sides
of every instruction, so that number is mostly waiting for neighbours. max4's real cost is
~2 x 7.2 = **14 ms**, not 334 ms.

Therefore **"max4 is 13.4% of the graph" is an instrumentation artefact** and is withdrawn, together
with the claim that max4 is the single largest defect. The instrumented category ranking is biased
towards kernels that happen to sit beside slow ones; the convolution block's position is
corroborated independently and stands, but the reduce block does not.

## Stage B — root cause of the convolution cost (diagnosis, no fix)

Dumping the TIR of `fused_conv2d1_add_relu` (3x3, 64->64, stride 1, 512x512) gives:

```
conv2d_nchw_intermediate_reindex_local = T.sblock_alloc_buffer((1, 262144, 64),  scope="local")
pad_temp_reindex_shared                 = T.sblock_alloc_buffer((1, 262144, 576), scope="shared")
param_0_reindex_shared                  = T.sblock_alloc_buffer((1, 64, 576),     scope="shared")
```

`pad_temp` is **576 MiB**. It is labelled `scope="shared"`, but a Baffin workgroup has
kilobytes of LDS, so it is in practice a global scratch buffer: the schedule materialises
the entire im2col matrix and reads it straight back, once per convolution.

Arithmetic: 1152 MiB per conv, 17 convolutions per pass, 2 passes = **41.1 GB** of memory
traffic. At 100 GB/s that is 411 ms, at 160 GB/s 257 ms, against a measured 791 ms for those
same 17 convolutions. The microbenchmark meanwhile already reaches 383 GFLOPS of a 1120
GFLOPS peak, i.e. 34% of peak.

**So the convolution block is memory-bound, not compute-bound, and tuning the tile sizes
cannot fix it.** The fix is to keep the padded window in registers or in a genuine
per-block shared tile instead of materialising im2col. That is a hand-written `s_tir`
schedule for this one convolution shape, reusable across the 17 instances.

Limit of this diagnosis: the bandwidth arithmetic accounts for 257-411 ms of the 791 ms. The
remainder is likely the same 576 MiB buffer being trafficked through LDS inside each block,
which the estimate does not model. The direction is established; the size of the prize is not.

## Stage B — the fix, tuned, measured in isolation

`scripts/opt/direct_conv.py` is a hand-written `tirx` PrimFunc for the same convolution
(3x3, 64->64, stride 1, 512x512), padding supplied by the caller. Each block computes a 32x32
spatial tile for 16 output channels; 256 threads, each owning 2x2 pixels and 16 channels (64
accumulators), looping over the 64 input channels and their 3x3 taps. Nothing is materialised
and no shared memory is used.

`scripts/opt/direct_conv_sweep.py` searches the same family over `(TILE, TH, TW, OCB)`:

| TILE | threads | OCB | acc/thread | time | throughput |
|---:|---:|---:|---:|---:|---:|
| 32 | 256 | 8 | 32 | 39.66 ms | 487 GFLOPS |
| **32** | **256** | **16** | **64** | **19.59 ms** | **987 GFLOPS** |
| 16 | 256 | 32 | 32 | 29.99 ms | 645 GFLOPS |
| 16 | 256 | 64 | 64 | 34.80 ms | 555 GFLOPS |
| 32 | 1024 | 16 | 16 | 31.08 ms | 622 GFLOPS |
| 64 | 256 | 8 | 128 | 80.27 ms | 241 GFLOPS |
| 64 | 256 | 16 | 256 | 1001.83 ms | 19 GFLOPS |

**987 GFLOPS is 88% of the 1120 GFLOPS peak, against 383 GFLOPS for the pipeline's own schedule
on the same shape — 2.58x.** Spot-checked against a numpy reference: OK.

The deciding variable is **accumulators per thread** (`OCB*PH*PW`), not the tile size:
`acc <= 64` is healthy and `acc = 64` is optimal, while `acc >= 128` spills to scratch and
collapses to 19-241 GFLOPS. Tile size and thread count matter much less.

Across the 17 convolutions this is roughly 520 ms, which would put the whole graph near 2000 ms,
well under the 2250 ms target.

**This is an isolated-kernel number and is not integrated into the graph.** DLight's conv rule is
what emits the 576 MiB buffer, so integration means replacing that rule for this shape inside the
pipeline, not swapping a call. No whole-graph speedup is claimed from it.

### A bad number, retracted

An earlier version of the sweep reported **1771 GFLOPS** at `OCB=24` and 1150 at `TILE=48`. Both
were wrong, and the bug was mine. The grid was computed as `OC // OCB` and `H // TILE`, so
`OCB=24` gave `64//24=2` and silently computed only 48 of the 64 output channels, and `TILE=48`
gave `512//48=10` and computed only 480 of 512 rows. 1771 GFLOPS also exceeds the physical peak,
which is what exposed it. The sweep now asserts both divide exactly. **The corrected best is 987
GFLOPS, not 1771**, and the 11 ms timings that accompanied the bad numbers are void.

## B4d — hand-written NHWC to NCHW, 1.37x

Two of the largest members of the transpose class are pure layout changes with no epilogue,
`out[c,h,w] = in[h*W+w,c]`: `fused_transpose7_reshape8` at 6 calls and `fused_transpose24_reshape25`
at 8. DLight maps the thread to the channel first, so it coalesces the load and scatters the store
across 128*128 floats per thread; measured 34.8 ms for 16.8 MB, 0.96 GB/s.

`scripts/opt/nhwc2nchw.py` inverts the mapping so the thread runs along the spatial axis and the
stores land contiguously: 10.73 ms, 3.13 GB/s, 3.24x in isolation. `swap_conv.py` matches the
two-buffer `(1,H*W,C) -> (1,C,H,W)` signature and substitutes it.

| configuration | subs | median | min | p90 | map corr |
|---|---:|---:|---:|---:|---:|
| original pipeline | 0 | 2476.8 ms | 2447.6 | 2490.6 | 1.0000000 |
| + hand-written conv | 2 | 1890.0 ms | 1873.5 | 1907.7 | 1.0000000 |
| **+ NHWC to NCHW** | **13** | **1779.5 ms** | 1751.4 | 1787.4 | 1.0000000 |

**2476.8 -> 1779.5 ms, saving 697.3 ms or 28.2%, which is 1.37x against ONNX Runtime CPU.** det err
2.46e-05 throughout.

Known limit: 3.13 GB/s is still far from the ~100 GB/s available. A shared-memory tile would
coalesce both the load and the store and needs a barrier inside the kernel; the tirx layer's
shared-memory and barrier API is not established here, so it is stated rather than attempted
blind. There is likely another 100 ms in this kernel alone.

## Final result

Three-way, one session, warmup 3, 15 timed runs, median:

| configuration | median | min | p90 | map corr |
|---|---:|---:|---:|---:|
| original pipeline | 2473.3 ms | 2457.2 | 2506.4 | 1.0000000 |
| **+ hand-written convolutions** | **1887.2 ms** | 1867.2 | 1893.0 | 1.0000000 |
| + vulkan matmul config (B2) | 1932.1 ms | 1914.4 | 1955.0 | 1.0000000 |

det err 2.46e-05 in all three. **2473.3 -> 1887.2 ms, saving 586.1 ms or 23.7%, which is
1.29x against ONNX Runtime CPU at 2438.3 ms.** The whole gain comes from two hand-written
kernels, substituted by buffer signature: 3x3 64->64 at 512x512, called 15 times, and 1x1
2048->512 at 128x128, called twice. Neither requires patching TVM; both are substituted from
outside it.

**B2 is reversed.** In this session the vulkan matmul config makes the whole graph 44.9 ms
slower. Its earlier +0.7% was a cross-session artefact and does not reproduce. Do not enable it.
That is the second cross-session comparison in this project that turned out wrong, after the CPU
baseline drift in A0; everything above is now measured inside one session.

### A limit of the ablation method

The 353 ms "other" class is layer_norm and take. Ablating the largest five individually returned
**negative** savings of -45.7 to -47.1 ms: a no-op body leaves the output buffers holding
uninitialised or denormal data, denormal floats run at a small fraction of normal speed, and the
downstream consumers get slower. Ablation is sound where a kernel's cost is dominated by its own
memory traffic, which is why it worked for conv and transpose, and unreliable where uninitialised
data feeds elementwise consumers. The positive class figures should be read as conservative.

### Remaining headroom

| block | upper bound | why it is hard |
|---|---:|---|
| transpose | 546 ms | the largest single item is 254.5 ms of interpolation and concatenation wearing a transpose's name; it runs at 1.05 GB/s against ~100 GB/s, but the fix is a different algorithm, not a faster copy |
| other | 353 ms | layer_norm and indexing; 826 MB of traffic over 14 calls is about 16 ms against a 1945 ms graph |
| matmul | 203 ms | the one available config change made it worse |

## Not attempted
- **Stage C, layout** (22.3%; the dlight `Transpose` rule is fixed and gives -79.5% on the
  standalone (1,16384,512) transpose, but it is inert on the full graph because
  `ApplyDefaultSchedule` runs after `FuseOps`).
- **Stage D** deliverables: DEPLOY.md update, review package.

## Withdrawn

- "Root cause: TVM's transpose kernel" — only part of the backbone gap, and inert after fusion.
- "0 of 13 cases lost / GPU never useful" — invalidated by transfer-dominated measurement.
- "Fusion is not a cause" — that experiment removed 0 passes.
- Peak bandwidth 56 GB/s — should be 112 GB/s (quad-rate).
- 218 GB/s copy baseline — exceeds the physical peak, invalid.
- max4 as "the detection head's global reduction" — it is a log_softmax stabiliser.

## Reproduce

```bash
export PYTHONPATH=$PWD/vendor/tvm-src/python
export TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build
export LD_LIBRARY_PATH=$PWD/vendor/mesa-ab/root/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=$PWD/vendor/mesa-ab/root/usr/share/vulkan/icd.d/radeon_icd.json

vendor/tvm-venv/bin/python scripts/opt/bench.py --tag A3 --reps 15
vendor/tvm-venv/bin/python scripts/repro_single_output_reduce.py --outdir /tmp/reproout
TVM_VM_PROFILE=1 vendor/tvm-venv/bin/python scripts/opt/profile_graph.py   # per-kernel
```

Artefacts: `work/LOG.md` (append-only work log), `work/last_experiment.json`,
`patches/` (TVM diffs against `4d0ac168c7`), `upstream/` (workspace repro + take/select material).
