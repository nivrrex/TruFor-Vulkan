# CAPABILITY-REPORT.md — RX 560 + TVM Vulkan: corrected measurement and root cause

**Supersedes the first version of this report in its entirety.** That version was wrong in ways
that inverted the conclusion, and the corrections are recorded below rather than quietly applied.

The single defect that broke everything: **the synthetic operator sweep timed H2D + D2H transfer
together with execution, and the transfer dominated every case by 30-100×.** For instance
`Sqrt` on 1×64×1024² took 184.0 ms measured end-to-end versus 190.9 ms for an `Identity` that
TVM had eliminated entirely — both numbers were transfer, not compute. Every "GPU is N× slower"
ratio in the first version is void.

Re-measured with the input resident on the device and only `dev.sync()` inside the timed region:

| workload | ONNX Runtime CPU | TVM Vulkan | speedup | GPU efficiency |
|---|---:|---:|---:|---|
| Sqrt 64ch 1024² | 31.3 ms | 5.95 ms | **5.26×** | |
| Erf 64ch 1024² | 30.3 ms | 5.90 ms | **5.14×** | |
| softmax 5×4096² | 34.6 ms | 7.90 ms | **4.39×** | |
| conv 3→64 3×3 @512² | 13.7 ms | 3.30 ms | **4.15×** | 275 GFLOPS |
| conv 64→64 3×3 s1 @512² | 69.6 ms | 50.5 ms | **1.38×** | **383 GFLOPS** |
| conv 1×1 2048→512 @128² | 185.5 ms | 89.6 ms | **2.07×** | 96 GFLOPS |
| identity 1×64×128² (copy) | 0.17 ms | 0.04 ms | 4.46× | 218 GB/s |
| matmul 16384×512×512 | 24.4 ms | 22.2 ms | 1.10× | **386 GFLOPS** |
| **transpose 1×64×128×128 NCHW→NHWC** | 0.32 ms | **1.60 ms** | **0.20×** | **5.2 GB/s** |
| **transpose 1×320×32×32** | 0.07 ms | **0.71 ms** | **0.10×** | **3.7 GB/s** |
| **transpose 1×64×256×256** | 2.09 ms | **6.04 ms** | **0.35×** | **5.6 GB/s** |
| depthwise conv (block1 dwconv) | 5.70 ms | 17.20 ms | 0.33× | 1.1 GFLOPS |

**The hardware is not weak.** Convolution reaches 383 GFLOPS, a plain copy 218 GB/s, elementwise
and softmax are 4-5× faster than four CPU cores, and large matmul is at parity. Every arithmetic
class is healthy **except the layout transform**.

## Root cause

```
TVM's Vulkan transpose / layout kernel has no effective tiling
    5 GB/s, i.e. ~40x slower than a plain copy of the same bytes (218 GB/s)
        ↓
the SegFormer backbone contains 320 Transpose + 288 Reshape
    (CMX repeatedly converts between NCHW and NHWC around every cross-attention)
        ↓
each layout op is 3-10x slower on the GPU than on the CPU
    320 x ~0.4 ms ≈ 128 ms of pure loss
        ↓
backbone 0.85x, decoder 0.79x, full model 0.94x
```

The arithmetic is self-consistent: the backbone's GPU-minus-CPU gap is
`1040.0 - 881.8 = 158.2 ms`, and 320 transposes at (0.5 - 0.1) ms accounts for ~128 ms of it, with
the 288 reshapes covering the remainder.

## Module-level measurement (device-resident)

Extracted with `onnx.utils.extract_model` from the production ONNX; the model itself is untouched.

| sub-graph | nodes | CPU | GPU | speedup |
|---|---:|---:|---:|---:|
| A: `rgb → tile` (Noiseprint++, 17 conv) | 34 | 934.4 ms | 792.3 ms | **1.18×** |
| B: `rgb → cat_24` (A + SegFormer backbone) | 1852 | 1816.2 ms | 1832.3 ms | 0.99× |
| C: full model | 1918 | 2316.8 ms | 2464.5 ms | **0.94×** |
| backbone = B − A | 1818 | 881.8 ms | 1040.0 ms | **0.85×** |
| decoder + heads = C − B | 66 | 500.7 ms | 632.2 ms | **0.79×** |

Noiseprint++'s 791 ms is fully accounted for by 17 × 50.5 ms of 64→64 convolution — it was never
the problem. The loss is concentrated in the backbone and the decoder heads, and the backbone is
where the transposes are.

## Where the FLOPs are

Computed from ONNX shape inference over the whole graph:

| op | count | GFLOP | share |
|---|---:|---:|---:|
| Conv | 111 | 117.72 | 54.3% |
| MatMul | 192 | 66.55 | 30.7% |
| Gemm | 10 | 32.51 | 15.0% |
| **total** | 345 | **216.78** | |

Full model averages: GPU 216.78 GFLOP / 2.464 s = **88 GFLOPS**, CPU 216.78 / 2.317 = **94 GFLOPS**.
The two are within 7% of each other, and the residual gap is layout overhead, not arithmetic.

Backbone operator mix: Transpose 320, Add 304, Reshape 288, MatMul 268, Mul 136,
LayerNormalization 114, Conv 107, Gather 80.

## Hypotheses tested and rejected

| hypothesis | experiment | result |
|---|---|---|
| memory clock is stuck low | `power_dpm_force_performance_level=high`; sysfs `*` marker confirmed it really was at 300 MHz under load, and radeontop agreed (0.42–0.48 GHz) | **rejected.** mclk went to 1.75 GHz (3.6–4.2×) and the numbers moved by −1.5% to −11% |
| `thread_warp_size=1` forces scalar codegen | set to the real `subgroupSize` (64), rebuilt | **rejected.** elementwise 190.9→191.9 ms, matmul 5.39→5.42, softmax 279.8→278.4. Reverted; `vendor/tvm-src` is byte-identical to upstream |
| operator fusion is at fault | removed `FuseOps`/`FuseTIR` from the pipeline | **rejected.** −0.4% / −0.2% / +0.5% on the three sub-graphs |
| depthwise convolution | measured; also counted its share | **not the cause.** 0.33× but only 0.3% of conv MACs |
| convolution kernel quality | measured 3→64, 64→64, 1×1 from the real graph, and dumped TIR | **healthy.** shared-memory buffers, `blockIdx 2×8192`, `threadIdx 8×8`, `T.grid(8,4,4)`, vectorized inner loops; 275–383 GFLOPS |

## Corrections to the previous version of this report

1. **Peak figures.** The old report used ~240 GB/s and ~794 GFLOPS. RX 560 is 128-bit GDDR5;
   TechPowerUp lists 112 GB/s (7 Gbps). This card's DPM table tops out at **1750 MHz**, which
   corresponds to 56 GB/s. Measured copy bandwidth of 218 GB/s exceeds both, so the numbers are
   not mutually consistent — the small-tensor copy is partly L2-resident, and the
   effective-bandwidth figures above should be read as ratios, not absolute ceilings.
2. **"0 of 13 cases won" → void.** All of those numbers were transfer. With device-resident timing
   the GPU wins most classes; `total` in the old JSON was 13, not the 23 the prose claimed.
3. **"1.2% of peak bandwidth" → void.** That was `536 MB / 190.9 ms`, i.e. transfer.
4. **Gate 2 wording.** `d2h = 2477 ms` is a synchronisation wait, and
   `timestamp_query_available: false` — there is no GPU-side timing. The JSON's
   `host_overhead_share: 0.9997` contradicts the 0.35% in the prose, and the script's
   `A_kernel_bound`/`B_overhead_bound` were both `true`, so the classification code was wrong too.
   The defensible statement is only: host-visible overhead outside `d2h` is ~8.5 ms.
5. **`vgt ≈ 0%` is not evidence.** It is a graphics-pipeline counter and is expected near zero
   under pure compute.
6. **"never worth using the GPU at any size" → withdrawn.** Withdraw. The GPU is faster on
   elementwise, softmax, convolution and large matmul, and is within 7% of the CPU overall.

## What this points at

The optimisation target is now specific and it is not the hardware, not fusion, not the clock:
**TVM's Vulkan transpose / layout codegen.** Three ways forward, cheapest first:

1. **Check why the transposes exist at all.** `ConvertLayout` should absorb NCHW↔NHWC around
   convolutions. If 320 explicit transposes survive into the executable, a layout pass is not
   firing, and removing them is cheaper than making them fast.
2. **Tile the transpose kernel.** 5 GB/s versus 218 GB/s for a copy is what an untiled strided
   access looks like. Shared-memory staging, the same fix the conv kernel already uses, is the
   obvious candidate.
3. **A/B against another Vulkan backend** (ncnn) on the same driver and the same model, now that
   the measurement is trustworthy. This is finally a meaningful experiment.

## Known gaps in the harness

- Conv coverage in the first sweep failed for a reason that was misreported: the JSON says
  `Ndim > 5 not supported for convolution`, which is a rank-inference failure on the Conv operands.
  Convolution was subsequently measured properly by extracting real sub-graphs from the model.
- VRAM cannot be read from `/sys/class/drm/card0/mem_info_vram_*` on this kernel; `radeontop` is
  the only source (25.95% / 1057 MB observed under load).
- Gate 6 is only partially answered: batch 8/16/32 were never attempted because TVM's ONNX
  frontend rejects batch > 1 on the detection head's dynamic `Reshape`. ONNX Runtime runs those
  batches correctly, so the model is fine; the toolchain is not.
- Timing protocol: warm-up 3, 15 repeats (6 for the sweeps), min/p10/median/p90/max. Scripts in
  `scripts/bench/`.
