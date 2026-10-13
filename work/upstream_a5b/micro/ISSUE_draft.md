# [Draft] Vulkan: `record_upload` corrupts a Mat whose plane stride (cstep) is smaller than the aligned cstep

## Title

Vulkan: uploading a `Mat` with compact external data (e.g. `ncnn.Mat(numpy)` from pyncnn) silently shifts every channel plane when `H*W*elemsize` is not 16-byte aligned and `c >= 2`

## Environment

- ncnn 1.0.20260526 pypi wheel and a local build from master (git HEAD `9f9d4ec`, version string 1.0.20261002) reproduce the issue **bit-for-bit identically**.
- OS: Debian-family Linux (Devuan 5), x86_64.
- GPU A: NVIDIA Tesla P4, proprietary driver.
- GPU B: AMD Radeon RX 460 2GiB, RADV / Mesa.
- CPU path (`use_vulkan_compute = False`) is clean on the same machine, same build.
- All fp16 options disabled (`use_fp16_packed/storage/arithmetic = False`).
- Deterministic across fresh processes. Two physically different GPUs from different vendors produce the **same** corrupted output → not driver-specific.

## Symptoms

An `Input -> Split` two-layer network reproduces it (`Split_vulkan::forward()` is a pure refcount alias and never touches data). Reading back either Split output blob from the GPU returns corrupted data whenever:

```
corrupted  <=>  c >= 2  AND  (H * W) % 4 != 0     (fp32, pack1)
```

`c == 1` is clean at every size; any `c` is clean when `(H*W) % 4 == 0`.

Observed output for input planes `[1..9]` / `[10..18]` at c=2, 3x3 (identical on both GPUs, both builds):

```
[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 13. 14. 15. 16. 17. 18.  0.  0.  0.]
```

Plane 0 is correct; plane 1 is shifted by `align(9,4) - 9 = 3` elements, its first 3 values are gone, and the tail reads bytes that were never written (zeros).

Note: `extract("in0")` looks clean — but it is a **misleading control**: the net returns the cached host-side `Mat` for the input blob, so it never reads the GPU buffer and cannot show the corruption.

## Root cause

`VkCompute::record_upload()` (`src/command.cpp:414` at `9f9d4ec`):

```cpp
const Mat src_b = src_fp16.batch(b);
VkMat staging_b = dst_staging.batch(b);
memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
```

- `dst_staging` is created with `create_like(src_fp16, ...)`, so `VkMat::create()` recomputes the **aligned** plane stride: `cstep = alignSize(w*h*elemsize, 16) / elemsize`.
- A host `Mat` that wraps external data in compact layout has a **smaller** `cstep`. This is not hypothetical: pyncnn does exactly that when wrapping numpy arrays — `python/src/main.cpp` sets `v->cstep = shape[2] * shape[1]` with the comment "the buffer from numpy is not aligned, so we set the cstep as numpy's cstep".
- `Mat::total()` is `cstep * c`, i.e. the *source* footprint. The single contiguous memcpy therefore writes plane `q` at offset `q * w*h` while every GPU consumer reads it at `q * align(w*h,4)` — a per-plane shift — and the last `c * (cstep_aligned - cstep_compact)` bytes of the staging buffer are never written at all.

When `w*h*elemsize` is already 16-byte aligned the two strides agree and everything is fine, which is exactly why the trigger is `(H*W) % 4 != 0`, and why `c == 1` is invisible (only the unused tail padding is affected).

## Suggested fix

Copy plane by plane when the strides differ (single-file patch tested against the mini repro grid, the 967x714 photo case, and a ~2000-layer real network; it also fixes the real-network numerics, see below):

```cpp
        if (src_b.cstep == (size_t)staging_b.cstep && src_b.c == staging_b.c)
        {
            memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
        }
        else
        {
            // src mat may wrap external data in compact layout (plane stride < aligned cstep)
            // copy plane by plane to match staging layout
            const size_t planesize = src_b.total() / src_b.c * src_b.elemsize;
            char* dst_data = (char*)staging_b.mapped_ptr();
            const char* src_data = (const char*)src_b.data;
            for (int q = 0; q < src_b.c; q++)
            {
                memcpy(dst_data + (size_t)q * staging_b.cstep * src_b.elemsize, src_data + (size_t)q * src_b.cstep * src_b.elemsize, planesize);
            }
        }
```

(`record_upload()` for the image path, `command.cpp:614`, and the flatten helper around `command.cpp:2653/2692` look like they carry the same assumption — not verified, buffer path is what we exercised.)

Alternatively/additionally the binding contract could be documented: constructing `ncnn::Mat` over user memory whose `cstep` is smaller than `alignSize(w*h*elemsize,16)/elemsize` is unsafe for Vulkan upload as it stands today.

## Rule grid (before fix)

c = 1..8 x 17 sizes, CORRUPT/ok matches `c >= 2 and (h*w) % 4 != 0` with **zero mismatches** (see `repro_split_pack4.py` section [2]):

| size | h*w % 4 | c=1 | c>=2 |
|---|---|---|---|
| 3x3 | 1 | ok | CORRUPT |
| 4x4 | 0 | ok | ok |
| 3x5 | 3 | ok | CORRUPT |
| 967x714 | 2 | ok | CORRUPT |
| 967x712 | 0 | ok | ok |

Random data at 967x714, c=3: `max_abs = 0.999`, `frac(diff > 1e-6) = 0.667`; at 967x712: exact zero.

## Impact on a real network (before fix → after fix)

Found while running a ~2000-layer pnnx-exported forensics network (TruFor) at native resolution. On an input with `H*W % 4 != 0` (967x714, c=3), the uploaded input planes are already wrong, so everything downstream diverges from the CPU result:

- final `out0` element-wise rel_L2 (GPU vs CPU) = 0.229; content-independent (uniform noise input: 0.268). Aggregate mean/std badly underestimates the damage — it is plane-shift corruption, only element-wise comparison exposes it.
- Task-level on a tamper-detection sample (714x967): AUC 0.4817 (broken GPU) vs 0.6641 (ORT-CPU reference).
- **With the suggested patch**: micro grid clean at all c/size combinations, 967x714 `max_abs = 0`, and the same sample gives AUC 0.6645 / ΔAUC 4.2e-4 vs the CPU reference.

Sizes with `H*W % 4 == 0` were never affected, which is how this survived: most benchmark sizes are round numbers.

## Separate issue (still open, seems unrelated)

On tiny nets (`Input+Split+Conv`, `Input+ReLU`) extracting the output blob of a layer that actually executed (e.g. the conv output, or the ReLU output) SIGSEGVs (exit 139), while extracting Split alias blobs works. It persists with the upload fix above, does not occur on the real ~2000-layer network at the same per-plane shapes, and we have not localized it. Happy to file separately if useful.

## Repro package

- `split_roundtrip.param` (3 lines, above)
- `repro_split_pack4.py` (runs all sections on a chosen device; no weights needed)
