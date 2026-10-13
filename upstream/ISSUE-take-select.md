# [Vulkan] dim-0 integer indexing (take/select) returns wrong elements when the source tensor's second-to-last dimension is 1

## Summary

On the TVM Vulkan backend, `t[k]` (dim-0 integer index, lowered to a take/select)
computes the wrong result when the source tensor's **second-to-last** dimension is
`1`. The same graph is **bit-exact** on the LLVM backend, and exact on Vulkan for
shapes that differ only in that dimension.

No model, weights or framework dependency is involved — the reproducer uses
`torch.randn` and a one-line module.

## Environment

| | |
|---|---|
| TVM | 0.27.0 (`4d0ac168c7`, built from source with `USE_VULKAN=ON`) |
| LLVM | 18.1.8 |
| Vulkan SDK | 1.4.341.1 |
| GPU | AMD Radeon RX 460 Graphics (RADV POLARIS11), Baffin, `device_type=discrete` |
| Driver | Mesa 22.3.6, `DRIVER_ID_MESA_RADV`, Vulkan API 1.3.41 |
| OS | Debian 12, x86_64 |
| Python / torch | 3.11.2 / 2.7.1+cpu |

Device capabilities reported by `{"kind":"vulkan","from_device":0}`:
`supports_int64=true`, `supports_float16=false`, `thread_warp_size=1`,
`max_threads_per_block=256`, `max_shared_memory_per_block=65536`.

## Reproducer

`tvm-vulkan-take-select-repro.py` (attached / in this gist). Run with:

```bash
python tvm-vulkan-take-select-repro.py
```

It builds each case twice from the same ONNX — once for `{"kind":"vulkan","from_device":0}`
and once for `{"kind":"llvm"}` — and reports the relative error against eager PyTorch.

## Observed

```
shape              op         vulkan_rel    llvm_rel      status
[2, 1, 64, 1, 1]   t[0]       9.992e-01     0.000e+00     WRONG
[2, 1, 64, 1, 1]   t[1]       1.000e+00     0.000e+00     WRONG
[2, 1, 64, 1, 1]   t[0],t[1]  1.009e+00     0.000e+00     WRONG
[2, 1, 1, 128, 128] t[0]      0.000e+00     0.000e+00     OK
[2, 1, 1, 128, 128] t[1]      0.000e+00     0.000e+00     OK
[2, 1, 1, 128, 128] t[0],t[1] 0.000e+00     0.000e+00     OK
[2, 1, 64, 128, 1]  t[0]      0.000e+00     0.000e+00     OK
[2, 1, 64, 128, 1]  t[1]      0.000e+00     0.000e+00     OK
[2, 1, 64, 128, 1]  t[0],t[1] 0.000e+00     0.000e+00     OK
[2, 1, 64, 1, 2]    t[0]      1.189e+00     0.000e+00     WRONG
[2, 1, 64, 1, 2]    t[1]      1.000e+00     0.000e+00     WRONG
[2, 1, 64, 1, 2]    t[0],t[1] 1.039e+00     0.000e+00     WRONG
[2, 1, 64, 1]       t[0]      9.993e-01     0.000e+00     WRONG
[2, 1, 64, 1]       t[1]      1.008e+00     0.000e+00     WRONG
[2, 1, 64, 1]       t[0],t[1] 1.000e+00     0.000e+00     WRONG
```

Across a wider sweep of 10 shapes x 3 ops, 24/30 combinations are wrong on Vulkan and
**30/30 are exact on LLVM**. The wrong values are not noise: for `t[0], t[1]` on
`[2,1,64,1,1]` the correlation against the reference is about **-0.10**, i.e. the two
halves come back essentially unrelated to the expected data.

## Trigger condition

`t[k]` is wrong exactly when the **second-to-last** dimension of the source is `1`.

| shape | dim -2 | result |
|---|---:|---|
| `[2,1,64,1,1]` | 1 | WRONG |
| `[2,1,32,1,1]` | 1 | WRONG |
| `[2,1,64,2,1]` | 1 | WRONG |
| `[2,1,64,1,2]` | 1 | WRONG |
| `[2,1,64,1]` (4-D) | 1 | WRONG |
| `[2,3,64,1,1]` | 1 | WRONG |
| `[8,1,64,1,1]` | 1 | WRONG |
| `[2,1,64,128,1]` | 128 | **OK** |
| `[2,1,1,128,128]` | 128 | **OK** |

The variation along dim 0, dim 1 and the last dimension does not matter
(`[2,...]`, `[8,...]`, `[2,3,...]` all fail); the last dimension being 1 is
sufficient on its own. Note `[2,1,64,128,1]` is correct despite `dim4 == 1`, so it
is specifically the *second-to-last* extent that matters.

## It is only the index

On the offending `[2,1,64,1,1]` tensor every other operation is exact on Vulkan:

| op | rel error |
|---|---:|
| identity | 0.0 |
| elementwise `c*2+1` | 0.0 |
| `sum` reduction | 7.1e-08 |
| `reshape(-1)` | 0.0 |
| `slice` `c[:,:,0:64,0:1,0:1]` | 0.0 |
| **`c[1]`** | **1.095e+00** |

## Workaround

Replacing the take with a slice plus a squeeze is exact:

| formulation | result |
|---|---|
| `c[1]` | **WRONG** (rel 1.382) |
| `c[1:2].squeeze(0)` | **OK** (rel 0.0) |
| `c.narrow(0, 1, 1).squeeze(0)` | **OK** (rel 0.0) |
| `c.squeeze(1)[1]` | **WRONG** (still a take) |
| `c.reshape(2,64,1,1)[1]` | **WRONG** (still a take) |
| `c.flatten(0,1)[1].reshape(1,64,1,1)` | **WRONG** (still a take) |

In ONNX terms, `Gather(X, [k], axis=0)` can be replaced by
`Slice(X, starts=[k], ends=[k+1], axes=[0])` followed by `Squeeze(axis=0)`, which is
semantically identical and produces bit-identical results on the CPU backend.

## Notes

- Enabling `VK_LAYER_KHRONOS_validation` produces **no** diagnostics for these cases,
  so the generated SPIR-V and the Vulkan API usage appear legal; the defect is not
  reported at the validation layer.
- This report is deliberately scoped to the indexing bug alone. It is
  **not** a claim that it explains any larger model failure — in the workload that
  motivated it, removing every such take from a ~1900-node graph changed the final
  result by less than 0.003 in correlation, i.e. not at all. That separate,
  still-unresolved behaviour (correct individual kernels, correct individual
  sub-graphs, wrong result and host heap corruption at whole-graph scale) is being
  tracked separately and is out of scope here.
- Only one driver (RADV 22.3.6) was available, so a driver interaction has not been
  excluded. The fact that the *same binary and the same kernels* are bit-exact on the
  LLVM backend, and that the failure is selective in tensor shape rather than
  uniform, points at TVM's Vulkan codegen, but that is inference, not a measurement.
