# PHASE8-STAGE1-LOCALISATION.md — where TVM-Vulkan first goes wrong

**Root cause found, with a minimal reproducer that has nothing to do with TruFor, and a working workaround.**

`TVM 0.27.0`'s Vulkan backend computes **dim-0 integer indexing (`take`/`select`)** incorrectly when the
source tensor has any size-1 dimension. On the RX 560 it returns the wrong elements; the LLVM backend
on the same graph is bit-exact.

## The chain of evidence

### 8A / 8C — walk stage 1 in execution order

Each stage-1 sub-module was exported on its own from the **real** captured inputs and run on
TVM-Vulkan (`scripts/phase8_stage1_bisect.py`):

| sub-module | rel_max vs max | corr | verdict |
|---|---:|---:|---|
| `patch_embed1` | 5.78e-07 | 1.00000000 | OK |
| `extra_patch_embed1` | 9.08e-07 | 1.00000000 | OK |
| `block1[0]` | 4.05e-07 | 1.00000000 | OK |
| `block1[1]` | 4.08e-07 | 1.00000000 | OK |
| `block1[2]` | 2.17e-07 | 1.00000000 | OK |
| `extra_block1[0]` | 1.91e-07 | 1.00000000 | OK |
| `extra_block1[1]` | 1.80e-07 | 1.00000000 | OK |
| `extra_block1[2]` | 1.85e-07 | 1.00000000 | OK |
| `norm1` | 2.58e-07 | 1.00000000 | OK |
| `extra_norm1` | 5.38e-07 | 1.00000000 | OK |
| **`FRMs[0]`** | **1.66e-01** | **0.98026444** | **DIVERGES** |

Last correct tensor: `extra_norm1` output. First wrong tensor: `FRMs[0]` output. The 17-layer
Noiseprint++ and all six SegFormer transformer blocks are exact on the GPU.

### Dissect `FRMs[0]` (`FeatureRectifyModule`, `lib/models/cmx/net_utils.py`)

`scripts/phase8_frm_bisect.py`, shapes from the real forward
(`x1, x2 = [1,64,128,128]`, `channel_weights = [2,1,64,1,1]`, `spatial_weights = [2,1,1,128,128]`):

| piece | rel_max | corr | verdict |
|---|---:|---:|---|
| `channel_weights(x1, x2)` — the cross-attention | 1.18e-07 | 1.00000000 | **OK** |
| `spatial_weights(x1, x2)` — the 1x1 conv stack | 2.09e-07 | 1.00000000 | **OK** |
| the broadcast arithmetic only | 1.57e-01 | 0.98656177 | **DIVERGES** |

So neither attention nor the conv stack is at fault. What remains is the arithmetic
`x1 + 0.5*cw[1]*x2 + 0.5*sw[1]*x2`, which slices the 5-D weight tensors.

### Every broadcast shape on its own is exact

`scripts/phase8_broadcast_bisect.py`, 9 combinations, **9/9 bit-exact (`max_abs = 0.0`)**:

```
[1,64,1,1] * [1,64,128,128]        [1,1,128,128] * [1,64,128,128]
[1,64,1,1] * [1,64,1,1]             [1,1,128,128] * [1,1,128,128]
[1,1,128,128] * [1,64,1,1]         0.5 * x2        0.5*cw1*x2
x1 + sw1 (broadcast add)            x2 * x2
```

The only thing never exercised: slicing the 5-D weight **inside** the graph.

### The culprit: dim-0 integer indexing

`scripts/phase8_select_bisect.py`:

| case | rel_max | corr | verdict |
|---|---:|---:|---|
| `cw5d -> cw5d[1]`, `cw5d = [2,1,64,1,1]` | 4.96e-01 | 0.99999447 | **DIVERGES** |
| `cw5d -> (cw5d[0], cw5d[1])` | 9.95e-01 | **-0.10020612** | **DIVERGES** |
| `sw5d -> (sw5d[0], sw5d[1])`, `sw5d = [2,1,1,128,128]` | 0.0 | 1.00000000 | **OK** |
| FRM arithmetic, slice inside the graph | 5.99e-02 | 0.99978151 | DIVERGES |
| FRM arithmetic, slice outside the graph | 0.0 | 1.00000000 | **OK** |

Same operation, same device, same backend: correct on `[2,1,1,128,128]`, wrong on
`[2,1,64,1,1]`. Slicing the weights **outside** the graph makes the whole thing exact — which is
why the assembled network is wrong while every operator test passed.

## 8D — minimal reproducer and which layer is at fault

`scripts/phase8_minimal_repro.py` — synthetic `torch.randn`, no model, no weights, no TruFor.
**24 of 30** shape+op combinations are wrong on Vulkan, and **every one of the 30 is exact on
LLVM** (`llvm_rel = 0.000e+00` throughout).

| source shape | `sel0` | `sel1` | `sel_both` | pattern |
|---|:---:|:---:|:---:|---|
| `[2,1,64,1,1]` (channel_weights) | WRONG | WRONG | WRONG | dim3 = 1 |
| `[2,1,1,128,128]` (spatial_weights) | OK | OK | OK | dim3 = 128 |
| `[2,1,32,1,1]` | WRONG | WRONG | WRONG | dim3 = 1 |
| `[2,1,64,2,1]` | WRONG | WRONG | WRONG | dim4 = 1 |
| `[2,1,64,1,2]` | WRONG | WRONG | WRONG | dim3 = 1 |
| `[2,1,64,128,1]` | OK | OK | OK | dim3 = 128, dim4 = 1 |
| `[2,1,64,1]` (4-D) | WRONG | WRONG | WRONG | last dim = 1 |
| `[2,3,64,1,1]` | WRONG | WRONG | WRONG | dim3 = 1 |
| `[8,1,64,1,1]` | WRONG | WRONG | WRONG | dim3 = 1 |

**Trigger: `t[k]` is wrong exactly when the tensor's second-to-last dimension is 1.**
`[2,1,64,128,1]` is fine; `[2,1,64,1,1]`, `[2,1,64,2,1]`, `[2,1,64,1,2]` are not.

It is *only* the index. On the same offending tensor every other operation is exact:

| op on `[2,1,64,1,1]` | result |
|---|---|
| identity | OK (0.0) |
| elementwise `c*2+1` | OK (0.0) |
| `sum` reduction | OK (7.1e-08) |
| `reshape(-1)` | OK (0.0) |
| `slice` `c[:,:,0:64,0:1,0:1]` | OK (0.0) |
| **`c[1]`** | **WRONG (1.095e+00)** |

### Which layer

| candidate | verdict | basis |
|---|---|---|
| ① TVM Vulkan operator/codegen bug | **CONFIRMED** | only the take/select kernel miscomputes; the identical graph is exact on LLVM; reproduced with synthetic data |
| ② fusion / layout transformation | ruled out | `identity` on the same tensor is exact, so it is not the graph transforms |
| ③ TVM Vulkan + RADV 22.3.6 incompatibility | unlikely | the driver is fine for every other kernel on the same device; LLVM is a different codegen but the *shapes* are the discriminator, and a driver bug would not be shape-selective in this way |
| ④ model-specific numerical edge | ruled out | reproduced with `torch.randn`, no model involved |

I am labelling ③ "unlikely" rather than "ruled out" — I did not test another Vulkan driver, so I
cannot separate "TVM codegen" from "TVM codegen that happens to trip over RADV's limits" without
doing that A/B.

## Workaround: `Slice` + `Squeeze` instead of `Select`

| formulation | on `[2,1,64,1,1]` |
|---|---|
| `c[1]` | **WRONG** (rel 1.382) |
| `c[1:2].squeeze(0)` | **OK** (rel 0.0) |
| `c.narrow(0,1,1).squeeze(0)` | **OK** (rel 0.0) |
| `c.flatten(0,1)[1].reshape(...)` | **WRONG** (rel 1.848 — still a select) |
| both halves via `slice + squeeze` | **OK** (rel 0.0) |

Earlier attempts that all failed (rel ≈ 1.0): `c.squeeze(1)[1]`, `c.reshape(2,64,1,1)[1]`,
`c.reshape(2,64)[1].reshape(...)` — all still lower to a take.

The rewrite is expressible **in the ONNX graph alone** (`Gather(t,[k],axis=0)` →
`Slice(t, starts=[k], ends=[k+1], axes=[0])` + `Squeeze(axis=0)`), so it needs no change to
TruFor and no change to the model definition. I have **not** yet applied it to the full 1918-node
graph and re-validated end to end — that is the next step, not a claim.

## Reproduce

```bash
export PYTHONPATH=$PWD/vendor/tvm-src/python
export TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

vendor/tvm-venv/bin/python scripts/phase8_stage1_bisect.py   --model weights/trufor.pth.tar --image truFor-src/test_docker/images/tampered2.png --outdir reports/phase8
vendor/tvm-venv/bin/python scripts/phase8_frm_bisect.py      --model weights/trufor.pth.tar --image truFor-src/test_docker/images/tampered2.png --outdir reports/phase8
vendor/tvm-venv/bin/python scripts/phase8_broadcast_bisect.py --model weights/trufor.pth.tar --image truFor-src/test_docker/images/tampered2.png --outdir reports/phase8
vendor/tvm-venv/bin/python scripts/phase8_select_bisect.py  --model weights/trufor.pth.tar --image truFor-src/test_docker/images/tampered2.png --outdir reports/phase8
vendor/tvm-venv/bin/python scripts/phase8_minimal_repro.py  --outdir reports/phase8
```

Artefacts: `reports/phase8/{stage1-bisect,frm-bisect,broadcast-bisect,select-bisect,minimal-repro}.json`.
