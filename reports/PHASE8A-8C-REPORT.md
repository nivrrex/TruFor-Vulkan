# PHASE8A-8C-REPORT.md (SUPERSEDED — see FINAL-REPORT.md)

**Direct answer to the question Phase 8A/8B were meant to settle: no.** The workaround is correct and
does not break the model, but the full graph on the RX 560 is still wrong — and disabling fusion
exposed a second, deeper defect: heap corruption.

## 8A — the ONNX rewrite: correct, and it does not fix the model

`scripts/onnx_gather_to_slice.py` rewrites every `Gather(X, [k], axis=0)` into
`Slice(X, starts=[k], ends=[k+1], axes=[0])` + `Squeeze(axis=0)`. Scope was kept narrow: only
`axis == 0` with a single-element constant index. Axis≠0 gathers, vector/negative indices, and
`Slice`/`Split`/`Squeeze`/`Expand`/`Reshape`/`Transpose` are untouched.

### Census (also serves as the Phase 8C residual check)

| | count |
|---|---:|
| `Gather` nodes in the graph | 80 |
| dim-0 scalar-index candidates | **80** |
| — of which Phase 8 risky (`src[-2] == 1`) | **8** |
| — of which other shapes | 72 |
| **patched** | **80** |
| **remaining dim-0 scalar `Gather` after the rewrite** | **0** |

### Gates

| gate | result |
|---|---|
| `onnx.checker.check_model(full_check=True)` | **OK** |
| ORT CPU, localization map | `max_abs = 0.000e+00`, `mean_abs = 0.0`, `rmse = 0.0`, `corr = 1.000000000000` |
| ORT CPU, confidence map | `max_abs = 0.000e+00`, `corr = 1.000000000000` |
| ORT CPU, integrity score | 0.9858689661 → 0.9858689661, err `0.0` |
| **CPU GATE (bit-identical + checker)** | **PASS** |

The rewrite is semantically exact — the model is untouched in every other way.

### But the full model on Vulkan is still wrong

| model | map corr | map max_abs | score err |
|---|---:|---:|---:|
| A: original ONNX | -0.2901420146 | 9.823e-01 | 3.660e-01 |
| B: patched ONNX | -0.2873656449 | 9.815e-01 | 3.552e-01 |

Moving from -0.290 to -0.287 is noise. The patched model also **passes** the TVM-LLVM gate
(`corr = 0.9999999999`, IoU = Dice = 1.0), so the rewrite is not the problem — the Vulkan backend
is.

## 8B — fusion is not the cause; there is heap corruption

`scripts/phase8_fusion_variants.py` rebuilds the same graph through a customised
`gpu_generic` pipeline with individual passes removed:

| variant | config | map corr | verdict |
|---|---|---:|---|
| C | `FuseOps` + `FuseTIR` removed | **-0.28736564** | DIVERGES |
| D | `ApplyDefaultSchedule` removed | **aborted: `corrupted double-linked list`** | — |

Variant C returns a number **identical to the default pipeline's**, so operator fusion is not the
culprit. Variant D does not return a number at all: glibc's heap checker fires
(`corrupted double-linked list`) and the process aborts. Variant E never ran because D took the
process down.

That is a memory-safety defect, not a numerical one. It is the same class of symptom as the
ExecuTorch failure on this machine (`free(): invalid pointer` during GPU allocation).

### Scale does not save it

Same model exported and run at 128×128 instead of 512×512:

| input | map corr | map max_abs | IoU | Dice |
|---|---:|---:|---:|---:|
| 512×512 | -0.290 | 9.823e-01 | 0.000000 | 0.000000 |
| 128×128 | **+0.189** | **1.777e-01** | 1.000000 | 1.000000 |

The error shrinks by ~5× but does not vanish. So this is not a pure out-of-bounds that only
manifests at large sizes — there is a size-dependent component and a persistent one.

## Consolidated picture

What is now established, all on the same machine / same TVM / same RX 560:

| fact | evidence |
|---|---|
| TVM 0.27 Vulkan miscomputes `t[k]` when `src[-2] == 1` | synthetic repro, 24/30 wrong, 30/30 exact on LLVM |
| `Slice`+`Squeeze` is a correct workaround *for that op* | `rel = 0.0` on the offending shapes |
| the model can be rewritten to remove every such take | 80/80 patched, ORT CPU bit-identical, checker OK |
| the rewritten model is still wrong on Vulkan | corr -0.287 |
| the rewritten model is correct on TVM LLVM | corr 0.9999999999, IoU = Dice = 1.0 |
| operator fusion is not the cause | variant C identical to default |
| the Vulkan runtime corrupts the heap on this graph | `corrupted double-linked list` |
| every operator is individually correct | 20/23 op suite, plus large-tensor and 5-D attention cases |
| every stage-1 sub-module is individually correct | 10/10 exact on Vulkan as separate sub-graphs |

The gap that remains is: *individually-correct sub-graphs and individually-correct kernels compose
into a wrong result only when lowered as one large graph, and doing so also corrupts the heap.*

## Honest verdict on the Phase 8 root cause

The take/select bug found in Phase 8 is **real and independently reproducible**, but it is **not the
cause of the full-graph divergence** — removing all 80 instances changed nothing measurable. I am
recording that explicitly because the earlier report implied a single root cause, and the
workaround test refutes that reading.

The remaining defect is inside TVM's Vulkan runtime itself (heap corruption while executing a
~1900-node graph), and locating it needs tooling this project has not used: a sanitizer build of
TVM, or the `vulkan-troubleshooting` path with `VK_LAYER` validation enabled.

## Options from here

| option | cost | what it would settle |
|---|---|---|
| rebuild TVM with ASan, rerun the full graph | hours; needs a second build tree | whether it is a CPU-side host buffer overrun in the Vulkan runtime |
| rerun with `VK_LAYER_KHRONOS_validation` | minutes | whether the SPIR-V kernels violate the Vulkan spec on this driver |
| A/B a newer Mesa/RADV | ~1 h, changes the driver variable | separates TVM codegen from RADV 22.3.6 |
| file the take/select bug upstream now | minutes, material is ready | the independently-proven half of the story |
| stop here | — | conclusion: TVM Vulkan is not usable for this model on this driver |

My recommendation: do the validation layer first (cheapest, and it can produce a driver-side
attribution), then file the take/select bug upstream with the minimal reproducer regardless of what
the other experiments show — that one is unambiguous and does not depend on the unresolved part.

## Reproduce

```bash
export PYTHONPATH=$PWD/vendor/tvm-src/python
export TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

# 8A  rewrite + checker + CPU gate
python3 scripts/onnx_gather_to_slice.py \
  --onnx reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx \
  --outdir reports/phase8a --image truFor-src/test_docker/images/tampered2.png

# 8A  CPU gate on the rewritten model
vendor/tvm-venv/bin/python scripts/tvm_cpu_gate.py \
  --onnx reports/phase8a/trufor-512x512-slicefix.onnx \
  --image truFor-src/test_docker/images/tampered2.png --outdir reports/phase8b

# 8B  A/B on Vulkan
vendor/tvm-venv/bin/python scripts/tvm_vulkan_run.py \
  --onnx reports/phase8a/trufor-512x512-slicefix.onnx \
  --image truFor-src/test_docker/images/tampered2.png --outdir reports/phase8b --skip-6e

# 8B  fusion variants (C and D; D aborts on heap corruption)
vendor/tvm-venv/bin/python scripts/phase8_fusion_variants.py \
  --original reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx \
  --patched  reports/phase8a/trufor-512x512-slicefix.onnx \
  --image    truFor-src/test_docker/images/tampered2.png --outdir reports/phase8b/fusion
```

Artefacts: `reports/phase8a/{trufor-512x512-slicefix.onnx,slicefix-report.json}`,
`reports/phase8b/{tvm-cpu-report.json,tvm-vulkan-report.json}`,
`reports/phase8b/fusion-run.log`, `reports/phase8b/s128/`.
