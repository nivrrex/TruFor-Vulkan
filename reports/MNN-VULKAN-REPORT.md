# MNN-VULKAN-REPORT.md — Phase 7

**Result: stopped before Vulkan. MNN converts the TruFor ONNX, but its shape inference fails on one Concat (`cat_26`), so the CPU backend cannot even be evaluated, let alone the Vulkan one.**

Per the plan, Phase 7 was entered because TVM's Vulkan output was wrong (Phase 6F/6G).
The gate for this phase was "MNN CPU correctness first" — that gate is not reachable.

## What worked

**ONNX → MNN conversion succeeds**, after two changes to the *exported artefact* (not to
TruFor) plus one converter flag. All three are recorded in
`reports/mnn/canonicalise-report.json`.

1. **Empty optional Resize inputs.** The dynamo exporter writes absent optional
   inputs as empty strings: `Resize(X, "", "", sizes)`. Legal ONNX, TVM accepts it, MNN does
   not — it reports `upsample_bilinear2d_N has empty input, the index is 1 / 2` and then cannot
   infer anything downstream. Replaced with real empty tensors of the correct dtype
   (roi/scales `float32`, sizes `int64`). Note that filling the holes with int64 makes the
   whole model unloadable (`Type 'tensor(int64)' of input parameter (node_Resize_4562_empty1)
   of operator (Resize) ... is invalid`) — the dtype per slot matters.
2. **13 negative `Concat` axes** rewritten to positive (`-1` → `rank-1`, all rank 3). MNN's
   converter resolves `axis` without handling the negative case.
3. **`--convertMatmulToConv 0`.** MNN's `Merge::ConvertMatMulToConv2D` pass rewrites 202
   MatMul nodes as Conv2D; that was what produced `Error for concat size of op [ cat_26 ],
   the 0 input not match output` *during conversion*. With the pass off, conversion is clean
   (`Converted Success!`, no warnings).

Every rewrite was re-verified on ONNX Runtime CPU against the unmodified graph:
`map` / `conf` / integrity score all `max_abs = 0.0`, i.e. **bit-identical**.

Result: `reports/mnn/trufor_nomatmul2conv.mnn`, 277 MB, inputs `[rgb]`, outputs
`[out, conf, det]`.

## What blocks it

`MNN.Interpreter(...).createSession({"backend":"CPU"})` **returns a Session** — so the
failure is not a hard session error — but MNN's shape inference never completes:

```
session: Session
input names : ['rgb']
output names: ['out', 'conf', 'det']
Error for concat size of op [ cat_26 ], the 0 input not match output
Compute Shape Error for cat_26
getSessionOutputAll -> {'conf': [], 'det': [], 'out': []}     # every shape empty
```

Every output tensor's shape is an empty array, so nothing can be preallocated and
`runSessionWithInputs` cannot even be called (`PyMNNTensor_init: Tensor Dim not match`).

`cat_26` is MNN's internal op name; the counter is per-op-type, and the model has 25 ONNX
`Concat` nodes, so it is not simply "the 26th Concat". Which one it is could not be
determined from the Python binding: `MNN.Interpreter` exposes no operator listing
(`getSessionInputAll` / `getSessionOutputAll` only cover graph I/O, and `MNN.expr` in the pip
wheel has no `Net` to load a `.mnn` and walk it). Identifying it needs MNN's C++ API
(`Interpreter::getSessionInfoAll`) with the headers from the MNN repository, or a flatbuffer
dump of the `.mnn`.

## Not a version bug

Re-tested on three releases, each converted with its own converter:

| MNN | conversion | CPU session |
|---|---|---|
| 3.6.1 | Converted Success | `Compute Shape Error for cat_26` |
| 3.5.0 | Converted Success | `Compute Shape Error for cat_26` |
| 3.4.1 | Converted Success | `Compute Shape Error for cat_26` |

Identical message on all three, so this is a structural limitation for this graph rather
than a regression in a particular release.

## What this says about the three routes tried

| route | reaches GPU execution? | correctness on GPU |
|---|---|---|
| ExecuTorch Vulkan 0.6.0 | no — aborts in GPU allocation (`free(): invalid pointer`) | n/a |
| TVM Vulkan 0.27.0 | **yes** — full graph runs on the RX 560 | **no** — backbone diverges |
| MNN 3.6.1 | **no** — blocked on the CPU backend by shape inference | n/a |

MNN never got far enough for its Vulkan backend to be exercised, so this report says
nothing about MNN's Vulkan support on the RX 560 — only that this model does not reach it.

## Honest caveats

- The `cat_26` root cause is **unidentified**. It could be a MNN converter/renderer gap, or
  something about how this graph expresses the concat (the CMX `ChannelWeights`
  `cat((avg, max), dim=1)` and the dual-modality `cat((x1, x2), dim=1)` are the plausible
  suspects, but that is inference from the source, not measured).
- `--transformerFuse` and `--transformerFuseC4 0` were also tried. They do not change the
  outcome.
- Nothing here is a claim about MNN on models it handles well. This is one graph.

## Reproduce

```bash
# 1. canonicalise the exported ONNX (Resize holes + negative axes), verified on ORT CPU
python3 scripts/onnx_canonicalize.py \
  --onnx reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx \
  --outdir reports/mnn --image truFor-src/test_docker/images/tampered2.png

# 2. convert, with the MatMul->Conv2D pass disabled
python3 -c "
import sys
from MNN.tools import mnnconvert as mc
sys.argv = ['mnnconvert','-f','ONNX','--convertMatmulToConv','0',
  '--modelFile','reports/mnn/trufor-512x512-canon.onnx',
  '--MNNModel','reports/mnn/trufor_nomatmul2conv.mnn','--bizCode','trufor']
mc.main()"

# 3. observe the shape-inference failure
python3 -c "
import MNN
it = MNN.Interpreter('reports/mnn/trufor_nomatmul2conv.mnn')
s = it.createSession({'precision':'normal','backend':'CPU','numThread':4})
print(it.getSessionOutputAll(s))"
```

Environment note: installing `MNN` pinned `protobuf` to 5.29.6, which `onnx` (needs >= 6.31.1)
rejects; protobuf was moved back to `>=6.31.1,<7` for this phase.

Artefacts: `reports/mnn/trufor-512x512-canon.onnx`, `reports/mnn/trufor_nomatmul2conv.mnn`,
`reports/mnn/canonicalise-report.json`, `scripts/mnn_cpu_gate.py`.
