# TVM-IMPORT.md — Phase 6B: ONNX → TVM Relax

**Result: import succeeds. No unsupported operators. The blocker found here was in the legacy-tracer ONNX, not in TVM.**

## First attempt: legacy-tracer ONNX — FAILED

Input: `reports/onnx-probe/trufor_512x512_legacy.onnx` (5133 nodes, 39 ops, 275.7 MB).

```
ValueError: Node /Size cannot handle ShapeExpr inputs.
```

The single offending node, located exactly (one of 5133):

```
/dncnn/dncnn.47/Conv_output_0  : [1, 1, 512, 512]     (Noiseprint++ last conv)
/Shape        = Shape(/dncnn/dncnn.47/Conv_output_0)
/Size         = Size(/Shape)          -> 262144
/Constant    = 4
/Sub          = /Size_output_0 - /Constant_output_0
/Sub_output_0 consumed by: Tile
```

`/Sub` feeds a `Tile`, so the number flows into a `repeats` argument. Because the
input is static at 512×512, that value is a compile-time constant, but the
exporter emitted it as a runtime op and TVM's Relax `/Size` refuses a
`ShapeExpr` operand.

Two other attempts on the same file:

- **ORT constant folding** (`onnxruntime.transformers.optimizer`) — rejected:
  `Unsupported model type: onnx for optimization, directly return model`.
  That transformers module only handles `bert`/`gpt2`-style models.
- **`onnxsim.simplify`** — failed with `RuntimeError: Input /Sub_output_0 is
  undefined!`, i.e. its own dead-code pass disagreed with this graph.
- **Manual `Size` → `Constant(262144)`** — the constant was inserted as a 0-D
  scalar, and ONNX Size returns a 1-D `[1]` tensor. With a 1-D replacement the
  file loads further but ORT then rejects it at session init:
  `Node (/Tile) Op (Tile) [ShapeInferenceError] 'Repeats' input has incorrect
  number of values`. The legacy graph's `If` nodes (the `adaptive_*_pool2d`
  dynamic-shape branches) leave `Tile`'s operand rank unresolvable, and
  replacing the shape op turned a lenient merge into a hard failure.

Patching around this would have meant fighting the exporter's baked-in shapes.

## Second attempt: dynamo exporter — SUCCEEDED

The dynamo exporter emits shapes symbolically instead of as 1608 `Constant` +
272 `Shape` + 291 `Gather` nodes, and it has no `If` nodes at all.

It needed two export-time rewrites, both applied to the **loaded model object**,
never to the sources on disk, and both verified bit-identical before export
(`scripts/export_onnx_dynamo.py`):

1. **`Adaptive{Max,Avg}Pool2d(1)` → `mean` / `amax` over dims (2,3)**.
   Source: `lib/models/cmx/net_utils.py:13-14`, CMX `ChannelWeights`.
   Reason: `ConversionError: No ONNX function found for
   aten.adaptive_max_pool2d ... No decompositions registered`.
   8 substitutions (4 FRMs × avg/max).
   Note: my first attempt used `nn.AvgPool2d(kernel_size=1)`, which is **wrong** —
   it keeps the spatial extent and produced 2097152 elements instead of 256.
   Global `mean`/`amax` over dims (2,3) is the correct equivalent.
2. **Drop the trailing `None` from `forward`.** `EncoderDecoder.forward` returns
   `(out, conf, det, npp)` with `npp=None`; the legacy tracer tolerates it, the
   dynamo exporter raises `AttributeError: 'NoneType' object has no attribute
   'name'` while translating the return node. An `ExportWrapper` returns the
   three real tensors.

Substitution check before export: `out`, `conf`, `det` all `max_abs = 0.000e+00`.

## Resulting model

| | legacy tracer | dynamo exporter |
|---|---:|---:|
| nodes | 5133 | **1918** |
| distinct ops | 39 | **30** |
| `Constant` | 1608 | **0** |
| `Shape` | 272 | **0** |
| `If` | 2 | **0** |
| file | 275.7 MB | 4.4 MB + 275.9 MB external data (inlined to 280.2 MB) |

dynamo op census: `Transpose` 324, `Add` 310, `Reshape` 297, `MatMul` 272,
`Mul` 140, `LayerNormalization` 114, `Conv` 111, `Gather` 80, `Softmax` 40,
`Relu` 39, `Div` 33, `Erf` 32, `Concat` 25, `Slice` 18, `Squeeze` 16, `Gemm` 10,
`Sigmoid` 9, `Split` 8, `Resize` 8, `Sub` 4, `ReduceMean` 4, `ReduceMax` 4,
`BatchNormalization` 4, `ReduceLogSumExp` 4, `ReduceSum` 4, `LogSoftmax` 2,
`Neg` 2, `Exp` 2, `Tile` 1, `Log` 1.

## Import

```
tvm 0.27.0  (libinfo Vulkan=OFF LLVM=ON)
onnx loaded: 1918 nodes, 30 distinct ops, 930 initializers
inputs : [('rgb', [1, 3, 512, 512])]
outputs: [('out', [1, 2, 512, 512]), ('conf', [1, 1, 512, 512]), ('det', [1, 1])]
import OK in 1.1s
relax funcs = ['tile', 'main']   blocks = 1   bindings = 2685
```

- **unsupported operators: none**
- **dynamic-shape problems: none** (shape fully static, as required by Phase 6B)
- import warnings: one `UserWarning` that a TensorProto is stored in
  `*.onnx.data`, which TVM rejects with `ValidationError: Data of TensorProto ...
  should be stored in ... but it is not regular file`. Fixed by inlining the
  external data with `onnx.save_model(..., save_as_external_data=False)`
  (280.2 MB single file). **TVM cannot read ONNX external data** — worth
  remembering for any future model.

API notes for TVM 0.27 (the annotations are wrong / the API moved):
- `from_onnx(model: GraphProto, ...)` is annotated as `GraphProto` but actually
  requires the **`ModelProto`**; passing `model.graph` gives
  `AttributeError: ir_version`.
- there is no `from_onnx(...) -> (mod, params)` pair; it returns an `IRModule`.
- `ir_mod["main"]` is already a `relax.Function`; its `.body` is a `SeqExpr` and
  the operators live in `blocks[0].bindings[i].var.op`.
- `tvm.__version__` no longer exists; use `tvm._version.__version__`.

## Reproduce

```bash
python3 scripts/export_onnx_dynamo.py --model weights/trufor.pth.tar \
  --outdir reports/onnx-dynamo --size 512
# inline the external data, then:
vendor/tvm-venv/bin/python scripts/tvm_import_probe.py \
  --onnx reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx --outdir reports/tvm
```

Artefacts: `reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx`,
`reports/tvm/tvm-import.json`, `reports/tvm/relax-main.txt` (the imported Relax IR).
