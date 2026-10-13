#!/usr/bin/env python3
"""Canonicalise the exported ONNX so that MNN's converter can consume it.

Two export artefacts block MNN but not TVM:

1. **Empty optional Resize inputs.** The dynamo exporter writes absent optional
   inputs as empty strings: `Resize(X, "", "", sizes)`. Legal ONNX; MNN reports
   `upsample_bilinear2d_N has empty input, the index is 1 / 2` and then cannot
   infer shapes downstream. Replaced by real empty tensors of the correct dtype
   (roi/scales float32, sizes int64) — same semantics, something to bind.

2. **Negative axes.** 13 `Concat` nodes carry `axis = -1`. MNN's converter
   resolves the axis without handling the negative case and ends with
   `Error for concat size of op [ cat_26 ], the 0 input not match output` /
   `Compute Shape Error for cat_26`, which makes `createSession` fail on the CPU
   backend, before Vulkan is even in play. Rewritten to the equivalent positive
   axis using each node's inferred rank.

Both edits touch only the exported artefact. After writing, the model is re-run
on ONNX Runtime CPU and compared against the unmodified graph, so a rewrite that
changed numerics cannot pass unnoticed.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

# ops whose `axis` attribute is affected, and where it is found
AXIS_OPS = {"Concat", "Gather", "Reshape", "Slice", "Split", "Unsqueeze",
            "Squeeze", "ReduceMean", "ReduceMax", "ReduceSum", "ReduceLogSumExp",
            "ReduceProd", "ArgMax", "ArgMin", "Transpose"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default="trufor-512x512-canon")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model = onnx.load(args.onnx)
    g = model.graph

    img = Image.open(args.image).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    def input_name_of(path):
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    ref = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"]).run(
        None, {input_name_of(args.onnx): x})

    inferred = onnx.shape_inference.infer_shapes(model)
    shape_of, rank_of = {}, {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            shape_of[vi.name] = dims
            rank_of[vi.name] = len(dims)

    initializers = {t.name for t in g.initializer}
    resize_fixes, axis_fixes, skipped = [], [], []

    for n in g.node:
        # ---- 1. Resize: replace empty-string optional inputs with empty tensors
        if n.op_type == "Resize" and any(i == "" for i in n.input):
            data_shape = shape_of.get(n.input[0]) if n.input and n.input[0] else None
            if data_shape is None:
                skipped.append((n.name, "Resize data shape unknown"))
            else:
                slots = list(n.input) + [""] * (4 - len(n.input))
                filled, notes = [], []
                for idx, name in enumerate(slots[:4]):
                    if name:
                        filled.append(name)
                        continue
                    if idx == 3 and not any(filled[1:3]):
                        sizes = [1, data_shape[1], 512, 512]
                        cn = n.name + "_sizes"
                        g.initializer.append(numpy_helper.from_array(np.array(sizes, dtype=np.int64), cn))
                        filled.append(cn)
                        notes.append("synthesised sizes=%s" % sizes)
                    else:
                        dt = np.int64 if idx == 3 else np.float32
                        en = n.name + "_empty%d" % idx
                        g.initializer.append(numpy_helper.from_array(np.array([], dtype=dt), en))
                        filled.append(en)
                        notes.append("empty %s slot %d" % (np.dtype(dt).name, idx))
                n.input[:] = filled
                resize_fixes.append({"node": n.name, "data_shape": data_shape, "notes": notes})

        # ---- 2. negative axis -> positive axis
        if n.op_type in AXIS_OPS:
            rank = None
            if n.input and n.input[0]:
                rank = rank_of.get(n.input[0])
            if rank is None and n.output:
                rank = rank_of.get(n.output[0])
            for a in n.attribute:
                if a.name != "axis" or a.type != onnx.AttributeProto.INT:
                    continue
                if a.i < 0 and rank:
                    new = a.i + rank
                    axis_fixes.append({"node": n.name, "op": n.op_type, "rank": rank,
                                       "from": a.i, "to": new})
                    a.i = new
                elif a.i < 0:
                    skipped.append((n.name, "%s axis=%d with unknown rank" % (n.op_type, a.i)))
                break

    out_path = os.path.join(args.outdir, args.name + ".onnx")
    onnx.save(model, out_path)

    print("Resize nodes fixed  : %d" % len(resize_fixes))
    for r in resize_fixes[:3]:
        print("   %-22s %s" % (r["node"], "; ".join(r["notes"])))
    print("negative axes fixed : %d" % len(axis_fixes))
    for a in axis_fixes[:5]:
        print("   %-22s %-8s rank=%d  %d -> %d" % (a["node"], a["op"], a["rank"], a["from"], a["to"]))
    if skipped:
        print("skipped: %s" % skipped[:5])

    # ---- verify numerics on ORT CPU ----
    in_name = input_name_of(out_path)
    got = ort.InferenceSession(out_path, so, providers=["CPUExecutionProvider"]).run(None, {in_name: x})

    def post(res):
        out, conf, det = res
        o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy()
        c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy()
        d = float(np.asarray(det).reshape(-1)[0])
        return o, c, 1.0 / (1.0 + np.exp(-d))

    def stats(name, a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            return {"name": name, "shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        d = np.abs(a - b)
        return {"name": name, "shape_match": True, "max_abs_err": float(d.max()),
                "mean_abs_err": float(d.mean()),
                "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0}

    r_map, r_conf, r_det = post(ref)
    g_map, g_conf, g_det = post(got)
    report = {
        "source": args.onnx, "output": out_path,
        "resize_fixes": resize_fixes, "axis_fixes": axis_fixes, "skipped": skipped,
        "equivalence": {
            "map": stats("map", r_map, g_map),
            "conf": stats("conf", r_conf, g_conf),
            "integrity_score": {"before": r_det, "after": g_det, "abs_err": abs(r_det - g_det)},
        },
    }
    worst = max(report["equivalence"]["map"]["max_abs_err"], report["equivalence"]["conf"]["max_abs_err"])
    report["numerically_identical"] = bool(worst == 0.0 and report["equivalence"]["integrity_score"]["abs_err"] == 0.0)

    print("\nORT CPU equivalence (before vs canonicalised):")
    for k in ("map", "conf"):
        s = report["equivalence"][k]
        print("  %-5s max_abs=%.3e mean=%.3e corr=%.12f" % (k, s["max_abs_err"], s["mean_abs_err"], s["corr"]))
    print("  det   before=%.10f after=%.10f err=%.3e" % (r_det, g_det, report["equivalence"]["integrity_score"]["abs_err"]))
    print("  identical: %s" % report["numerically_identical"])

    with open(os.path.join(args.outdir, "canonicalise-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % out_path)
    return 0 if report["numerically_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
