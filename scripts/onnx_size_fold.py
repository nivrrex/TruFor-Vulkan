#!/usr/bin/env python3
"""Phase 6B: fold shape-derived constants in the exported ONNX, iteratively.

TVM's Relax frontend aborts on

    ValueError: Node /Size cannot handle ShapeExpr inputs

For the only such node, ONNX computes

    /Sub = Size(Shape(/dncnn/dncnn.47/Conv_output_0)) - /Constant

and the Size input has static shape [1, 1, 512, 512], so its value is 262144
for every input -- a compile-time constant that the exporter left as a runtime
op. This pass replaces such nodes with Constant nodes, then re-verifies on ONNX
Runtime CPU that all three outputs are bit-identical before anything is handed
to TVM.

It only touches the exported artefact. TruFor sources are never read here.
Repeatable: safe to run again on an already-folded model.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT_HW = 512


def shapes_of(model):
    """input name -> static shape (list) for every value with a static shape."""
    out = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            out[vi.name] = dims
    return out


def run_ort(path, image_path):
    import onnxruntime as ort
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if img.size != (INPUT_HW, INPUT_HW):
        img = img.resize((INPUT_HW, INPUT_HW), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})


def fold_once(model):
    """Replace Size(Shape(x)) with a Constant when x has a static shape.

    Returns (n_folded, names). Non-destructive: rebuilds graph.node.
    """
    inferred = onnx.shape_inference.infer_shapes(model)
    shp = shapes_of(inferred)

    size_nodes = [n for n in model.graph.node if n.op_type == "Size"]
    const_by_out = {}
    for n in model.graph.node:
        if n.op_type == "Constant":
            const_by_out[n.output[0]] = n

    shape_of_node = {}
    for n in model.graph.node:
        if n.op_type == "Shape":
            shape_of_node[n.output[0]] = n

    new_nodes = []
    folded = []
    replaced_outputs = {}
    for n in model.graph.node:
        if n.op_type != "Size" or len(n.input) != 1:
            new_nodes.append(n)
            continue
        sh_node = shape_of_node.get(n.input[0])
        src = sh_node.input[0] if sh_node is not None else None
        dims = shp.get(src) if src else None
        if not dims:
            new_nodes.append(n)
            continue
        value = int(np.prod(dims))
        # ONNX Size returns a 1-D tensor [n]; a 0-D scalar here would break the
        # downstream shape inference (Tile's `repeats` length check).
        cst = numpy_helper.from_array(np.array([value], dtype=np.int64), n.output[0] + "_folded")
        cnode = helper.make_node("Constant", [], [n.output[0]], name=n.name + "_const", value=cst)
        new_nodes.append(cnode)
        replaced_outputs[n.output[0]] = ("Size->Constant", value, src, dims)
        folded.append({"node": n.name, "input": src, "input_shape": dims, "value": value})

    if folded:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return folded


def prune_unused(model):
    """Drop nodes whose outputs nothing consumes and that are not graph outputs."""
    g = model.graph
    live = set()
    changed = True
    removed = 0
    while changed:
        changed = False
        consumers = collections.Counter()
        for n in g.node:
            for i in n.input:
                consumers[i] += 1
        needed = set(o.name for o in g.output) | set(o.name for o in g.initializer)
        for n in g.node:
            for o in n.output:
                if consumers[o] > 0:
                    needed.add(o)
        keep = []
        for n in g.node:
            if any(consumers[o] == 0 and o not in needed for o in n.output) and n.op_type not in ("Constant",):
                removed += 1
                changed = True
                continue
            keep.append(n)
        if changed:
            del g.node[:]
            g.node.extend(keep)
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    import torch.nn.functional as F

    model = onnx.load(args.onnx)
    report = {"source": args.onnx, "input_hw": INPUT_HW,
              "nodes_before": len(model.graph.node),
              "op_census_before": dict(collections.Counter(n.op_type for n in model.graph.node).most_common())}

    ref = run_ort(args.onnx, args.image)

    folded = fold_once(model)
    report["folded"] = folded
    print("folded %d Size node(s):" % len(folded))
    for f in folded:
        print("   %s: Size(Shape(%s)) shape=%s -> %d" % (f["node"], f["input"], f["input_shape"], f["value"]))

    out_path = os.path.join(args.outdir, "trufor-%dx%d-folded.onnx" % (INPUT_HW, INPUT_HW))
    onnx.save(model, out_path)
    report["nodes_after"] = len(model.graph.node)
    report["op_census_after"] = dict(collections.Counter(n.op_type for n in model.graph.node).most_common())
    report["folded_bytes"] = os.path.getsize(out_path)
    print("nodes %d -> %d, size %.1f MB" % (report["nodes_before"], report["nodes_after"],
                                            report["folded_bytes"] / 1e6))

    got = run_ort(out_path, args.image)

    def post(res):
        out, conf, det = res
        o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy() if out.ndim == 4 and out.shape[1] > 1 else out
        c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy() if conf.ndim == 4 else conf
        d = float(np.asarray(det).reshape(-1)[0])
        return o, c, 1.0 / (1.0 + np.exp(-d))

    def stats(name, a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            return {"name": name, "shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        d = np.abs(a - b)
        return {"name": name, "shape_match": True, "shape": list(a.shape),
                "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
                "rmse": float(np.sqrt((d ** 2).mean())),
                "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0}

    r_out, r_conf, r_det = post(ref)
    g_out, g_conf, g_det = post(got)
    report["equivalence"] = {"map": stats("map", r_out, g_out), "conf": stats("conf", r_conf, g_conf),
                             "integrity_score": {"before": r_det, "after": g_det, "abs_err": abs(r_det - g_det)}}
    worst = max(report["equivalence"]["map"]["max_abs_err"], report["equivalence"]["conf"]["max_abs_err"])
    report["numerically_identical"] = bool(worst == 0.0 and report["equivalence"]["integrity_score"]["abs_err"] == 0.0)

    print("\nORT CPU equivalence (original vs folded):")
    for k in ("map", "conf"):
        s = report["equivalence"][k]
        print("  %-5s max_abs=%.3e mean_abs=%.3e corr=%.12f" % (k, s["max_abs_err"], s["mean_abs_err"], s["corr"]))
    print("  det   before=%.10f after=%.10f err=%.3e" % (r_det, g_det, report["equivalence"]["integrity_score"]["abs_err"]))
    print("  numerically identical: %s" % report["numerically_identical"])

    with open(os.path.join(args.outdir, "folding-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % out_path)
    return 0 if report["numerically_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
