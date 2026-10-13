#!/usr/bin/env python3
"""Give every 2-input ONNX Resize an explicit `sizes` input.

The dynamo exporter emits Resize with only the data input plus empty optional
inputs, so the output extent is not derivable from the graph. TVM tolerates
this, MNN does not: its converter reports

    Error for concat size of op [ cat_26 ], the 0 input not match output
    Compute Shape Error for cat_26

at run time, because it cannot infer the shape of the Resize output and the
Concat that consumes it.

Since TruFor is inference-only and the input size is fixed at 512x512 here,
the sizes are compile-time constants. This walks the graph, infers each Resize
data input's shape, and appends a Constant `sizes` tensor. It edits the exported
artefact only; the ONNX numerics are then re-checked against ONNX Runtime CPU
before anything is handed to MNN.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model = onnx.load(args.onnx)
    g = model.graph

    # ---- reference from the unpatched graph ----
    img = Image.open(args.image).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x})

    # ---- infer shapes ----
    inferred = onnx.shape_inference.infer_shapes(model)
    shape_of = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            shape_of[vi.name] = dims

    initializers = {t.name: t for t in g.initializer}

    # The dynamo exporter writes optional Resize inputs that are absent as empty
    # strings: Resize(X, "", "", sizes). That is legal ONNX and TVM accepts it,
    # but MNN's converter rejects it --
    #   upsample_bilinear2d_6 has empty input, the index is 1 / 2
    # and then cannot infer the shape of the consuming Concat. Replacing each
    # hole with a real empty tensor (shape [0]) keeps the semantics identical
    # ("this optional input is absent") while giving the converter something
    # concrete to bind.
    patched, skipped = [], []
    new_nodes, new_inits = [], []
    for n in g.node:
        if n.op_type != "Resize" or not any(i == "" for i in n.input):
            new_nodes.append(n)
            continue
        data_shape = shape_of.get(n.input[0]) if n.input and n.input[0] else None
        if data_shape is None:
            skipped.append((n.name, "unknown data shape"))
            new_nodes.append(n)
            continue

        # normalise to exactly 4 slots: X, roi, scales, sizes
        slots = list(n.input) + [""] * (4 - len(n.input))
        filled = []
        for idx, name in enumerate(slots[:4]):
            if name:
                filled.append(name)
                continue
            if idx == 3 and not any(filled[1:3]):
                # neither roi nor scales present -> synthesise `sizes`
                sizes = [1, data_shape[1], 512, 512]
                cn = n.name + "_sizes"
                new_inits.append(numpy_helper.from_array(np.array(sizes, dtype=np.int64), cn))
                filled.append(cn)
                patched.append({"node": n.name, "data_shape": data_shape,
                                "sizes": sizes, "what": "synthesised sizes"})
            else:
                # ONNX types the optional inputs strictly: roi float32,
                # scales float32, sizes int64. An int64 hole in slots 1/2 makes
                # the whole model unloadable.
                dt = np.int64 if idx == 3 else np.float32
                en = n.name + "_empty%d" % idx
                new_inits.append(numpy_helper.from_array(np.array([], dtype=dt), en))
                filled.append(en)
                patched.append({"node": n.name, "data_shape": data_shape,
                                "what": "empty %s for slot %d" % (np.dtype(dt).name, idx)})
        # mutate in place: rebuilding the node would mean re-encoding all 10
        # Resize attributes, and make_node() refuses AttributeProto values
        n.input[:] = filled
        new_nodes.append(n)

    if patched:
        del g.node[:]
        g.node.extend(new_nodes)
        g.initializer.extend(new_inits)
        out_path = os.path.join(args.outdir, "trufor-512x512-resizefix.onnx")
        onnx.save(model, out_path)
        print("patched %d Resize node(s):" % len(patched))
        for p in patched:
            print("   %-24s data=%s  %s" % (p["node"], p["data_shape"], p["what"]))
    else:
        out_path = args.onnx
        print("no Resize node needed patching")
    if skipped:
        print("skipped: %s" % skipped)

    # ---- verify the patched model is numerically identical on ORT CPU ----
    sess2 = ort.InferenceSession(out_path, so, providers=["CPUExecutionProvider"])
    got = sess2.run(None, {sess2.get_inputs()[0].name: x})

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
        "resize_patched": patched, "resize_skipped": skipped,
        "equivalence": {
            "map": stats("map", r_map, g_map),
            "conf": stats("conf", r_conf, g_conf),
            "integrity_score": {"before": r_det, "after": g_det, "abs_err": abs(r_det - g_det)},
        },
    }
    worst = max(report["equivalence"]["map"]["max_abs_err"], report["equivalence"]["conf"]["max_abs_err"])
    report["numerically_identical"] = bool(worst == 0.0 and report["equivalence"]["integrity_score"]["abs_err"] == 0.0)
    print("\nORT CPU equivalence (before vs after Resize patch):")
    for k in ("map", "conf"):
        s = report["equivalence"][k]
        print("  %-5s max_abs=%.3e mean=%.3e corr=%.12f" % (k, s["max_abs_err"], s["mean_abs_err"], s["corr"]))
    print("  det   before=%.10f after=%.10f err=%.3e"
          % (r_det, g_det, report["equivalence"]["integrity_score"]["abs_err"]))
    print("  identical: %s" % report["numerically_identical"])
    with open(os.path.join(args.outdir, "resize-fix-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % out_path)
    return 0 if report["numerically_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
