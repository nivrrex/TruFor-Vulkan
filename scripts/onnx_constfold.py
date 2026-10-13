#!/usr/bin/env python3
"""Phase 6B fix: constant-fold the exported ONNX before handing it to TVM.

TVM's Relax frontend rejected the model with

    ValueError: Node /Size cannot handle ShapeExpr inputs

traced to a single node, Size(Shape(conv_out)), whose result is only used to
compute a subtraction constant. With a fixed 512x512 input every shape-derived
value in this graph is a compile-time constant, so the fix is to let the
standard ONNX constant folding remove it. The exporter bakes shapes in as
Constant nodes (1608 Constant / 272 Shape / 291 Gather), so there is a lot for
the folder to do.

This rewrites the *exported artefact*, not the TruFor sources. Correctness of
the folded model is re-checked against ONNX Runtime CPU against the
already-verified reference, so a folding pass that changes numerics cannot slip
through unnoticed.

Outputs: <outdir>/trufor-512x512-folded.onnx and folding-report.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnx
from PIL import Image

INPUT_HW = 512


def census(model):
    import collections

    return collections.Counter(n.op_type for n in model.graph.node)


def opt_onnxruntime(src, dst):
    from onnxruntime.transformers.optimizer import optimize_model

    t0 = time.perf_counter()
    model = optimize_model(src, model_type="onnx", opt_level=99)
    # optimize_model returns an OnnxModel wrapper; the protobuf is on .model
    proto = getattr(model, "model", model)
    onnx.save(proto, dst)
    return time.perf_counter() - t0


def ort_reference(src, image_path, workdir):
    """Run the model on ORT CPU and return the three outputs."""
    import onnxruntime as ort

    img = Image.open(image_path).convert("RGB")
    if img.size != (INPUT_HW, INPUT_HW):
        img = img.resize((INPUT_HW, INPUT_HW), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    res = sess.run(None, {iname: x})
    return res, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import torch.nn.functional as F

    report = {"source": args.onnx, "input_hw": INPUT_HW}

    src = onnx.load(args.onnx, load_external_data=False)
    report["op_census_before"] = dict(census(src).most_common())
    report["nodes_before"] = len(src.graph.node)
    print("before: %d nodes, %d distinct ops" % (report["nodes_before"], len(report["op_census_before"])))

    ref, x = ort_reference(args.onnx, args.image, args.outdir)
    report["ort_reference_output_shapes"] = [list(r.shape) for r in ref]

    folded_path = os.path.join(args.outdir, "trufor-%dx%d-folded.onnx" % (INPUT_HW, INPUT_HW))
    report["fold_seconds"] = opt_onnxruntime(args.onnx, folded_path)
    folded = onnx.load(folded_path, load_external_data=False)
    report["op_census_after"] = dict(census(folded).most_common())
    report["nodes_after"] = len(folded.graph.node)
    report["folded_bytes"] = os.path.getsize(folded_path)
    print("after : %d nodes, %d distinct ops (%.1fs folding)"
          % (report["nodes_after"], len(report["op_census_after"]), report["fold_seconds"]))

    # verify the folded model is numerically identical on ORT CPU
    got, _ = ort_reference(folded_path, args.image, args.outdir)

    def np_stats(name, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            return {"name": name, "shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        d = np.abs(a - b)
        return {
            "name": name,
            "shape_match": True,
            "shape": list(a.shape),
            "max_abs_err": float(d.max()),
            "mean_abs_err": float(d.mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0,
        }

    def postprocess(res):
        out, conf, det = res
        if out.ndim == 4 and out.shape[1] > 1:
            o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy()
        else:
            o = out
        if conf.ndim == 4:
            c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy()
        else:
            c = conf
        d = float(np.asarray(det).reshape(-1)[0])
        return o, c, 1.0 / (1.0 + np.exp(-d))

    import torch

    r_out, r_conf, r_det = postprocess(ref)
    g_out, g_conf, g_det = postprocess(got)
    report["equivalence"] = {
        "map": np_stats("map", r_out, g_out),
        "conf": np_stats("conf", r_conf, g_conf),
        "integrity_score": {"pytorch_orc_before": r_det, "onnx_folded": g_det,
                            "abs_err": abs(r_det - g_det)},
    }
    worst = max(report["equivalence"]["map"]["max_abs_err"], report["equivalence"]["conf"]["max_abs_err"])
    report["folding_is_numerically_safe"] = bool(worst < 1e-4 and report["equivalence"]["integrity_score"]["abs_err"] < 1e-5)
    print("\nfolding equivalence (ORT CPU, before vs after):")
    print("  map : max_abs=%.3e mean_abs=%.3e corr=%.10f" % (
        report["equivalence"]["map"]["max_abs_err"], report["equivalence"]["map"]["mean_abs_err"],
        report["equivalence"]["map"]["corr"]))
    print("  conf: max_abs=%.3e mean_abs=%.3e corr=%.10f" % (
        report["equivalence"]["conf"]["max_abs_err"], report["equivalence"]["conf"]["mean_abs_err"],
        report["equivalence"]["conf"]["corr"]))
    print("  det : before=%.8f after=%.8f err=%.3e" % (
        r_det, g_det, report["equivalence"]["integrity_score"]["abs_err"]))
    print("  numerically safe: %s" % report["folding_is_numerically_safe"])

    with open(os.path.join(args.outdir, "folding-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % folded_path)
    return 0 if report["folding_is_numerically_safe"] else 1


if __name__ == "__main__":
    sys.exit(main())
