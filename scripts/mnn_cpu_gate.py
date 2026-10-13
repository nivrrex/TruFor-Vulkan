#!/usr/bin/env python3
"""Phase 7C: MNN CPU correctness, before any Vulkan attempt.

Runs the converted .mnn through MNN's CPU backend and compares against the
ONNX Runtime CPU reference on all three outputs. Per the plan, Vulkan is only
attempted if this passes -- otherwise a Vulkan-only discrepancy would be
indistinguishable from a bad conversion.

Uses the pip `MNN` python binding (statically linked, CPU). The Vulkan backend
needs the release libMNN.so and is exercised separately by a C++ driver.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def postprocess(out, conf, det):
    import torch
    import torch.nn.functional as F

    out = np.asarray(out, np.float32)
    conf = np.asarray(conf, np.float32)
    # both carry a leading batch dim of 1: out [1,2,H,W], conf [1,1,H,W]
    if out.ndim == 4:
        o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy()
    else:
        o = F.softmax(torch.from_numpy(out), dim=0)[1].numpy()
    c = torch.sigmoid(torch.from_numpy(conf[0] if conf.ndim == 4 else conf))[0].numpy()
    d = float(np.asarray(det).reshape(-1)[0])
    return o, c, 1.0 / (1.0 + np.exp(-d))


def stats(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return {"name": name, "shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = np.abs(a - b)
    return {
        "name": name, "shape_match": True, "shape": list(a.shape),
        "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnn", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--backend", default="CPU")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import MNN  # noqa: F401  (import for its side-effect prints / availability)
    import onnxruntime as ort
    from PIL import Image

    img = Image.open(args.image).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    ref = sess.run(None, {sess.get_inputs()[0].name: x})
    ort_s = time.perf_counter() - t0
    ref_map, ref_conf, ref_det = postprocess(*ref)
    print("ORT CPU reference: %.2fs det=%.8f" % (ort_s, ref_det), flush=True)

    interp = MNN.Interpreter(args.mnn)
    from MNN import Tensor

    config = {"precision": "normal", "backend": args.backend, "numThread": 4}
    ret = interp.createSession(config)
    print("createSession(%s) -> %r" % (args.backend, ret), flush=True)
    if isinstance(ret, int) and ret != 0:
        print("session creation failed")
        return 1
    print("session info:", interp.getSessionInfo(0) if hasattr(interp, "getSessionInfo") else "n/a", flush=True)

    n_in = len(interp.getSessionInputNames())
    n_out = len(interp.getSessionOutputNames())
    print("inputs : %d  outputs: %d" % (n_in, n_out), flush=True)

    in_tensor = Tensor(x, MNN.Halide_Type_Float, [1, 3, 512, 512], MNN.Tensor_DimensionType_Caffe)
    in_tensors = [in_tensor]
    out_tensors = [Tensor(np.zeros((1, 2, 512, 512), np.float32), MNN.Halide_Type_Float,
                            [1, 2, 512, 512], MNN.Tensor_DimensionType_Caffe),
                   Tensor(np.zeros((1, 1, 512, 512), np.float32), MNN.Halide_Type_Float,
                            [1, 1, 512, 512], MNN.Tensor_DimensionType_Caffe),
                   Tensor(np.zeros((1, 1), np.float32), MNN.Halide_Type_Float,
                            [1, 1], MNN.Tensor_DimensionType_Caffe)]

    times = []
    arrs = None
    for _ in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        st = interp.runSessionWithInputs(in_tensors, out_tensors)
        if st != 0:
            print("runSessionWithInputs -> %d" % st)
            return 1
        times.append(time.perf_counter() - t0)
        arrs = [np.array(t.getNumpyData(), dtype=np.float32).reshape(tuple(t.getShape()))
                for t in out_tensors]

    print("mnn outputs:", [list(a.shape) for a in arrs], flush=True)
    if len(arrs) < 3:
        print("expected 3 outputs, got %d" % len(arrs))
        return 1
    g_map, g_conf, g_det = postprocess(arrs[0], arrs[1], arrs[2])

    report = {
        "backend": args.backend, "mnn_model": args.mnn,
        "ort_cpu_seconds": ort_s, "mnn_seconds": times,
        "mnn_seconds_min": min(times),
        "reference_integrity_score": ref_det,
        "equivalence": {
            "map": stats("map", ref_map, g_map),
            "conf": stats("conf", ref_conf, g_conf),
            "integrity_score": {"ref": ref_det, "mnn": g_det, "abs_err": abs(ref_det - g_det)},
        },
    }
    worst = max(report["equivalence"]["map"]["max_abs_err"], report["equivalence"]["conf"]["max_abs_err"])
    report["correctness_pass"] = bool(worst < 1e-2 and report["equivalence"]["integrity_score"]["abs_err"] < 1e-2)

    print("\n=== MNN %s vs ONNX Runtime CPU ===" % args.backend)
    print("  latency: %s" % " ".join("%.2fs" % t for t in times))
    print("  score  ref=%.8f mnn=%.8f err=%.3e" % (ref_det, g_det, report["equivalence"]["integrity_score"]["abs_err"]))
    for k in ("map", "conf"):
        s = report["equivalence"][k]
        print("  %-6s max_abs=%.3e mean=%.3e rmse=%.3e corr=%.10f" % (k, s["max_abs_err"], s["mean_abs_err"], s["rmse"], s["corr"]))
    print("  CORRECTNESS: %s" % ("PASS" if report["correctness_pass"] else "FAIL"))

    with open(os.path.join(args.outdir, "mnn-%s-report.json" % args.backend.lower()), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "mnn-%s-report.json" % args.backend.lower()))
    return 0 if report["correctness_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
