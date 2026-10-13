#!/usr/bin/env python3
"""Phase 6C: TVM CPU (LLVM) build + run, compared against the verified reference.

This is the correctness gate for TVM as a whole. Vulkan cannot be attempted
until TVM reproduces the ONNX Runtime / PyTorch results, because a Vulkan-only
discrepancy would be indistinguishable from a compiler bug.

Uses the installed apache-tvm wheel (USE_LLVM=ON, USE_VULKAN=OFF), so it runs
with the venv interpreter and needs no extra build.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

TVM_VERSION = None


def postprocess(out, conf, det):
    """Map raw network outputs to (localization map, confidence map, integrity score)."""
    import torch
    import torch.nn.functional as F

    o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy() if out.ndim == 4 and out.shape[1] > 1 else out
    c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy() if conf.ndim == 4 else conf
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


def iou_dice(a, b, thr=0.5):
    ma = (a > thr).astype(bool)
    mb = (b > thr).astype(bool)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return 1.0, 1.0
    return float(inter / union), float(2 * inter / (ma.sum() + mb.sum()))


def main():
    global TVM_VERSION

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    # source build exposes tvm.__version__; the wheel only tvm._version.__version__
    TVM_VERSION = getattr(tvm, "__version__", None)
    if TVM_VERSION is None:
        TVM_VERSION = tvm._version.__version__
    print("tvm %s  target=%s" % (TVM_VERSION, args.target), flush=True)

    # ---- reference: ONNX Runtime CPU on the same input ----
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
    ref_s = time.perf_counter() - t0
    ref_map, ref_conf, ref_det = postprocess(*ref)
    print("ORT CPU reference: %.2fs  det=%.8f" % (ref_s, ref_det), flush=True)

    # ---- import + build ----
    import onnx

    m = onnx.load(args.onnx, load_external_data=False)
    ir = from_onnx(m, shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
    print("import ok", flush=True)

    t0 = time.perf_counter()
    build_ok = True
    rt_mod = None
    try:
        rt_mod = relax.build(ir, target=args.target)
    except Exception as e:  # noqa: BLE001
        import traceback

        build_ok = False
        print("BUILD FAILED: %s: %s" % (type(e).__name__, str(e)[:400]), flush=True)
        tb = traceback.format_exc()
        with open(os.path.join(args.outdir, "tvm-build-error.txt"), "w") as f:
            f.write(tb)
        print(tb[-1200:], flush=True)
    build_s = time.perf_counter() - t0
    print("build %s in %.1fs" % ("ok" if build_ok else "FAILED", build_s), flush=True)

    report = {
        "tvm_version": TVM_VERSION, "target": args.target,
        "onnx": args.onnx, "image": args.image,
        "import_ok": True, "build_ok": build_ok, "build_seconds": build_s,
        "ort_cpu_seconds": ref_s,
        "reference": {"integrity_score": ref_det,
                      "map_mean": float(ref_map.mean()), "conf_mean": float(ref_conf.mean())},
    }

    if not build_ok:
        with open(os.path.join(args.outdir, "tvm-cpu-report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote report (build failed)")
        return 1

    # ---- run ----
    # Official 0.27 pattern (tests/python/relax/test_transform_codegen_pass.py):
    #   vm = relax.VirtualMachine(exe.jit(), dev);  out = vm["main"](*inputs)
    # Executable.__call__ does not work for a VMExecutable: the bytecode entry
    # point is not exported as a plain module function.
    vm = relax.VirtualMachine(rt_mod.jit(), tvm.cpu())
    tvm_input = tvm.runtime.tensor(x)
    times = []
    out = None
    for _ in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        out = vm["main"](tvm_input)
        times.append(time.perf_counter() - t0)
    # relax.VirtualMachine returns a tvm_ffi.Array container holding the 3
    # output tensors; each element is a tvm Tensor exposing .numpy().
    import tvm_ffi
    if isinstance(out, tvm_ffi.Array):
        got = [out[i].numpy() for i in range(len(out))]
    elif isinstance(out, (list, tuple)):
        got = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for a in out]
    else:
        got = [out.numpy() if hasattr(out, "numpy") else np.asarray(out)]
    print("outputs: %s" % [list(g.shape) for g in got], flush=True)
    g_map, g_conf, g_det = postprocess(*got)

    report["run_seconds"] = times
    report["run_seconds_min"] = min(times)
    report["equivalence"] = {
        "map": stats("map", ref_map, g_map),
        "conf": stats("conf", ref_conf, g_conf),
        "integrity_score": {"ref": ref_det, "tvm": g_det, "abs_err": abs(ref_det - g_det)},
    }
    i, d = iou_dice(ref_map, g_map)
    report["equivalence"]["map_iou"] = i
    report["equivalence"]["map_dice"] = d

    worst = max(report["equivalence"]["map"]["max_abs_err"], report["equivalence"]["conf"]["max_abs_err"])
    report["correctness_pass"] = bool(worst < 1e-3 and report["equivalence"]["integrity_score"]["abs_err"] < 1e-4)

    print("\n=== TVM %s vs ONNX Runtime CPU ===" % args.target)
    print("  build: %.1fs   run: %s" % (build_s, " ".join("%.2fs" % t for t in times)))
    print("  integrity score : ref=%.8f  tvm=%.8f  err=%.3e"
          % (ref_det, g_det, report["equivalence"]["integrity_score"]["abs_err"]))
    for k in ("map", "conf"):
        s = report["equivalence"][k]
        print("  %-14s : max_abs=%.3e mean_abs=%.3e rmse=%.3e corr=%.10f"
              % (k, s["max_abs_err"], s["mean_abs_err"], s["rmse"], s["corr"]))
    print("  map IoU=%.6f Dice=%.6f" % (i, d))
    print("  CORRECTNESS: %s" % ("PASS" if report["correctness_pass"] else "FAIL"))

    with open(os.path.join(args.outdir, "tvm-cpu-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "tvm-cpu-report.json"))
    return 0 if report["correctness_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
