#!/usr/bin/env python3
"""Final benchmark: TruFor inference on the RX 560, TVM Vulkan, Mesa 25.0.7.

The earlier timings were not usable: they stopped at the VM call and never
waited for the device, so they measured dispatch, not computation. Here the
output is copied back to host inside the timed region, which forces the Vulkan
queue to drain, and every configuration is warmed up before the steady-state
runs.

Configurations:
    pytorch-cpu  : unmodified TruFor, PyTorch, CPU (the BASELINE.md reference)
    ort-cpu      : the verified ONNX model on ONNX Runtime CPU
    tvm-llvm     : the same ONNX through TVM's LLVM backend
    tvm-vulkan   : the same ONNX through TVM's Vulkan backend on the RX 560

Correctness of every configuration is checked against the ONNX Runtime CPU
result for the same input, so a "fast but wrong" result cannot be reported as a
speedup.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TruFor checkpoint (.pth.tar)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", default="", help="comma list to restrict configs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    only = {c.strip() for c in args.only.split(",") if c.strip()}

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import onnxruntime as ort
    import tvm
    from PIL import Image

    print("tvm %s" % (getattr(tvm, "__version__", None) or tvm._version.__version__), flush=True)
    dev = tvm.vulkan()
    print("device: %s  driver=%s  api=%s" % (dev.device_name, dev.driver_version, dev.api_version), flush=True)
    vk = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    llvm = tvm.target.Target({"kind": "llvm"})

    import onnx
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    import tvm_ffi

    def load_img(path):
        img = Image.open(path).convert("RGB")
        if img.size != (args.size, args.size):
            img = img.resize((args.size, args.size), Image.BILINEAR)
        return (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)

    xs = {p: load_img(p) for p in args.images}

    def post(out, conf, det):
        o = F.softmax(torch.from_numpy(np.asarray(out, np.float32)[0]), dim=0)[1].numpy()
        c = torch.sigmoid(torch.from_numpy(np.asarray(conf, np.float32)[0]))[0].numpy()
        d = float(np.asarray(det).reshape(-1)[0])
        return o, c, 1.0 / (1.0 + np.exp(-d))

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    iname = ort_sess.get_inputs()[0].name

    om = onnx.load(args.onnx, load_external_data=False)
    ir = from_onnx(om, shape_dict={"rgb": [1, 3, args.size, args.size]}, dtype_dict="float32")

    print("building tvm-llvm ...", flush=True)
    exe_llvm = relax.build(ir, target=llvm)
    print("building tvm-vulkan ...", flush=True)
    t0 = time.perf_counter()
    exe_vk = relax.build(ir, target=vk)
    print("  vulkan build %.1fs" % (time.perf_counter() - t0), flush=True)

    def to_np(out):
        if isinstance(out, tvm_ffi.Array):
            return [out[i].numpy() for i in range(len(out))]
        if isinstance(out, (list, tuple)):
            return [o.numpy() for o in out]
        return [out.numpy()]

    results = []

    for path, x in xs.items():
        print("\n=== %s (%dx%d) ===" % (os.path.basename(path), args.size, args.size), flush=True)

        # ---- reference ----
        ref = ort_sess.run(None, {iname: x})
        ref_map, ref_conf, ref_det = post(*ref)

        rows = {}

        def timeit(fn):
            for _ in range(args.warmup):
                fn()
            ts = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                r = fn()
                ts.append(time.perf_counter() - t0)
            return r, ts

        if not only or "ort-cpu" in only:
            r, ts = timeit(lambda: ort_sess.run(None, {iname: x}))
            rows["ort-cpu"] = (r, ts)

        if not only or "tvm-vulkan" in only:
            tx = tvm.runtime.tensor(x, dev)

            def run_vk():
                out = relax.VirtualMachine(exe_vk.jit(), dev)["main"](tx)
                a = to_np(out)          # host copy inside the timed region => sync
                return a
            r, ts = timeit(run_vk)
            rows["tvm-vulkan"] = (r, ts)

        if not only or "tvm-llvm" in only:
            tx = tvm.runtime.tensor(x)

            def run_llvm():
                out = relax.VirtualMachine(exe_llvm.jit(), tvm.cpu())["main"](tx)
                return to_np(out)
            r, ts = timeit(run_llvm)
            rows["tvm-llvm"] = (r, ts)

        ort_ms = min(rows["ort-cpu"][1]) * 1000 if "ort-cpu" in rows else None
        print("%-12s %10s %10s %10s %10s" % ("config", "min(ms)", "median", "max", "map_corr"))
        entry = {"image": path, "size": args.size, "configs": {}}
        for name, (r, ts) in rows.items():
            g_map, g_conf, g_det = post(*r)
            m = ref_map.astype(np.float64); gg = g_map.astype(np.float64)
            corr = float(np.corrcoef(m.ravel(), gg.ravel())[0, 1])
            maxerr = float(np.abs(m - gg).max())
            score_err = abs(g_det - ref_det)
            ok = corr > 0.999 and maxerr < 1e-2
            print("%-12s %10.1f %10.1f %10.1f %10.7f  %s"
                  % (name, min(ts) * 1000, float(np.median(ts)) * 1000, max(ts) * 1000, corr,
                     "OK" if ok else "WRONG"))
            entry["configs"][name] = {
                "min_s": min(ts), "median_s": float(np.median(ts)), "max_s": max(ts),
                "all_s": ts, "map_corr": corr, "map_max_abs": maxerr,
                "integrity_score": g_det, "integrity_score_ref": ref_det,
                "integrity_score_abs_err": score_err, "correct": bool(ok),
            }
        if ort_ms and "tvm-vulkan" in entry["configs"] and entry["configs"]["tvm-vulkan"]["correct"]:
            sp = ort_ms / (entry["configs"]["tvm-vulkan"]["min_s"] * 1000)
            entry["speedup_vulkan_vs_ort_cpu"] = sp
            print("speedup (ORT CPU / TVM Vulkan) = %.2fx" % sp)
        results.append(entry)

    with open(os.path.join(args.outdir, "benchmark.json"), "w") as f:
        json.dump({"device": dev.device_name, "driver": str(dev.driver_version),
                   "api": str(dev.api_version), "size": args.size,
                   "warmup": args.warmup, "repeats": args.repeats, "results": results}, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "benchmark.json"))


if __name__ == "__main__":
    sys.exit(main())
