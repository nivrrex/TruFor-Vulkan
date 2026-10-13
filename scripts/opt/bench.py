"""Unified experiment driver for the optimisation stages.

One place that builds a whole-graph executable and measures it, so every stage is
judged the same way:

  * median of 15 timed runs (un-instrumented; profiling is a separate script)
  * correctness against the ONNX Runtime CPU reference: map correlation, IoU,
    Dice, integrity score error
  * prints the per-shape kernels of interest, read from the TIR, never from names

The reduction split is applied in memory only and can be turned off with
TRUFOR_SPLIT_REDUCE=0, so the default path stays the unmodified model.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "opt"))

DEFAULT_ONNX = os.path.join(REPO, "reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx")
DEFAULT_IMAGE = os.path.join(REPO, "truFor-src/test_docker/images/tampered2.png")


def load_input(path, size=512):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return (np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)


def reference(onnx_path, x, threads=4):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})


def post(res):
    import torch
    import torch.nn.functional as F

    out, conf, det = res
    o = F.softmax(torch.from_numpy(np.asarray(out, np.float32)[0]), 0)[1].numpy()
    c = torch.sigmoid(torch.from_numpy(np.asarray(conf, np.float32)[0]))[0].numpy()
    d = float(np.asarray(det).reshape(-1)[0])
    return o, c, 1.0 / (1.0 + np.exp(-d))


def build(model, target, extra_sched=False):
    """Lower and compile. The split, if any, is already inside `model`."""
    import tvm
    import tvm.ir.transform as T
    from tvm.relax.backend.gpu_generic import pipeline as GP
    from tvm.relax.transform import (
        AnnotateTIROpPattern,
        FoldConstant,
        FuseOps,
        FuseTIR,
        LegalizeOps,
    )
    from tvm.s_tir import dlight as dl
    from tvm.relax.frontend.onnx import from_onnx

    ir = from_onnx(model, shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
    rules = [dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(), dl.gpu.GeneralReduction()]
    if extra_sched:
        rules.append(dl.gpu.Transpose())
    rules.append(dl.gpu.Fallback())
    seq = T.Sequential(
        GP.library_dispatch_passes(target)
        + [LegalizeOps(), AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR(),
           dl.ApplyDefaultSchedule(*rules)]
        + GP.dataflow_lower_passes(target)
        + GP.finalize_passes(target)
    )
    with target:
        mod = seq(ir)
    return tvm.relax.build(mod, target=target), mod.script()


def run(exe, dev, x, reps=15, warmup=3):
    import tvm
    from tvm import relax

    vm = relax.VirtualMachine(exe.jit(), dev)
    tin = tvm.runtime.tensor(x, dev)
    for _ in range(warmup):
        vm["main"](tin)
        dev.sync()
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = vm["main"](tin)
        dev.sync()
        ts.append(time.perf_counter() - t0)
    return [o.numpy() for o in out], ts


def tirstats(tir_text, names):
    """Read a kernel's grid and shapes out of the TIR, not out of its name."""
    out = {}
    blocks = re.split(r"\n    def ", tir_text)
    for blk in blocks:
        m = re.match(r"(\w+)\((.*?)\)", blk.split("\n")[0])
        if not m or m.group(1) not in names:
            continue
        body = blk
        bx = re.search(r'T\.thread_binding\(T\.int64\((\d+)\), thread="blockIdx\.x"\)', body)
        tx = re.findall(r'T\.thread_binding\(T\.int64\((\d+)\), thread="threadIdx\.(\w+)"\)', body)
        dims = re.findall(r"T\.int64\((\d+)\)", m.group(2))
        out[m.group(1)] = {"grid_x": int(bx.group(1)) if bx else None,
                           "threads": [int(v) for v, _ in tx],
                           "arg_dims": [int(v) for v in dims[:5]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--extra-sched", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=os.path.join(REPO, "work", "last_experiment.json"))
    args = ap.parse_args()

    import onnx
    import tvm
    import split_reduce

    dev = tvm.vulkan()
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("device %s  driver %s" % (dev.device_name, dev.driver_version), flush=True)

    x = load_input(args.image)
    ref_raw = reference(args.onnx, x)
    r_map, r_conf, r_det = post(ref_raw)
    ref_ma = (r_map > 0.5)
    print("reference: det=%.8f  map>0.5 pixels=%d" % (r_det, int(ref_ma.sum())), flush=True)

    results = {}
    for label, split in (("baseline", False), ("split", True)):
        os.environ["TRUFOR_SPLIT_REDUCE"] = "1" if split else "0"
        model = onnx.load(args.onnx)
        recs = []
        if split:
            model, recs = split_reduce.split(model)
        exe, tir = build(model, tgt, args.extra_sched)
        outs, ts = run(exe, dev, x, args.reps, args.warmup)
        g_map, g_conf, g_det = post(outs)
        corr = float(np.corrcoef(r_map.ravel().astype(np.float64), g_map.ravel().astype(np.float64))[0, 1])
        g_ma = g_map > 0.5
        inter = np.logical_and(ref_ma, g_ma).sum()
        union = np.logical_or(ref_ma, g_ma).sum()
        iou = float(inter / union) if union else 1.0
        dice = float(2 * inter / (ref_ma.sum() + g_ma.sum())) if (ref_ma.sum() + g_ma.sum()) else 1.0
        nprim = len(re.findall(r"@T\.prim_func", tir))
        results[label] = {
            "median_ms": statistics.median(ts) * 1e3,
            "min_ms": min(ts) * 1e3, "max_ms": max(ts) * 1e3,
            "map_corr": corr, "iou": iou, "dice": dice,
            "score_abs_err": abs(g_det - r_det),
            "prim_funcs": nprim, "rewrites": recs,
        }
        print("%-9s median=%8.1f ms  min=%8.1f  kernels=%d  corr=%.7f  IoU=%.6f Dice=%.6f  det_err=%.2e"
              % (label, results[label]["median_ms"], results[label]["min_ms"], nprim, corr, iou, dice,
                 results[label]["score_abs_err"]), flush=True)
        for r in recs:
            print("    split %-24s %s -> %s into %s" % (r["op"], r["input_shape"], r["output_shape"],
                                                         r["split_into"]), flush=True)

    if "baseline" in results and "split" in results:
        d = results["baseline"]["median_ms"] - results["split"]["median_ms"]
        print("\nwhole-graph change: %+.1f ms (%.1f%%)  speedup vs baseline %.4fx"
              % (-d, 100.0 * d / results["baseline"]["median_ms"],
                 results["baseline"]["median_ms"] / results["split"]["median_ms"]), flush=True)
        results["delta_ms"] = d

    names = ["fused_conv2d1_add_relu", "max4", "fused_transpose64_reshape62_transpose65_reshape"]
    results["tir_stats"] = tirstats(tir, names)
    for k, v in results["tir_stats"].items():
        print("TIR %-44s grid_x=%s threads=%s" % (k, v["grid_x"], v["threads"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
