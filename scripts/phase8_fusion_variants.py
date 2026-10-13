#!/usr/bin/env python3
"""Phase 8B: is the full-graph divergence caused by fusion or layout?

The Phase 8A rewrite (all 80 dim-0 scalar Gather -> Slice + Squeeze) is
bit-identical on ORT CPU and passes the TVM-LLVM gate, but the patched model is
still wrong on Vulkan (map corr -0.287). So the take/select bug is real and
reproducible, but it is not the only thing breaking the full graph.

The stage-1 sub-module tests were run on *separate* sub-graphs, where each
module is lowered on its own. The full 1918-node graph instead goes through the
whole gpu_generic pipeline, whose legalize stage contains

    LegalizeOps, AnnotateTIROpPattern, FoldConstant, FuseOps, FuseTIR,
    ApplyDefaultSchedule(gpu.Matmul / GEMV / Reduction / GeneralReduction /
    Fallback)

FuseOps/FuseTIR rewrite elementwise chains into fused TIR kernels, so a correct
operator can still be wrong once a neighbour is absorbed into it. This rebuilds
the same model with those passes removed and re-measures, which separates
"fused kernel is wrong" from "individual kernels are wrong".

Variants:
    A  original ONNX, default pipeline   (known wrong)
    B  patched  ONNX, default pipeline   (known wrong)
    C  patched  ONNX, FuseOps/FuseTIR removed
    D  patched  ONNX, ApplyDefaultSchedule removed
    E  patched  ONNX, both removed
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--variants", default="C,D,E")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import tvm
    from tvm import relax
    from tvm.relax.backend.gpu_generic import pipeline as GP
    from tvm.relax.frontend.onnx import from_onnx

    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from PIL import Image

    dev = tvm.vulkan()
    vk = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("device: %s" % dev.device_name, flush=True)

    img = Image.open(args.image).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ref_session = ort.InferenceSession(args.original, so, providers=["CPUExecutionProvider"])
    ref = ref_session.run(None, {ref_session.get_inputs()[0].name: x})

    def post(res):
        out, conf, det = res
        o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy()
        c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy()
        d = float(np.asarray(det).reshape(-1)[0])
        return o, c, 1.0 / (1.0 + np.exp(-d))

    def stats(a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        d = np.abs(a - b)
        return {"shape_match": True, "max_abs_err": float(d.max()),
                "mean_abs_err": float(d.mean()),
                "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else None}

    ref_map, ref_conf, ref_det = post(ref)

    def custom_pipeline(drop_fuse: bool, drop_schedule: bool):
        def getter(target):
            @tvm.transform.module_pass(opt_level=0)
            def _p(mod, _ctx):
                leg = []
                for p in GP.legalize_passes(target):
                    cls = type(p).__name__
                    if drop_fuse and cls in ("FuseOps", "FuseTIR"):
                        continue
                    if drop_schedule and cls.startswith("ApplyDefaultSchedule"):
                        continue
                    leg.append(p)
                seq = tvm.transform.Sequential(
                    GP.library_dispatch_passes(target) + leg
                    + GP.dataflow_lower_passes(target) + GP.finalize_passes(target)
                )
                return seq(mod)
            return _p
        return getter

    variants = {
        "C": (args.patched, dict(drop_fuse=True, drop_schedule=False)),
        "D": (args.patched, dict(drop_fuse=False, drop_schedule=True)),
        "E": (args.patched, dict(drop_fuse=True, drop_schedule=True)),
    }

    rows = []
    print("\n%-4s %-34s %-9s %-11s %-11s %s" % ("var", "config", "build_s", "max_abs", "corr", "verdict"), flush=True)
    for key in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if key not in variants:
            continue
        path, flags = variants[key]
        row = {"variant": key, "model": path, **flags}
        try:
            m = onnx.load(path, load_external_data=False)
            ir = from_onnx(m, shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
            t0 = time.perf_counter()
            exe = relax.build(ir, target=vk, relax_pipeline=custom_pipeline(**flags)(vk))
            build_s = time.perf_counter() - t0
            import tvm_ffi
            vm = relax.VirtualMachine(exe.jit(), dev)
            out = vm["main"](tvm.runtime.tensor(x, dev))
            got = ([out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array)
                   else ([o.numpy() for o in out] if isinstance(out, (list, tuple)) else [out.numpy()]))
            g_map, g_conf, g_det = post(got)
            sm, sc = stats(ref_map, g_map), stats(ref_conf, g_conf)
            corr = sm.get("corr") if sm.get("corr") is not None else -1.0
            ok = corr > 0.99 and sm.get("max_abs_err", 1.0) < 1e-2
            row.update({"build_s": build_s, "map": sm, "conf": sc,
                        "integrity_score": {"ref": ref_det, "vulkan": g_det, "abs_err": abs(ref_det - g_det)},
                        "verdict": "OK" if ok else "DIVERGES"})
            cfg = "fuse=%s sched=%s" % (not flags["drop_fuse"], not flags["drop_schedule"])
            print("%-4s %-34s %-9.1f %-11.3e %-11s %s"
                  % (key, cfg, build_s, sm.get("max_abs_err", float("nan")),
                     ("%.8f" % corr), row["verdict"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["verdict"] = "ERROR"
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            with open(os.path.join(args.outdir, "variant-%s-error.txt" % key), "w") as f:
                f.write(traceback.format_exc())
            print("%-4s %-34s ERROR %s" % (key, "-", row["error"][:70]), flush=True)
        rows.append(row)

    with open(os.path.join(args.outdir, "fusion-variants.json"), "w") as f:
        json.dump({"device": dev.device_name, "rows": rows}, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "fusion-variants.json"))


if __name__ == "__main__":
    sys.exit(main())
