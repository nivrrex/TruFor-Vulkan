#!/usr/bin/env python3
"""Phase 6E + 6F: actually execute on the RX 560, then the full TruFor model.

6E first runs a small graph made of exactly the operator kinds TruFor uses
(conv -> activation -> reshape -> matmul/linear -> norm -> softmax) on
    eager PyTorch / TVM llvm / TVM vulkan
and reports the error of each against eager. Only if that passes does 6F run the
full model. The point of 6E is to tell "Vulkan execution path is broken" apart
from "something in this particular 1918-node model is broken".

Run with the source build on the path:
  PYTHONPATH=vendor/tvm-src/python TVM_LIBRARY_PATH=vendor/tvm-src/build \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def run_exe(exe, dev, x, warmup=1, repeats=3):
    """Build the relax VM, run warmup + repeats, return (outputs, times)."""
    from tvm import relax

    import tvm_ffi

    vm = relax.VirtualMachine(exe.jit(), dev)
    t = time.perf_counter()
    times = []
    out = None
    for _ in range(warmup + repeats):
        t0 = time.perf_counter()
        out = vm["main"](x)
        times.append(time.perf_counter() - t0)
    # VM output is a tvm_ffi.Array for multi-output graphs and a bare Tensor for
    # single-output ones; Tensor is not iterable, so branch on the type.
    if isinstance(out, tvm_ffi.Array):
        arrs = [to_numpy(out[i]) for i in range(len(out))]
    elif isinstance(out, (list, tuple)):
        arrs = [to_numpy(o) for o in out]
    else:
        arrs = [to_numpy(out)]
    setup = time.perf_counter() - t
    return arrs, {"setup_s": setup, "warmup_s": times[:warmup],
                  "steady_s": times[warmup:], "steady_min_s": min(times[warmup:]) if len(times) > warmup else None}


def dev_tensor(x, dev):
    """Allocate the input on the same device the kernels run on.

    A CPU tensor handed to a Vulkan kernel aborts with
    'Mismatched x.device_type on argument #0'.
    """
    import tvm
    return tvm.runtime.tensor(x, dev)


def stats(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return {"name": name, "shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = np.abs(a - b)
    return {"name": name, "shape_match": True, "shape": list(a.shape),
            "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0}


def phase6e(tvm, relax, report, outdir_hint):
    """Small operator-coverage model, built straight with tvm.relax.from_function."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    ).eval()
    ref_out = None
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        ref = m(x)

    # tvm.relax.frontend.torch was removed in 0.27, so go through ONNX -- the
    # same path the real model takes, which keeps the smoke test honest.
    import onnx  # noqa: F401
    from tvm.relax.frontend.onnx import from_onnx

    tmp_onnx = os.path.join(outdir_hint, "phase6e-smoke.onnx")
    torch.onnx.export(m, x, tmp_onnx, input_names=["x"], output_names=["y"],
                      opset_version=18, dynamo=True, verbose=False)
    om = onnx.load(tmp_onnx, load_external_data=True)
    tmod = from_onnx(om, shape_dict={"x": [1, 3, 32, 32]}, dtype_dict="float32")
    xarr = x.numpy().astype("float32")

    row = {"name": "conv-relu-conv-relu", "input_shape": list(x.shape)}
    for label, dev in (("tvm-llvm", tvm.cpu()), ("tvm-vulkan", tvm.vulkan())):
        target = {"kind": "llvm"} if label == "tvm-llvm" else {"kind": "vulkan", "from_device": 0}
        try:
            t0 = time.perf_counter()
            exe = relax.build(tmod, target=tvm.target.Target(target))
            build_s = time.perf_counter() - t0
            outs, tm = run_exe(exe, dev, dev_tensor(xarr, dev), warmup=1, repeats=3)
            o = outs[0]
            row[label] = {"build_s": build_s, **tm, "out_shape": list(o.shape),
                          "ref_shape": list(ref.shape),
                          "equivalence": stats("out", ref.numpy(), o)}
            print("  [6E] %-11s build=%.1fs steady=%.4fs  max_abs=%.3e corr=%.10f"
                  % (label, build_s, tm["steady_min_s"] or -1,
                     row[label]["equivalence"]["max_abs_err"], row[label]["equivalence"]["corr"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row[label] = {"error": "%s: %s" % (type(e).__name__, str(e)[:400])}
            print("  [6E] %-11s FAILED: %s" % (label, row[label]["error"]), flush=True)
    report["phase6e"] = row
    ok = all("error" not in row.get(k, {}) for k in ("tvm-llvm", "tvm-vulkan"))
    report["phase6e_pass"] = ok
    return ok


def phase6f(tvm, relax, onnx_path, image, outdir, report, target_json, repeats=5):
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tvm.relax.frontend.onnx import from_onnx

    # read the input resolution from the graph rather than assuming 512
    _om = onnx.load(onnx_path, load_external_data=False)
    _ishape = [d.dim_value for d in _om.graph.input[0].type.tensor_type.shape.dim]
    hw = (int(_ishape[2]), int(_ishape[3]))
    img = Image.open(image).convert("RGB")
    if img.size != hw:
        img = img.resize(hw, Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    ref = sess.run(None, {sess.get_inputs()[0].name: x})
    ort_s = time.perf_counter() - t0

    def post(res):
        out, conf, det = res
        o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy()
        c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy()
        d = float(np.asarray(det).reshape(-1)[0])
        return o, c, 1.0 / (1.0 + np.exp(-d))

    ref_map, ref_conf, ref_det = post(ref)

    m = onnx.load(onnx_path, load_external_data=False)
    ir = from_onnx(m, shape_dict={"rgb": [1, 3, hw[0], hw[1]]}, dtype_dict="float32")
    target = tvm.target.Target(target_json)

    row = {"ort_cpu_seconds": ort_s, "reference_integrity_score": ref_det}
    t0 = time.perf_counter()
    exe = relax.build(ir, target=target)
    row["build_seconds"] = time.perf_counter() - t0
    print("  [6F] build ok in %.1fs" % row["build_seconds"], flush=True)

    try:
        vk = tvm.vulkan()
        outs, tm = run_exe(exe, vk, dev_tensor(x, vk), warmup=1, repeats=repeats)
        g_map, g_conf, g_det = post(outs)
        row["vulkan"] = tm
        row["equivalence"] = {
            "map": stats("map", ref_map, g_map),
            "conf": stats("conf", ref_conf, g_conf),
            "integrity_score": {"ref": ref_det, "vulkan": g_det, "abs_err": abs(ref_det - g_det)},
        }
        ma = (ref_map > 0.5)
        mb = (g_map > 0.5)
        inter = np.logical_and(ma, mb).sum()
        union = np.logical_or(ma, mb).sum()
        row["equivalence"]["map_iou"] = float(inter / union) if union else 1.0
        row["equivalence"]["map_dice"] = float(2 * inter / (ma.sum() + mb.sum())) if (ma.sum() + mb.sum()) else 1.0
        worst = max(row["equivalence"]["map"]["max_abs_err"], row["equivalence"]["conf"]["max_abs_err"])
        row["correctness_pass"] = bool(worst < 1e-2 and row["equivalence"]["integrity_score"]["abs_err"] < 1e-2)
        print("  [6F] VULKAN RAN. steady=%s" % " ".join("%.3fs" % t for t in tm["steady_s"]), flush=True)
        print("       score  ref=%.6f vk=%.6f err=%.3e" % (ref_det, g_det, row["equivalence"]["integrity_score"]["abs_err"]))
        print("       map    max_abs=%.3e mean=%.3e rmse=%.3e corr=%.10f IoU=%.6f Dice=%.6f"
              % (row["equivalence"]["map"]["max_abs_err"], row["equivalence"]["map"]["mean_abs_err"],
                 row["equivalence"]["map"]["rmse"], row["equivalence"]["map"]["corr"],
                 row["equivalence"]["map_iou"], row["equivalence"]["map_dice"]))
        print("       conf   max_abs=%.3e mean=%.3e corr=%.10f"
              % (row["equivalence"]["conf"]["max_abs_err"], row["equivalence"]["conf"]["mean_abs_err"],
                 row["equivalence"]["conf"]["corr"]))
    except Exception as e:  # noqa: BLE001
        import traceback
        row["vulkan_error"] = "%s: %s" % (type(e).__name__, str(e)[:800])
        tb = traceback.format_exc()
        with open(os.path.join(outdir, "tvm-vulkan-run-error.txt"), "w") as f:
            f.write(tb)
        print("  [6F] VULKAN RUN FAILED: %s" % row["vulkan_error"], flush=True)
        print(tb[-1200:], flush=True)
    report["phase6f"] = row
    return "vulkan_error" not in row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", default='{"kind":"vulkan","from_device":0}')
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-6e", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import tvm
    from tvm import relax

    tv = getattr(tvm, "__version__", None) or tvm._version.__version__
    print("tvm %s" % tv, flush=True)

    dev = tvm.vulkan()
    print("Vulkan device: %s  (exists=%s)" % (dev.device_name, dev.exist), flush=True)
    if "llvmpipe" in str(dev.device_name).lower():
        print("REFUSING to continue on llvmpipe (software Vulkan)")
        return 2

    report = {"tvm_version": tv, "device_name": dev.device_name, "target": args.target}

    print("\n=== Phase 6E: operator smoke test on Vulkan ===", flush=True)
    if args.skip_6e:
        print("  (skipped)")
        ok6e = True
    else:
        try:
            ok6e = phase6e(tvm, relax, report, args.outdir)
        except Exception as e:  # noqa: BLE001
            import traceback
            report["phase6e_error"] = "%s: %s" % (type(e).__name__, str(e)[:500])
            print("  [6E] FAILED: %s" % report["phase6e_error"], flush=True)
            print(traceback.format_exc()[-800:], flush=True)
            ok6e = False

    if not ok6e:
        print("\nPhase 6E did not pass; stopping before the full model (per the plan).")
        with open(os.path.join(args.outdir, "tvm-vulkan-report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return 1

    print("\n=== Phase 6F: full TruFor on Vulkan ===", flush=True)
    run6f = phase6f(tvm, relax, args.onnx, args.image, args.outdir, report, args.target, args.repeats)

    with open(os.path.join(args.outdir, "tvm-vulkan-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "tvm-vulkan-report.json"))
    return 0 if run6f else 1


if __name__ == "__main__":
    sys.exit(main())
