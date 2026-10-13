"""Gate 0 + Gate 1: fix the environment, then establish the 512x512 baseline.

Gate 0  environment, driver, runtime and model identity are recorded so that
        every later gate is anchored to something reproducible.
Gate 1  the 512x512 batch-1 number, measured with the agreed protocol
        (warmup 3, 15 repeats, min/p10/median/p90/max), for both engines:
        ONNX Runtime CPU and TVM Vulkan on the RX 560.

Correctness is re-checked every run, so a "fast but wrong" measurement cannot
survive into the report.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ONNX = os.path.join(REPO, "reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx")
DEFAULT_IMAGE = os.path.join(REPO, "truFor-src/test_docker/images/tampered2.png")


def post(out, conf, det):
    import torch
    import torch.nn.functional as F
    o = F.softmax(torch.from_numpy(np.asarray(out, np.float32)[0]), dim=0)[1].numpy()
    c = torch.sigmoid(torch.from_numpy(np.asarray(conf, np.float32)[0]))[0].numpy()
    d = float(np.asarray(det).reshape(-1)[0])
    return o, c, 1.0 / (1.0 + np.exp(-d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--outdir", default=os.path.join(REPO, "reports/capability"))
    ap.add_argument("--repeats", type=int, default=C.REPEATS)
    ap.add_argument("--warmup", type=int, default=C.WARMUP)
    args = ap.parse_args()

    # ---------------------------------------------------------------- Gate 0
    print("=" * 70)
    print("GATE 0  environment / identity")
    print("=" * 70)
    env = C.collect_env()
    model = C.model_facts(args.onnx)
    for k in ("tvm", "tvm_use_vulkan", "cpu_count", "vram_total_bytes"):
        print("  %-22s %s" % (k, env.get(k)))
    for k, v in (env.get("vulkan") or {}).items():
        print("  vulkan.%-18s %s" % (k, v))
    print("  model.sha256         %s" % model["sha256"])
    print("  model.bytes          %d" % model["bytes"])
    print("  model.inputs         %s" % model["inputs"])

    v = env.get("vulkan") or {}
    gate0 = {
        "pass": bool(v.get("exists")) and model["bytes"] > 0,
        "driver_is_2507": "25.0.7" in str(v.get("driver_version", "")),
        "model_sha256": model["sha256"],
        "no_slice_squeeze_in_path": "slicefix" not in args.onnx,
    }
    print("\n  driver is 25.0.7     %s" % gate0["driver_is_2507"])
    print("  production ONNX       %s" % gate0["no_slice_squeeze_in_path"])
    print("  GATE 0: %s" % ("PASS" if gate0["pass"] and gate0["driver_is_2507"] else "FAIL"))
    if not (gate0["pass"] and gate0["driver_is_2507"]):
        print("  refusing to continue on an unverified environment")
        C.write_json(os.path.join(args.outdir, "gate0_env.json"), {"env": env, "model": model, "gate0": gate0})
        return 2

    # ---------------------------------------------------------------- Gate 1
    print("\n" + "=" * 70)
    print("GATE 1  512x512 batch-1 baseline")
    print("=" * 70)

    import onnx
    import onnxruntime as ort
    import tvm
    import tvm_ffi
    from PIL import Image
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    img = Image.open(args.image).convert("RGB").resize((512, 512), Image.BILINEAR)
    x = (np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    iname = ort_sess.get_inputs()[0].name

    m = onnx.load(args.onnx, load_external_data=False)
    ir = from_onnx(m, shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
    dev = tvm.vulkan()
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})

    import time
    t0 = time.perf_counter()
    exe = relax.build(ir, target=tgt)
    build_s = time.perf_counter() - t0
    print("  tvm vulkan build: %.1f s" % build_s)

    results = {"gate0": gate0, "env": env, "model": model, "vulkan_build_s": build_s}

    # ---- correctness reference first
    ref = ort_sess.run(None, {iname: x})
    r_map, r_conf, r_det = post(*ref)

    # ---- CPU
    peak = C.vram_peak_sample()
    t = C.Timer(args.warmup, args.repeats)
    t(lambda: ort_sess.run(None, {iname: x}))
    peak["stop"] = True
    cpu = t.report()
    print("\n  ONNX Runtime CPU   min=%7.1f ms  p10=%7.1f  med=%7.1f  p90=%7.1f  max=%7.1f  (sd=%.1f)"
          % (cpu["min"] * 1e3, cpu["p10"] * 1e3, cpu["median"] * 1e3,
             cpu["p90"] * 1e3, cpu["max"] * 1e3, cpu["stdev"] * 1e3))
    results["cpu"] = {"summary": cpu, "vram_peak_bytes": peak["v"]}

    # ---- GPU
    vms = [relax.VirtualMachine(exe.jit(), dev) for _ in range(1)]
    vm = vms[0]

    def run_gpu():
        tin = tvm.runtime.tensor(x, dev)
        out = vm["main"](tin)
        return [out[i].numpy() if isinstance(out, tvm_ffi.Array) else out.numpy()
                for i in range(len(out))] if isinstance(out, tvm_ffi.Array) else [out.numpy()]

    peak = C.vram_peak_sample()
    t = C.Timer(args.warmup, args.repeats)
    outs = t(run_gpu)
    peak["stop"] = True
    gpu = t.report()
    g_map, g_conf, g_det = post(*t.last_result)
    print("  TVM Vulkan RX560   min=%7.1f ms  p10=%7.1f  med=%7.1f  p90=%7.1f  max=%7.1f  (sd=%.1f)"
          % (gpu["min"] * 1e3, gpu["p10"] * 1e3, gpu["median"] * 1e3,
             gpu["p90"] * 1e3, gpu["max"] * 1e3, gpu["stdev"] * 1e3))
    print("  vram peak during GPU: %.0f MB of %.0f MB"
          % (peak["v"] / 2**20, (env.get("vram_total_bytes") or 0) / 2**20))

    corr = float(np.corrcoef(r_map.ravel().astype(np.float64), g_map.ravel().astype(np.float64))[0, 1])
    maxerr = float(np.abs(r_map.astype(np.float64) - g_map.astype(np.float64)).max())
    results["gpu"] = {"summary": gpu, "vram_peak_bytes": peak["v"]}
    results["correctness"] = {"map_corr": corr, "map_max_abs": maxerr,
                              "score_ref": r_det, "score_gpu": g_det,
                              "score_abs_err": abs(r_det - g_det)}

    speed_med = cpu["median"] / gpu["median"]
    speed_min = cpu["min"] / gpu["min"]
    print("\n  map correlation  %.7f   max_abs %.3e   score err %.3e"
          % (corr, maxerr, abs(r_det - g_det)))
    print("  speedup (median) %.3fx      speedup (min) %.3fx" % (speed_med, speed_min))
    results["speedup"] = {"median": speed_med, "min": speed_min}

    gate1 = {
        "correct": bool(corr > 0.999 and maxerr < 1e-2),
        "beats_cpu": bool(speed_min > 1.0),
        "verdict": ("GPU FASTER" if speed_min > 1.0 else "NO MATERIAL GPU GAIN"),
    }
    print("\n  GATE 1: correctness=%s  gpu_faster=%s  -> %s"
          % (gate1["correct"], gate1["beats_cpu"], gate1["verdict"]))
    results["gate1"] = gate1

    C.write_json(os.path.join(args.outdir, "gate1_baseline.json"), results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
