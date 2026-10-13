"""N0a measurement body, run under a shell trap that restores governor and DPM."""
import os, sys, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "tvm-venv", "lib", "python3.11", "site-packages"))
import numpy as np

GOV = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
DPM = "/sys/class/drm/card0/device/power_dpm_force_performance_level"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vendor", "tvm-src", "python"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "opt"))

M = os.path.join(ROOT, "reports", "onnx-dynamo", "trufor_512x512_dynamo_inline.onnx")
IMGS = os.path.join(ROOT, "truFor-src", "test_docker", "images")

def set_gov(v):
    with open(GOV, "w") as f: f.write(v)

def set_dpm(v):
    with open(DPM, "w") as f: f.write(v)

def load_tiles(n=3):
    from PIL import Image
    out = []
    for f in ("tampered2.png", "tampered1.png", "pristine1.jpg"):
        img = Image.open(os.path.join(IMGS, f)).convert("RGB").resize((512, 512))
        out.append((np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.).astype(np.float32))
    return out

def build():
    import tvm, onnx
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir import dlight as dl
    from tvm.relax.backend.gpu_generic import pipeline as GP
    import tvm.ir.transform as T
    from tvm.relax.transform import (LegalizeOps, AnnotateTIROpPattern,
                                     FoldConstant, FuseOps, FuseTIR)
    import swap_conv
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    ir = from_onnx(onnx.load(M), shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
    seq = T.Sequential(
        GP.library_dispatch_passes(tgt) +
        [LegalizeOps(), AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR(),
         dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                                 dl.gpu.GeneralReduction(), dl.gpu.Fallback())] +
        GP.dataflow_lower_passes(tgt) + GP.finalize_passes(tgt))
    with tgt:
        mod = seq(ir)
    n, _, _ = swap_conv.swap(mod, use_depthwise=True, layout=True,
                             verify=(tvm.vulkan(), tgt), verbose=False)
    print("substitutions: %d" % n, flush=True)
    return relax.build(mod, target=tgt), tvm.vulkan()

def main():
    import tvm, onnxruntime as ort
    tiles = load_tiles()
    print("ORT: intra_op_num_threads=%d inter_op=%d graph_opt=%s"
          % (ort.SessionOptions().intra_op_num_threads,
             ort.SessionOptions().inter_op_num_threads,
             ort.GraphOptimizationLevel.ORT_ENABLE_ALL), flush=True)
    print("building TVM module (once, before any switching) ...", flush=True)
    exe, dev = build()
    from tvm import relax
    vm = relax.VirtualMachine(exe.jit(), dev)
    tins = [tvm.runtime.tensor(t, dev) for t in tiles]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(M, so, providers=["CPUExecutionProvider"])

    def run_cpu():
        t0 = time.perf_counter(); sess.run(None, {"rgb": tiles[0]}); return time.perf_counter() - t0
    def run_gpu():
        t0 = time.perf_counter(); vm["main"](tins[0]); dev.sync(); return time.perf_counter() - t0

    # warm both paths once
    run_cpu(); run_gpu()
    res = {k: [] for k in ("cpu_pow", "cpu_perf", "gpu_auto", "gpu_high")}
    for rep in range(15):
        # ABAB on every single run, with settle + one discarded run per switch
        for key, gov, dpm, fn in (
                ("cpu_pow", "powersave", "auto", run_cpu),
                ("cpu_perf", "performance", "auto", run_cpu),
                ("gpu_auto", "powersave", "auto", run_gpu),
                ("gpu_high", "powersave", "high", run_gpu)):
            set_gov(gov); set_dpm(dpm)
            time.sleep(1.5)
            fn()                      # discarded: settles frequency/DPM
            res[key].append(fn())
        if rep == 0:
            print("  first rep done, continuing", flush=True)
    def stat(v):
        v = sorted(v); n = len(v)
        return (statistics.median(v) * 1e3, (v[3 * n // 4] - v[n // 4]) * 1e3,
                v[0] * 1e3, v[-1] * 1e3)
    print("\n%-10s %9s %9s %9s %9s" % ("config", "median", "IQR", "min", "max"))
    out = {}
    for k in ("cpu_pow", "cpu_perf", "gpu_auto", "gpu_high"):
        m, iqr, lo, hi = stat(res[k]); out[k] = m
        print("%-10s %9.1f %9.1f %9.1f %9.1f" % (k, m, iqr, lo, hi))
    print("\nCPU performance vs powersave : %.1f%% faster" % (100 * (out["cpu_pow"] - out["cpu_perf"]) / out["cpu_pow"]))
    print("GPU high vs auto            : %.1f%% faster" % (100 * (out["gpu_auto"] - out["gpu_high"]) / out["gpu_auto"]))
    print("\nspeedup over CPU, default config (powersave) : %.3fx" % (out["cpu_pow"] / out["gpu_auto"]))
    print("speedup over CPU, best CPU config           : %.3fx" % (out["cpu_perf"] / out["gpu_auto"]))

main()
