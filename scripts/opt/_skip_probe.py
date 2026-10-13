"""One process, one TVM_VM_SKIP setting: build the current graph and time it.

The skip list is read once at library load, so each configuration needs its own
process; this script is the body and skip_probe.sh is the driver. Prints median
and IQR over the requested number of runs, and echoes any TVMSKIP lines the
runtime emitted so a refused skip cannot pass unnoticed.
"""
import os, sys, time, statistics
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vendor", "tvm-src", "python"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "opt"))

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    import tvm, onnx
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir import dlight as dl
    from tvm.relax.backend.gpu_generic import pipeline as GP
    import tvm.ir.transform as T
    from tvm.relax.transform import (LegalizeOps, AnnotateTIROpPattern,
                                     FoldConstant, FuseOps, FuseTIR)
    import swap_conv
    from PIL import Image
    M = os.path.join(ROOT, "reports", "onnx-dynamo", "trufor_512x512_dynamo_inline.onnx")
    img = Image.open(os.path.join(ROOT, "truFor-src", "test_docker", "images", "tampered2.png"))
    x = (np.asarray(img.convert("RGB").resize((512, 512)), np.float32)
         .transpose(2, 0, 1)[None] / 255.).astype(np.float32)
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
    dev = tvm.vulkan()
    k, _, _ = swap_conv.swap(mod, use_depthwise=True, layout=True, verify=(dev, tgt), verbose=False)
    e = relax.build(mod, target=tgt)
    vm = relax.VirtualMachine(e.jit(), dev)
    tin = tvm.runtime.tensor(x, dev)
    for _ in range(3):
        vm["main"](tin); dev.sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); vm["main"](tin); dev.sync()
        ts.append(time.perf_counter() - t0)
    v = sorted(ts)
    print("SKIPLIST=%r subs=%d runs=%d median=%.1f IQR=%.1f min=%.1f max=%.1f"
          % (os.environ.get("TVM_VM_SKIP", ""), k, n, statistics.median(v) * 1e3,
             (v[3 * n // 4] - v[n // 4]) * 1e3, v[0] * 1e3, v[-1] * 1e3))

main()
