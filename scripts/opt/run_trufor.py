#!/usr/bin/env python3
"""Production entry point: TruFor on TVM Vulkan with the hand-written kernels.

The substitutions happen here, on the Python side, immediately before
`relax.build`. Nothing inside TVM is modified for them, so this script works
against an unmodified TVM 0.27.0; the only upstream change the project carries
is the per-kernel profiler in src/runtime/vm/vm.cc, which is diagnostic only and
is not required to run.

The substitution is by buffer signature, never by name, because the fusion pass
assigns the names. Every candidate is verified against the PrimFunc it replaces,
full tensor, before it is accepted:

  * depthwise 3x3  the ReLU epilogue is chosen by trying both and keeping the
    one that reproduces the original output, because half of the fused
    depthwise convolutions end in a bare residual add and guessing wrong gives
    rel_L2 0.71 instead of 2.2e-08
  * NCHW/NHWC layout change and the same fused with GELU, verified on both
    buffers

If a signature does not match, or verification fails, the function is simply
left alone and the graph still builds and runs. That is the fallback path: a
model whose fusion differs from the one these numbers were taken on degrades to
the stock DLight schedules rather than to a wrong answer.

Environment:
  TRUFOR_HANDWRITTEN  1 (default) enable the substitutions, 0 disable
  TRUFOR_DECODER      1 (default) enable the decoder multi-scale split rewrite
  TRUFOR_VERIFY       1 (default) verify every candidate, 0 skip (unsafe:
                      the depthwise ReLU cannot be inferred without it)
  TVM_VULKAN_TARGET   default {"kind":"vulkan","from_device":0}

Usage:
    python3 scripts/opt/run_trufor.py IMAGE [IMAGE ...]
    python3 scripts/opt/run_trufor.py --bench IMAGE --reps 15
"""

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "vendor", "tvm-src", "python"))
sys.path.insert(0, _HERE)

MODEL = os.path.join(_ROOT, "reports", "onnx-dynamo", "trufor_512x512_dynamo_inline.onnx")


def env_flag(name, default="1"):
    return os.environ.get(name, default).lower() not in ("0", "off", "false", "no")


def build_vm(verbose=True):
    import tvm
    import onnx
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir import dlight as dl
    from tvm.relax.backend.gpu_generic import pipeline as GP
    import tvm.ir.transform as T
    from tvm.relax.transform import (LegalizeOps, AnnotateTIROpPattern,
                                     FoldConstant, FuseOps, FuseTIR)
    import swap_conv

    target = tvm.target.Target(eval(os.environ.get(
        "TVM_VULKAN_TARGET", '{"kind":"vulkan","from_device":0}')))
    ir = from_onnx(onnx.load(MODEL), shape_dict={"rgb": [1, 3, 512, 512]},
                   dtype_dict="float32")
    # The pipeline is deliberately split around decoder_split.rewrite: the
    # rewrite edits main's DataflowBlock, which only exists before
    # dataflow_lower_passes turns it into a lambda. swap_conv's flat-TIR
    # kernels must run AFTER lowering, because the lowering pass asserts an
    # s_tir body for kernels that read exactly one buffer.
    seq_pre = T.Sequential(
        GP.library_dispatch_passes(target) +
        [LegalizeOps(), AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR(),
         dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                                 dl.gpu.GeneralReduction(), dl.gpu.Fallback())])
    with target:
        mod = seq_pre(ir)

    if env_flag("TRUFOR_DECODER"):
        import decoder_split
        try:
            mod, hits = decoder_split.rewrite(mod, verbose=verbose)
            if verbose:
                print("decoder_split rewrote %d call site(s)" % hits, file=sys.stderr)
        except Exception as e:
            print("decoder_split not applied: %s" % e, file=sys.stderr)

    seq_post = T.Sequential(GP.dataflow_lower_passes(target) + GP.finalize_passes(target))
    with target:
        mod = seq_post(mod)

    if env_flag("TRUFOR_HANDWRITTEN"):
        dev = tvm.vulkan()
        n, names, log = swap_conv.swap(
            mod, use_depthwise=True, layout=True,
            verify=(dev, target) if env_flag("TRUFOR_VERIFY") else None,
            verbose=verbose)
        if verbose:
            print("substituted %d PrimFuncs:" % n, file=sys.stderr)
            for ln in log:
                print("   %s" % ln, file=sys.stderr)
            if not n:
                print("   (none matched; running the stock DLight schedules)",
                      file=sys.stderr)
    else:
        if verbose:
            print("TRUFOR_HANDWRITTEN=0, stock schedules", file=sys.stderr)

    return relax.build(mod, target=target), tvm.vulkan(), target


def load(path):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((512, 512))
    import numpy as np
    return (np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.).astype(np.float32)


def main():
    args = [a for a in sys.argv[1:]]
    reps, bench = 15, False
    if "--bench" in args:
        bench = True
        args.remove("--bench")
    if "--reps" in args:
        i = args.index("--reps")
        reps = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 1
    import torch
    import torch.nn.functional as F
    import tvm
    import numpy as np

    exe, dev, _ = build_vm()
    vm = relax.VirtualMachine(exe.jit(), dev)
    for path in args:
        x = load(path)
        tin = tvm.runtime.tensor(x, dev)
        for _ in range(3):
            o = vm["main"](tin)
            dev.sync()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            o = vm["main"](tin)
            dev.sync()
            ts.append(time.perf_counter() - t0)
        arr = [o[i].numpy() for i in range(len(o))]
        m = F.softmax(torch.from_numpy(arr[0][0]), 0)[1].numpy()
        d = 1.0 / (1.0 + np.exp(-float(np.asarray(arr[2]).reshape(-1)[0])))
        v = sorted(ts)
        print("%-40s median %7.1f ms  IQR %.1f  det=%.6f  map[min,max]=[%.4f,%.4f]"
              % (os.path.basename(path), statistics.median(v) * 1e3,
                 (v[3 * len(v) // 4] - v[len(v) // 4]) * 1e3, d, m.min(), m.max()))
        if not bench:
            continue
    return 0


if __name__ == "__main__":
    from tvm import relax  # noqa: F401  (used inside build_vm's caller frame)
    sys.exit(main())
