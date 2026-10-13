#!/usr/bin/env python3
"""Minimal reproducer: TVM Vulkan degenerates a reduction whose output has a
single element into a single-threaded serial loop.

No model, no ONNX export of a network, no TruFor. One ReduceMax, one shape.

    input  (1, 1, 262144)  float32   -- 1.0 MB
    output (1, 1, 1)       float32

Observed on RX 560 (Baffin) / Mesa 25.0.7 / TVM 0.27.0, build with USE_VULKAN=ON:

    default pipeline (Matmul, GEMV, Reduction, GeneralReduction, Fallback)
        83.95 ms     blockIdx.x = 1     <-- one block
    Reduction + Fallback only
        83.75 ms     blockIdx.x = 1
    GeneralReduction + Fallback only
        84.67 ms     blockIdx.x = 1

    1.0 MB / 0.084 s = 0.012 GB/s, against 112 GB/s of peak bandwidth.
    The generated TIR is:

        for ax0_fused_0 in T.thread_binding(T.int64(1),    thread="blockIdx.x"):
          for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            T.where(ax0_fused_0 * 256 + ax0_fused_1 < 1)      # only thread 0 survives
            lv[0,0,0] = -inf
            for ax1 in range(T.int64(262144)):
              lv[0,0,0] = T.max(lv[0,0,0], x[0,0,ax1])

    i.e. 255 of 256 threads are masked off and one thread walks the axis serially.

Contrast: the same operator with a per-channel output is fine, because the
reduction's parallelism is derived from the output size.

    (1, 128, 128, 128) -> (1, 128, 1, 1)
         0.24 ms       blockIdx.x = 64

This matters for TruFor because the detection head's global pooling reduces the
full-resolution (1,1,512,512) localisation map to a scalar, and that single
PrimFunc accounts for 13.4% of the whole-graph GPU time (329 ms of 2464 ms).

The reduction schedule in gpu/reduction.py does contain a cross-thread path

    _, tx = sch.split(r, factors=[None, len_tx])
    rf = sch.rfactor(tx, 0)

but it does not fire here: the emitted TIR contains no rfactor block for any of
the three rule sets tried. Deriving grid size from the output extents is the
structural problem -- a fix has to pick a split of the *reduce* axis
independently of the output extent, and then combine the partial results
(two-stage, atomic, or a persistent second kernel).
"""

import argparse
import re
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    import onnx
    import tvm
    from onnx import TensorProto, helper
    from tvm.relax.frontend.onnx import from_onnx
    import tvm.ir.transform as T
    from tvm.s_tir import dlight as dl
    from tvm.relax.backend.gpu_generic import pipeline as GP
    from tvm.relax.transform import (
        AnnotateTIROpPattern,
        FoldConstant,
        FuseOps,
        FuseTIR,
        LegalizeOps,
    )

    dev = tvm.vulkan()
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("device: %s  driver: %s" % (dev.device_name, dev.driver_version), flush=True)

    import os

    os.makedirs(args.outdir, exist_ok=True)

    def make(shape, out_shape, tag):
        node = helper.make_node("ReduceMax", ["x"], ["y"], axes=[2], keepdims=1)
        g = helper.make_graph(
            [node],
            "reduce",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        p = os.path.join(args.outdir, "reduce_%s.onnx" % tag)
        onnx.save(m, p)
        return p

    def build(path, x, rules):
        ir = from_onnx(onnx.load(path), shape_dict={"x": list(x.shape)}, dtype_dict="float32")
        seq = T.Sequential(
            GP.library_dispatch_passes(tgt)
            + [
                LegalizeOps(),
                AnnotateTIROpPattern(),
                FoldConstant(),
                FuseOps(),
                FuseTIR(),
                dl.ApplyDefaultSchedule(*rules),
            ]
            + GP.dataflow_lower_passes(tgt)
            + GP.finalize_passes(tgt)
        )
        with tgt:
            mod = seq(ir)
        return tvm.relax.build(mod, target=tgt), mod.script()

    def measure(exe, x):
        vm = tvm.relax.VirtualMachine(exe.jit(), dev)
        tin = tvm.runtime.tensor(x, dev)

        def run():
            vm["main"](tin)
            dev.sync()

        for _ in range(3):
            run()
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run()
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    cases = [
        ("trufor_global", (1, 1, 262144), (1, 1, 1)),
        ("per_channel", (1, 128, 128, 128), (1, 128, 1, 1)),
    ]
    rule_sets = [
        ("default", [dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                     dl.gpu.GeneralReduction(), dl.gpu.Fallback()]),
        ("reduction_only", [dl.gpu.Reduction(), dl.gpu.Fallback()]),
        ("general_only", [dl.gpu.GeneralReduction(), dl.gpu.Fallback()]),
    ]

    for tag, shape, out_shape in cases:
        path = make(shape, out_shape, tag)
        x = np.random.rand(*[int(v) for v in shape]).astype(np.float32)
        nbytes = x.size * 4
        print("\n%s  %s -> %s   (%.2f MB)" % (tag, shape, out_shape, nbytes / 1e6))
        for name, rules in rule_sets:
            exe, txt = build(path, x, rules)
            ms = measure(exe, x)
            bx = re.findall(r'thread="blockIdx\.x"\)', txt)
            dims = re.findall(
                r'T\.thread_binding\(T\.int64\((\d+)\), thread="blockIdx\.x"\)', txt
            )
            print("   %-16s %8.2f ms   %6.1f ms/MB   blockIdx.x=%s  rfactor=%s"
                  % (name, ms, nbytes / 1e6 / (ms / 1e3), dims[:1], "rfactor" in txt))


if __name__ == "__main__":
    sys.exit(main())
