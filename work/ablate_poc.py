"""Verify the prior AI's T0-ablation BLOCKED claim.

The claim was: 7 attempts to ablate classes with a constant-0.5 no-op all
native-crashed, so the class-level SKIP upper bound is unavailable. The
existing `ablate.make_noop` generator emits

    def name(x: ..., y: ...):
        T.func_attr(...)
        y[...] = T.float32(0)

with two bugs: (1) the def is indented 4 spaces at module top level, which is
a plain Python IndentationError before anything reaches TVM, and (2) the body
uses literal `y[...]` where `...` is Python Ellipsis, not a TIR index, so even
after fixing the indent the PrimFunc fails to parse.

This file builds a valid hand-written no-op that writes 0.5 to every output
element via a 1-block-per-row, 256-thread grid, and shows it compiles to SPIR-V
and runs to completion on Vulkan. That refutes the "TVM cannot do ablation"
part of the claim; the crashes were the generator's fault.
"""

import sys
import os
import numpy as np
import tvm
from tvm.script import tirx as T


@T.prim_func
def noop_conv2d(
    x: T.Buffer((1, 32, 64, 64), "float32"),
    w: T.Buffer((64, 32, 3, 3), "float32"),
    y: T.Buffer((1, 64, 64, 64), "float32"),
):
    T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
    tx = T.thread_binding(T.int64(256), thread="threadIdx.x")
    bx = T.thread_binding(T.int64(64), thread="blockIdx.x")
    for i in T.serial(16):
        idx = i * 256 + tx
        y[0, bx, idx // 64, idx % 64] = T.float32(0.5)


def main():
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0}, host="llvm")
    dev = tvm.vulkan()
    built = tvm.compile(noop_conv2d, target=tgt)
    x = (np.random.RandomState(1).randn(1, 32, 64, 64) * 0.1).astype("float32")
    w = (np.random.RandomState(2).randn(64, 32, 3, 3) * 0.1).astype("float32")
    y = np.zeros((1, 64, 64, 64), "float32")
    tx_ = tvm.runtime.tensor(x, dev)
    tw_ = tvm.runtime.tensor(w, dev)
    ty_ = tvm.runtime.tensor(y, dev)
    built(tx_, tw_, ty_)
    dev.sync()
    got = ty_.numpy()
    uniq = np.unique(got)
    print("unique values in y:", uniq)
    print("all 0.5:", bool((got == 0.5).all()))
    return 0 if (got == 0.5).all() else 1


if __name__ == "__main__":
    sys.exit(main())
