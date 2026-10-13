"""Hand-written depthwise 3x3 convolution for the Vulkan target.

T0 measured the DLight schedules for these at 3.12-3.20 GB/s, against roughly
100 GB/s of memory bandwidth and a 40 GB/s target. The three largest account for
293 ms of a 1768 ms graph:

    fused_conv2d5_add10    33.6 MB  21.01 ms  3.20 GB/s   6 calls  126.1 ms
    fused_conv2d12_add23   16.8 MB  10.76 ms  3.12 GB/s   8 calls   86.1 ms
    fused_conv2d19_add35   10.5 MB   6.76 ms  3.12 GB/s  12 calls   81.1 ms

An earlier note in this project dismissed these as "under 1 GFLOP each, not worth
a kernel". That was the wrong criterion: depthwise convolution is memory bound
and its FLOP count is irrelevant. The 502.8 MB of depthwise traffic here is
almost exactly the 503 MB figure given independently.

Layout: the thread index runs along W so that both the load and the store are
contiguous, which is the whole point. Each thread handles VEC consecutive
columns and slides the three taps over them, so the three input rows are read
once each per output row rather than nine times. The nine weights live in
registers, loaded from a per-block staged copy rather than re-read per pixel.
Padding uses an integer mask multiplied into a clamped load, not a bool, because
a predicated version of this kernel was measured at 3.85x slower on the dense
convolution and a select-based one would not build.
"""

import os

_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def depthwise3x3(
    data: T.Buffer((1, {C}, {H}, {W}), "float32"),
    weight: T.Buffer(({C}, 1, 3, 3), "float32"),
    bias: T.Buffer((1, {C}, 1, 1), "float32"),
    out: T.Buffer((1, {C}, {OH}, {OW}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    c = T.thread_binding(T.int64({CB}), thread="blockIdx.x")   # channel
    by = T.thread_binding(T.int64({HB}), thread="blockIdx.y")   # row block
    bx = T.thread_binding(T.int64({BX}), thread="blockIdx.z")   # column block
    ty = T.thread_binding(T.int64({TH}), thread="threadIdx.y")  # row in block
    tx = T.thread_binding(T.int64({TX}), thread="threadIdx.x")  # column in block
    for v in T.serial({VEC}):
        y = by * {TH} + ty
        x = (bx * {TX} + tx) * {VEC} + v
        acc = T.float32(0)
        for kh in T.serial(3):
            iy = y + kh - 1
            for kw in T.serial(3):
                ix = x + kw - 1
                cyi = T.min(T.max(iy, 0), {H} - 1)
                cxi = T.min(T.max(ix, 0), {W} - 1)
                m = T.Cast("float32", T.min(iy + 1, 1) * T.min({H} - iy, 1)
                           * T.min(ix + 1, 1) * T.min({W} - ix, 1))
                acc = acc + weight[c, 0, kh, kw] * T.Cast("float32", data[0, c, cyi, cxi]) * m
        r = acc + bias[0, c, 0, 0]
        # y is always in range by construction (integer divide by OW, and
        # OH % TH == 0), so only x needs a guard, and it is a plain
        # comparison: a &-chained bool expression is what fails to build here.
        if (bx * {TX} + tx) * {VEC} + v < {OW}:
            # T.max, not T.if_then_else(r > 0, r, 0): the select form fails to
            # build in this tirx layer (SPIR-V error at a fixed instruction
            # index) and max is the same value for every input except NaN and
            # signed zero, neither of which the padding mask can produce.
{EPILOGUE}
'''


def build(C, H, W, OH, OW, TH=8, VEC=4, relu=True, name="depthwise3x3"):
    """TH output rows per block; VEC consecutive columns per thread."""
    import importlib.util

    # Rows and columns get separate axes. An earlier version folded both into
    # threadIdx.x as y = ty*TH + (tx*VEC+v)//OW, which silently computed only
    # ty*TH rows whenever TH < OW: the integer divide was always zero. It
    # reported 446 GB/s, above the physical bandwidth, and a full-tensor
    # comparison against numpy gave rel_L2 0.996. Always verify coverage.
    HB = OH // TH
    cols = (OW + VEC - 1) // VEC
    assert OH % TH == 0, "OH %d not divisible by TH %d" % (OH, TH)
    assert TH <= 64, "threadIdx.y %d exceeds the hardware limit" % TH
    # relu=False for the fused convolutions that end in a residual add with no
    # activation. Choosing wrongly is not a rounding error: verifying against the
    # real PrimFunc gives rel_L2 2.2e-08 when right and 0.71 when wrong, because
    # clamping the negatives to zero shifts every output by up to 0.17.
    if relu:
        EPILOGUE = "            out[0, c, y, x] = T.max(r, T.float32(0))"
    else:
        EPILOGUE = "            out[0, c, y, x] = r"
    src = _TEMPLATE.format(C=C, H=H, W=W, OH=OH, OW=OW, TH=TH, VEC=VEC, EPILOGUE=EPILOGUE,
                           CB=C, HB=HB, TX=min(cols, 256),
                           BX=(cols + min(cols, 256) - 1) // min(cols, 256)).strip()
    path = "/tmp/_dw_%d_%d_%d_%d_%d_%d.py" % (C, H, W, OH, TH, VEC)
    os.makedirs("/tmp", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns[name]


def bytes_moved(C, H, W, OH, OW):
    return 4 * C * (H * W + OH * OW)
