"""Hand-written NCHW <-> NHWC layout change, optionally with a GELU epilogue.

Measured with the DLight schedules, the pure layout change runs at 0.96 GB/s
because the thread index follows the channel: consecutive threads read
consecutive channels, which coalesces, and then store to dst[c, h, w] where c is
the outermost axis, putting consecutive threads 128*128 floats apart. Every store
is its own cache line. Inverting the mapping, so the thread follows the spatial
axis and the stores are contiguous, is the whole fix.

This module serves two shapes in the graph:
  * pure layout change, 3-D token input (1, H*W, C) -> (1, C, H, W)
  * the same change fused with an exact GELU, 4-D NCHW input
    (1, C, H, W) -> (1, H*W, C), as in
    fused_reshape10_transpose8_divide1_erf_add11_multiply1_multiply2 whose TIR
    reads
        out[p, c] = x * 0.5 * (erf(x / 1.4142135381698608) + 1.0)
    i.e. 0.5 * x * (1 + erf(x / sqrt(2))).

VEC gives each thread VEC consecutive columns; VEC=8 measured 8.19 GB/s against
5.57 for VEC=1. Still short of the 20 GB/s target, because the load stays
strided by C. Closing that needs a shared-memory tile, which needs the s_tir
block form and a working barrier; in the flat form T.alloc_shared with
T.tvm_storage_sync compiles and runs but returns zeros, so that is left stated
rather than attempted.
"""

import os

_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def nhwc2nchw(
    src: T.Buffer(({SRC}), "float32"),
    dst: T.Buffer(({DST}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    cb = T.thread_binding(T.int64({C}), thread="blockIdx.x")
    pb = T.thread_binding(T.int64({NPB}), thread="blockIdx.y")
    t = T.thread_binding(T.int64({NT}), thread="threadIdx.x")
    p0 = pb * {NT} * {VEC} + t * {VEC}
    for k in T.serial({VEC}):
        q = p0 + k
{BODY}
'''

# One body per (src_mode, act). Kept as whole lines rather than format strings
# because these contain braces and quotes that fight with .format().
_BODY_NHWC_PLAIN = """        dst[0, cb, q // {W}, q % {W}] = src[0, q, cb]"""
_BODY_NCHW_PLAIN = """        dst[0, q, cb] = src[0, cb, q // {W}, q % {W}]"""
_BODY_NHWC_GELU = """        xv = src[0, q, cb]
        dst[0, q, cb] = T.Cast("float32", xv) * T.float32(0.5) * (
            T.erf(T.Cast("float32", xv) / T.float32(1.4142135381698608))
            + T.float32(1.0))"""
_BODY_NCHW_GELU = """        xv = src[0, cb, q // {W}, q % {W}]
        dst[0, q, cb] = T.Cast("float32", xv) * T.float32(0.5) * (
            T.erf(T.Cast("float32", xv) / T.float32(1.4142135381698608))
            + T.float32(1.0))"""


def build(C, H, W, NT=256, VEC=8, act=None, src_mode="nhwc", tiled=False, name="nhwc2nchw"):
    """tiled=True routes the plain NHWC->NCHW case through the shared-memory
    kernel, which is 3.25x the scalar one: 28.30 against 8.71 GB/s on
    (1,16384,256) -> (1,256,128,128), measured with max_abs 0.0. It is not used
    for the GELU variant, which runs the other way round and has an erf
    epilogue the tile kernel does not carry.

    act=None for a plain layout change, act="gelu" to fuse the activation.

    src_mode="nhwc" means src is (1, H*W, C) and dst is (1, C, H, W);
    src_mode="nchw" means src is (1, C, H, W) and dst is (1, H*W, C).
    """
    HW = H * W
    if tiled and act is None and src_mode == "nhwc":
        # TC and NT come from the sweep on (1,16384,256)->(1,256,128,128):
        #   TC=128 TS=8 NT=256  1.19 ms  28.30 GB/s
        #   TC=64  TS=8 NT=512  1.28 ms  26.28 GB/s
        #   TC=32  TS=16 NT=128 2.18 ms  15.41 GB/s
        # TC cannot exceed the channel count, and it must divide it, so for
        # narrow layers the block is widened in threads instead.
        tc = 128
        while tc > 16 and C % tc != 0:
            tc //= 2
        assert C % tc == 0, "no usable tile width for C=%d" % C
        nt = 256 if tc >= 128 else 512
        return build_tiled(C, H, W, TC=tc, TS=8, NT=nt, name=name)
    assert HW % NT == 0, "spatial extent %d not divisible by %d threads" % (HW, NT)
    # VEC cannot exceed what the extent divides into; the assert rejects a
    # mismatch rather than silently computing fewer elements, which is how a
    # 1771 GFLOPS phantom got reported earlier in this project.
    VEC = max(1, min(VEC, HW // NT))
    per = NT * VEC
    assert HW % per == 0, "spatial extent %d not divisible by %d threads x %d" % (HW, NT, VEC)

    # Both buffers keep a leading batch of 1. nchw mode reads (1,C,H,W) and
    # writes (1,H*W,C); nhwc mode reads (1,H*W,C) and writes (1,C,H,W).
    if src_mode == "nchw":
        src_shape, dst_shape = (1, C, H, W), (1, HW, C)
    else:
        src_shape, dst_shape = (1, HW, C), (1, C, H, W)
    body = {
        ("nhwc", None): _BODY_NHWC_PLAIN,
        ("nchw", None): _BODY_NCHW_PLAIN,
        ("nhwc", "gelu"): _BODY_NHWC_GELU,
        ("nchw", "gelu"): _BODY_NCHW_GELU,
    }[(src_mode, act)].format(W=W)
    fmt = lambda sh: ", ".join("T.int64(%d)" % d for d in sh)
    src = _TEMPLATE.format(SRC=fmt(src_shape), DST=fmt(dst_shape),
                           C=C, NT=NT, NPB=HW // per, VEC=VEC, BODY=body).strip()
    path = "/tmp/_n2c_%d_%d_%d_%s_%s_%d.py" % (C, H, W, src_mode, act, VEC)
    os.makedirs("/tmp", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    # The template always defines nhwc2nchw_tiled; the caller's `name` is the
    # global symbol it should carry, not the Python-level function name.
    return ns[name]



def bytes_moved(C, H, W, src_mode="nhwc"):
    HW = H * W
    return 4 * C * HW * 2


# ---------------------------------------------------------------------------
# Shared-memory tiled version.
#
# The scalar version above is capped at 8.71 GB/s because the thread index runs
# along W to make the store contiguous, which leaves the load strided by C. A
# tile in shared memory coalesces both: read with the channel varying fastest,
# then write with the column varying fastest, and the two orderings differ
# inside shared memory rather than in global memory. The barrier between the
# phases is inserted by ThreadSync; an earlier note in this project claimed
# shared memory "returns zeros in the flat tirx form", which was a bug in the
# probe rather than a property of the form. A related rule that IS real, and is
# why no explicit tvm_storage_sync appears below: in TIRX a comparison bound to
# a variable is normalised to int8 while if_then_else wants bool, so every
# condition here is written inline.
#
# The shared rows are padded to 33 so that a column walk hits 32 banks plus one
# and does not serialise.

_TEMPLATE_TILED = '''
from tvm.script import tirx as T

@T.prim_func
def nhwc2nchw_tiled(
    src: T.Buffer((1, {HW}, {C}), "float32"),
    dst: T.Buffer((1, {C}, {H}, {W}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    cb = T.thread_binding(T.int64({CBLK}), thread="blockIdx.x")
    sb = T.thread_binding(T.int64({SBLK}), thread="blockIdx.y")
    tx = T.thread_binding(T.int64({NT}), thread="threadIdx.x")
    # Row padded by one so a column walk spreads over the banks.
    sh = T.alloc_shared(({TC}, {TILE} + 1), "float32")
    c0 = cb * {TC}
    p0 = sb * {TILE}
    # Load phase: the flat index makes the channel the fastest-varying term, so
    # consecutive threads read consecutive channels, which is coalesced.
    for it in T.serial({NIT}):
        idx = it * {NT} + tx
        cc = idx % {TC}
        pp = idx // {TC}
        sh[cc, pp] = src[0, p0 + pp, c0 + cc]
    # Store phase: the flat index makes the column the fastest-varying term, so
    # consecutive threads write consecutive columns of one output row, which is
    # also coalesced. The reordering between the two phases happened in shared
    # memory, not in global memory. ThreadSync inserts the barrier.
    for it2 in T.serial({NIT}):
        idx = it2 * {NT} + tx
        xx = idx % {TS}
        yy = (idx // {TS}) % {TS}
        cc = idx // {TSPP}
        pi = p0 + yy * {TS} + xx
        dst[0, c0 + cc, pi // {W}, pi % {W}] = sh[cc, yy * {TS} + xx]
'''


def build_tiled(C, H, W, TC=32, TS=32, NT=256, name="nhwc2nchw_tiled"):
    """Tiled NHWC->NCHW. TC channels x TS*TS spatial per block."""
    HW = H * W
    assert C % TC == 0, "C %d not divisible by TC %d" % (C, TC)
    assert HW % (TS * TS) == 0, "HW %d not divisible by %d" % (HW, TS * TS)
    TILE = TS * TS
    nit = (TC * TILE) // NT
    assert TC * TILE == nit * NT, "tile %dx%d does not divide over %d threads" % (TC, TILE, NT)
    src = _TEMPLATE_TILED.format(
        HW=HW, C=C, H=H, W=W, TC=TC, TS=TS, TILE=TILE, TSPP=TILE,
        CBLK=C // TC, SBLK=HW // TILE, NT=NT, NIT=nit).strip()
    path = "/tmp/_n2ct_%d_%d_%d_%d_%d.py" % (C, H, W, TC, TS)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    # The template always defines nhwc2nchw_tiled; the caller's `name` is the
    # global symbol it should carry, not the Python-level function name.
    fn = ns["nhwc2nchw_tiled"]
    return fn.with_attr("global_symbol", name) if hasattr(fn, "with_attr") else fn
