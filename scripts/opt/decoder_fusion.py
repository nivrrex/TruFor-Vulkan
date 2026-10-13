"""Hand-written decoder multi-scale fusion for the Vulkan target.

Replaces
    fused_transpose64_reshape62_transpose65_reshape63_transpose24_reshape25_
    transpose66_resize2d_resize2d1_resize2d2_reshape64_concatenate12
which the B4b ablation measured at 254.5 ms over 2 calls, moving 268 MB at
1.05 GB/s against a memory bandwidth of roughly 100 GB/s.

Semantics recovered from the TIR, not guessed. Four sources of 512 channels each
are resized to 128x128 and concatenated into one 2048-channel output:

    v0 in [1536, 2048)  lv2575 (16384 tokens = 128x128)  scale 1     c - 1536
    v0 in [1024, 1536)  lv2569 ( 4096 tokens =  64x64)   scale 0.5   c - 1024
    v0 in [ 512, 1024)  lv2563 ( 1024 tokens =  32x32)   scale 0.25  c -  512
    v0 in [   0,  512)  lv2557 (  256 tokens =  16x16)   scale 0.125 c

The 128x128 branch appears as a bare copy in the TIR, which is consistent with
scale 1. The others are bilinear with align_corners=false, i.e.

    s  = (dst + 0.5) * scale - 0.5
    i0 = clamp(floor(s), 0, n-1);  i1 = clamp(i0 + 1, 0, n-1);  w = s - i0

applied on each axis, and the four corner samples combined with the usual
weights. The sources are in token layout (n, C), so the read of a row is strided
by C; the write of a row is contiguous. Splitting into a transpose kernel plus a
resize kernel would fix the read, but a signature substitution cannot introduce
an intermediate kernel, so this stays single-kernel and is verified instead.
"""

import os

_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def decoder_fusion(
    lv2557: T.Buffer((1, 256, 512), "float32"),
    lv2563: T.Buffer((1, 1024, 512), "float32"),
    lv2569: T.Buffer((1, 4096, 512), "float32"),
    lv2575: T.Buffer((1, 16384, 512), "float32"),
    out: T.Buffer((1, 2048, 128, 128), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    c = T.thread_binding(T.int64(2048), thread="blockIdx.x")
    yb = T.thread_binding(T.int64(16), thread="blockIdx.y")
    # xb covers 128/16 = 8 blocks, not 16. With 16, x = xb*16 + tx runs to 255
    # while the output is only 128 wide, and every thread with x >= 128 writes
    # out of bounds, which is what made all four channel bands read as mixtures
    # of their neighbours.
    xb = T.thread_binding(T.int64(8), thread="blockIdx.z")
    ty = T.thread_binding(T.int64(8), thread="threadIdx.y")
    tx = T.thread_binding(T.int64(16), thread="threadIdx.x")
    y = yb * 8 + ty
    x = xb * 16 + tx

    if c >= 1536:
        out[0, c, y, x] = lv2575[0, y * 128 + x, c - 1536]
    elif c >= 1024:
        out[0, c, y, x] = T.Cast("float32", _bilerp(
            lv2569, c - 1024, y, x, 64, 64, 4096, 0.5))
    elif c >= 512:
        out[0, c, y, x] = T.Cast("float32", _bilerp(
            lv2563, c - 512, y, x, 32, 32, 1024, 0.25))
    else:
        out[0, c, y, x] = T.Cast("float32", _bilerp(
            lv2557, c, y, x, 16, 16, 256, 0.125))
'''


def _load_module():
    import importlib.util
    path = "/tmp/_decfuse_impl.py"
    src = '''from tvm.script import tirx as T


def _bilerp(buf, ch, y, x, SH, SW, NTOK, scale):
    """Bilinear sample of buf[0, iy*SW+ix, ch] at output (y, x).

    align_corners=false, matching the expression recovered from the original
    PrimFunc: s = (d + 0.5) * scale - 0.5, i0 = clamp(floor(s), 0, n-1),
    i1 = clamp(i0 + 1, 0, n-1), w = s - i0, corners combined with the usual
    product weights.
    """
    fy = T.Cast("float32", y) + T.float32(0.5)
    fx = T.Cast("float32", x) + T.float32(0.5)
    sy = fy * T.Cast("float32", scale) - T.float32(0.5)
    sx = fx * T.Cast("float32", scale) - T.float32(0.5)
    syi = T.Cast("int32", T.floor(sy))
    sxi = T.Cast("int32", T.floor(sx))
    y0 = T.max(T.min(syi, SH - 1), 0)
    x0 = T.max(T.min(sxi, SW - 1), 0)
    y1 = T.max(T.min(syi + 1, SH - 1), 0)
    x1 = T.max(T.min(sxi + 1, SW - 1), 0)
    wy = sy - T.Cast("float32", y0)
    wx = sx - T.Cast("float32", x0)
    v00 = T.Cast("float32", buf[0, y0 * SW + x0, ch])
    v01 = T.Cast("float32", buf[0, y0 * SW + x1, ch])
    v10 = T.Cast("float32", buf[0, y1 * SW + x0, ch])
    v11 = T.Cast("float32", buf[0, y1 * SW + x1, ch])
    a = v00 + (v01 - v00) * wx
    b = v10 + (v11 - v10) * wx
    return a + (b - a) * wy
'''
    os.makedirs("/tmp", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    spec = importlib.util.spec_from_file_location("_decfuse_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(name="decoder_fusion"):
    src = _TEMPLATE.format().strip()
    path = "/tmp/_decfuse_%s.py" % name
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {"_bilerp": _load_module()._bilerp}
    exec(compile(src, path, "exec"), ns)
    return ns[name]


def traffic_mb():
    src = 4 * (256 + 1024 + 4096 + 16384) * 512
    dst = 4 * 2048 * 128 * 128
    return (src + dst) / 1e6


# ---------------------------------------------------------------------------
# The five-kernel split.
#
# The single fused kernel above is limited to 183.76 ms against the original's
# 209.06 ms because the sources are in token layout, so an output row reads with
# a constant stride of 512 floats and only the store coalesces. A signature
# substitution cannot introduce the intermediate buffer that a transpose-then-
# resize split needs, so this part is written here and verified here, and only
# the graph-level wiring is left.
#
# Stage 1 is the existing scripts/opt/nhwc2nchw.py, four calls, one per source,
# producing NCHW: (1,512,16,16), (1,512,32,32), (1,512,64,64), (1,512,128,128).
# Stage 2 is the kernel below: four NCHW inputs, both reads and writes
# contiguous, resize and concatenate to (1, 2048, 128, 128).

_TEMPLATE_SPLIT = '''
from tvm.script import tirx as T

@T.prim_func
def resize_concat(
    s0: T.Buffer((1, 512, 16, 16), "float32"),
    s1: T.Buffer((1, 512, 32, 32), "float32"),
    s2: T.Buffer((1, 512, 64, 64), "float32"),
    s3: T.Buffer((1, 512, 128, 128), "float32"),
    out: T.Buffer((1, 2048, 128, 128), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    c = T.thread_binding(T.int64(2048), thread="blockIdx.x")
    yb = T.thread_binding(T.int64(16), thread="blockIdx.y")
    # xb covers 128/16 = 8 blocks, not 16. With 16, x = xb*16 + tx runs to 255
    # while the output is only 128 wide, and every thread with x >= 128 writes
    # out of bounds, which is what made all four channel bands read as mixtures
    # of their neighbours.
    xb = T.thread_binding(T.int64(8), thread="blockIdx.z")
    ty = T.thread_binding(T.int64(8), thread="threadIdx.y")
    tx = T.thread_binding(T.int64(16), thread="threadIdx.x")
    y = yb * 8 + ty
    x = xb * 16 + tx

    if c >= 1536:
        out[0, c, y, x] = s3[0, c - 1536, y, x]
    elif c >= 1024:
        out[0, c, y, x] = T.Cast("float32", _bilerp(
            s2, c - 1024, y, x, 64, 0.5))
    elif c >= 512:
        out[0, c, y, x] = T.Cast("float32", _bilerp(
            s1, c - 512, y, x, 32, 0.25))
    else:
        out[0, c, y, x] = T.Cast("float32", _bilerp(
            s0, c, y, x, 16, 0.125))
'''


def _bilerp_nchw():
    """Bilinear sample of an NCHW buffer.

    Same formula as _bilerp, align_corners=false, but indexing the NCHW layout
    so that a thread walking x reads contiguously. The two index expressions
    differ only in the last term.
    """
    src = '''def _bilerp(buf, ch, y, x, S, scale):
    fy = T.Cast("float32", y) + T.float32(0.5)
    fx = T.Cast("float32", x) + T.float32(0.5)
    sy = fy * T.Cast("float32", scale) - T.float32(0.5)
    sx = fx * T.Cast("float32", scale) - T.float32(0.5)
    syi = T.Cast("int32", T.floor(sy))
    sxi = T.Cast("int32", T.floor(sx))
    y0 = T.max(T.min(syi, S - 1), 0)
    x0 = T.max(T.min(sxi, S - 1), 0)
    y1 = T.max(T.min(syi + 1, S - 1), 0)
    x1 = T.max(T.min(sxi + 1, S - 1), 0)
    wy = sy - T.Cast("float32", y0)
    wx = sx - T.Cast("float32", x0)
    v00 = T.Cast("float32", buf[0, ch, y0, x0])
    v01 = T.Cast("float32", buf[0, ch, y0, x1])
    v10 = T.Cast("float32", buf[0, ch, y1, x0])
    v11 = T.Cast("float32", buf[0, ch, y1, x1])
    a = v00 + (v01 - v00) * wx
    b = v10 + (v11 - v10) * wx
    return a + (b - a) * wy
'''
    path = "/tmp/_decfuse_nchw.py"
    with open(path, "w") as fh:
        fh.write("from tvm.script import tirx as T\n\n\n" + src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("_decfuse_nchw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._bilerp


# Chosen by sweep against the scalar version, bit-exact on all four bands:
#     VEC=1 (scalar)          24.58 ms   7.28 GB/s
#     VEC=2  64x8 threads      6.23 ms  28.71 GB/s
#     VEC=4  32x8 threads      5.45 ms  32.82 GB/s   <- used
#     VEC=8  16x8 threads      6.94 ms  25.78 GB/s
#     VEC=8  16x4 threads     10.98 ms  16.28 GB/s
#     VEC=16  8x8 threads     13.02 ms  13.73 GB/s
# 4.51x, and 5.45 ms is within a factor of 1.2 of the 6.7 ms that the 134 MB
# of contiguous output alone would need at 20 GB/s, so the store is no longer the
# limit. Vectorising the store also lets neighbouring columns share L1 lines,
# since the bilinear sampling windows overlap along x.
RESIZE_VEC, RESIZE_NTX, RESIZE_NTY = 4, 32, 8


def build_split(name="resize_concat"):
    return build_split_vec(VEC=RESIZE_VEC, NTX=RESIZE_NTX, NTY=RESIZE_NTY)
    path = "/tmp/_decfuse_split_%s.py" % name
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {"_bilerp": _bilerp_nchw()}
    exec(compile(src, path, "exec"), ns)
    return ns[name]


# ---------------------------------------------------------------------------
# Two-kernel form, for the Relax-level rewrite.
#
# The lowering pass runs
#   DataflowReshapeRewriter::VisitExpr_(CallNode*) -> IsCallingTIRReshape
#     -> HasReshapePattern -> ICHECK(body is SBlockRelizeNode)
# on any call_tir whose PrimFunc reads exactly one buffer: before it descends
# into that check it calls GetUsedTensorArgIndices and returns early when the
# answer is not 1, and that helper counts the buffers a function body actually
# walks. Declaring op_pattern = 0 does not help, because IsCallingTIRReshape
# calls HasReshapePattern directly rather than consulting the attribute.
#
# So: register nothing that reads a single buffer. Stage 1 below reads all four
# token-layout sources and writes all four NCHW outputs in one multi-output
# call_tir, and stage 2 already reads four. Both are skipped by the check, and
# no fake argument or unread parameter is needed.
#
# Measured separately: 20.5 ms for stage 1 (four transposes, 178 MB) and
# 23.50 ms for stage 2, against 209.06 ms for the original fused kernel.

_TEMPLATE_TRANS4 = '''
from tvm.script import tirx as T

@T.prim_func
def trans4(
    a0: T.Buffer((1, 256, 512), "float32"),
    a1: T.Buffer((1, 1024, 512), "float32"),
    a2: T.Buffer((1, 4096, 512), "float32"),
    a3: T.Buffer((1, 16384, 512), "float32"),
    b0: T.Buffer((1, 512, 16, 16), "float32"),
    b1: T.Buffer((1, 512, 32, 32), "float32"),
    b2: T.Buffer((1, 512, 64, 64), "float32"),
    b3: T.Buffer((1, 512, 128, 128), "float32"),
):
    T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
    cb = T.thread_binding(T.int64(512), thread="blockIdx.x")
    pb = T.thread_binding(T.int64(64), thread="blockIdx.y")
    t = T.thread_binding(T.int64(256), thread="threadIdx.x")
    p = pb * 256 + t
    y0 = p // 16
    x0 = p % 16
    y1 = p // 32
    x1 = p % 32
    y2 = p // 64
    x2 = p % 64
    y3 = p // 128
    x3 = p % 128
    # Each output is only written where p is in range for it: 16*16 = 256,
    # 32*32 = 1024, 64*64 = 4096, 128*128 = 16384. Without the guards the
    # smaller three write past their end and the first three come back wrong.
    if p < 256:
        b0[0, cb, y0, x0] = a0[0, p, cb]
    if p < 1024:
        b1[0, cb, y1, x1] = a1[0, p, cb]
    if p < 4096:
        b2[0, cb, y2, x2] = a2[0, p, cb]
    b3[0, cb, y3, x3] = a3[0, p, cb]
'''


def build_trans4():
    """Stage 1: four token-layout (1, n, 512) sources to four NCHW outputs.

    One flat index p in [0, 16384) serves all four outputs, each decoding its
    own (y, x) from p with its own extent. Sharing a single (y, x) across the
    four does not work: they are 16x16, 32x32, 64x64 and 128x128, so any common
    (y, x) is out of range for three of them.
    """
    src = _TEMPLATE_TRANS4.strip()
    path = "/tmp/_decfuse_trans4.py"
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns["trans4"]


# ---------------------------------------------------------------------------
# Single-output stage 1.
#
# The Relax rewrite cannot use one four-output call_tir: the multi-output value
# is a Tuple, and TupleGetItem off it has no inferred type, so EmitNormalized
# rejects the binding. Four single-output calls avoid the tuple entirely.
#
# Each of them therefore reads exactly ONE source, which puts it back under the
# lowering pass's check: rewrite_dataflow_reshape.cc returns early unless
# GetUsedTensorArgIndices is exactly 1, and HasReshapePattern then asserts an
# SBlockRealizeNode body, which flat TIR is not. So each kernel takes a second
# buffer and reads it inside a data-dependent identity select, which the
# simplifier cannot fold away:
#
#     out[...] = T.if_then_else(guard[0, 0, 0, 0] == guard[0, 0, 0, 0], acc, acc)
#
# The comparison is true for every finite input, so the result is unchanged, and
# because it reads a buffer the simplifier has nothing to fold. The condition is
# written INLINE: binding it to a variable is what triggers the failure, since
# TIRX normalises a bound comparison to int8 while if_then_else wants bool.
# That was measured, not assumed: the same kernel with the condition inlined
# builds and runs, and with it bound to a variable it fails at
# codegen_spirv.h:173 with T.int8 against T.bool.

_TEMPLATE_TRANS1 = '''
from tvm.script import tirx as T

@T.prim_func
def trans1(
    src: T.Buffer((1, {NT}, 512), "float32"),
    guard: T.Buffer((1, {NT}, 512), "float32"),
    out: T.Buffer((1, 512, {S}, {S}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    cb = T.thread_binding(T.int64(512), thread="blockIdx.x")
    pb = T.thread_binding(T.int64({S}), thread="blockIdx.y")
    ty = T.thread_binding(T.int64({S}), thread="threadIdx.y")
    p = pb * {S} + ty
    acc = src[0, p, cb]
    out[0, cb, p // {S}, p % {S}] = T.if_then_else(
        guard[0, 0, 0] == guard[0, 0, 0], acc, acc)
'''


def build_trans1(NT, S, name="trans1"):
    """One source (1, NT, 512) transposed to NCHW (1, 512, S, S), S*S == NT."""
    assert NT == S * S, "spatial extent %d is not %dx%d" % (NT, S, S)
    src = _TEMPLATE_TRANS1.format(NT=NT, S=S).strip()
    path = "/tmp/_decfuse_t1_%d_%d.py" % (NT, S)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns[name]


# ---------------------------------------------------------------------------
# Vector-store variant of resize_concat, kept separate from the template above
# so the verified one stays intact.
#
# The output row is 128 wide and contiguous, so NTX * VEC is chosen to divide it
# exactly and no bounds test is needed inside the loop. An earlier attempt did
# carry an inner `if x < 128` inside each elif branch and did not compile; this
# form removes the construct entirely rather than debugging it.
# ---------------------------------------------------------------------------

_TEMPLATE_SPLIT_VEC = '''
from tvm.script import tirx as T

@T.prim_func
def resize_concat_vec(
    s0: T.Buffer((1, 512, 16, 16), "float32"),
    s1: T.Buffer((1, 512, 32, 32), "float32"),
    s2: T.Buffer((1, 512, 64, 64), "float32"),
    s3: T.Buffer((1, 512, 128, 128), "float32"),
    out: T.Buffer((1, 2048, 128, 128), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    c = T.thread_binding(T.int64(2048), thread="blockIdx.x")
    yb = T.thread_binding(T.int64({HB}), thread="blockIdx.y")
    xb = T.thread_binding(T.int64({XB}), thread="blockIdx.z")
    ty = T.thread_binding(T.int64({NTY}), thread="threadIdx.y")
    tx = T.thread_binding(T.int64({NTX}), thread="threadIdx.x")
    y = yb * {NTY} + ty
    x0 = (xb * {NTX} + tx) * {VEC}

    if c >= 1536:
        for v in T.serial({VEC}):
            out[0, c, y, x0 + v] = s3[0, c - 1536, y, x0 + v]
    elif c >= 1024:
        for v in T.serial({VEC}):
            out[0, c, y, x0 + v] = T.Cast("float32", _bilerp(s2, c - 1024, y, x0 + v, 64, 0.5))
    elif c >= 512:
        for v in T.serial({VEC}):
            out[0, c, y, x0 + v] = T.Cast("float32", _bilerp(s1, c - 512, y, x0 + v, 32, 0.25))
    else:
        for v in T.serial({VEC}):
            out[0, c, y, x0 + v] = T.Cast("float32", _bilerp(s0, c, y, x0 + v, 16, 0.125))
'''


def build_split_vec(VEC=8, NTX=16, NTY=8, name="resize_concat_vec"):
    """NTX * VEC must divide the 128-wide output row exactly."""
    per = NTX * VEC
    assert 128 % per == 0, "NTX*VEC=%d does not divide the 128-wide row" % per
    src = _TEMPLATE_SPLIT_VEC.format(
        VEC=VEC, NTX=NTX, NTY=NTY,
        HB=128 // NTY, XB=128 // per).strip()
    path = "/tmp/_decfuse_vec_%d_%d_%d.py" % (VEC, NTX, NTY)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {"_bilerp": _bilerp_nchw()}
    exec(compile(src, path, "exec"), ns)
    return ns[name]
