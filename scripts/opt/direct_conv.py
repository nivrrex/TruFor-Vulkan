"""Hand-written direct convolution for 3x3, 64->64, stride 1, 512x512 on Vulkan.

Motivation (see work/LOG.md, 2026-09-30, "Stage B root cause"): the schedule the
production pipeline emits allocates

    pad_temp_reindex_shared = T.sblock_alloc_buffer((1, 262144, 576), scope="shared")

which is 576 MiB, labelled "shared" although a Baffin workgroup has kilobytes of
LDS.  In practice it is a global scratch buffer holding the whole im2col matrix,
written and read back once per convolution: 1152 MiB of traffic per convolution
and 41.1 GB across the 17 convolutions, which is what makes the block
memory-bound rather than compute-bound.

This module keeps the padded window in registers instead and materialises nothing.
Each block computes a 32x32 spatial tile for 8 output channels, streaming the 64
input channels and their 3x3 taps, so the input is read from DRAM about once.

Layout: 16x16 = 256 threads per block, each thread owning a 2x2 pixel quadrant and
16 output channels, i.e. 64 accumulators.  Weights are read per (ci, kh, kw) into
registers.  No shared memory is used, which keeps the block inside a Baffin
workgroup's register budget and avoids relying on the mislabelled "shared" scope.

Measured: 19.59 ms, 987 GFLOPS, 88% of the 1120 GFLOPS peak, against 50.47 ms
and 383 GFLOPS for the pipeline's own schedule on the same shape, i.e. 2.58x.

The input is UNPADDED, (1, 64, 512, 512), and bias plus ReLU are folded into
the epilogue, so the signature is exactly that of the ``fused_conv*_add_relu``
PrimFuncs this kernel replaces: (data, weight (64,64,3,3) untransformed,
bias (1,64,1,1), out). Zero padding is handled by an integer mask inside the
tap loop, see the comment there.

"""

OC = 64
IC = 64
H = W = 512
K = 3
# Chosen by exhaustive sweep over every configuration in which TILE divides the
# spatial extent, OCB divides the channel count, and TH divides TILE.
# Accumulators per thread, OCB*PH*PW, decide almost everything: acc=64 is the
# sweet spot, acc=128 drops to about 320 GFLOPS, acc=256 collapses to 17-20
# GFLOPS as the accumulators spill out of the register file. On the 3x3
# 64->64 at 512x512 shape the sweep gives:
#     TILE=16 OCB=16 8x8 threads  acc=64  16.02 ms  1206 GFLOPS  57% of peak
#     TILE=32 OCB=16 16x16       acc=64  16.52 ms  1170 GFLOPS  (previous default)
#     TILE=32 OCB=32 16x16       acc=128 60.86 ms   318 GFLOPS
#     TILE=32 OCB=16 8x8         acc=256 1127 ms     17 GFLOPS
# TILE=16 looked 3% better than TILE=32 in isolation (16.02 vs 16.52 ms) but a
# same-session whole-graph A/B could not tell them apart: TILE=32 measured
# 1490.9 and then 1469.5 ms on two runs of the identical configuration, and
# TILE=16 landed between them at 1471.2 ms. TILE=32 is kept because it is what
# the earlier records and patches refer to, not because it measured faster.
# The 1x1 kernel prefers a different point and is configured separately.
TILE = 32      # spatial tile per block, both directions
OCB = 16       # output channels per block
TH = TW = 16   # threads per block
PH = PW = 2    # pixels per thread


def build(OC_=None, IC_=None, H_=None, W_=None, TILE_=None, OCB_=None,
          TH_=None, TW_=None):
    """Return a tirx PrimFunc for the given shape. Compile with the Vulkan target.

    Defaults to the 3x3 64->64 at 512x512 that was tuned. TH_/TW_ and the tile
    size are independent of the shape, so the same register blocking
    (OCB*PH*PW accumulators per thread) transfers to other shapes; the sweep on
    the 64->64 case found 64 optimal and anything above 128 spilling.
    """
    from tvm.script import tirx as T

    OC = OC_ or OC
    IC = IC_ or IC
    H = H_ or H
    W = W_ or W
    TILE = TILE_ or TILE
    OCB = OCB_ or OCB
    TH = TH_ or TH
    TW = TW_ or TW
    # Both must divide exactly; integer division would silently drop channels or
    # rows and the kernel would only look fast because it computed less. This
    # bit me once and produced a 1771 GFLOPS phantom.
    assert OC % OCB == 0, "OCB %d does not divide %d" % (OCB, OC)
    assert H % TILE == 0 and W % TILE == 0, "TILE %d does not divide %dx%d" % (TILE, H, W)
    n_ty = H // TILE
    n_bz = OC // OCB

    @T.prim_func
    def direct_conv3x3(  # pylint: disable=too-many-arguments
        data: T.Buffer((1, IC, H, W), "float32"),
        weight: T.Buffer((OC, IC, K, K), "float32"),
        bias: T.Buffer((1, OC, 1, 1), "float32"),
        out: T.Buffer((1, OC, H, W), "float32"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        tx = T.thread_binding(T.int64(TW), thread="threadIdx.x")
        ty = T.thread_binding(T.int64(TH), thread="threadIdx.y")
        bxi = T.thread_binding(T.int64(n_ty), thread="blockIdx.x")
        byi = T.thread_binding(T.int64(n_ty), thread="blockIdx.y")
        bzi = T.thread_binding(T.int64(n_bz), thread="blockIdx.z")

        acc = T.alloc_buffer((OCB, PH, PW), dtype="float32", scope="local")
        for co in T.serial(OCB):
            for py in T.serial(PH):
                for px in T.serial(PW):
                    acc[co, py, px] = T.float32(0)

        for co in T.serial(OCB):
            for py in T.serial(PH):
                gy = byi * TILE + ty * PH + py
                for px in T.serial(PW):
                    gx = bxi * TILE + tx * PW + px
                    acc[co, py, px] = T.float32(0)

        for ci in T.serial(IC):
            for kh in T.serial(K):
                for kw in T.serial(K):
                    for py in T.serial(PH):
                        gy = byi * TILE + ty * PH + py
                        iy = gy + kh - 1
                        for px in T.serial(PW):
                            gx = bxi * TILE + tx * PW + px
                            ix = gx + kw - 1
                            # Integer-mask zero padding. Two earlier forms failed:
                            # a predicated tap loop ran 3.85x slower because the
                            # else arm made the accumulator a load/store pair, and
                            # a branchless clamp+select would not build because
                            # TVMScript yields an int8 where CodeGenSPIRV requires
                            # bool (codegen_spirv.h:173). This form has no bool and
                            # no control flow: clamp the index so the load is
                            # always in range, and multiply the loaded value by a
                            # 0/1 float mask derived purely from the index. The
                            # mask depends only on the pixel, not on the output
                            # channel, so it is hoisted out of the channel loop.
                            cy = T.min(T.max(iy, 0), H - 1)
                            cx = T.min(T.max(ix, 0), W - 1)
                            m = T.Cast("float32", T.min(iy + 1, 1) * T.min(H - iy, 1)
                                       * T.min(ix + 1, 1) * T.min(W - ix, 1))
                            xv = T.Cast("float32", data[0, ci, cy, cx]) * m
                            for co in T.serial(OCB):
                                acc[co, py, px] = (
                                    acc[co, py, px]
                                    + weight[bzi * OCB + co, ci, kh, kw] * xv
                                )

        for co in T.serial(OCB):
            for py in T.serial(PH):
                gy = byi * TILE + ty * PH + py
                for px in T.serial(PW):
                    gx = bxi * TILE + tx * PW + px
                    r = acc[co, py, px] + bias[0, bzi * OCB + co, 0, 0]
                    out[0, bzi * OCB + co, gy, gx] = T.if_then_else(r > T.float32(0), r, T.float32(0))

    return direct_conv3x3


def flops():
    return 2 * OC * IC * K * K * H * W / 1e9


_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def direct_conv3x3(
    data: T.Buffer((1, {IC}, {H}, {W}), "float32"),
    weight: T.Buffer(({OC}, {IC}, 3, 3), "float32"),
    bias: T.Buffer((1, {OC}, 1, 1), "float32"),
    out: T.Buffer((1, {OC}, {H}, {W}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    tx = T.thread_binding(T.int64({TW}), thread="threadIdx.x")
    ty = T.thread_binding(T.int64({TH}), thread="threadIdx.y")
    bxi = T.thread_binding(T.int64({NTY}), thread="blockIdx.x")
    byi = T.thread_binding(T.int64({NTY}), thread="blockIdx.y")
    bzi = T.thread_binding(T.int64({NBZ}), thread="blockIdx.z")

    acc = T.alloc_buffer(({OCB}, {PH}, {PW}), dtype="float32", scope="local")
    for co in T.serial({OCB}):
        for py in T.serial({PH}):
            for px in T.serial({PW}):
                acc[co, py, px] = T.float32(0)

    for ci in T.serial({IC}):
        for kh in T.serial(3):
            for kw in T.serial(3):
                for py in T.serial({PH}):
                    gy = byi * {TILE} + ty * {PH} + py
                    iy = gy + kh - 1
                    for px in T.serial({PW}):
                        gx = bxi * {TILE} + tx * {PW} + px
                        ix = gx + kw - 1
                        cy = T.min(T.max(iy, 0), {H} - 1)
                        cx = T.min(T.max(ix, 0), {W} - 1)
                        m = T.Cast("float32", T.min(iy + 1, 1) * T.min({H} - iy, 1)
                                   * T.min(ix + 1, 1) * T.min({W} - ix, 1))
                        xv = T.Cast("float32", data[0, ci, cy, cx]) * m
                        for co in T.serial({OCB}):
                            acc[co, py, px] = (
                                acc[co, py, px]
                                + weight[bzi * {OCB} + co, ci, kh, kw] * xv
                            )

    for co in T.serial({OCB}):
        for py in T.serial({PH}):
            gy = byi * {TILE} + ty * {PH} + py
            for px in T.serial({PW}):
                gx = bxi * {TILE} + tx * {PW} + px
                r = acc[co, py, px] + bias[0, bzi * {OCB} + co, 0, 0]
                out[0, bzi * {OCB} + co, gy, gx] = T.if_then_else(
                    r > T.float32(0), r, T.float32(0))
'''


def build_for(OC, IC, H, W, TILE=32, OCB=16, TH=16, TW=16, name="direct_conv3x3"):
    """Build a kernel for an arbitrary 3x3 convolution shape.

    The source is generated as text and imported from a temporary module,
    because TVMScript parses source text: passing the shape in as runtime
    variables to a hand-written @T.prim_func does not work, the parser has no
    way to see them.
    """
    import importlib.util

    assert OC % OCB == 0, "OCB %d does not divide %d" % (OCB, OC)
    assert H % TILE == 0 and W % TILE == 0, "TILE %d does not divide %dx%d" % (TILE, H, W)
    src = _TEMPLATE.format(OC=OC, IC=IC, H=H, W=W, TILE=TILE, OCB=OCB, TH=TH, TW=TW,
                           PH=TILE // TH, PW=TILE // TW,
                           NTY=H // TILE, NBZ=OC // OCB).strip()
    path = "/tmp/_dc_%d_%d_%d_%d_%d_%d.py" % (OC, IC, H, W, TILE, OCB)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    tag = "%d_%d_%d_%d_%d_%d_%d" % (OC, IC, H, W, TILE, OCB, TH * 100 + TW)
    spec = importlib.util.spec_from_file_location("dc_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def flops_for(OC, IC, H, W):
    return 2 * OC * IC * 3 * 3 * H * W / 1e9


_1x1_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def direct_conv1x1(
    data: T.Buffer((1, {IC}, {H}, {W}), "float32"),
    weight: T.Buffer(({OC}, {IC}, 1, 1), "float32"),
    bias: T.Buffer((1, {OC}, 1, 1), "float32"),
    out: T.Buffer((1, {OC}, {H}, {W}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    tx = T.thread_binding(T.int64({TW}), thread="threadIdx.x")
    ty = T.thread_binding(T.int64({TH}), thread="threadIdx.y")
    bxi = T.thread_binding(T.int64({NTY}), thread="blockIdx.x")
    byi = T.thread_binding(T.int64({NTY}), thread="blockIdx.y")
    bzi = T.thread_binding(T.int64({NBZ}), thread="blockIdx.z")

    acc = T.alloc_buffer(({OCB}, {PH}, {PW}), dtype="float32", scope="local")
    for co in T.serial({OCB}):
        for py in T.serial({PH}):
            for px in T.serial({PW}):
                acc[co, py, px] = T.float32(0)

    for ci in T.serial({IC}):
        for py in T.serial({PH}):
            gy = byi * {TILE} + ty * {PH} + py
            for px in T.serial({PW}):
                gx = bxi * {TILE} + tx * {PW} + px
                xv = T.Cast("float32", data[0, ci, gy, gx])
                for co in T.serial({OCB}):
                    acc[co, py, px] = (
                        acc[co, py, px]
                        + weight[bzi * {OCB} + co, ci, 0, 0] * xv
                    )

    for co in T.serial({OCB}):
        for py in T.serial({PH}):
            gy = byi * {TILE} + ty * {PH} + py
            for px in T.serial({PW}):
                gx = bxi * {TILE} + tx * {PW} + px
                r = acc[co, py, px] + bias[0, bzi * {OCB} + co, 0, 0]
                out[0, bzi * {OCB} + co, gy, gx] = T.if_then_else(
                    r > T.float32(0), r, T.float32(0))
'''


def build_1x1(OC, IC, H, W, TILE=32, OCB=16, TH=16, TW=16, name="direct_conv1x1"):
    """1x1 convolution, i.e. a per-pixel GEMM. No padding and no tap loop."""
    import importlib.util

    assert OC % OCB == 0, "OCB %d does not divide %d" % (OCB, OC)
    assert H % TILE == 0 and W % TILE == 0, "TILE %d does not divide %dx%d" % (TILE, H, W)
    src = _1x1_TEMPLATE.format(OC=OC, IC=IC, H=H, W=W, TILE=TILE, OCB=OCB, TH=TH, TW=TW,
                               PH=TILE // TH, PW=TILE // TW,
                               NTY=H // TILE, NBZ=OC // OCB).strip()
    path = "/tmp/_dc1x1_%d_%d_%d_%d_%d_%d.py" % (OC, IC, H, W, TILE, OCB)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    tag = "%d_%d_%d_%d_%d_%d_%d" % (OC, IC, H, W, TILE, OCB, TH * 100 + TW)
    spec = importlib.util.spec_from_file_location("dc1_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)
