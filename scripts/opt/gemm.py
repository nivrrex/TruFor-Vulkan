"""Hand-written GEMM + bias (+ optional ReLU) for the Vulkan target.

T0's re-derived gain table left one contiguous unoptimised block: six fused
matmuls, 86.7 ms of SKIP ceiling, 9.4 percent of a 926.8 ms graph. Measured in
isolation, DLight reaches 177 to 305 GFLOPS on them, which is 9 to 16 percent of
the 1903 GFLOPS peak, while the hand-written 3x3 convolution in this project
reaches 1175 GFLOPS, 62 percent. DLight already uses an 8x8 micro tile, so
accumulator count is not the lever here; the deficiency is memory access and
occupancy, since M/64 by N/64 yields only 80 blocks for 56 CUs.

This version trades accumulator count for block count, which is the right
trade at these sizes: a 32x32 block tile with 8x8 threads gives 4x4 = 16
accumulators per thread and 320 blocks instead of 80. No shared memory to start
with; the 3x3 convolution reached 62 percent of peak relying on L1 and L2 alone,
so the first question is whether these shapes are register-bound or
bandwidth-bound, and that is answered without a tile.

Conditions are written inline throughout. In TIRX a comparison bound to a
variable is normalised to int8 while if_then_else wants bool, which is a build
failure at a fixed instruction index; this was measured, not assumed.
"""

import os

_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def gemm_bias(
    a: T.Buffer((1, {M}, {K}), "float32"),
    b: T.Buffer(({K}, {N}), "float32"),
    bias: T.Buffer(({N},), "float32"),
    out: T.Buffer((1, {M}, {N}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    cb = T.thread_binding(T.int64({GBN}), thread="blockIdx.x")
    mb = T.thread_binding(T.int64({GBM}), thread="blockIdx.y")
    tx = T.thread_binding(T.int64({NTX}), thread="threadIdx.x")
    ty = T.thread_binding(T.int64({NTY}), thread="threadIdx.y")
    acc = T.alloc_buffer(({TM}, {TN}), dtype="float32", scope="local")
    for i in T.serial({TM}):
        for j in T.serial({TN}):
            acc[i, j] = T.float32(0)
    for kk in T.serial({NK}):
        for i in T.serial({TM}):
            for j in T.serial({TN}):
                acc[i, j] = acc[i, j] + a[0, mb * {BM} + ty * {TM} + i, kk] * b[kk, cb * {BN} + tx * {TN} + j]
    for i in T.serial({TM}):
        for j in T.serial({TN}):
            r = acc[i, j] + bias[cb * {BN} + tx * {TN} + j]
            out[0, mb * {BM} + ty * {TM} + i, cb * {BN} + tx * {TN} + j] = T.max(r, T.float32(0))
'''


def build(M, K, N, BM=32, BN=32, NTX=8, NTY=8, relu=True, name="gemm_bias"):
    """C = relu(A @ B + bias), A is (1,M,K) and B is (K,N).

    BM/BN are the block tile, NTX/NTY the thread grid inside it, so each thread
    owns (BM//NTY) x (BN//NTX) accumulators.
    """
    assert M % BM == 0, "M %d not divisible by BM %d" % (M, BM)
    assert N % BN == 0, "N %d not divisible by BN %d" % (N, BN)
    TM, TN = BM // NTY, BN // NTX
    ep = "T.max(r, T.float32(0))" if relu else "r"
    src = (_TEMPLATE.replace("T.max(r, T.float32(0))", "@@EPILOGUE@@")
           .format(M=M, K=K, N=N, BM=BM, BN=BN, GBN=N // BN, GBM=M // BM,
                   NTX=NTX, NTY=NTY, TM=TM, TN=TN, NK=K)).replace("@@EPILOGUE@@", ep)
    path = "/tmp/_gemm_%d_%d_%d_%d_%d_%d_%d.py" % (M, K, N, BM, BN, NTX, NTY)
    os.makedirs("/tmp", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns[name]


def gflop(M, K, N):
    return 2.0 * M * K * N / 1e9


# ---------------------------------------------------------------------------
# Shared-memory tiled GEMM.
#
# The register-only version above reaches 217 GFLOPS against DLight's 265, and
# the reason is that DLight's Matmul rule already sets use_shared=True and
# cache_reads A and B. So the second attempt stages both operands in shared
# memory, using the same flat-TIRX form the tiled transpose in this project
# proved works: T.alloc_shared, an explicit T.tvm_storage_sync between phases,
# and every condition written inline.
#
# BM x BN is the block tile, BK the K step, and each thread owns TM x TN
# accumulators with (BM/TM) * (BN/TN) threads. As is stored transposed and
# padded so the inner read is a straight stride-1 walk over k with a padded row,
# which keeps shared access free of bank conflicts without a swizzle.
# ---------------------------------------------------------------------------

_TEMPLATE_SMEM = '''
from tvm.script import tirx as T

@T.prim_func
def gemm_smem(
    a: T.Buffer((1, {M}, {K}), "float32"),
    b: T.Buffer(({K}, {N}), "float32"),
    bias: T.Buffer(({N},), "float32"),
    {RES_DECL}out: T.Buffer((1, {M}, {N}), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    mb = T.thread_binding(T.int64({GB_M}), thread="blockIdx.x")
    nb = T.thread_binding(T.int64({GB_N}), thread="blockIdx.y")
    ty = T.thread_binding(T.int64({NTY}), thread="threadIdx.y")
    tx = T.thread_binding(T.int64({NTX}), thread="threadIdx.x")
    As = T.alloc_shared(({BK}, {BM} + {PAD}), "float32")
    Bs = T.alloc_shared(({BK}, {BN} + {PAD}), "float32")
    acc = T.alloc_buffer(({TM}, {TN}), dtype="float32", scope="local")
    av = T.alloc_buffer(({TM},), dtype="float32", scope="local")
    bv = T.alloc_buffer(({TN},), dtype="float32", scope="local")
    nthreads = {NTX} * {NTY}

    for i in T.serial({TM}):
        for j in T.serial({TN}):
            acc[i, j] = T.float32(0)

    for k0 in T.serial({NK}):
        # Both the destination and the source k index use l % BK (not l % BM):
        # mixing them is a silent, deterministic wrong answer that still looks
        # fast. k0 is the K-block index, so the absolute k needs k0 * BK.
        for e in T.serial({LA}):
            l = e * nthreads + ty * {NTX} + tx
            if l < {BM} * {BK}:
                As[l % {BK}, l // {BK}] = a[0, mb * {BM} + l // {BK}, k0 * {BK} + l % {BK}]
        for e in T.serial({LB}):
            l = e * nthreads + ty * {NTX} + tx
            if l < {BK} * {BN}:
                Bs[l % {BK}, l // {BK}] = b[k0 * {BK} + l % {BK}, nb * {BN} + l // {BK}]
        T.tvm_storage_sync("shared")
        for kk in T.serial({BK}):
            for i in T.serial({TM}):
                av[i] = As[kk, ty * {TM} + i]
            for j in T.serial({TN}):
                bv[j] = Bs[kk, tx * {TN} + j]
            for i in T.serial({TM}):
                for j in T.serial({TN}):
                    acc[i, j] = acc[i, j] + av[i] * bv[j]
        T.tvm_storage_sync("shared")

    for i in T.serial({TM}):
        for j in T.serial({TN}):
            r = acc[i, j] + bias[nb * {BN} + tx * {TN} + j]{RES_TERM}
            out[0, mb * {BM} + ty * {TM} + i, nb * {BN} + tx * {TN} + j] = {EPILOGUE}
'''


def build_smem(M, K, N, BM=64, BN=64, BK=4, TM=4, TN=4, PAD=1, relu=False,
               residual=False, name="gemm_smem"):
    """C = epilogue(A @ B + bias [+ residual]). A is (1,M,K), B is (K,N).

    With residual=True the kernel takes a fifth buffer of shape (1,M,N) placed
    before `out` in the signature (matching the graph's data/weight/bias/
    residual/out order); with relu=True the epilogue is max(.,0). The default
    (relu=False, residual=False) matches what the field-built pipeline
    actually emits for the 6 fused_matmul* functions: none of them apply ReLU.
    """
    NTY, NTX = BM // TM, BN // TN
    nthreads = NTX * NTY
    la = (BM * BK + nthreads - 1) // nthreads
    lb = (BK * BN + nthreads - 1) // nthreads
    res_decl = "residual: T.Buffer((1, %d, %d), \"float32\"),\n    " % (M, N) if residual else ""
    res_term = " + residual[0, mb * %d + ty * %d + i, nb * %d + tx * %d + j]" % (BM, TM, BN, TN) if residual else ""
    epilogue = "T.max(r, T.float32(0))" if relu else "r"
    src = _TEMPLATE_SMEM.format(
        M=M, K=K, N=N, BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, PAD=PAD,
        GB_M=M // BM, GB_N=N // BN, NTX=NTX, NTY=NTY,
        LA=la, LB=lb, NK=K // BK,
        RES_DECL=res_decl, RES_TERM=res_term, EPILOGUE=epilogue).strip()
    path = "/tmp/_gemmsm_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d.py" % (
        M, K, N, BM, BN, BK, TM, TN, PAD, int(relu), int(residual))
    os.makedirs("/tmp", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns[name]
