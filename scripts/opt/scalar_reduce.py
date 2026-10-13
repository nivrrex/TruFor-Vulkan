"""Hand-written scalar-output reductions for the Vulkan target.

The pipeline's schedule for these is pathological, and the SKIP measurement puts
it at the top of the whole graph: max4 costs 330.1 ms of a 1456.9 ms run, 22.7
percent, 4 calls at 82.5 ms each. The TIR says why:

    def max4(lv2619: (1,1,262144), lv2619_red: (1,1,1)):
      for ax0_fused_0 in T.thread_binding(1, blockIdx.x):     # grid is 1
        for ax0_fused_1 in T.thread_binding(256, threadIdx.x):
          T.where(ax0_fused_0 * 256 + ax0_fused_1 < 1)       # thread 0 only
            for ax1 in range(262144):                        # serial scan

The reduction schedule derives its grid from the OUTPUT extents. A scalar output
means one block, the block's 256 threads are masked down to thread 0, and that
one thread walks 262144 elements.

THE TREE REDUCTION BELOW HAD TWO BUGS, and the second is the more serious one.

1. The stride started at NT rather than NT/2, so the first step read sh[t + NT],
   one past the end of the array. On the scalar shape that read returned
   something harmless and the kernel still verified bit-exact against the
   original PrimFunc, so it shipped.

2. There was no barrier between tree steps, and every thread wrote the output.
   ThreadSync cannot see the dependency: the reads sit inside an if_then_else, so
   its analysis does not connect step s's write to step s+1's read. A thread
   could therefore read a partner's half-updated value. Again the scalar shape
   verified correct, because with a single block the interleaving happened to
   work out; writing the same loop for a per-channel version, which has 128
   blocks, produced a wrong answer, and that is what exposed it.

Both are fixed: the stride starts at NT/2, an explicit T.tvm_storage_sync runs
at the end of every step, and only thread 0 writes the output. The lesson is
specific and worth keeping: a verification that passes against one shape is not
evidence that an out-of-bounds access or a missing barrier is absent, and the
shapes that share a kernel family are exactly where a silent bug hides. The
scalar kernel passed two independent full-tensor checks and was still wrong on
both counts. A maximum over 1 MB should take microseconds
at 100 GB/s; this takes 82.5 ms, about four orders of magnitude off, which is
what a serial single-thread scan costs.

This module instead keeps all NT threads busy on disjoint chunks and finishes
with a log2(NT)-step tree reduction through a shared-memory array. No barrier is
written by hand: a probe with and without an explicit tvm_storage_sync both give
the correct result, so the ThreadSync pass inserts what is needed.

That probe also settles a claim this project had recorded as settled. An earlier
note said shared memory "returns zeros in the flat tirx form". That was a bug in
the probe, not a property of the form: it never wrote its output buffer, and its
tree index ((t//st)*st*2 + (t%st)*2) skipped the whole upper half at the first
step. The form works; the flat form is not a dead end.
"""

import os

_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def scalar_max(
    data: T.Buffer((1, 1, {N}), "float32"),
    out: T.Buffer((1, 1, 1), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    t = T.thread_binding(T.int64({NT}), thread="threadIdx.x")
    sh = T.alloc_shared(({NT},), "float32")
    acc = T.alloc_buffer((1,), dtype="float32", scope="local")
    acc[0] = T.float32(-3.4e38)
    for i in T.serial({PER}):
        acc[0] = T.max(acc[0], data[0, 0, t * {PER} + i])
    sh[t] = acc[0]
    for s in T.serial({LOG2}):
        st = {NT} >> (s + 1)
        # Clamp before indexing: SPIR-V evaluates both arms of OpSelect, so on
        # a non-reader thread (t // st) % 2 == 1 the sh[t + st] read still
        # happens and reaches index >= NT. Mesa RADV silently returns 0 for the
        # OOB read; NVIDIA's Vulkan driver returns undefined memory, which is
        # how the same kernel that passed on RX460 failed verification on P4
        # with rel_L2 5e+02. Readers already keep t + st < NT, so clamping
        # only affects the arms that get discarded by if_then_else.
        v = T.if_then_else((t // st) % 2 == 0, sh[T.min(t + st, {NT} - 1)], T.float32(-3.4e38))
        sh[t] = T.max(sh[t], v)
        T.tvm_storage_sync("shared")
    if t == 0:
        out[0, 0, 0] = sh[0]
'''


def build_max(N, NT=256, name="scalar_max"):
    """Full-tree reduction over N elements with NT threads; N must be a multiple
    of NT and NT a power of two."""
    assert N % NT == 0, "N %d not divisible by NT %d" % (N, NT)
    assert NT & (NT - 1) == 0, "NT %d is not a power of two" % NT
    src = _TEMPLATE.format(N=N, NT=NT, PER=N // NT, LOG2=NT.bit_length() - 1).strip()
    path = "/tmp/_sred_%d_%d.py" % (N, NT)
    os.makedirs("/tmp", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns[name]


# ---------------------------------------------------------------------------
# Per-channel maximum, (1,C,H,W) -> (1,C,1,1).
#
# Same pathology as the scalar case and worse in a different way. DLight emits
#   for ax0_fused_0 in thread_binding(1, blockIdx.x):     # grid is 1
#     for ax0_fused_1 in thread_binding(256, threadIdx.x):
#       T.where(ax0_fused_0 * 256 + ax0_fused_1 < 128)   # 128 of 256 threads
#         for ax1, ax2 in T.grid(128, 128):              # serial, 16384 steps
# so one block, half its threads idle, each scanning 16384 elements, on a device
# with 56 CUs. The output here is NOT scalar, it is (1,128,1,1), so the grid has
# 128 independent channels available to exploit and DLight uses one.

_CHANNEL_TEMPLATE = '''
from tvm.script import tirx as T

@T.prim_func
def channel_max(
    data: T.Buffer((1, {C}, {H}, {W}), "float32"),
    out: T.Buffer((1, {C}, 1, 1), "float32"),
):
    T.func_attr({{"tirx.is_scheduled": True, "tirx.noalias": True}})
    c = T.thread_binding(T.int64({C}), thread="blockIdx.x")
    t = T.thread_binding(T.int64({NT}), thread="threadIdx.x")
    sh = T.alloc_shared(({NT},), "float32")
    acc = T.alloc_buffer((1,), dtype="float32", scope="local")
    acc[0] = T.float32(-3.4e38)
    for i in T.serial({NIT}):
        idx = i * {NT} + t
        acc[0] = T.max(acc[0], data[0, c, idx // {W}, idx % {W}])
    sh[t] = acc[0]
    for s in T.serial({LOG2}):
        st = {NT} >> (s + 1)
        # See the comment on the same line in _TEMPLATE: OOB read is UB even
        # though if_then_else discards the value, because SPIR-V evaluates both
        # arms of OpSelect. Clamp to keep the load in-bounds.
        v = T.if_then_else((t // st) % 2 == 0, sh[T.min(t + st, {NT} - 1)], T.float32(-3.4e38))
        sh[t] = T.max(sh[t], v)
        T.tvm_storage_sync("shared")
    if t == 0:
        out[0, c, 0, 0] = sh[0]
'''


def build_channel_max(C, H, W, NT=256, name="channel_max"):
    """Per-channel max over H*W. H*W must be a multiple of NT and NT a power of two.

    The index makes the column the fastest-varying term so the load is
    contiguous, and the shared rows are unpadded because a pure reduction only
    ever reads sh[t] and sh[t+st], never a column walk.
    """
    HW = H * W
    assert HW % NT == 0, "H*W %d not divisible by NT %d" % (HW, NT)
    assert NT & (NT - 1) == 0, "NT %d is not a power of two" % NT
    src = _CHANNEL_TEMPLATE.format(C=C, H=H, W=W, NT=NT,
                                    NIT=HW // NT, LOG2=NT.bit_length() - 1).strip()
    path = "/tmp/_cmax_%d_%d_%d_%d.py" % (C, H, W, NT)
    os.makedirs("/tmp", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src + "\n")
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns[name]
