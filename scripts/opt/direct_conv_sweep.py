"""Parameter sweep for the hand-written direct convolution.

The isolated kernel at (TILE=32, 16x16 threads, 2x2 pixels, OCB=8) runs at
39.71 ms / 487 GFLOPS.  Its arithmetic intensity is the thing to raise: per block
it streams 64 * 34 * 34 * 4 B = 296 KB of input against 18 KB of weights, so
every input element is only reused by 8 output channels.  Two knobs move that:

  OCB   output channels per block  -> directly multiplies the reuse of each
                                     input element, at the cost of one
                                     accumulator per channel per pixel
  TILE  spatial extent per block   -> more pixels per input element loaded,
                                     i.e. more weight reuse

So the sweep is over (TILE, TH, TW, OCB) with PH = TILE // TH, PW = TILE // TW.
Thread count is kept at a multiple of the wave size where possible; register
pressure is OCB * PH * PW accumulators per thread and is reported per row so an
overflow shows up as a failure rather than as a silently slow kernel.
"""

import itertools
import time

import numpy as np
import tvm

from direct_conv import flops, OC, IC, H, W, K

WAVE = 64  # Baffin (GCN5) wavefront size


def make(TILE, TH, TW, OCB):
    from tvm.script import tirx as T

    PH, PW = TILE // TH, TILE // TW
    # Both TILE and OCB must divide the problem exactly. Integer division would
    # silently drop channels (OCB=24 -> 64//24=2 -> 48 channels) or rows
    # (TILE=48 -> 512//48=10 -> 480 rows), and the kernel then looks fast only
    # because it computed less. Assert rather than divide.
    assert OC % OCB == 0, "OCB %d does not divide %d output channels" % (OCB, OC)
    assert H % TILE == 0 and W % TILE == 0, "TILE %d does not divide %dx%d" % (TILE, H, W)
    n_ty = H // TILE
    n_bz = OC // OCB

    @T.prim_func
    def conv(
        data: T.Buffer((1, IC, H + 2, W + 2), "float32"),
        weight: T.Buffer((OC, IC, K, K), "float32"),
        out: T.Buffer((OC, H, W), "float32"),
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

        for ci in T.serial(IC):
            for kh in T.serial(K):
                for kw in T.serial(K):
                    for co in T.serial(OCB):
                        wv = weight[bzi * OCB + co, ci, kh, kw]
                        for py in T.serial(PH):
                            gy = byi * TILE + ty * PH + py
                            for px in T.serial(PW):
                                gx = bxi * TILE + tx * PW + px
                                acc[co, py, px] = acc[co, py, px] + wv * data[0, ci, gy + kh, gx + kw]

        for co in T.serial(OCB):
            for py in T.serial(PH):
                gy = byi * TILE + ty * PH + py
                for px in T.serial(PW):
                    gx = bxi * TILE + tx * PW + px
                    out[bzi * OCB + co, gy, gx] = acc[co, py, px]

    return conv


def run(TILE, TH, TW, OCB, reps=8, dev=None, target=None):
    PH, PW = TILE // TH, TILE // TW
    reg = OCB * PH * PW
    thr = TH * TW
    try:
        lib = tvm.compile(make(TILE, TH, TW, OCB), target=target)
    except Exception as e:
        return dict(TILE=TILE, TH=TH, TW=TW, OCB=OCB, threads=thr, acc=reg,
                    ms=float("nan"), gflops=float("nan"), note="build failed: %s" % str(e)[:60])
    A = tvm.runtime.tensor(np.random.randn(1, IC, H + 2, W + 2).astype(np.float32), dev)
    Wt = tvm.runtime.tensor((np.random.randn(OC, IC, K, K) * 0.01).astype(np.float32), dev)
    O = tvm.runtime.tensor(np.zeros((OC, H, W), np.float32), dev)
    lib(A, Wt, O)
    dev.sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lib(A, Wt, O)
        dev.sync()
        ts.append(time.perf_counter() - t0)
    best = min(ts[1:])
    return dict(TILE=TILE, TH=TH, TW=TW, OCB=OCB, threads=thr, acc=reg,
                ms=best * 1e3, gflops=flops() / best, note="")


def check(TILE, TH, TW, OCB, dev, target, pts=((3, 0, 0), (27, 100, 200), (59, 511, 511), (43, 300, 17))):
    lib = tvm.compile(make(TILE, TH, TW, OCB), target=target)
    an = np.random.RandomState(0).randn(1, IC, H + 2, W + 2).astype(np.float32)
    wn = (np.random.RandomState(1).randn(OC, IC, K, K) * 0.01).astype(np.float32)
    A = tvm.runtime.tensor(an, dev)
    Wt = tvm.runtime.tensor(wn, dev)
    O = tvm.runtime.tensor(np.zeros((OC, H, W), np.float32), dev)
    lib(A, Wt, O)
    dev.sync()
    oh = O.numpy()
    bad = []
    for (co, y, x) in pts:
        ref = float(sum(wn[co, ci, kh, kw] * an[0, ci, y + kh, x + kw]
                        for ci in range(IC) for kh in range(K) for kw in range(K)))
        if abs(oh[co, y, x] - ref) > 1e-2 * max(1.0, abs(ref)):
            bad.append((co, y, x))
    return bad


def main():
    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    dev = tvm.vulkan()
    # TILE must divide 512, OCB must divide 64, PH=TILE//TH, PW=TILE//TW.
    grid = [
        (32, 16, 16, 8),    # the original configuration, for reference
        (32, 16, 16, 16),
        (16, 16, 16, 16),
        (16, 16, 16, 32),
        (16, 16, 16, 64),
        (32, 32, 32, 16),
        (32, 32, 32, 32),
        (64, 16, 16, 16),
        (64, 32, 32, 16),
        (64, 16, 16, 8),
    ]
    rows = []
    for cfg in grid:
        r = run(*cfg, dev=dev, target=target)
        rows.append(r)
        print("TILE=%-3d thr=%-5d OCB=%-3d acc=%-4d  %8.2f ms  %6.0f GFLOPS  %s"
              % (r["TILE"], r["threads"], r["OCB"], r["acc"], r["ms"], r["gflops"], r["note"]), flush=True)
    good = [r for r in rows if r["note"] == "" and r["ms"] == r["ms"]]
    if not good:
        print("\nno configuration built")
        return
    best = min(good, key=lambda r: r["ms"])
    print("\nbest: TILE=%d thr=%d OCB=%d acc=%d -> %.2f ms, %.0f GFLOPS"
          % (best["TILE"], best["threads"], best["OCB"], best["acc"], best["ms"], best["gflops"]))
    bad = check(best["TILE"], best["TH"], best["TW"], best["OCB"], dev, target)
    print("spot check on best:", "OK" if not bad else "MISMATCH at %s" % (bad,))
    print("\nreference: DLight 50.47 ms / 383 GFLOPS, first hand-written 39.71 ms / 487 GFLOPS")


if __name__ == "__main__":
    main()
