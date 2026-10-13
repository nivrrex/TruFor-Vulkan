import os, sys, time, statistics
import numpy as np, tvm
sys.path.insert(0, os.path.join(os.environ["ROOT"], "scripts", "opt"))
import gemm

dev = tvm.vulkan()
tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})

SHAPES = [
    (1024, 1280, 320, 265),
    (1024,  320, 1280, 305),
    (4096,  128,  512, 177),
    (16384,  64,  512, 0),
    (16384,  64,  256, 0),
    (1024,  320,  512, 0),
    (256,   512, 2048, 0),
]

def bench(f, a, b, bias, out, reps=15):
    ex = tvm.compile(f, target=tgt)
    for _ in range(3):
        ex(a, b, bias, out); dev.sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ex(a, b, bias, out); dev.sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3

def cfg(M, K, N, BM, BN, BK, TM, TN):
    rs = np.random.RandomState(1)
    a = (rs.randn(1, M, K) * 0.05).astype("float32")
    b = (rs.randn(K, N) * 0.05).astype("float32")
    bias = (rs.randn(N) * 0.01).astype("float32")
    exp = np.maximum(a.reshape(-1, K) @ b + bias, 0).reshape(1, M, N).astype("float32")
    f = gemm.build_smem(M, K, N, BM=BM, BN=BN, BK=BK, TM=TM, TN=TN)
    ta, tb, tbias = (tvm.runtime.tensor(x, dev) for x in (a, b, bias))
    out = tvm.runtime.tensor(np.zeros((1, M, N), "float32"), dev)
    tvm.compile(f, target=tgt)(ta, tb, tbias, out); dev.sync()
    err = float(np.abs(out.numpy() - exp).max())
    nbad = int((np.abs(out.numpy() - exp) > 0.05).sum())
    g = gemm.gflop(M, K, N)
    ms = bench(f, ta, tb, tbias, out)
    return ms, g * 1e3 / ms, nbad, err

for (M, K, N, dlight_g) in SHAPES:
    print("=== shape M=%d K=%d N=%d (DLight %.0f GF) ===" % (M, K, N, dlight_g))
    for BM, BN, BK, TM, TN in [
        (64, 64, 4, 4, 4),
        (64, 64, 8, 4, 4),
        (32, 32, 4, 4, 4),
    ]:
        if M % BM or N % BN or K % BK:
            print("  BM%d BN%d BK%d TM%d TN%d: skip (shape)" % (BM, BN, BK, TM, TN))
            continue
        try:
            ms, gf, nbad, err = cfg(M, K, N, BM, BN, BK, TM, TN)
            print("  BM%d BN%d BK%d TM%d TN%d: %.3f ms  %4d GF  nbad=%d max_abs=%.1e"
                  % (BM, BN, BK, TM, TN, ms, gf, nbad, err))
        except Exception as e:
            print("  BM%d BN%d BK%d TM%d TN%d: FAIL %s" % (BM, BN, BK, TM, TN, str(e)[:80]))
