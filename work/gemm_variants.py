"""Verify all 4 epilogue variants (relu x residual) against a numpy reference."""
import os, sys
import numpy as np, tvm
sys.path.insert(0, os.path.join(os.environ["ROOT"], "scripts", "opt"))
import gemm

dev = tvm.vulkan()
tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})

def test(M, K, N, BM, BN, BK, TM, TN, relu, residual):
    rs = np.random.RandomState(1)
    a = (rs.randn(1, M, K) * 0.05).astype("float32")
    b = (rs.randn(K, N) * 0.05).astype("float32")
    bias = (rs.randn(N) * 0.01).astype("float32")
    res = (rs.randn(1, M, N) * 0.01).astype("float32") if residual else None
    r = a.reshape(-1, K) @ b + bias
    if residual: r = r + res.reshape(M, N)
    if relu: r = np.maximum(r, 0)
    exp = r.reshape(1, M, N).astype("float32")
    f = gemm.build_smem(M, K, N, BM=BM, BN=BN, BK=BK, TM=TM, TN=TN,
                        relu=relu, residual=residual)
    ta, tb, tbias = (tvm.runtime.tensor(x, dev) for x in (a, b, bias))
    tout = tvm.runtime.tensor(np.zeros((1, M, N), "float32"), dev)
    args = (ta, tb, tbias, tvm.runtime.tensor(res, dev), tout) if residual else (ta, tb, tbias, tout)
    tvm.compile(f, target=tgt)(*args)
    dev.sync()
    got = tout.numpy()
    err = np.abs(got - exp)
    print("M%d K%d N%d relu=%d res=%d: nbad=%d max_abs=%.2e" %
          (M, K, N, int(relu), int(residual), int((err > 0.05).sum()), float(err.max())))

for (M, K, N) in [(32, 32, 32), (128, 64, 64), (1024, 320, 512), (1024, 1280, 320), (4096, 128, 512)]:
    for relu in (False, True):
        for res in (False, True):
            test(M, K, N, 64, 64, 4, 4, 4, relu, res)
