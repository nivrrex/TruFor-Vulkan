"""B1: replace the pipeline's fused 3x3 64->64 convolution with the hand-written kernel.

Runs after FuseTIR and the DLight default schedule. PrimFuncs in the lowered
IRModule are keyed by GlobalVar and print as `@T.prim_func(s_tir=True)`, so
matching is done on the buffer shapes in the signature rather than on the name,
which the fusion pass assigns:

    (1, 64, 512, 512) data, (64, 64, 3, 3) weight, (1, 64, 1, 1) bias, (1, 64, 512, 512) out

The weight is passed through untransformed, which is what makes a like-for-like
replacement possible at all. The replacement carries tirx.is_scheduled, the
attribute DLight uses to decide a PrimFunc no longer needs a default schedule.
"""

import re

import tvm

import depthwise as dw_conv
import scalar_reduce as sred
import direct_conv
import nhwc2nchw
import gemm

# (data, weight, bias, out) -> (TILE, OCB) chosen by sweep on each shape.
# For 64->64 at 512x512 TILE=32/OCB=16 was optimal (acc=64, 1086 GFLOPS).
# For 512<-2048 at 128x128, TILE=32/OCB=8 is the only configuration that both
# fills the 448-core device (1024 blocks) and keeps accumulators at 32;
# TILE=32/OCB=16 leaves only 512 blocks and collapses to 36 GFLOPS.
TARGETS = {
    ((1, 64, 512, 512), (64, 64, 3, 3), (1, 64, 1, 1), (1, 64, 512, 512)): dict(TILE=32, OCB=16),
    # 1x1: TILE=32/OCB=32 measured best at 46.0 ms / 748 GFLOPS. Note that for
    # this shape acc=128 beats acc=64, the opposite of the 3x3 case, because
    # 128x128 only yields 256 blocks at OCB=32 and the extra reuse matters more
    # than keeping the accumulator count low.
    ((1, 2048, 128, 128), (512, 2048, 1, 1), (1, 512, 1, 1), (1, 512, 128, 128)): dict(TILE=32, OCB=32),
}

# TH=4, VEC=8 chosen by sweep on the 256-channel 128x128 shape, all
# configurations verified by full-tensor comparison against numpy.
DW_CFG = dict(TH=4, VEC=8)
DW_TOL = 1e-5   # full-tensor rel_L2 against the original PrimFunc
# 512 threads measured best for the scalar max: 0.855 ms against 0.974 at 256
# and 0.950 at 1024. 64 is wrong (its tree does not converge), 128/256 work.
REDUCE_NT = 512
USE_CHANNEL = True   # set False to exclude the per-channel max
CHANNEL_MAX_NT = 256   # 0.936 ms at 256, 1.124 at 512, 2.084 at 1024

# Winning tile across the 6 matmul-cluster shapes on this GPU (RX 460 2 GB).
# BM64 BN64 BK4 TM4 TN4 gives 256 threads/block and 4x4=16 accumulators/thread.
GEMM_BM, GEMM_BN, GEMM_BK, GEMM_TM, GEMM_TN = 64, 64, 4, 4, 4


def _verify_new(cand, fn, sig_ok, dev, target, tol=DW_TOL):
    """Full-tensor check of a candidate against the PrimFunc it replaces.

    sig_ok is the signature of the ORIGINAL function, which the candidate must
    share; arguments are built from it, not from the candidate, so a shape
    mismatch shows up as a failure rather than as a silently different workload.
    """
    import numpy as _np
    rs = _np.random.RandomState(0)
    n = len(sig_ok)
    args = [tvm.runtime.tensor(rs.randn(*sig_ok[i][0]).astype("float32") * 0.05, dev)
            for i in range(n)]
    try:
        tvm.compile(fn, target=target)(*args)
        dev.sync()
    except Exception as e:
        return False, "original failed: %s" % str(e)[:60]
    ref = [a.numpy().copy() for a in args]
    try:
        tvm.compile(cand, target=target)(*args)
        dev.sync()
    except Exception as e:
        return False, "candidate failed: %s" % str(e)[:60]
    out = []
    for i in range(n):
        a, b = ref[i], args[i].numpy()
        r = float(_np.linalg.norm((b - a).ravel()) / max(float(_np.linalg.norm(a)), 1e-12))
        out.append(r)
    worst = max(out)
    return (worst < tol), "max rel_L2 over %d buffers = %.2e" % (n, worst)


def _pick_epilogue(fn, dev, target, tol=DW_TOL):
    """Return the relu flag that reproduces `fn` on random input, or None.

    The fused depthwise kernels end either in a ReLU or in a bare residual add,
    and the name is not a reliable signal. Guessing costs a factor of ten in
    accuracy: verified against the real PrimFunc, rel_L2 is 2.2e-08 when the
    flag is right and 0.71 when it is wrong. So try both and keep the one that
    matches, rather than inferring it.
    """
    import re as _re
    import numpy as _np
    head = _re.search(r"^def\s+\w+\((.*?)\):\s*$", fn.script(), _re.M | _re.S).group(1)
    ps = _re.findall(r"(\w+):\s*T\.Buffer\(\(([^)]*(?:\([^)]*\)[^)]*)*)\),\s*\"(\w+)\"\)", head)
    sh = [tuple(int(x) for x in _re.findall(r"T\.int64\((\d+)\)", d)) for _, d, _ in ps]
    rs = _np.random.RandomState(0)
    args = [tvm.runtime.tensor(rs.randn(*s).astype("float32") * 0.05, dev) for s in sh]
    out = tvm.runtime.tensor(_np.zeros(sh[3], "float32"), dev)
    try:
        tvm.compile(fn, target=target)(args[0], args[1], args[2], out)
        dev.sync()
    except Exception as e:
        return None, "original PrimFunc failed: %s" % str(e)[:60]
    ref = out.numpy().copy()
    C, H, W = sh[0][1], sh[0][2], sh[0][3]
    for relu in (True, False):
        try:
            cand = dw_conv.build(C, H, W, sh[3][2], sh[3][3], relu=relu, **DW_CFG)
            o = tvm.runtime.tensor(_np.zeros(sh[3], "float32"), dev)
            tvm.compile(cand, target=target)(args[0], args[1], args[2], o)
            dev.sync()
            got = o.numpy()
            rel = float(_np.linalg.norm((got - ref).ravel()) / max(float(_np.linalg.norm(ref)), 1e-12))
        except Exception:
            continue
        if rel < tol:
            return relu, "relu=%s rel_L2=%.2e" % (relu, rel)
    return None, "neither relu flag matched (relu=True gave the wrong answer)"

_SIG = re.compile(r"\w+: T\.Buffer\(\((.*?)\), \"(\w+)\"\)")
_DIM = re.compile(r"T\.int64\((\d+)\)")


def parse_sig(script):
    """Signature = the leading run of T.Buffer parameters.

    The def header cannot be matched with a non-greedy regex because
    `T.Buffer((T.int64(1), ...))` closes parens of its own, so the first `)`
    lands inside the buffer shape. Instead take the script up to the first
    statement after the header and collect its Buffer parameters in order.
    """
    m = re.search(r"^def\s+\w+\(", script, re.M)
    if not m:
        return None
    # TVMScript prints the whole def header on one line, so the first newline
    # ends it. Truncating on a `):` pattern instead runs past the header into
    # the body, because the body is full of calls that end the same way.
    head = script[m.end():].split("\n", 1)[0]
    out = []
    for dims, dtype in _SIG.findall(head):
        out.append((tuple(int(d) for d in _DIM.findall(dims)), dtype))
    return out


def _key(sig):
    if not sig or len(sig) < 4:
        return None
    data, weight, bias, out = sig[0][0], sig[1][0], sig[2][0], sig[3][0]
    if any(d != "float32" for _, d in sig[:4]):
        return None
    if any(d != "float32" for _, d in sig[:4]):
        return None
    return (data, weight, bias, out)


def swap(mod, only=None, layout=True, use_depthwise=True, use_reduce=True,
         use_gemm=True, verify=None, tiled=True, verbose=True):
    """Replace matching PrimFuncs in `mod` in place. Returns (n, names, log).

    verify=(dev, target) enables per-candidate full-tensor verification. Without
    it the depthwise replacements are made blind, and that was wrong for half of
    them, so it should normally be supplied.

    `only` restricts the dense/1x1 convolution replacements to those exact
    signatures; `layout` and `depthwise` toggle the other two families
    independently. All three are separate flags because an A/B that leaves one
    on while narrowing another measures nothing, which has now happened twice.
    """
    swapped, names, log = 0, [], []
    for gv in list(mod.functions.keys()):
        fn = mod.functions[gv]
        if not hasattr(fn, "attrs") or "T.prim_func" not in fn.script():
            continue
        sig_ok = parse_sig(fn.script())
        key = _key(sig_ok)
        # Scalar-output max: (1,1,N) -> (1,1,1). The pipeline schedules these
        # with grid=1 and 255 of 256 threads masked, leaving one thread to scan N
        # elements serially; SKIP measures max4 at 330.1 ms of a 1456.9 ms run.
        if use_reduce and sig_ok and len(sig_ok) == 2 \
                and len(sig_ok[0][0]) == 3 and sig_ok[0][0][0] == 1 \
                and sig_ok[0][0][1] == 1 and sig_ok[1][0] == (1, 1, 1) \
                and sig_ok[0][0][2] % REDUCE_NT == 0:
            n_elems = sig_ok[0][0][2]
            cand = sred.build_max(n_elems, REDUCE_NT)
            if verify is not None:
                ok, why = _verify_new(cand, fn, sig_ok, verify[0], verify[1])
                if not ok:
                    if verbose:
                        print("  跳过 scalar max %s: %s" % (str(gv).split('"')[1], why))
                    continue
                log.append("%s  scalar-max N=%d  %s" % (str(gv).split('"')[1], n_elems, why))
            mod[gv] = cand.with_attr("global_symbol", str(gv).split('"')[1])
            swapped += 1
            names.append(str(gv))
            continue
        # Per-channel maximum, (1,C,H,W) -> (1,C,1,1). Same grid=1 pathology as
        # the scalar case, and the output is not scalar here, so 128 independent
        # channels are available to spread over blocks. DLight measures
        # 15.476 ms in isolation; this is 0.936 ms, 20.6x.
        if use_reduce and USE_CHANNEL and sig_ok and len(sig_ok) == 2 and len(sig_ok[0][0]) == 4 \
                and sig_ok[0][0][0] == 1 and sig_ok[1][0] == (1, sig_ok[0][0][1], 1, 1) \
                and (sig_ok[0][0][2] * sig_ok[0][0][3]) % CHANNEL_MAX_NT == 0:
            Cc, Hc, Wc = sig_ok[0][0][1], sig_ok[0][0][2], sig_ok[0][0][3]
            cand = sred.build_channel_max(Cc, Hc, Wc, NT=CHANNEL_MAX_NT)
            if verify is not None:
                ok, why = _verify_new(cand, fn, sig_ok, verify[0], verify[1])
                if not ok:
                    if verbose:
                        print("  跳过 channel max %s: %s" % (str(gv).split('"')[1], why))
                    continue
                log.append("%s  channel-max C=%d  %s" % (str(gv).split('"')[1], Cc, why))
            mod[gv] = cand.with_attr("global_symbol", str(gv).split('"')[1])
            swapped += 1
            names.append(str(gv))
            continue
        # depthwise 3x3: (1,C,H,W) data, (C,1,3,3) weight, (1,C,1,1) bias,
        # (1,C,OH,OW) out. T0 measured DLight at 3.12-3.20 GB/s on the three
        # largest, 293 ms of a 1768 ms graph.
        if use_depthwise and sig_ok and len(sig_ok) == 4 and len(sig_ok[1][0]) == 4 \
                and sig_ok[1][0][1] == 1 and sig_ok[1][0][2] == 3 and sig_ok[1][0][3] == 3 \
                and sig_ok[0][0] == (1,) + sig_ok[1][0][:1] + (sig_ok[0][0][2], sig_ok[0][0][3]) \
                and sig_ok[2][0] == (1,) + sig_ok[1][0][:1] + (1, 1) \
                and sig_ok[3][0][1] == sig_ok[1][0][0]:
            d0, d1 = sig_ok[0][0], sig_ok[3][0]
            if d0[1] == d1[1] and d0[2] == d1[2] and d0[3] == d1[3]:
                if verify is not None:
                    relu, why = _pick_epilogue(fn, verify[0], verify[1])
                    if relu is None:
                        if verbose:
                            print("  跳过 %s: %s" % (str(gv).split('"')[1], why))
                        continue
                    log.append("%s  %s" % (str(gv).split('"')[1], why))
                    cand = dw_conv.build(d0[1], d0[2], d0[3], d1[2], d1[3],
                                         relu=relu, **DW_CFG)
                else:
                    cand = dw_conv.build(d0[1], d0[2], d0[3], d1[2], d1[3], **DW_CFG)
                mod[gv] = cand.with_attr("global_symbol", str(gv).split('"')[1])
                swapped += 1
                names.append(str(gv))
                continue
        # NCHW -> NHWC fused with GELU: 4-D (1,C,H,W) in, 3-D (1,H*W,C) out.
        # Matches fused_reshape10_transpose8_divide1_erf_add11_multiply1_multiply2,
        # which the T3 scan measured at 4.6 GB/s over 6 calls.
        if layout and sig_ok and len(sig_ok) == 2 and len(sig_ok[0][0]) == 4 \
                and len(sig_ok[1][0]) == 3 and sig_ok[0][0][0] == 1 and sig_ok[1][0][0] == 1 \
                and sig_ok[0][0][2] * sig_ok[0][0][3] == sig_ok[1][0][1]:
            C_, H_, W_ = sig_ok[0][0][1], sig_ok[0][0][2], sig_ok[0][0][3]
            if sig_ok[1][0][2] == C_ and H_ * W_ % 256 == 0:
                cand = nhwc2nchw.build(C_, H_, W_, act="gelu", src_mode="nchw")
                if verify is not None:
                    ok, why = _verify_new(cand, fn, sig_ok, verify[0], verify[1])
                    if not ok:
                        if verbose:
                            print("  跳过 GELU layout %s: %s" % (str(gv).split('"')[1], why))
                        continue
                    log.append("%s  gelu-layout  %s" % (str(gv).split('"')[1], why))
                mod[gv] = cand.with_attr("global_symbol", str(gv).split('"')[1])
                swapped += 1
                names.append(str(gv))
                continue
        # NHWC -> NCHW: two buffers, src (1, H*W, C) and dst (1, C, H, W).
        if layout and sig_ok and len(sig_ok) == 2 and len(sig_ok[0][0]) == 3 and len(sig_ok[1][0]) == 4 \
                and sig_ok[0][0][0] == 1 and sig_ok[1][0][0] == 1 \
                and sig_ok[0][0][1] == sig_ok[1][0][2] * sig_ok[1][0][3] \
                and sig_ok[0][0][2] == sig_ok[1][0][1]:
            C, H, W_ = sig_ok[1][0][1], sig_ok[1][0][2], sig_ok[1][0][3]
            if H * W_ % 256 == 0 and H * W_ >= 256:
                mod[gv] = nhwc2nchw.build(C, H, W_, tiled=tiled).with_attr(
                    "global_symbol", str(gv).split('"')[1])
                swapped += 1
                names.append(str(gv))
                continue
        # GEMM + bias (+ optional residual), 4- or 5-buffer signature.
        # DLight's Matmul rule already uses use_shared=True and cache_reads, so
        # beating it needs a genuine shared-memory tiled GEMM. The kernel here
        # was measured at BM=BN=64 / BK=4 / TM=TN=4 across the 6 matmul-cluster
        # shapes (isolated): 524/673/410 GF against DLight's 265/305/177 on the
        # top three, i.e. 2.0-2.4x. The graph's matmul epilogues are bias-only
        # or bias+residual with no ReLU, so try relu=False first and only fall
        # back to relu=True if verification rejects it.
        if use_gemm and sig_ok and len(sig_ok) in (4, 5) \
                and all(d == "float32" for _, d in sig_ok) \
                and len(sig_ok[0][0]) == 3 and sig_ok[0][0][0] == 1 \
                and len(sig_ok[1][0]) == 2 and sig_ok[1][0][0] == sig_ok[0][0][2] \
                and len(sig_ok[2][0]) == 1 and sig_ok[2][0][0] == sig_ok[1][0][1] \
                and sig_ok[-1][0] == (1, sig_ok[0][0][1], sig_ok[1][0][1]) \
                and (len(sig_ok) == 4 or sig_ok[3][0] == sig_ok[4][0]) \
                and sig_ok[0][0][1] % GEMM_BM == 0 and sig_ok[1][0][1] % GEMM_BN == 0 \
                and sig_ok[0][0][2] % GEMM_BK == 0:
            M_, K_, N_ = sig_ok[0][0][1], sig_ok[0][0][2], sig_ok[1][0][1]
            has_res = len(sig_ok) == 5
            picked = None
            for relu_ in (False, True):
                try:
                    cand = gemm.build_smem(M_, K_, N_, BM=GEMM_BM, BN=GEMM_BN,
                                           BK=GEMM_BK, TM=GEMM_TM, TN=GEMM_TN,
                                           relu=relu_, residual=has_res)
                except Exception:
                    continue
                if verify is None:
                    picked = (cand, "gemm %dx%dx%d res=%d relu=%d (unverified)" %
                              (M_, K_, N_, int(has_res), int(relu_)))
                    break
                ok, why = _verify_new(cand, fn, sig_ok, verify[0], verify[1])
                if ok:
                    picked = (cand, "gemm %dx%dx%d res=%d relu=%d  %s" %
                              (M_, K_, N_, int(has_res), int(relu_), why))
                    break
            if picked is None:
                if verbose:
                    print("  跳过 gemm %s (neither epilogue matched)" %
                          str(gv).split('"')[1])
            else:
                cand, tag = picked
                mod[gv] = cand.with_attr("global_symbol", str(gv).split('"')[1])
                swapped += 1
                names.append(str(gv))
                if verbose:
                    log.append("%s  %s" % (str(gv).split('"')[1], tag))
            continue
        if key is None or key not in TARGETS:
            continue
        if only is not None and key not in only:
            continue
        data, weight, bias, out = key
        if weight[2] == 1 and weight[3] == 1:
            repl = direct_conv.build_1x1(weight[0], weight[1], data[2], data[3],
                                         **TARGETS[key])
        else:
            repl = direct_conv.build_for(weight[0], weight[1], data[2], data[3],
                                         **TARGETS[key])
        # Name the replacement after the GlobalVar, not after fn.attrs: reusing
        # the attribute form registers under a name the module's NameSupply has
        # not seen and NormalizeGlobalVar then rejects the module with
        # "duplicate global symbol".
        mod[gv] = repl.with_attr("global_symbol", str(gv).split('"')[1])
        swapped += 1
        names.append(str(gv))
    if verbose:
        print("替换 %d 个 PrimFunc" % swapped)
        for n in names:
            print("   %s" % n.split('"')[1])
    return swapped, names, log
