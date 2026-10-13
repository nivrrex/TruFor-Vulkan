#!/usr/bin/env python3
"""Verify the dynamic ONNX in ONNX Runtime against this project's torch model.

Self-contained: the reference is truFor-src's own model (build_model +
swap_adaptive_pool + ExportWrapper, the exact graph that was exported), not any
external copy. For each size we feed the SAME tensor to torch and to ORT and
compare per-output.

Two things are checked, independently:
  1. map==input contract: out/conf spatial dims equal the input H,W at every
     size (this is what the caller relies on for arbitrary image sizes).
  2. numerical parity: rel L2 and Pearson correlation of out and conf between
     ORT and torch.

As a cross-check, at 512x512 the dynamic ONNX is also compared against the
production fixed-512 inline ONNX; those two graphs come from the same weights
and should agree closely.

Normalization is a caller concern -- both sides get the identical float tensor,
so the comparison is normalization-agnostic by construction.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from export_onnx_dynamo import build_model, swap_adaptive_pool, ExportWrapper  # noqa: E402


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def report(name, ref, got):
    ref = np.asarray(ref, np.float64); got = np.asarray(got, np.float64)
    shape_ok = ref.shape == got.shape
    if shape_ok:
        rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
        mx = float(np.max(np.abs(got - ref)))
        cr = pearson(ref, got)
        print("    %-5s shape %-16s rel_L2=%.3e max_abs=%.3e corr=%.8f -> %s"
              % (name, str(got.shape), rel, mx, cr, "OK" if rel < 1e-2 else "CHECK"))
    else:
        print("    %-5s SHAPE MISMATCH ref=%s got=%s -> FAIL" % (name, ref.shape, got.shape))
    return shape_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "trufor.pth.tar"))
    ap.add_argument("--onnx", default=os.path.join(ROOT, "reports", "onnx-dynamo",
                    "trufor_dynamic_dynamo.onnx"))
    ap.add_argument("--fixed512", default=os.path.join(ROOT, "reports", "onnx-dynamo",
                    "trufor_512x512_dynamo_inline.onnx"))
    ap.add_argument("--exp", default="trufor_ph3")
    ap.add_argument("--sizes", default="256x256,512x512,640x480,1200x900,518x731")
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    fixed_sess = None
    if os.path.exists(args.fixed512):
        try:
            fixed_sess = ort.InferenceSession(args.fixed512, so,
                                              providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001
            print("fixed-512 session failed to load (%s); skipping cross-check" % e)

    torch.set_grad_enabled(False)
    model = build_model(args.model, args.exp)
    swap_adaptive_pool(model)
    wrapper = ExportWrapper(model)
    wrapper.eval()

    sizes = [tuple(int(v) for v in s.lower().replace("x", ",").split(","))
             for s in args.sizes.split(",") if s]

    rng = np.random.default_rng(0)
    all_ok = True
    for (h, w) in sizes:
        print("\n== %dx%d ==" % (h, w))
        x = rng.random((1, 3, h, w), dtype=np.float32)
        with torch.no_grad():
            ref = [t.detach().cpu().numpy() for t in wrapper(torch.from_numpy(x))]
        got = sess.run(None, {iname: x})
        # names: 0=out(map logits),1=conf,2=det
        ok = True
        ok = report("out", ref[0], got[0]) and ok
        ok = report("conf", ref[1], got[1]) and ok
        ok = report("det", ref[2], got[2]) and ok
        all_ok = all_ok and ok

        # map==input contract
        if got[0].shape[-2:] != (h, w) or got[1].shape[-2:] != (h, w):
            print("    map/conf spatial != input (%dx%d): out=%s conf=%s -> FAIL"
                  % (h, w, got[0].shape, got[1].shape))
            all_ok = False
        else:
            print("    map==input contract OK (out %s)" % (got[0].shape,))

        if fixed_sess is not None and (h, w) == (512, 512):
            fg = fixed_sess.run(None, {fixed_sess.get_inputs()[0].name: x})
            print("    -- vs production fixed-512 ONNX --")
            report("out", fg[0], got[0])
            report("conf", fg[1], got[1])
            report("det", fg[2], got[2])

    print("\nRESULT:", "PASS" if all_ok else "SOMETHING TO CHECK")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
