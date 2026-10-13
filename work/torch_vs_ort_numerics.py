#!/usr/bin/env python3
"""This project's torch model (truFor-src) vs this project's ONNX+ORT:
normalization-convention check.

The ONNX graph is a plain float->float function: the /255-vs-/256 choice is
made by the caller, not baked into the graph. Upstream GRIP-UNINA
(`data_core.py`) and any torch path normalise RGB by 256.0; this project's
run_trufor.py has fed 255.0. The map_corr = 1.0 numbers on record compare this
project's ORT to this project's TVM, both on /255 -- they do not pin the
convention against a torch reference.

This resolves it self-contained: the torch reference here is truFor-src built
from weights/trufor.pth.tar (the exact weights the ONNX was exported from), fed
/256 (the upstream convention). We then compare the ORT result at /255 and /256
against that torch reference and each other:

  A. ONNX matches torch when fed /256 -> caller must feed /256 to reproduce the
     official numbers. Fix: set caller preprocessing to /256, no re-export.
  B. ONNX matches only at /255 -> accept 0.4% offset from the official convention.
  C. some other convention -> re-export.

The reference is this project's own model, so nothing outside this repo is
required to run the check.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from export_onnx_dynamo import build_model, swap_adaptive_pool, ExportWrapper  # noqa: E402


def load_and_resize(path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, np.uint8)


def torch_run(wrapper, arr):
    x = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1)[None] / 256.0)
    with torch.no_grad():
        out, conf, det = wrapper(x)
    m = F.softmax(out[0], dim=0)[1].numpy()
    c = torch.sigmoid(conf[0])[0].numpy()
    s = float(torch.sigmoid(det).item())
    return m, c, s



def ort_run(sess, x):
    iname = sess.get_inputs()[0].name
    out, conf, det = sess.run(None, {iname: x})
    m = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy()
    c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy()
    s = float(torch.sigmoid(torch.from_numpy(det)).item())
    return m, c, s


def stats(a, b):
    d = b - a
    rel_l2 = float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-12))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return rel_l2, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--pth", required=True)
    a = ap.parse_args()

    arr = load_and_resize(a.image)
    print("image", os.path.basename(a.image), "->", arr.shape)

    # --- this project's torch reference (truFor-src), /256 upstream convention
    torch.set_grad_enabled(False)
    model = build_model(a.pth, "trufor_ph3")
    swap_adaptive_pool(model)
    wrapper = ExportWrapper(model)
    wrapper.eval()
    m_torch, c_torch, s_torch = torch_run(wrapper, arr)
    print("torch map shape", m_torch.shape, "score", s_torch)

    # --- ORT, /255 (this project's convention) ---------------------------
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    x255 = arr.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    m_255, c_255, s_255 = ort_run(sess, x255)

    # --- ORT, /256 (matches torch path externally) -----------------------
    x256 = arr.astype(np.float32).transpose(2, 0, 1)[None] / 256.0
    m_256, c_256, s_256 = ort_run(sess, x256)

    print()
    print("%-40s %-12s %-12s %-12s" % ("pair", "rel_L2 (map)", "corr (map)", "score delta"))
    def row(label, mref, sref, m2, s2):
        rl, co = stats(mref, m2)
        print("%-40s %-12.3e %-12.8f %-12.3e" % (label, rl, co, abs(sref - s2)))

    row("torch(/256) vs ORT(/255)", m_torch, s_torch, m_255, s_255)
    row("torch(/256) vs ORT(/256)", m_torch, s_torch, m_256, s_256)
    row("ORT(/255)   vs ORT(/256)", m_255, s_255, m_256, s_256)

    print()
    print("score torch = %.6f" % s_torch)
    print("score 255   = %.6f" % s_255)
    print("score 256   = %.6f" % s_256)

    # also conf map
    if c_torch is not None:
        r5, c5 = stats(c_torch, c_255)
        r6, c6 = stats(c_torch, c_256)
        print()
        print("conf: torch vs ORT/255 rel_L2 %.3e corr %.8f" % (r5, c5))
        print("conf: torch vs ORT/256 rel_L2 %.3e corr %.8f" % (r6, c6))


if __name__ == "__main__":
    main()
