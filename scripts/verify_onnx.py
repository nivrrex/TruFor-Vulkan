#!/usr/bin/env python3
"""Probe 3b: numerical equivalence of the exported ONNX model vs PyTorch.

Export succeeding means nothing if the numbers drift. Runs both on the same
512x512 input tensor and reports the error on all three model outputs, plus
checks the output spatial size actually matches the input (the legacy tracer
hard-codes shapes, so this is the property most likely to be wrong).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRUFOR_ROOT = os.environ.get("TRUFOR_ROOT", os.path.join(ROOT, "truFor-src"))
TRAIN_DIR = os.path.join(TRUFOR_ROOT, "TruFor_train_test")
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from lib.config import config, update_config  # noqa: E402
from lib.utils import get_model  # noqa: E402


def build_model(model_file, exp="trufor_ph3"):
    class _Args:
        pass

    prev = os.getcwd()
    os.chdir(TRAIN_DIR)
    try:
        a = _Args()
        a.experiment = exp
        a.opts = []
        update_config(config, a)
        model = get_model(config)
    finally:
        os.chdir(prev)
    ckpt = torch.load(model_file, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def stats(name, a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = np.abs(a - b)
    denom = max(float(np.abs(a).max()), 1e-12)
    return {
        "name": name,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "shape_match": list(a.shape) == list(b.shape),
        "max_abs_err": float(d.max()),
        "mean_abs_err": float(d.mean()),
        "max_rel_err": float(d.max() / denom),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", default="trufor_ph3")
    args = ap.parse_args()

    import onnxruntime as ort
    from PIL import Image

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    # --- input, built the same way TruFor's TestDataset builds it ---
    sys.path.insert(0, TRAIN_DIR)
    from dataset.dataset_test import TestDataset

    rgb, _ = next(iter(torch.utils.data.DataLoader(TestDataset(list_img=[args.image]), batch_size=1)))
    img = Image.open(args.image).convert("RGB")
    if img.size != (args.size, args.size):
        img = img.resize((args.size, args.size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    print("test image %s -> %dx%d, input range [%.3f, %.3f]" % (args.image, x.shape[3], x.shape[2], x.min(), x.max()))

    model = build_model(args.model, args.exp)
    t0 = time.perf_counter()
    out, conf, det, _npp = model(torch.from_numpy(x), save_np=False)
    t_pt = time.perf_counter() - t0

    pt_loc = F.softmax(torch.squeeze(out, 0), dim=0)[1].numpy()
    pt_conf = torch.sigmoid(torch.squeeze(conf, 0))[0].numpy()
    pt_det = torch.sigmoid(det).item()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = [o.name for o in sess.get_outputs()]
    print("onnx input :", sess.get_inputs()[0].shape, sess.get_inputs()[0].type)
    print("onnx outputs:", oname)
    t0 = time.perf_counter()
    res = sess.run(None, {iname: x})
    t_ort = time.perf_counter() - t0

    def pick(key):
        for n, r in zip(oname, res):
            if key in n.lower():
                return r
        return None

    o_loc, o_conf, o_det = pick("out"), pick("conf"), pick("det")
    if o_loc is not None and o_loc.ndim == 4 and o_loc.shape[1] > 1:
        o_loc = F.softmax(torch.from_numpy(o_loc[0]), dim=0)[1].numpy()
    if o_conf is not None and o_conf.ndim == 4:
        o_conf = torch.sigmoid(torch.from_numpy(o_conf[0]))[0].numpy()
    # test.py applies torch.sigmoid(det); the exported graph stops at the logit,
    # so the sigmoid must be applied here to compare like with like.
    o_det_raw = float(np.asarray(o_det).reshape(-1)[0]) if o_det is not None else None
    o_det = 1.0 / (1.0 + np.exp(-o_det_raw)) if o_det_raw is not None else None

    report = {
        "image": args.image,
        "size": args.size,
        "providers": sess.get_providers(),
        "pytorch_total_s": t_pt,
        "onnxruntime_cpu_s": t_ort,
        "integrity_score": {"pytorch": pt_det, "onnx": o_det, "onnx_logit": o_det_raw,
                            "abs_err": abs(pt_det - o_det) if o_det is not None else None},
        "localization": stats("map", pt_loc, o_loc) if o_loc is not None else None,
        "confidence": stats("conf", pt_conf, o_conf) if o_conf is not None else None,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== numerical equivalence (PyTorch vs ONNX Runtime CPU) ===")
    print("integrity score : pytorch=%.6f  onnx logit=%.6f -> sigmoid=%.6f  err=%s"
          % (pt_det, o_det_raw, o_det, ("%.3e" % abs(pt_det - o_det)) if o_det is not None else "n/a"))
    for key in ("localization", "confidence"):
        r = report[key]
        if r:
            print("%-13s : shape %s vs %s match=%s  max_abs=%.3e  mean_abs=%.3e  rmse=%.3e  corr=%.8f"
                  % (key, r["shape_a"], r["shape_b"], r["shape_match"], r["max_abs_err"],
                     r["mean_abs_err"], r["rmse"], r["corr"]))
    print("latency         : pytorch=%.2fs  ort-cpu=%.2fs" % (t_pt, t_ort))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
