#!/usr/bin/env python3
"""Pointwise boundary comparison: torch eager sub-module outputs vs ncnn blobs.

Usage: pnnx_boundary_probe.py [--blobs 19,27] [--size 64x64] [--paramdir DIR]
CPU ncnn, fp32. Prints rel_L2 per boundary. Marker BOUNDARY_PROBE_DONE.
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(ROOT))

SITE = os.path.join(ROOT, "venv", "lib", "python3.11", "site-packages")
for _name, _path in (("timm", os.path.join(SITE, "timm")),
                     ("timm.models", os.path.join(SITE, "timm", "models"))):
    _mod = types.ModuleType(_name)
    _mod.__path__ = [_path]
    sys.modules[_name] = _mod

_spec = importlib.util.spec_from_file_location(
    "export_onnx_dynamo", os.path.join(ROOT, "scripts", "export_onnx_dynamo.py"))
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

conv_spec = importlib.util.spec_from_file_location(
    "pnnx_a1_convert_patches", os.path.join(HERE, "pnnx_a1_convert.py"))
_conv = importlib.util.module_from_spec(conv_spec)
conv_spec.loader.exec_module(_conv)


def rel_l2(a, b):
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b)) + 1e-12
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="64x64")
    ap.add_argument("--blobs", default="19,27")
    ap.add_argument("--dncnn", action="store_true",
                    help="layerwise DnCNN comparison (blobs 3..19)")
    ap.add_argument("--paramdir", default=os.path.join(HERE, "ncnn_a1"))
    args = ap.parse_args()
    h, w = (int(v) for v in args.size.split("x"))

    import torch
    torch.set_grad_enabled(False)
    model = exp.build_model(os.path.join(ROOT, "weights", "trufor.pth.tar"))
    model.eval()
    exp.swap_adaptive_pool(model)
    wrapper = exp.ExportWrapper(model)
    wrapper.eval()
    blob_t = torch.rand(1, 3, h, w)  # deterministic under fixed seed below
    torch.manual_seed(123)
    blob_t = torch.rand(1, 3, h, w)
    ed = model
    while not hasattr(ed, "dncnn") and hasattr(ed, "model"):
        ed = ed.model
    assert hasattr(ed, "dncnn"), "cannot locate EncoderDecoder"
    modal_x = ed.dncnn(blob_t)
    modal_x_raw = modal_x
    if modal_x.shape[1] == 1:
        modal_x = torch.tile(modal_x, (3, 1, 1))
    rgb_norm = ed.prepro(blob_t) if ed.prepro is not None else blob_t
    x = ed.backbone(rgb_norm, modal_x)
    x_list = list(x) if isinstance(x, (tuple, list)) else [x]
    print("backbone returns %d tensors: %s" %
          (len(x_list), [tuple(t.shape) for t in x_list]), flush=True)
    torch_outs = {"modal_x": modal_x_raw, "rgb_norm": rgb_norm}
    print("torch modal_x: %s rgb_norm: %s" %
          (tuple(modal_x_raw.shape), tuple(rgb_norm.shape)), flush=True)

    import ncnn
    n = ncnn.Net()
    n.opt.use_vulkan_compute = False
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    assert n.load_param(os.path.join(args.paramdir, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(args.paramdir, "trufor.ts.ncnn.bin")) == 0
    blob = blob_t.numpy()[0].copy()
    ex = n.create_extractor()
    ex.input("in0", ncnn.Mat(blob))

    want = [int(v) for v in args.blobs.split(",")]
    got = {}
    for bid in want:
        ret, m = ex.extract(str(bid))
        assert ret == 0, "extract %d failed" % bid
        got[bid] = np.array(m).astype("float64")

    if args.dncnn:
        # relu_i outputs <-> blob(3+i), i=0..15 ; dncnn return <-> blob19
        relus = [mm for mm in ed.dncnn.modules()
                 if type(mm).__name__ == "ReLU"]
        cap = []
        handles = []

        def hook(mod, inp, out):
            cap.append(out.detach().double().numpy()[0].copy())

        for r in relus:
            handles.append(r.register_forward_hook(hook))
        with torch.no_grad():
            out_raw = ed.dncnn(blob_t)
        for h in handles:
            h.remove()
        cap.append(out_raw.double().numpy()[0])
        print("dncnn relu hooks: %d (+final) " % len(cap), flush=True)
        exd = n.create_extractor()
        exd.input("in0", ncnn.Mat(blob))
        for i, ref in enumerate(cap):
            bid = 3 + i if i < 16 else 19
            ret, m = exd.extract(str(bid))
            if ret != 0:
                print("dncnn blob%d extract FAILED" % bid, flush=True)
                continue
            g = np.array(m).astype("float64")
            print("dncnn L%-2d blob%-3d torch%s ncnn%s rel_L2=%.3e" %
                  (i, bid, tuple(ref.shape), tuple(g.shape), rel_l2(g, ref)),
                  flush=True)
        print("DNCNN_LAYERWISE_DONE", flush=True)
        return 0

    b19 = {19: "modal_x", 27: "rgb_norm"}
    for bid, name in b19.items():
        if bid in got:
            ref = torch_outs[name].double().numpy()[0]
            g = got[bid]
            print("blob%-4d vs %-8s torch%s ncnn%s rel_L2=%.3e" %
                  (bid, name, tuple(ref.shape), tuple(g.shape), rel_l2(g, ref)),
                  flush=True)

    # locate backbone outputs: shape match on all blobs of the net
    refs = [t.double().numpy()[0] for t in x_list]
    ex2 = n.create_extractor()
    ex2.input("in0", ncnn.Mat(blob))
    nb = len(list(n.blobs()))
    hits = [None] * len(refs)
    for i in range(1, nb + 1):
        ret, m = ex2.extract(str(i))
        if ret != 0 or m.empty():
            continue
        a = np.array(m)
        if a.ndim != 3:
            continue
        for j, xref in enumerate(refs):
            if a.shape == xref.shape:
                v = rel_l2(a.astype("float64"), xref)
                if hits[j] is None or v < hits[j][1]:
                    hits[j] = (i, v)
    for j, xref in enumerate(refs):
        if hits[j]:
            print("backbone-out[%d] shape%s best blob%d rel_L2=%.3e" %
                  (j, tuple(xref.shape), hits[j][0], hits[j][1]), flush=True)
        else:
            print("backbone-out[%d]: no ncnn blob with shape %s" %
                  (j, tuple(xref.shape)), flush=True)
    print("BOUNDARY_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
