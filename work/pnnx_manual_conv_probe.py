#!/usr/bin/env python3
"""Manual single-layer forward: blob2 -> torch conv0(+BN fold)+relu -> vs blob3.

Also blob18 -> torch conv16 -> vs blob19. Distinguishes weight-mismatch from
ncnn-runtime-mismatch. Marker MANUAL_CONV_DONE.
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

import torch


def rel_l2(a, b):
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b)) + 1e-12
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="64x64")
    args = ap.parse_args()
    h, w = (int(v) for v in args.size.split("x"))

    torch.set_grad_enabled(False)
    model = exp.build_model(os.path.join(ROOT, "weights", "trufor.pth.tar"))
    model.eval()
    ed = model
    while not hasattr(ed, "dncnn") and hasattr(ed, "model"):
        ed = ed.model
    seq = list(ed.dncnn)
    conv0 = seq[0]
    relu0 = seq[1]
    conv_last = seq[-1]
    print("dncnn modules: %d, first=%s last=%s" %
          (len(seq), type(conv0).__name__, type(conv_last).__name__), flush=True)

    torch.manual_seed(123)
    blob_t = torch.rand(1, 3, h, w)
    x0 = blob_t[0].numpy().astype("float32")

    r0 = relu0(conv0(blob_t)).double().numpy()[0]

    import ncnn
    n = ncnn.Net()
    n.opt.use_vulkan_compute = False
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    pd = os.path.join(HERE, "ncnn_a1")
    assert n.load_param(os.path.join(pd, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(pd, "trufor.ts.ncnn.bin")) == 0
    ex = n.create_extractor()
    ex.input("in0", ncnn.Mat(x0))

    ret, m2 = ex.extract("2")
    assert ret == 0
    a2 = np.array(m2).astype("float64")
    print("blob2 vs input: rel_L2=%.3e" % rel_l2(a2, x0.astype("float64")),
          flush=True)

    # manual conv0 on ncnn's blob2 value
    mn = conv0(torch.from_numpy(a2.astype("float32")[None])).double()
    mn = torch.relu(mn)[0].numpy()
    ret, m3 = ex.extract("3")
    assert ret == 0
    a3 = np.array(m3).astype("float64")
    print("blob3 vs ncnnconv:  rel_L2=%.3e" % rel_l2(a3, mn), flush=True)
    print("blob3 vs torchchain: rel_L2=%.3e" % rel_l2(a3, r0), flush=True)

    # layer 16: feed ncnn's own blob18 into torch conv_last
    ret, m18 = ex.extract("18")
    assert ret == 0
    a18 = np.array(m18).astype("float64")
    ml = conv_last(torch.from_numpy(a18.astype("float32")[None])).double()[0].numpy()
    ret, m19 = ex.extract("19")
    assert ret == 0
    a19 = np.array(m19).astype("float64")
    print("blob19 vs manualconv(blob18): rel_L2=%.3e" % rel_l2(a19, ml),
          flush=True)
    print("blob18 dims/c/h/w = %d/%d/%d/%d" % (m18.dims, m18.c, m18.h, m18.w),
          flush=True)

    # scan the whole bin for a fragment of torch conv0 weight to find its offset
    raw = open(os.path.join(pd, "trufor.ts.ncnn.bin"), "rb").read()
    frag = conv0.weight.detach().float().numpy().ravel()[:8].astype("<f4").tobytes()
    pos = raw.find(frag)
    print("conv0 weight first-8-float fragment found at byte offset: %s "
          "(bin size %d)" % (pos, len(raw)), flush=True)
    frag_last = (conv0.bias is None)
    b = conv0.bias.detach().float().numpy().ravel()[:8].astype("<f4").tobytes()
    posb = raw.find(b)
    print("conv0 bias fragment offset: %s" % posb, flush=True)
    print("MANUAL_CONV_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
