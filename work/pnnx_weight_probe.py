#!/usr/bin/env python3
"""Verify DnCNN conv weights/BIAS in ncnn .bin vs torch (BN folded).

ncnn Convolution bin layout: bias float32 (pad4) then weight float32 (pad4),
per layer, in param layer order. Marker WEIGHT_PROBE_DONE.
"""
import importlib.util
import os
import struct
import sys
import types

import numpy as np
import torch

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

MODELDIR = os.path.join(HERE, "ncnn_a1")


def pad4(n):
    return n + (-n % 4)


def main():
    torch.set_grad_enabled(False)
    param = open(os.path.join(MODELDIR, "trufor.ts.ncnn.param")).read().splitlines()
    binf = open(os.path.join(MODELDIR, "trufor.ts.ncnn.bin"), "rb")

    convs = []
    for ln in param[2:]:
        parts = ln.split()
        if parts and parts[0] in ("Convolution", "ConvolutionDepthWise"):
            d = {}
            for tok in parts[3:]:
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    try:
                        d[int(k)] = float(v)
                    except ValueError:
                        pass
            convs.append((parts[1], int(d.get(0, 0)), int(d.get(1, 0)),
                          int(d.get(2, 1)), int(d.get(5, 0)), d.get(6, 0)))
    print("param Convolution layers: %d" % len(convs), flush=True)

    model = exp.build_model(os.path.join(ROOT, "weights", "trufor.pth.tar"))
    model.eval()
    ed = model
    while not hasattr(ed, "dncnn") and hasattr(ed, "model"):
        ed = ed.model
    torch_convs = [m for m in ed.dncnn.modules() if isinstance(m, torch.nn.Conv2d)]
    torch_bns = [m for m in ed.dncnn.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    print("torch dncnn convs: %d bns: %d" %
          (len(torch_convs), len(torch_bns)), flush=True)

    # walk bin: read each conv's bias+weight with pad4
    off = 0
    tci = 0
    bad = 0
    for li, (name, outch, ktype, ksize_k, bias_n, weight_count) in enumerate(convs[:20]):
        # dncnn convs are the first 17 Convolution layers in param
        in_ch = int(weight_count // 9 // outch) if ksize_k == 3 and ktype == 1 else None
        nb = pad4(outch) if bias_n else 0
        binf.seek(off)
        bias = np.frombuffer(binf.read(nb * 4), dtype="float32")[:outch] if nb else None
        nw = pad4(int(weight_count))
        w = np.frombuffer(binf.read(nw * 4), dtype="float32")[:int(weight_count)]
        off += (nb + nw) * 4
        if li >= len(torch_convs):
            print("conv%d(%s): beyond torch dncnn (17) -- stop compare" % (li, name),
                  flush=True)
            break
        tc = torch_convs[li]
        wt = tc.weight.detach().double().numpy().ravel()
        # BN fold if next module is BatchNorm
        bn = None
        seq = list(ed.dncnn)
        idx = seq.index(tc)
        if idx + 1 < len(seq) and isinstance(seq[idx + 1], torch.nn.BatchNorm2d):
            bn = seq[idx + 1]
        if bn is not None:
            gamma = bn.weight.detach().double().numpy()
            beta = bn.bias.detach().double().numpy()
            mean = bn.running_mean.detach().double().numpy()
            var = bn.running_var.detach().double().numpy()
            eps = bn.eps
            s = (gamma / np.sqrt(var + eps))  # per OUTPUT channel
            w_in = wt.reshape(outch, -1)
            wt = (w_in * s[:, None]).ravel()
            b0 = (tc.bias.detach().double().numpy()
                  if tc.bias is not None else np.zeros(outch))
            bias_t = s * (b0 - mean) + beta
        else:
            bias_t = (tc.bias.detach().double().numpy()
                      if tc.bias is not None else np.zeros(outch))
        w_ok = np.allclose(w.astype("float64"), wt, rtol=0, atol=1e-5)
        b_ok = (bias is None and np.allclose(bias_t, 0)) or (
            bias is not None and
            np.allclose(bias.astype("float64"), bias_t, rtol=0, atol=1e-5))
        if not (w_ok and b_ok):
            dw = np.abs(w.astype("float64") - wt).max() if wt.size else 0
            print("conv%d(%s) WEIGHT_OK=%s BIAS_OK=%s maxdiff_w=%.4e "
                  "bin_bias[:3]=%s torch_bias[:3]=%s" %
                  (li, name, w_ok, b_ok, dw,
                   None if bias is None else bias[:3], bias_t[:3]), flush=True)
            bad += 1
        else:
            print("conv%d(%s) ok" % (li, name), flush=True)
    print("WEIGHT_PROBE_MISMATCH=%d" % bad, flush=True)
    print("WEIGHT_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
