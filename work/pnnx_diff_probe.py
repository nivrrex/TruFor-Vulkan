#!/usr/bin/env python3
"""Locate first divergent subgraph: torch-eager module outputs vs ncnn blobs.

Match by (dims,c,h,w) then stats (mean,std). Prints per-anchor status and
A1_DIFF_PROBE_DONE. Usage: pnnx_diff_probe.py [--device 1] [--size 512x512]
"""
import argparse
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(ROOT))

# same timm namespace-package shim as pnnx_a1_convert.py (torchvision.datasets
# is absent; torch 2.14.1 has no matching torchvision build)
SITE = os.path.join(ROOT, "venv", "lib", "python3.11", "site-packages")
for _name, _path in (("timm", os.path.join(SITE, "timm")),
                     ("timm.models", os.path.join(SITE, "timm", "models"))):
    _mod = types.ModuleType(_name)
    _mod.__path__ = [_path]
    sys.modules[_name] = _mod

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "export_onnx_dynamo", os.path.join(ROOT, "scripts", "export_onnx_dynamo.py"))
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

# apply the same pnnx convert patches so eager == traced semantics
conv_spec = importlib.util.spec_from_file_location(
    "pnnx_a1_convert_patches", os.path.join(HERE, "pnnx_a1_convert.py"))
_conv = importlib.util.module_from_spec(conv_spec)
conv_spec.loader.exec_module(_conv)


FINALISTS = False


def collect_torch(blob):
    import torch
    model = exp.build_model(os.path.join(ROOT, "weights", "trufor.pth.tar"))
    model.eval()
    exp.swap_adaptive_pool(model)
    wrapper = exp.ExportWrapper(model)
    wrapper.eval()
    anchors = []

    def hook(mod, inp, out):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for i, o in enumerate(outs):
            if isinstance(o, torch.Tensor) and o.dim() >= 2:
                t = o.detach().double()
                anchors.append((type(mod).__name__ + ":" + str(len(anchors)),
                                tuple(t.shape), float(t.mean()), float(t.std())))

    handles = [m.register_forward_hook(hook) for m in wrapper.modules()]
    with torch.no_grad():
        finals = wrapper(torch.from_numpy(blob).float())
    for h in handles:
        h.remove()
    fin = []
    for o in finals:
        t = o.detach().double()
        fin.append((tuple(t.shape), float(t.mean()), float(t.std())))
    return anchors, fin


def collect_ncnn(blob, device, param_dir):
    import ncnn
    n = ncnn.Net()
    n.opt.use_vulkan_compute = device >= 0
    if device >= 0:
        ncnn.create_gpu_instance()
        n.set_vulkan_device(device)
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    assert n.load_param(os.path.join(param_dir, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(param_dir, "trufor.ts.ncnn.bin")) == 0
    if FINALISTS:
        # finals-only mode: no sweep at all in this process
        ex = n.create_extractor()
        ex.input("in0", ncnn.Mat(blob))
        finals = []
        for nm in ("out0", "out1", "out2"):
            print("finals extract %s" % nm, flush=True)
            ret, m = ex.extract(nm)
            a = np.array(m).astype("float64")
            finals.append((nm, m.dims, m.c, m.h, m.w,
                           float(a.mean()), float(a.std())))
        print("finals done", flush=True)
        if device >= 0:
            ncnn.destroy_gpu_instance()
        return [], finals
    ex = n.create_extractor()
    ex.input("in0", ncnn.Mat(blob))
    blobs = []
    nb = len(list(n.blobs()))
    for i in range(1, nb + 1):
        if i % 100 == 0:
            print("sweep %d/%d" % (i, nb), flush=True)
        ret, m = ex.extract(str(i))
        if ret != 0 or m.empty() or m.dims < 2:
            continue
        a = np.array(m).astype("float64")
        if a.size == 0:
            continue
        blobs.append((i, m.dims, m.c, m.h, m.w,
                      float(a.mean()), float(a.std())))
    print("sweep done nb=%d kept=%d" % (nb, len(blobs)), flush=True)
    finals = []
    ex = n.create_extractor()  # fresh extractor: reusing the all-blob sweep
    ex.input("in0", ncnn.Mat(blob))  # pipeline segfaults on final extract
    for i, nm in enumerate(["out0", "out1", "out2"]):
        print("finals extract %s" % nm, flush=True)
        ret, m = ex.extract(nm)
        a = np.array(m).astype("float64")
        finals.append((nm, m.dims, m.c, m.h, m.w, float(a.mean()), float(a.std())))
    print("finals done", flush=True)
    if device >= 0:
        ncnn.destroy_gpu_instance()
    return blobs, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="1")
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--paramdir", default=os.path.join(HERE, "ncnn_a1"))
    ap.add_argument("--finalists", action="store_true",
                    help="extract only out0/out1/out2 (no sweep)")
    args = ap.parse_args()
    global FINALISTS
    FINALISTS = args.finalists
    device = -1 if args.device == "cpu" else int(args.device)
    h, w = (int(v) for v in args.size.split("x"))
    blob = np.random.RandomState(0).rand(1, 3, h, w).astype("float32")

    ta, tf = collect_torch(blob)
    print("torch anchors: %d" % len(ta), flush=True)
    nb, nf = collect_ncnn(blob[0], device, args.paramdir)
    print("ncnn blobs (dims>=2): %d" % len(nb), flush=True)

    index = {}
    for rec in nb:
        _, dims, c, bh, bw, _, _ = rec
        index.setdefault((dims, bh, bw, c), []).append(rec)
        # also try transposed/reshaped hypotheses: skip, shape match is primary

    def find(shape, mean, std):
        if len(shape) == 3:
            _, nh, nc_ = shape
            key = (3, nc_, nh, 1)
            cands = index.get(key, [])
        elif len(shape) == 4:
            _, nc_, nh, nw = shape
            cands = index.get((4, nc_, nh, nw), []) + index.get((3, nc_, nh, nw), [])
        elif len(shape) == 2:
            nc_, nh = shape
            cands = index.get((2, nc_, nh, 1), []) if False else []
            for k, v in index.items():
                if k[0] == 2 and (k[2], k[1]) == (nc_, nh) or k[0] == 2 and (k[1], k[2]) == (nc_, nh):
                    cands.extend(v)
        else:
            cands = []
        best = None
        for cd in cands:
            if std and abs(cd[6] - std) <= max(1e-3, 0.02 * std):
                return cd, True
            if best is None or abs(cd[6] - std) < abs(best[6] - std):
                best = cd
        return (best, False) if best else (None, False)

    first_bad = None
    for (name, shape, mean, std) in ta:
        rec, ok = find(shape, mean, std)
        status = "OK" if ok else ("NEAR" if rec else "NOMATCH")
        if status != "OK" and first_bad is None:
            first_bad = (name, shape, mean, std)
        if status != "OK":
            print("%-28s shape=%-18s std=%.5f -> %s %s" %
                  (name, str(shape), std, status,
                   ("blob%d(dims=%d,c=%d,h=%d,w=%d,std=%.5f)" %
                    (rec[0], rec[1], rec[3], rec[2], rec[4], rec[6])) if rec else ""),
                  flush=True)
    if first_bad is not None:
        print("FIRST_DIVERGENT_ANCHOR:", first_bad, flush=True)
    print("finals torch:", tf, flush=True)
    print("finals ncnn :", nf, flush=True)
    print("A1_DIFF_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
