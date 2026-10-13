#!/usr/bin/env python3
"""Gemm-section verification split into crash-isolated stages.

 --stage torch    eager reference (decode_head linear_c1..4 in/out + weights)
 --stage ncnn     single-blob extract on vulkan device, os._exit (no teardown)
 --stage compare  offline numpy: bin W vs torch W, blob2533 vs seq_in,
                  manual gemm vs blob2534
Markers: GEMM_TORCH_DONE / GEMM_NCNN_DONE / GEMM_COMPARE_DONE
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

REF192 = "/tmp/gemm_ref192.npz"


def _load_exp():
    spec = importlib.util.spec_from_file_location(
        "export_onnx_dynamo", os.path.join(ROOT, "scripts", "export_onnx_dynamo.py"))
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    return exp


def stage_torch(h, w):
    import torch
    exp = _load_exp()
    model = exp.build_model(os.path.join(ROOT, "weights", "trufor.pth.tar"))
    model.eval()
    exp.swap_adaptive_pool(model)
    wrapper = exp.ExportWrapper(model)
    wrapper.eval()

    cap = {}

    def mkhook(key):
        def hook(mod, inp, out):
            if key not in cap:
                cap[key] = (inp[0].detach().double().cpu().numpy(),
                            out.detach().double().cpu().numpy())
        return hook

    targets = {}
    for name, mod in wrapper.named_modules():
        for head in ("decode_head", "decode_head_conf"):
            for i in (1, 2, 3, 4):
                if name.endswith("%s.linear_c%d" % (head, i)):
                    targets[name] = mod
            if name.endswith("%s.linear_fuse" % head) or \
               name.endswith("%s.linear_pred" % head):
                targets[name] = mod
    assert len(targets) == 12, sorted(targets)
    handles = [mod.register_forward_hook(mkhook(name)) for name, mod in targets.items()]

    blob = np.random.RandomState(0).rand(1, 3, h, w).astype("float32")
    with torch.no_grad():
        wrapper(torch.from_numpy(blob))
    for x in handles:
        x.remove()

    dec = None
    for name, mod in wrapper.named_modules():
        if name.endswith("decode_head"):
            dec = mod
    dmod = None
    for name, mod in wrapper.named_modules():
        if name.endswith("decode_head"):
            dmod = mod
    sd = {k: v for k, v in dmod.state_dict().items()}
    out = {"input": blob}
    for name in targets:
        key = name.split("model.")[-1].replace(".", "_")
        xin, xout = cap[name]
        out["in_" + key] = xin.astype("float32")
        out["out_" + key] = xout.astype("float32")
    for k, v in sd.items():
        out["sd_" + k] = v.detach().double().cpu().numpy().astype("float32")
    # decode_head input = list of 4 multiscale feats; grab c4 via pre-hook
    got = {}

    def pre(mod, inp):
        got["feats"] = [t.detach().double().cpu().numpy() for t in inp[0]]
    hh = dmod.register_forward_pre_hook(pre)
    with torch.no_grad():
        wrapper(torch.from_numpy(blob))
    hh.remove()
    for i, f in enumerate(got["feats"]):
        out["feat%d" % i] = f.astype("float32")
    np.savez(REF192, **out)
    print("saved %s keys=%d" % (REF192, len(out)), flush=True)
    for i, f in enumerate(got["feats"]):
        print("feat%d shape=%s" % (i, f.shape), flush=True)
    print("GEMM_TORCH_DONE", flush=True)
    return 0


def stage_ncnn(device, blob_id, ref):
    import ncnn
    d = np.load(ref)
    inp = d["input"][0]
    ncnn.create_gpu_instance()
    n = ncnn.Net()
    n.opt.use_vulkan_compute = True
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    n.set_vulkan_device(device)
    md = os.path.join(HERE, "ncnn_a1")
    assert n.load_param(os.path.join(md, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(md, "trufor.ts.ncnn.bin")) == 0
    ex = n.create_extractor()
    ex.input("in0", ncnn.Mat(inp))
    ret, m = ex.extract(blob_id)
    assert ret == 0, blob_id
    a = np.array(m)
    print("blob %s dims=%d c=%d h=%d w=%d dtype=%s" %
          (blob_id, m.dims, m.c, m.h, m.w, a.dtype), flush=True)
    ad = a.astype("float64")
    print("blob %s mean=%.6f std=%.6f" % (blob_id, ad.mean(), ad.std()), flush=True)
    np.savez("/tmp/gemm_blob%s.npz" % blob_id, a=a)
    print("GEMM_NCNN_DONE", flush=True)
    sys.stdout.flush()
    os._exit(0)


# bin offsets measured by pnnx_manual_conv_probe (fp32 bin, row-major out,in)
OFF4 = dict(W=262164908, B=263213488)


def stage_compare(blob_dir):
    r = np.load(REF192)
    binpath = os.path.join(HERE, "ncnn_a1", "trufor.ts.ncnn.bin")
    f = open(binpath, "rb")
    f.seek(OFF4["W"])
    W_bin = np.frombuffer(f.read(512 * 512 * 4), dtype=np.float32).reshape(512, 512)
    f.seek(OFF4["B"])
    B_bin = np.frombuffer(f.read(512 * 4), dtype=np.float32)
    f.close()

    # torch linear_c4 weight lives in state_dict under key 'linear_c4.*'
    wk = [k for k in r.files if k.startswith("sd_linear_c4")]
    print("ref sd_linear_c4 keys:", wk, flush=True)
    W_t = None
    B_t = None
    for k in wk:
        arr = r[k]
        if arr.shape == (512, 512):
            W_t = arr.astype("float64")
        elif arr.size == 512:
            B_t = arr.astype("float64")
    if W_t is not None:
        same = np.array_equal(W_t.astype("float32"), W_bin)
        rel = np.linalg.norm(W_bin - W_t) / max(np.linalg.norm(W_t), 1e-12)
        print("W bin-vs-torch exact=%s rel=%.3e" % (same, rel), flush=True)
        relb = np.linalg.norm(B_bin - B_t) / max(np.linalg.norm(B_t), 1e-12)
        print("B bin-vs-torch rel=%.3e" % relb, flush=True)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))

    seq_in = r["in_linear_c4"]  # (1,36,512)
    lin_out = r["out_linear_c4"]
    if os.path.exists("/tmp/gemm_blob2533.npz"):
        g = np.load("/tmp/gemm_blob2533.npz")["a"].astype("float64")
        flat = g.ravel()
        tgt = seq_in.astype("float64").ravel()
        cands = {"flat": tgt, "transpose": seq_in.astype("float64").transpose(0, 2, 1).ravel()}
        for name, tv in cands.items():
            if flat.size == tv.size:
                print("2533 vs seq_in(%s) rel=%.3e" % (name, rel(flat, tv)), flush=True)
            else:
                print("2533 vs seq_in(%s) size %d vs %d" % (name, flat.size, tv.size), flush=True)
    if os.path.exists("/tmp/gemm_blob2534.npz"):
        g4 = np.load("/tmp/gemm_blob2534.npz")["a"].astype("float64")
        man = seq_in.astype("float64").reshape(-1, 512) @ W_bin.astype("float64").T + B_bin
        outs = lin_out.astype("float64").reshape(-1, 512)
        print("manual-vs-torch rel=%.3e" % rel(man, outs), flush=True)
        if g4.size == man.size:
            print("2534-vs-manual rel=%.3e" % rel(g4.ravel(), man.ravel()), flush=True)
            t4 = outs.ravel()
            print("2534-vs-torch rel=%.3e" % rel(g4.ravel(), t4), flush=True)
        else:
            print("2534 size %d vs %d" % (g4.size, man.size), flush=True)
    print("GEMM_COMPARE_DONE", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["torch", "ncnn", "compare"])
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--size", default="192x192")
    ap.add_argument("--blob", default="2533")
    ap.add_argument("--ref", default=REF192)
    args = ap.parse_args()
    if args.stage == "torch":
        h, w = (int(v) for v in args.size.split("x"))
        return stage_torch(h, w)
    if args.stage == "ncnn":
        stage_ncnn(args.device, args.blob, args.ref)
        return 0
    return stage_compare(None)


if __name__ == "__main__":
    sys.exit(main())
