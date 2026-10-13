#!/usr/bin/env python3
"""Locate the first op where ncnn-Vulkan diverges from ncnn-CPU.

Modes:
  --side cpu|gpu --size HxW --out FILE   sweep all blobs, write stats TSV
  --side dump --blob N --size HxW --device D --out FILE.npy
  --side cmp A.tsv B.tsv                 print first divergent blobs + op info

Stats are float64 (mean, std, L2-norm) per blob; input is RandomState(0).
Marker GPU_BLOB_DIFF_DONE.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM = os.path.join(HERE, "ncnn_a1", "trufor.ts.ncnn.param")
MBIN = os.path.join(HERE, "ncnn_a1", "trufor.ts.ncnn.bin")


def parse_param(path):
    """blob index str -> (line_no, optype, layer_name, tokens_after_counts)."""
    owners = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for ln, line in enumerate(lines[2:], start=2):
        t = line.split()
        if len(t) < 4:
            continue
        optype, name = t[0], t[1]
        nin, nout = int(t[2]), int(t[3])
        ins = t[4:4 + nin]
        outs = t[4 + nin:4 + nin + nout]
        for o in outs:
            owners[o] = (ln, optype, name, " ".join(t[4:]))
        _ = ins
    return owners


def make_net(side, device):
    import ncnn
    n = ncnn.Net()
    n.opt.use_vulkan_compute = (side == "gpu")
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    for a in ("use_bf16_packed", "use_bf16_storage", "use_bf16_arithmetic"):
        if hasattr(n.opt, a):
            setattr(n.opt, a, False)
    if side == "gpu":
        ncnn.create_gpu_instance()
        n.set_vulkan_device(device)
    assert n.load_param(PARAM) == 0
    assert n.load_model(MBIN) == 0
    return n, ncnn


def rand_input(h, w):
    return np.random.RandomState(0).rand(3, h, w).astype("float32")


def png_input(path):
    import cv2
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    x = img.transpose(2, 0, 1).astype(np.float32) / np.float32(256.0)
    return np.ascontiguousarray(x)


def stats_of(a):
    d = a.astype("float64")
    return float(np.mean(d)), float(np.std(d)), float(np.linalg.norm(d))


def sweep(side, h, w, device, out, lo=1, hi=None, png=None):
    n, ncnn = make_net(side, device)
    x = png_input(png) if png else rand_input(h, w)
    ex = n.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    nb = len(list(n.blobs()))
    hi = nb if hi is None else min(hi, nb)
    with open(out, "a") as f:
        for i in range(lo, hi + 1):
            if i % 200 == 0:
                print("sweep %d/%d" % (i, nb), flush=True)
            ret, m = ex.extract(str(i))
            if ret != 0 or m.empty() or m.dims < 2:
                continue
            a = np.array(m)
            if a.size == 0:
                continue
            mean, std, norm = stats_of(a)
            f.write("%d\t%d\t%d\t%d\t%d\t%.9e\t%.9e\t%.9e\n"
                    % (i, m.dims, m.c, m.h, m.w, mean, std, norm))
            del m, a
    print("sweep done range=%d-%d nb=%d" % (lo, hi, nb), flush=True)
    print("GPU_BLOB_DIFF_DONE", flush=True)


def dump_blobs(blob_list, h, w, device, out, png=None):
    side = "cpu" if device < 0 else "gpu"
    n, ncnn = make_net(side, device)
    x = png_input(png) if png else rand_input(h, w)
    data = {}
    numeric = [b for b in blob_list if b.isdigit()]
    named = [b for b in blob_list if not b.isdigit()]
    if numeric:
        ex = n.create_extractor()
        ex.input("in0", ncnn.Mat(x))
        for b in numeric:
            ret, m = ex.extract(b)
            assert ret == 0 and not m.empty(), "extract %s failed" % b
            data[b] = np.array(m)
        del ex
    for nb_ in named:
        # fresh extractor per named output: sweep-then-final reuse SIGSEGVs
        ex = n.create_extractor()
        ex.input("in0", ncnn.Mat(x))
        ret, m = ex.extract(nb_)
        assert ret == 0 and not m.empty(), "extract %s failed" % nb_
        data[nb_] = np.array(m)
        del ex
    np.savez(out, input=x, **data)
    for k, v in data.items():
        mean, std, norm = stats_of(v)
        print("blob%s shape=%s mean=%.6e std=%.6e norm=%.6e" % (k, v.shape, mean, std, norm))
    print("GPU_BLOB_DIFF_DONE", flush=True)


def cmp_files(pa, pb, rel_thr=1e-3):
    owners = parse_param(PARAM)

    def load(p):
        d = {}
        with open(p) as f:
            for line in f:
                t = line.rstrip("\n").split("\t")
                d[int(t[0])] = dict(
                    dims=int(t[1]), c=int(t[2]), h=int(t[3]), w=int(t[4]),
                    mean=float(t[5]), std=float(t[6]), norm=float(t[7]))
        return d
    A, B = load(pa), load(pb)  # A=cpu B=gpu
    shared = sorted(set(A) & set(B))
    assert len(A) == len(B) == len(shared), \
        "blob sets differ: cpu=%d gpu=%d common=%d" % (len(A), len(B), len(shared))
    diffs = []
    for i in shared:
        a, b = A[i], B[i]
        assert (a["dims"], a["c"], a["h"], a["w"]) == (b["dims"], b["c"], b["h"], b["w"])
        worst = 0.0
        which = ""
        for k in ("std", "norm", "mean"):
            den = max(abs(a[k]), 1e-30)
            v = abs(b[k] - a[k]) / den
            if v > worst:
                worst, which = v, k
        if worst > rel_thr:
            diffs.append((i, worst, which))
    if not diffs:
        print("NO blob exceeds rel diff %.0e -> this size is clean CPU-vs-GPU" % rel_thr)
    owners_first = diffs[:15]
    for i, v, which in owners_first:
        ln, optype, name, rest = owners.get(str(i), ("?", "?", "?", "?"))
        print("blob%-5d rel_%s=%.3e  <- L%-4d %-24s %s" % (i, which, v, ln, optype, rest))
    print("total divergent blobs: %d / %d" % (len(diffs), len(shared)))
    print("GPU_BLOB_DIFF_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["cpu", "gpu", "dump", "cmp"])
    ap.add_argument("--size", default="714x967")
    ap.add_argument("--device", default="1", help="vulkan device for gpu/dump, -1=cpu")
    ap.add_argument("--blob", default="0", help="csv blob ids/names for dump")
    ap.add_argument("--png", default=None, help="real image input for dump/sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--range", default=None, help="lo-hi, fresh pipeline per process")
    ap.add_argument("files", nargs="*", help="two TSVs for cmp mode")
    args = ap.parse_args()
    h, w = (int(v) for v in args.size.split("x"))
    lo, hi = 1, None
    if args.range:
        lo, hi = (int(v) for v in args.range.split("-"))
    if args.side in ("cpu", "gpu"):
        sweep(args.side, h, w, -1 if args.side == "cpu" else int(args.device),
              args.out, lo, hi, args.png)
    elif args.side == "dump":
        dump_blobs([b.strip() for b in args.blob.split(",")], h, w,
                   int(args.device), args.out, args.png)
    else:
        cmp_files(args.files[0], args.files[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
