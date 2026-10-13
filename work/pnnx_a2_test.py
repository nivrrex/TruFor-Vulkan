#!/usr/bin/env python3
"""A2 gate: ncnn(CPU, fp16 off) vs ORT-CPU fp32 on identical inputs.

Judgement criteria (review plan): map rel_L2 <= 1e-3, conf abs <= 1e-2,
det abs <= 2e-2 (det logit scale, see prior torch<->ONNX tables).
"""
import os
import sys

import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ncnn_a1")
REFDIR = os.path.join(HERE, "ncnn_a2")
CASES = sys.argv[2:] or ["512x512", "384x512"]


def main():
    n = ncnn.Net()
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    n.opt.use_bf16_storage = False
    n.opt.use_vulkan_compute = False
    if n.load_param(os.path.join(MODELDIR, "trufor.ts.ncnn.param")) != 0:
        print("load_param FAILED"); return 1
    if n.load_model(os.path.join(MODELDIR, "trufor.ts.ncnn.bin")) != 0:
        print("load_model FAILED"); return 1
    in_name = list(n.input_names())[0]
    out_names = list(n.output_names())
    # pnnx names outputs out0/out1/... in ExportWrapper order (out, conf, det)
    ref_names = ["out", "conf", "det"]

    worst = None
    for tag in CASES:
        d = np.load(os.path.join(REFDIR, "ref_%s.npz" % tag))
        blob = d["input"]
        h, w = blob.shape[1], blob.shape[2]
        m = ncnn.Mat(blob)
        ex = n.create_extractor()
        ex.input(in_name, m)
        print("== %s ==" % tag, flush=True)
        for i, oname in enumerate(out_names):
            rname = ref_names[i] if i < len(ref_names) else oname
            ret, out = ex.extract(oname)
            if ret != 0:
                print("%s(%s) extract FAILED rc=%d" % (oname, rname, ret)); worst = (tag, rname, None); continue
            got = np.array(out).astype("float64")  # Mat -> numpy, layout follows dims
            ref = np.asarray(d[rname], dtype="float64")
            if got.shape != ref.shape and got.size > ref.size and got.size % ref.size == 0:
                # ncnn channel packing (NC1HW1->NC4HW4) appends filler planes; real data is first
                got = got.ravel()[:ref.size].reshape(ref.shape)
            if got.shape != ref.shape and got.size == ref.size:
                got = got.reshape(ref.shape)
            if got.shape != ref.shape:
                print("%s SHAPE MISMATCH ncnn=%s ref=%s" % (rname, got.shape, ref.shape))
                worst = (tag, rname, "shape"); continue
            # gate criteria: out map rel_L2<=1e-3; conf map max_abs<=1e-2; det logit |d|<=2e-2
            if rname == "out":
                rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))
                metric = ("rel_L2=%.3e (<=1e-3)" % rel, rel <= 1e-3)
            else:
                mx = float(np.abs(got - ref).max())
                thr = 1e-2 if rname == "conf" else 2e-2
                metric = ("max_abs=%.3e (<=%g)" % (mx, thr), mx <= thr)
            print("%s: %s -> %s" % (rname, metric[0], "OK" if metric[1] else "FAIL"), flush=True)
            if not metric[1] and worst is None:
                worst = (tag, rname, metric[0])

    if worst is None:
        print("A2_GATE=PASS")
    else:
        print("A2_GATE=FAIL %s" % (worst,))
    print("A2_TEST_DONE", flush=True)
    return 0 if worst is None else 2


if __name__ == "__main__":
    sys.exit(main())
