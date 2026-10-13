#!/usr/bin/env python3
"""A3 gate: ncnn(Vulkan) vs ORT-CPU fp32 reference, per device and fp16 mode.

Usage: pnnx_a3_vulkan_test.py --device 1 [--fp16] [--sizes 512x512 384x512]
Same judgement criteria as A2 (out rel_L2<=1e-3, conf max_abs<=1e-2,
det <=2e-2). Run once under system Mesa and once under vendor driver to
compare. Marker: A3_GATE=PASS/FAIL.
"""
import argparse
import os
import sys

import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "ncnn_a2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeldir", default=os.path.join(HERE, "ncnn_a1"))
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--sizes", nargs="*", default=["512x512", "384x512"])
    args = ap.parse_args()

    ncnn.create_gpu_instance()
    gi = ncnn.get_gpu_info(args.device)
    print("device %d: %s" % (args.device, gi.device_name()), flush=True)

    n = ncnn.Net()
    n.opt.use_vulkan_compute = True
    # ncnn Vulkan winograd path is nondeterministic on this graph (2026-10-02:
    # fresh processes produce different outputs with defaults; winograd-off is
    # bit-stable on P4 and RX460 at 192^2 and 512^2)
    n.opt.use_winograd_convolution = False
    n.opt.use_winograd23_convolution = False
    n.opt.use_winograd43_convolution = False
    n.opt.use_winograd63_convolution = False
    n.opt.use_fp16_packed = args.fp16
    n.opt.use_fp16_storage = args.fp16
    n.opt.use_fp16_arithmetic = args.fp16
    n.set_vulkan_device(args.device)
    if n.load_param(os.path.join(args.modeldir, "trufor.ts.ncnn.param")) != 0:
        print("load_param FAILED"); return 1
    if n.load_model(os.path.join(args.modeldir, "trufor.ts.ncnn.bin")) != 0:
        print("load_model FAILED"); return 1
    in_name = list(n.input_names())[0]
    out_names = list(n.output_names())
    ref_names = ["out", "conf", "det"]
    failed = False
    for tag in args.sizes:
        d = np.load(os.path.join(REFDIR, "ref_%s.npz" % tag))
        blob = d["input"]
        m = ncnn.Mat(blob)
        ex = n.create_extractor()
        ex.input(in_name, m)
        print("== %s ==" % tag, flush=True)
        for i, oname in enumerate(out_names):
            rname = ref_names[i] if i < len(ref_names) else oname
            ret, out = ex.extract(oname)
            if ret != 0:
                print("%s(%s) extract FAILED rc=%d" % (oname, rname, ret))
                failed = True
                continue
            got = np.array(out).astype("float64")
            ref = np.asarray(d[rname], dtype="float64")
            if got.shape != ref.shape and got.size > ref.size and got.size % ref.size == 0:
                got = got.ravel()[:ref.size].reshape(ref.shape)
            if got.shape != ref.shape and got.size == ref.size:
                got = got.reshape(ref.shape)
            if got.shape != ref.shape:
                print("%s SHAPE MISMATCH ncnn=%s ref=%s" % (rname, got.shape, ref.shape))
                failed = True
                continue
            if rname == "out":
                rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
                ok = rel <= 1e-3
                print("out rel_L2=%.3e %s" % (rel, "OK" if ok else "FAIL"), flush=True)
            elif rname == "conf":
                ma = float(np.max(np.abs(got - ref)))
                ok = ma <= 1e-2
                print("conf max_abs=%.3e %s" % (ma, "OK" if ok else "FAIL"), flush=True)
            else:
                ma = float(np.max(np.abs(got - ref)))
                ok = ma <= 2e-2
                print("det max_abs=%.3e %s" % (ma, "OK" if ok else "FAIL"), flush=True)
            if not ok:
                failed = True
    print("A3_GATE=%s" % ("FAIL" if failed else "PASS"), flush=True)
    ncnn.destroy_gpu_instance()
    return 0


if __name__ == "__main__":
    sys.exit(main())
