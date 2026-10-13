#!/usr/bin/env python3
"""A1 gate test: does the converted ncnn model actually run dynamic H/W?

Loads work/ncnn_a1/trufor.ts.{param,bin} with every fp16 flag off (CPU),
runs three input geometries -- 256x256, 384x512, 1000x1500 (the last two
are not multiples-of-32 square) -- and checks that the primary output map
carries the input's spatial size. Constant-baked Reshape/interpolate sizes
from the TorchScript trace show up here as wrong shapes or extract errors.
"""
import os
import sys
import time

import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(HERE, "ncnn_a1")

CASES = [(256, 256), (384, 512), (1000, 1500)]  # (H, W)


def main():
    n = ncnn.Net()
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    n.opt.use_bf16_storage = False
    n.opt.use_vulkan_compute = False

    rc = n.load_param(os.path.join(MODELDIR, "trufor.ts.ncnn.param"))
    if rc != 0:
        print("load_param FAILED rc=%d" % rc)
        return 1
    rc = n.load_model(os.path.join(MODELDIR, "trufor.ts.ncnn.bin"))
    if rc != 0:
        print("load_model FAILED rc=%d" % rc)
        return 1
    print("loaded: %d layers, inputs=%s outputs=%s"
          % (len(n.layers()), list(n.input_names()), list(n.output_names())), flush=True)

    in_name = list(n.input_names())[0]
    out_names = list(n.output_names())
    n_outs = len(out_names)
    fails = 0

    for (h, w) in CASES:
        rng = np.random.RandomState(0)
        blob = rng.rand(3, h, w).astype("float32")
        m = ncnn.Mat(blob)
        ex = n.create_extractor()
        ex.input(in_name, m)
        t0 = time.perf_counter()
        shapes = []
        ok = True
        for oname in out_names:
            ret, out = ex.extract(oname)
            if ret != 0:
                print("%dx%d: extract %s FAILED rc=%d" % (h, w, oname, ret), flush=True)
                ok = False
                break
            shapes.append((oname, out.dims, out.w, out.h, out.c))
        dt = time.perf_counter() - t0
        for nm, dims, ow, oh, oc in shapes:
            print("%4dx%-4d %s: dims=%d w=%d h=%d c=%d" % (h, w, nm, dims, ow, oh, oc), flush=True)
        if ok:
            primary = shapes[0]
            if primary[2] != w or primary[3] != h:
                print("GATE FAIL: primary output spatial (%d,%d) != input (%d,%d)"
                      % (primary[2], primary[3], w, h), flush=True)
                fails += 1
            else:
                print("GATE OK %dx%d (%.1fs)" % (h, w, dt), flush=True)
        else:
            fails += 1

    print("A1_TEST_DONE fails=%d" % fails, flush=True)
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
