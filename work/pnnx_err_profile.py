#!/usr/bin/env python3
"""Spatial profile of ncnn-vs-ORT conf/out error. Marker ERRPROFILE_DONE."""
import argparse
import os
import sys

import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "ncnn_a2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--tag", default="512x512")
    args = ap.parse_args()
    h, w = (int(v) for v in args.tag.split("x"))

    ncnn.create_gpu_instance()
    n = ncnn.Net()
    n.opt.use_vulkan_compute = True
    n.set_vulkan_device(args.device)
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    n.opt.use_winograd_convolution = False
    n.opt.use_winograd23_convolution = False
    n.opt.use_winograd43_convolution = False
    n.opt.use_winograd63_convolution = False
    md = os.path.join(HERE, "ncnn_a1")
    assert n.load_param(os.path.join(md, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(md, "trufor.ts.ncnn.bin")) == 0
    in_name = list(n.input_names())[0]
    out_names = list(n.output_names())

    d = np.load(os.path.join(REFDIR, "ref_%s.npz" % args.tag))
    blob = d["input"]
    ex = n.create_extractor()
    ex.input(in_name, ncnn.Mat(blob))
    for nm, key in zip(out_names, ("out", "conf", "det")):
        ret, m = ex.extract(nm)
        assert ret == 0
        got = np.array(m).astype("float64")
        ref = d[key].astype("float64")
        got = got.ravel()[:ref.size].reshape(ref.shape)
        while got.ndim > 3:
            got = got[0]
            ref = ref[0]
        err = np.abs(got - ref)
        if err.ndim == 2:
            print("%s: max=%.4e at %s" % (key, err.max(),
                                          np.unravel_index(err.argmax(), err.shape)))
            # distance from image border of the argmax
            iy, ix = np.unravel_index(err.argmax(), err.shape)
            print("%s: border_dist=%d px" % (key, min(iy, ix, h - 1 - iy, w - 1 - ix)))
            inner = err[16:-16, 16:-16]
            band = err - inner.min()
            print("%s: interior(16px crop) max=%.4e mean=%.3e" %
                  (key, inner.max(), inner.mean()))
            print("%s: count>1e-2: total=%d interior=%d" %
                  (key, int((err > 1e-2).sum()), int((inner > 1e-2).sum())))
        elif err.ndim == 3:
            for c in range(err.shape[0]):
                e = err[c]
                print("%s[%d]: max=%.4e interior16=%.4e count>1e-3=%d" %
                      (key, c, e.max(), e[16:-16, 16:-16].max(),
                       int((e > 1e-3).sum())))
        else:
            print("%s: abs=%.4e" % (key, err.max()))
    print("ERRPROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
