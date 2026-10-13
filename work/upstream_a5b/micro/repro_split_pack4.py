#!/usr/bin/env python3
"""Minimal repro: ncnn-Vulkan corrupts Split output blobs when channels>=2 and
(H*W)%4 != 0. CPU path and upload roundtrip are clean; both physical GPUs and
both the master build and the pypi wheel agree bit-for-bit.

Run: python repro_split_pack4.py [device]   (default 1 = Tesla P4)
Marker: REPRO_SPLIT_PACK4_DONE
"""
import os
import sys

import numpy as np
import ncnn

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def make_net():
    n = ncnn.Net()
    n.opt.use_vulkan_compute = True
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    assert n.load_param(os.path.join(HERE, "split_roundtrip.param")) == 0
    return n


def extract_array(net, name, x):
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    ret, m = ex.extract(name)
    assert ret == 0
    return np.array(m)


def main():
    ncnn.create_gpu_instance()
    net = make_net()
    net.set_vulkan_device(DEVICE)

    print("ncnn version:", ncnn.__version__)
    print("device:", DEVICE, "=", ncnn.get_gpu_info(DEVICE).device_name())

    # 1) ordered values, c=2, 3x3 -> shows exact corruption shape
    x = np.ascontiguousarray(
        np.arange(2 * 3 * 3, dtype="float32").reshape(2, 3, 3) + 1.0)
    a_in0 = extract_array(net, "in0", x)
    a_1 = extract_array(net, "1", x)
    print("\n[1] c=2 3x3, input = 1..18")
    print("    in0  :", np.asarray(a_in0, dtype="float32").ravel().tolist())
    print("    split:", np.asarray(a_1, dtype="float32").ravel().tolist())
    ok_in0 = np.array_equal(np.asarray(a_in0, dtype="float32").ravel(),
                            x.ravel())
    bad_split = not np.array_equal(np.asarray(a_1, dtype="float32").ravel(),
                                   x.ravel())
    print("    in0 clean: %s ; split corrupted: %s" % (ok_in0, bad_split))

    # 2) channel count x size grid
    print("\n[2] grid: c in 1..8, sizes; CORRUPT iff c>=2 and (h*w)%4!=0")
    sizes = [(1, 1), (2, 1), (1, 2), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2),
             (3, 3), (4, 4), (2, 6), (6, 2), (3, 5), (5, 3), (4, 1),
             (967, 714), (967, 712)]
    rs = np.random.RandomState(0)
    mism = 0
    for c in range(1, 9):
        row = []
        for (h, w) in sizes:
            xg = np.ascontiguousarray(
                rs.rand(c, h, w).astype("float32"))
            g = np.asarray(extract_array(net, "1", xg), dtype="float32")
            bad = not np.array_equal(g, xg)
            expect = c >= 2 and (h * w) % 4 != 0
            row.append("BAD" if bad else "ok")
            if bad != expect:
                mism += 1
                row[-1] = "RULE-MISMATCH(%s)" % ("BAD" if expect else "ok")
        print("    c=%d: %s" % (c, " ".join("%-4s" % v for v in row)))
    print("    rule mismatches:", mism)

    # 3) the two 967-contiguous sizes side by side (h*w%4 boundary)
    print("\n[3] photo-shaped contrast, c=3")
    for (h, w) in [(967, 714), (967, 712)]:
        xp = np.ascontiguousarray(rs.rand(3, h, w).astype("float32"))
        g = np.asarray(extract_array(net, "1", xp), dtype="float32")
        d = np.abs(g - xp)
        print("    %dx%d (hw%%4=%d): max_abs=%.3e frac>1e-6=%.3f" %
              (h, w, (h * w) % 4, float(d.max()),
               float((d > 1e-6).mean())))

    # 4) CPU path clean at the worst size
    print("\n[4] same net on CPU (use_vulkan_compute=False), 3x3 c=2")
    nc = ncnn.Net()
    nc.opt.use_vulkan_compute = False
    nc.opt.use_fp16_packed = False
    nc.opt.use_fp16_storage = False
    nc.opt.use_fp16_arithmetic = False
    assert nc.load_param(os.path.join(HERE, "split_roundtrip.param")) == 0
    exc = nc.create_extractor()
    exc.input("in0", ncnn.Mat(x))
    ret, mc = exc.extract("1")
    assert ret == 0
    print("    CPU split output clean:",
          np.array_equal(np.asarray(mc, dtype="float32").ravel(), x.ravel()))

    ncnn.destroy_gpu_instance()
    print("\nREPRO_SPLIT_PACK4_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
