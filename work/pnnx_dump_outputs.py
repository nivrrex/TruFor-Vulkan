#!/usr/bin/env python3
"""Dump ncnn outputs (CPU or Vulkan) for a given input npz. Marker DUMP_DONE."""
import argparse
import os
import sys

import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ref", required=True, help="npz with 'input'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vulkan = args.device != "cpu"
    if vulkan:
        ncnn.create_gpu_instance()
    n = ncnn.Net()
    n.opt.use_vulkan_compute = vulkan
    if vulkan:
        n.set_vulkan_device(int(args.device))
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    md = os.path.join(HERE, "ncnn_a1")
    assert n.load_param(os.path.join(md, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(md, "trufor.ts.ncnn.bin")) == 0
    in_name = list(n.input_names())[0]
    out_names = list(n.output_names())
    d = np.load(args.ref)
    arr = d["input"]
    d.close()
    ex = n.create_extractor()
    # ncnn.Mat(numpy) 是零拷贝包装且 Vulkan 上传延迟到 extract：数组必须存活至全部抽取完成
    ex.input(in_name, ncnn.Mat(arr))
    res = {}
    for nm, key in zip(out_names, ("out", "conf", "det")):
        ret, m = ex.extract(nm)
        assert ret == 0
        res[key] = np.array(m).astype("float32")
    del arr
    np.savez(args.out, **res)
    print("DUMP_DONE %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
