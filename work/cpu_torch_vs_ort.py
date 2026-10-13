#!/usr/bin/env python3
"""Same-session ABAB bench: onnxruntime-CPU vs torch-CPU.

Motivation: the torch/timm/yacs stack is the heaviest dependency a consumer of
this ONNX would carry. If ORT-CPU is not materially slower than torch-CPU on the
same graph, a CPU-fallback deployment can drop torch entirely and route both
Vulkan and CPU-fallback through ONNX.

Fairness:
  * one process, one env
  * same 512x512 input array shape, same numpy/PIL version
  * torch path uses this project's own truFor-src model (the exact weights the
    ONNX was exported from), fed x/256; ORT path uses this project's export with
    the dynamo-inline ONNX, fed x/255. The absolute pixel semantics differ by
    0.4% but the compute graph is the same, which is all a latency bench needs.
  * threads=4 in ORT (intra_op); torch pinned via torch.set_num_threads(4) so
    both stacks use 4 cores.
  * median of 15 reps per arm per image, after 3 warmups per arm per image,
    ABAB interleaved at the image-loop level (three images cycled).
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True,
                    help="path to trufor_512x512_dynamo_inline.onnx")
    ap.add_argument("--pth", required=True,
                    help="path to trufor.pth.tar")
    ap.add_argument("--images", nargs="+", required=True,
                    help="absolute paths of images (any size; resized to 512x512)")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="-")
    return ap.parse_args()


def load_ort(onnx_path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def run(x_255):
        sess.run(None, {iname: x_255})

    return run


def load_torch(pth_path, threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    from export_onnx_dynamo import build_model, swap_adaptive_pool, ExportWrapper
    model = build_model(pth_path, "trufor_ph3")
    swap_adaptive_pool(model)
    wrapper = ExportWrapper(model)
    wrapper.eval()

    def run(x_torch):
        wrapper(x_torch)

    return run


def load_images(paths):
    """Return list of np.uint8 arrays at 512x512."""
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((512, 512), Image.BILINEAR)
        out.append(np.asarray(img, np.uint8))
    return out


def arm_median(runner, tensors, reps):
    ts = []
    for x in tensors:
        runner(x)
    ts = []
    for _ in range(reps):
        for x in tensors:
            t0 = time.perf_counter()
            runner(x)
            ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    q1 = ts[len(ts) // 4]
    q3 = ts[3 * len(ts) // 4]
    return med * 1e3, (q3 - q1) * 1e3


def main():
    a = parse_args()
    ort_run = load_ort(a.onnx, a.threads)
    torch_run = load_torch(a.pth, a.threads)

    imgs = load_images(a.images)
    x_255 = [(img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
             .astype(np.float32) for img in imgs]
    import torch
    x_256_t = [(torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)[None])
                / 256.0) for img in imgs]

    # 3 warmups each, no timing
    for _ in range(3):
        for x in x_255:
            ort_run(x)
        for x in x_256_t:
            torch_run(x)

    # ABAB at the reps-loop level: interleave per cycle to fight drift.
    ort_ts = []
    torch_ts = []
    for _ in range(a.reps):
        for x in x_255:
            t0 = time.perf_counter(); ort_run(x)
            ort_ts.append(time.perf_counter() - t0)
        for x in x_256_t:
            t0 = time.perf_counter(); torch_run(x)
            torch_ts.append(time.perf_counter() - t0)

    def summarize(ts):
        ts = sorted(ts)
        return (statistics.median(ts) * 1e3,
                (ts[3 * len(ts) // 4] - ts[len(ts) // 4]) * 1e3,
                ts[0] * 1e3, ts[-1] * 1e3)

    o = summarize(ort_ts)
    t = summarize(torch_ts)
    lines = [
        "threads=%d  reps=%d  images=%d  per-arm samples=%d"
        % (a.threads, a.reps, len(imgs), a.reps * len(imgs)),
        "ORT   CPU: median %7.1f ms  IQR %5.1f  min %7.1f  max %7.1f" % o,
        "Torch CPU: median %7.1f ms  IQR %5.1f  min %7.1f  max %7.1f" % t,
        "ORT / Torch = %.3f  (ORT 1.000 means parity; < 1.0 means ORT faster)"
        % (o[0] / t[0]),
    ]
    text = "\n".join(lines)
    if a.out == "-":
        print(text)
    else:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
