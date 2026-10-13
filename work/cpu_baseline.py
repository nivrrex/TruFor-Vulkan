#!/usr/bin/env python3
"""ORT CPU baseline: median + IQR over N reps of the production ONNX.

Keeps the whole run (host->session->host) inside the timed region so the CPU
pool is drained, matching how run_trufor times the GPU side.
"""
import os, sys, time, statistics, argparse
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=os.path.join(os.environ.get("ROOT", os.getcwd()),
                "reports", "onnx-dynamo", "trufor_512x512_dynamo_inline.onnx"))
ap.add_argument("image")
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = a.threads
so.inter_op_num_threads = 1
sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
x = np.asarray(Image.open(a.image).convert("RGB").resize((512, 512)),
               np.float32).transpose(2, 0, 1)[None] / 255.0
for _ in range(3):
    sess.run(None, {iname: x})
ts = []
for _ in range(a.reps):
    t0 = time.perf_counter()
    sess.run(None, {iname: x})
    ts.append(time.perf_counter() - t0)
v = sorted(ts)
print("ORT CPU  %-30s  median %7.1f ms  IQR %.1f  threads=%d"
      % (os.path.basename(a.image), statistics.median(v) * 1e3,
         (v[3 * len(v) // 4] - v[len(v) // 4]) * 1e3, a.threads))
