#!/usr/bin/env python3
"""A2 reference: run the dynamic ONNX under ORT-CPU fp32 and freeze inputs/outputs.

Saves npz per case so the ncnn side can be fed byte-identical float32 CHW
blobs. Gate (from the review plan): map rel_L2 <= 1e-3, conf <= 1e-2.
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "reports", "onnx-dynamo", "trufor_dynamic_dynamo.onnx")
OUTDIR = os.path.join(HERE, "ncnn_a2")

CASES = [("512x512", 512, 512), ("384x512", 384, 512)]

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print("ort", ort.__version__, "in:", in_name, "outs:", out_names, flush=True)

    for tag, h, w in CASES:
        rng = np.random.RandomState(0)
        blob = rng.rand(1, 3, h, w).astype("float32")
        t0 = time.perf_counter()
        outs = sess.run(None, {in_name: blob})
        dt = time.perf_counter() - t0
        np.savez_compressed(
            os.path.join(OUTDIR, "ref_%s.npz" % tag),
            input=blob[0], **{n: o for n, o in zip(out_names, outs)})
        print("%s done %.1fs shapes=%s" % (
            tag, dt, [tuple(o.shape) for o in outs]), flush=True)
    print("A2_REFS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
