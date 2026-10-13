#!/usr/bin/env python3
"""#44-②: ORT CUDA EP cross-process determinism probe (single run per process).

Run twice from fresh processes and compare OUT_MD5 lines.
Usage: ort_cuda_crossproc.py [--provider CUDAExecutionProvider|CPUExecutionProvider]
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(os.path.dirname(HERE), "reports", "onnx-dynamo",
                     "trufor_dynamic_dynamo.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    ap.add_argument("--ref", default=os.path.join(HERE, "ncnn_a2", "ref_512x512.npz"))
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL, sess_options=so,
                                providers=[args.provider])
    print("EP_USED:", sess.get_providers()[0], flush=True)
    d = np.load(args.ref)
    x = d["input"][None].astype("float32")
    name = sess.get_inputs()[0].name
    outs = sess.run(None, {name: np.ascontiguousarray(x)})
    for o, n in zip(outs, [x.name for x in sess.get_outputs()]):
        a = np.asarray(o, dtype="float32")
        md5 = hashlib.md5(a.tobytes()).hexdigest()[:12]
        print("OUT_MD5 %s=%s sum=%.8e" % (n, md5, float(a.astype("float64").sum())),
              flush=True)
    print("CUDA_XPROC_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
