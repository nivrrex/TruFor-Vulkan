#!/usr/bin/env python3
"""Probe: can the ORT WebGPU EP (Dawn -> Vulkan) actually initialise and use the
RX 560 on this machine, and does it enumerate the discrete GPU rather than a
software adapter?

Deliberately does not touch TruFor. Builds a tiny Conv+Relu ONNX graph so the
result answers only "is the runtime path alive", independent of model coverage.
"""

import os
import sys
import time

import numpy as np

os.environ.setdefault("ORT_DISABLE_ALL_LOGS", "0")

import onnxruntime as ort  # noqa: E402


def build_tiny_model(path, outdir):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    os.makedirs(outdir, exist_ok=True)
    w = numpy_helper.from_array(np.random.RandomState(0).rand(8, 3, 3, 3).astype(np.float32), "w")
    b = numpy_helper.from_array(np.random.RandomState(1).rand(8).astype(np.float32), "b")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 62, 62])
    node = helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3])
    graph = helper.make_graph([node], "tiny", [x], [y], [w, b])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, path)
    return path


def main():
    ep_lib = os.path.join(
        os.path.dirname(ort.__file__), "..", "onnxruntime_ep_webgpu", "libonnxruntime_providers_webgpu.so"
    )
    ep_lib = os.path.abspath(ep_lib)
    print("ort version           :", ort.__version__)
    print("webgpu ep library     :", ep_lib, "exists =", os.path.exists(ep_lib))
    print("static providers      :", ort.get_available_providers())
    print("all providers (incl. builtins):", ort.get_all_providers())

    try:
        ort.register_execution_provider_library("WebGpuExecutionProvider", ep_lib)
        print("register_execution_provider_library: ok (no exception)")
    except Exception as e:  # noqa: BLE001
        print("register FAILED:", type(e).__name__, e)
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    model = build_tiny_model(os.path.join(here, "tiny_conv.onnx"), here)
    x = np.random.RandomState(7).rand(1, 3, 64, 64).astype(np.float32)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 1
    try:
        so.register_execution_provider_library("WebGpuExecutionProvider", ep_lib)
    except Exception as e:  # noqa: BLE001
        print("session-level register unsupported:", type(e).__name__, e)

    result = {}
    for label, providers in (
        ("CPU", ["CPUExecutionProvider"]),
        ("WebGPU", ["WebGpuExecutionProvider", "CPUExecutionProvider"]),
    ):
        t0 = time.perf_counter()
        try:
            sess = ort.InferenceSession(model, so, providers=providers)
            out = sess.run(None, {"x": x})[0]
            dt = time.perf_counter() - t0
            result[label] = (out, dt)
            print("[%s] ok  used=%s  out=%s  checksum=%.6f  %.3fs" % (
                label, sess.get_providers(), out.shape, float(out.astype(np.float64).sum()), dt))
        except Exception as e:  # noqa: BLE001
            print("[%s] FAILED: %s: %s" % (label, type(e).__name__, str(e)[:600]))
            result[label] = None

    if result.get("CPU") and result.get("WebGPU"):
        a, b = result["CPU"][0], result["WebGPU"][0]
        print("\nmax abs diff CPU vs WebGPU: %.3e" % float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))

    return 0 if result.get("WebGPU") else 1


if __name__ == "__main__":
    sys.exit(main())
