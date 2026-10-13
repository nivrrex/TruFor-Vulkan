"""Gate 4: is RX 560 out of compute, or is TVM's schedule leaving compute unused?

Gate 2 showed 99.65% of the 512x512 inference is real GPU kernel time, so
launch/copy/sync tuning is worthless. That leaves one question: would *any*
Vulkan kernel beat ORT's CPU on this device, at some size?

Each operator is swept from small to large, on the same TVM Vulkan backend and
the same ONNX Runtime CPU backend, with the same input data. The crossover
point is the answer:

    small  GPU < CPU,  large GPU > CPU   -> the hardware is fine, the fixed
                                           per-op cost of this graph is the
                                           problem, and batching/larger tiles
                                           will pay off
    small  GPU < CPU,  large GPU < CPU   -> the software stack is not reaching
                                           the hardware's capability, and
                                           kernel work is justified

Shapes follow the ones TruFor actually uses, so the crossover is relevant to the
real workload rather than to a synthetic best case.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _uniq(tag):
    import re as _re
    return "/tmp/_g4_%s.onnx" % _re.sub(r"[^0-9a-zA-Z]+", "_", tag)


def onnx_matmul(tag, a_shape, b_shape):
    from onnx import TensorProto, helper, numpy_helper
    node = helper.make_node("MatMul", ["a", "b"], ["c"])
    g = helper.make_graph([node], "matmul",
                          [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                           helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape))],
                          [helper.make_tensor_value_info("c", TensorProto.FLOAT, list(a_shape))])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    p = _uniq(tag)
    import onnx
    onnx.save(m, p)
    return p


def onnx_one_arg(tag, kind, shape, out_shape=None, **attrs):  # noqa: C901
    from onnx import TensorProto, helper
    # helper.make_node does not accept AttributeProto values; pass them through
    # the node's attribute list instead
    node = helper.make_node(kind, ["a"], ["c"])
    for a, v in attrs.items():
        node.attribute.append(helper.make_attribute(a, v))
    oshape = list(out_shape) if out_shape is not None else list(shape)
    g = helper.make_graph([node], kind.lower(),
                          [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(shape))],
                          [helper.make_tensor_value_info("c", TensorProto.FLOAT, oshape)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    p = _uniq(tag)
    import onnx
    onnx.save(m, p)
    return p


def hw_of(shape):
    return shape[2]


def onnx_conv(in_shape, cout, k, pad):
    """Weights stay initializers; only 'a' is a graph input.

    TVM's ONNX frontend needs every Conv operand's rank spelled out, so the
    shapes of the initializers are carried in the graph's value_info instead of
    being left for inference.
    """
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(cout, in_shape[1], k, k).astype(np.float32), "w")
    b = numpy_helper.from_array(np.zeros(cout, np.float32), "b")
    node = helper.make_node("Conv", ["a", "w", "b"], ["c"], kernel_shape=[k, k], pads=[pad] * 4)
    g = helper.make_graph([node], "conv",
                          [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(in_shape))],
                          [helper.make_tensor_value_info("c", TensorProto.FLOAT,
                                                          [in_shape[0], cout, in_shape[2], in_shape[3]])],
                          [w, b])
    g.value_info.append(helper.make_tensor_value_info(
        "w", TensorProto.FLOAT, [cout, in_shape[1], k, k]))
    g.value_info.append(helper.make_tensor_value_info("b", TensorProto.FLOAT, [cout]))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    p = _uniq("conv_%d_%d" % (in_shape[1], hw_of(in_shape)))
    import onnx
    onnx.save(m, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "reports/capability"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    import onnxruntime as ort  # noqa: F401
    import tvm
    import tvm_ffi  # noqa: F401
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    dev = tvm.vulkan()
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("=" * 78)
    print("GATE 4  synthetic operator scaling: does the GPU ever win?")
    print("=" * 78)
    print("device %s, driver %s, fp32, thread_warp_size=%s"
          % (dev.device_name, dev.driver_version, dev.warp_size))

    cases = []
    for hw in (64, 128, 256, 512, 1024):
        cases.append(("elementwise 64ch %dx%d" % (hw, hw),
                      onnx_one_arg("id_%d" % hw, "Identity", (1, 64, hw, hw)),
                      [np.random.rand(1, 64, hw, hw).astype(np.float32)]))
    for hw in (32, 64, 128, 256, 512):
        p = onnx_conv((1, 64, hw, hw), 64, 3, 1)
        cases.append(("conv3x3 64ch %dx%d" % (hw, hw), p, [None]))
    for m_, k, n in ((512, 64, 64), (2048, 128, 128), (8192, 256, 256),
                     (16384, 512, 512)):
        a = np.random.rand(1, m_, k).astype(np.float32)
        b = np.random.rand(1, k, n).astype(np.float32)
        cases.append(("matmul %dx%dx%d" % (m_, k, n),
                      onnx_matmul("mm_%d" % m_, (1, m_, k), (1, k, n)), [a, b]))
    for n_ in (4096, 65536, 1048576, 2097152):
        cases.append(("reduce_sum n=%d" % n_,
                      onnx_one_arg("rs_%d" % n_, "ReduceSum", (1, 1, n_), out_shape=(1, 1), axes=[2], keepdims=0),
                      [np.random.rand(1, 1, n_).astype(np.float32)]))
    for heads, n_ in ((4, 256), (5, 1024), (5, 2048), (5, 4096)):
        shp = (1, heads, n_, n_)
        cases.append(("softmax %dx%dx%d" % (heads, n_, n_),
                      onnx_one_arg("sm_%d" % n_, "Softmax", shp, axis=-1),
                      [np.random.rand(*shp).astype(np.float32)]))

    rows = []
    print("\n%-28s %11s %11s %9s %10s" % ("case", "CPU ms", "GPU ms", "speedup", "rel_err"))
    for name, path, inputs in cases:
        try:
            import onnx as _ox
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            gnames = [i.name for i in sess.get_inputs()]
            # only graph inputs are fed; Conv weights/bias stay initializers in
            # both runtimes, exactly as the real model keeps them
            tvals = []
            for k_, nm in enumerate(gnames):
                src = inputs[k_] if k_ < len(inputs) and inputs[k_] is not None \
                    else np.asarray(0.0, np.float32)
                tvals.append(np.ascontiguousarray(src, np.float32))
            shape_dict = {nm: list(v.shape) for nm, v in zip(gnames, tvals)}

            ir = from_onnx(_ox.load(path, load_external_data=True),
                           shape_dict=shape_dict, dtype_dict="float32")
            exe = relax.build(ir, target=tgt)
            vm = relax.VirtualMachine(exe.jit(), dev)
            feed = {nm: v for nm, v in zip(gnames, tvals)}
            targs = [tvm.runtime.tensor(v, dev) for v in tvals]

            def cpu():
                return sess.run(None, feed)

            def gpu():
                out = vm["main"](*targs)
                if isinstance(out, tvm_ffi.Array):
                    return [out[i].numpy() for i in range(len(out))]
                return [out.numpy()]

            t = C.Timer(args.warmup, args.repeats)
            t(cpu)
            cs, rc = t.report(), t.last_result
            t = C.Timer(args.warmup, args.repeats)
            t(gpu)
            gs, gc = t.report(), t.last_result

            a = np.asarray(rc[0], np.float64).ravel()
            b = np.asarray(gc[0], np.float64).ravel()
            scale = max(float(np.abs(a).max()), 1e-9)
            err = float(np.abs(a - b).max()) / scale if a.shape == b.shape else float("inf")
            sp = cs["median"] / gs["median"] if gs["median"] else float("inf")
            rows.append({"case": name, "cpu_ms": cs["median"] * 1e3, "gpu_ms": gs["median"] * 1e3,
                         "speedup": sp, "rel_err": err,
                         "elements": int(np.prod(tvals[0].shape)) if tvals else 0})
            print("%-28s %11.3f %11.3f %8.3fx %10.2e%s"
                  % (name, cs["median"] * 1e3, gs["median"] * 1e3, sp, err,
                     "   MISMATCH" if err > 1e-2 else ""))
        except Exception as e:  # noqa: BLE001
            rows.append({"case": name, "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
            print("%-28s ERROR %s" % (name, str(e)[:70]))
        finally:
            # 15 GB box: a compiled executable plus its host copies is tens of MB
            # each, and the sweep keeps every one alive otherwise
            for _n in ("exe", "vm", "ir", "sess", "targs", "tvals", "feed"):
                globals().pop(_n, None)
            import gc
            gc.collect()

    ok = [r for r in rows if "speedup" in r]
    wins = [r for r in ok if r["speedup"] > 1.0]
    print("\n%d/%d cases where the GPU is faster; max speedup %.3fx (case: %s)"
          % (len(wins), len(ok), max((r["speedup"] for r in ok), default=0),
             max(ok, key=lambda r: r["speedup"])["case"] if ok else "-"))
    C.write_json(os.path.join(args.outdir, "gate4_kernel_scaling.json"),
                 {"device": dev.device_name, "rows": rows,
                  "gpu_wins": len(wins), "total": len(ok)})
    return 0


if __name__ == "__main__":
    import tvm_ffi
    sys.exit(main())
