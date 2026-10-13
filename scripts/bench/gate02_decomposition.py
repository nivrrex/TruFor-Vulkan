"""Gate 2: where does the 2.5 s go?

The question Gate 1 raised is not "is the GPU slower" but "slower at what".
This splits one 512x512 batch-1 inference into host-visible phases and
separates fixed per-launch cost from work that scales with the graph:

    t_h2d   input tensor upload
    t_exec  the relax VM call  (dispatch + kernels + any implicit wait)
    t_d2h   draining the three outputs back to host
    t_total the sum, which must reproduce the Gate 1 number

plus two reference points that turn those numbers into an explanation:

    t_fixed  launch + sync overhead of a trivial graph, measured with the same
             VM path -- the cost a 1900-node graph pays once per layer it does
             not amortise
    t_sync_residual  a second, immediate drain after the first: whatever time
             it takes is work the first drain did not wait for

It also probes whether Vulkan timestamp queries are reachable from this TVM
build, because a real GPU timestamp would settle the kernel-vs-overhead
question without inference.
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ONNX = os.path.join(REPO, "reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx")
DEFAULT_IMAGE = os.path.join(REPO, "truFor-src/test_docker/images/tampered2.png")


def phase_split(exe, dev, x, warmup, repeats):
    import tvm
    import tvm_ffi
    from tvm import relax

    vm = relax.VirtualMachine(exe.jit(), dev)

    def drain(out):
        if isinstance(out, tvm_ffi.Array):
            return [out[i].numpy() for i in range(len(out))]
        return [out.numpy()]

    rows = []
    for _ in range(warmup):
        tin = tvm.runtime.tensor(x, dev)
        drain(vm["main"](tin))
    for _ in range(repeats):
        t0 = time.perf_counter()
        tin = tvm.runtime.tensor(x, dev)
        t1 = time.perf_counter()
        out = vm["main"](tin)
        t2 = time.perf_counter()
        arr = drain(out)
        t3 = time.perf_counter()
        # second drain: anything left over is work the first drain did not wait for
        t4 = time.perf_counter()
        drain(out)
        t5 = time.perf_counter()
        rows.append({
            "h2d": t1 - t0, "exec": t2 - t1, "d2h": t3 - t2,
            "total": t3 - t0, "second_drain": t5 - t4,
        })
    agg = {}
    for k in rows[0]:
        agg[k] = C.summarize([r[k] for r in rows])
    return agg, arr


def fixed_overhead_probe(dev, warmup, repeats):
    """A one-op graph through the same VM path: pure launch + sync + drain.

    Built via ONNX rather than TVMScript because that is the exact route the
    real model takes, so the measured overhead is comparable.
    """
    import onnx
    import tvm
    import tvm_ffi
    from onnx import TensorProto, helper, numpy_helper
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    n = 4
    # both operands stay graph inputs: an initializer would be folded into a
    # constant and main would end up taking zero arguments
    node = helper.make_node("Mul", ["a", "a"], ["b"])
    g = helper.make_graph([node], "one_op",
                          [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, n])],
                          [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, n])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    path = "/tmp/_gate2_one_op.onnx"
    onnx.save(m, path)

    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    exe = relax.build(from_onnx(m, shape_dict={"a": [1, n]}, dtype_dict="float32"), target=tgt)
    vm = relax.VirtualMachine(exe.jit(), dev)
    av = tvm.runtime.tensor(np.ones((1, n), np.float32), dev)
    bv = tvm.runtime.tensor(np.zeros((1, n), np.float32), dev)

    def one():
        t0 = time.perf_counter()
        out = vm["main"](av)
        arr = [out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array) else [out.numpy()]
        return time.perf_counter() - t0, arr

    for _ in range(warmup):
        one()
    ts = [one()[0] for _ in range(repeats)]
    return C.summarize(ts), ts


def host_cpu_reference_overhead(repeats):
    """What a comparable no-op costs on the CPU side, for scale."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.empty((1, 4), np.float32)
        ts.append(time.perf_counter() - t0)
    return C.summarize(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--outdir", default=os.path.join(REPO, "reports/capability"))
    ap.add_argument("--repeats", type=int, default=C.REPEATS)
    ap.add_argument("--warmup", type=int, default=C.WARMUP)
    args = ap.parse_args()

    import onnx
    import tvm
    from PIL import Image
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    print("=" * 70)
    print("GATE 2  GPU time decomposition (512x512, batch 1)")
    print("=" * 70)

    img = Image.open(args.image).convert("RGB").resize((512, 512), Image.BILINEAR)
    x = (np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)

    dev = tvm.vulkan()
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("  target keys: %s" % sorted(tgt.keys))
    print("  timestamp query exposed by TVM target: %s"
          % ("yes" if any("timestamp" in k for k in tgt.keys) else "no"))
    print("  device create_stream: %s"
          % (repr(tvm.runtime.device(dev.type, dev.index).create_raw_stream)
             if hasattr(tvm.runtime.device(dev.type, dev.index), "create_raw_stream") else "n/a"))

    m = onnx.load(args.onnx, load_external_data=False)
    ir = from_onnx(m, shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
    t0 = time.perf_counter()
    exe = relax.build(ir, target=tgt)
    print("  build %.1f s" % (time.perf_counter() - t0))

    phases, _ = phase_split(exe, dev, x, args.warmup, args.repeats)
    print("\n  phase            median      p10        p90        share")
    med = phases["total"]["median"]
    for k in ("h2d", "exec", "d2h", "second_drain"):
        s = phases[k]
        share = (s["median"] / med * 100) if med else 0
        print("  %-14s %8.1f ms %8.1f %8.1f    %5.1f%%"
              % (k, s["median"] * 1e3, s["p10"] * 1e3, s["p90"] * 1e3, share))
    print("  %-14s %8.1f ms %8.1f %8.1f"
          % ("total", phases["total"]["median"] * 1e3,
             phases["total"]["p10"] * 1e3, phases["total"]["p90"] * 1e3))

    fixed, fixed_ts = fixed_overhead_probe(dev, args.warmup, args.repeats)
    print("\n  trivial 1-op graph, same VM path (launch + sync + drain):")
    print("    median %.3f ms   p10 %.3f   p90 %.3f"
          % (fixed["median"] * 1e3, fixed["p10"] * 1e3, fixed["p90"] * 1e3))
    n_nodes = len(m.graph.node)
    per_node = phases["exec"]["median"] / max(n_nodes, 1)
    print("    TruFor graph has %d ONNX nodes -> %.4f ms/node if all exec were per-node"
          % (n_nodes, per_node * 1e3))
    print("    fixed cost as a fraction of exec: %.2f%%"
          % (fixed["median"] / phases["exec"]["median"] * 100))

    overhead = phases["h2d"]["median"] + phases["d2h"]["median"] + phases["second_drain"]["median"]
    exec_median = phases["exec"]["median"]
    variable = max(exec_median - fixed["median"], 0.0)
    classification = {
        "A_kernel_bound": variable / exec_median > 0.7,
        "B_overhead_bound": overhead / med > 0.5,
    }
    print("\n  host-visible overhead (h2d + d2h + residual) = %.1f ms of %.1f ms (%.1f%%)"
          % (overhead * 1e3, med * 1e3, overhead / med * 100))
    print("  exec attributable to graph work      = %.1f ms" % (variable * 1e3))
    if classification["A_kernel_bound"]:
        print("  -> CASE A: mostly kernel/graph work")
    elif classification["B_overhead_bound"]:
        print("  -> CASE B: mostly launch / copy / synchronisation overhead")
    else:
        print("  -> CASE C: mixed; the split is in the report")

    out = {
        "phases": phases, "fixed_overhead": fixed, "n_onnx_nodes": n_nodes,
        "per_node_exec_ms": per_node * 1e3,
        "host_overhead_share": overhead / med if med else None,
        "graph_work_ms": variable * 1e3,
        "classification": classification,
        "timestamp_query_available": any("timestamp" in k for k in tgt.keys),
    }
    C.write_json(os.path.join(args.outdir, "gate2_decomposition.json"), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
