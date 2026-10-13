"""Gate 3: batch scaling -- is the GPU a latency device or a throughput device?

Gate 2 established that 99.65% of a 512x512 batch-1 inference is kernel time
and only 8.5 ms is host-visible overhead, which predicts batching will not help
much: there is almost no fixed cost to amortise. That is a prediction, not a
measurement, so this tests it.

The production graph is batch 1, so batched variants are exported from the
unmodified model with the two exact export-time rewrites already established
(Adaptive{Max,Avg}Pool2d -> mean/amax, drop the trailing None) and the
Slice+Squeeze rewrite is deliberately NOT applied.

For each batch the same weights and the same tiles are run through ONNX Runtime
CPU and TVM Vulkan, and per-image cost is reported, because a GPU that wins only
on throughput is still useless for a single-image forensic call.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_SCRIPT = os.path.join(REPO, "scripts/export_onnx_dynamo.py")


def ensure_export(batch, outdir):
    import onnx
    path = os.path.join(outdir, "batch%d" % batch, "trufor_512x512_dynamo.onnx")
    inlined = path.replace("_dynamo.onnx", "_inline.onnx")
    m0 = onnx.load(inlined, load_external_data=False) if os.path.exists(inlined) else None
    if m0 is not None:
        dim = m0.graph.input[0].type.tensor_type.shape.dim[0].dim_value
        if dim == batch:
            return inlined
        os.remove(inlined)  # stale export from an earlier run with a different batch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = subprocess.run([sys.executable, EXPORT_SCRIPT, "--model",
                        os.path.join(REPO, "weights/trufor.pth.tar"),
                        "--outdir", os.path.dirname(path), "--size", "512",
                        "--batch", str(batch)],
                       capture_output=True, text=True)
    if not os.path.exists(path):
        raise RuntimeError("export failed for batch %d (r=%s): %s"
                           % (batch, r.returncode, r.stderr[-300:]))
    import onnx
    m = onnx.load(path, load_external_data=True)
    onnx.save_model(m, inlined, save_as_external_data=False)
    os.remove(path)
    for ext in (".data",):
        p = path + ext
        if os.path.exists(p):
            os.remove(p)
    return inlined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "reports/capability"))
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(REPO, "scripts/bench"))
    import common as C

    import onnx
    import onnxruntime as ort
    import tvm
    import tvm_ffi
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    dev = tvm.vulkan()
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = args.cpu_threads

    print("=" * 78)
    print("GATE 3  batch scaling (throughput vs latency)")
    print("=" * 78)
    print("exporting batch variants from the unmodified model ...")
    for b in (int(v) for v in args.batches.split(",")):
        p = ensure_export(b, args.outdir)
        print("  batch %-2d %s" % (b, os.path.basename(p)))

    rows = []
    print("\n%-6s %10s %10s %10s %10s %9s %9s"
          % ("batch", "CPU tot", "GPU tot", "CPU/img", "GPU/img", "cpu img/s", "gpu img/s"))
    for b in (int(v) for v in args.batches.split(",")):
        path = ensure_export(b, args.outdir)
        try:
            m = onnx.load(path, load_external_data=False)
            ir = from_onnx(m, shape_dict={"rgb": [b, 3, 512, 512]}, dtype_dict="float32")
            t0 = time.perf_counter()
            exe = relax.build(ir, target=tgt)
            build_s = time.perf_counter() - t0
            vm = relax.VirtualMachine(exe.jit(), dev)

            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name
            x = np.random.rand(b, 3, 512, 512).astype(np.float32)
            tx = tvm.runtime.tensor(x, dev)

            def cpu():
                return sess.run(None, {iname: x})

            def gpu():
                out = vm["main"](tx)
                if isinstance(out, tvm_ffi.Array):
                    return [out[i].numpy() for i in range(len(out))]
                return [out.numpy()]

            tc = C.Timer(args.warmup, args.repeats)
            tc(cpu)
            rc = tc.report()
            tg = C.Timer(args.warmup, args.repeats)
            tg(gpu)
            rg = tg.report()
            a = np.asarray(tc.last_result[0], np.float64).ravel()
            c = np.asarray(tg.last_result[0], np.float64).ravel()
            err = float(np.abs(a - c).max()) / max(float(np.abs(a).max()), 1e-9) if a.shape == c.shape else float("inf")

            cpu_img, gpu_img = rc["median"] / b, rg["median"] / b
            print("%-6d %9.0fms %9.0fms %9.0fms %9.0fms %9.3f %9.3f%s"
                  % (b, rc["median"] * 1e3, rg["median"] * 1e3,
                     cpu_img * 1e3, gpu_img * 1e3,
                     1.0 / cpu_img, 1.0 / gpu_img,
                     "   MISMATCH" if err > 1e-2 else ""))
            rows.append({"batch": b, "build_s": build_s,
                         "cpu_total_s": rc["median"], "gpu_total_s": rg["median"],
                         "cpu_per_image_s": cpu_img, "gpu_per_image_s": gpu_img,
                         "cpu_img_per_s": 1.0 / cpu_img, "gpu_img_per_s": 1.0 / gpu_img,
                         "speedup_total": rc["median"] / rg["median"],
                         "speedup_per_image": cpu_img / gpu_img, "rel_err": err})
            del exe, vm, ir, sess, tx
            import gc
            gc.collect()
        except Exception as e:  # noqa: BLE001
            rows.append({"batch": b, "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
            print("%-6d ERROR %s" % (b, str(e)[:70]))

    ok = [r for r in rows if "speedup_per_image" in r]
    if ok:
        base = ok[0]
        print("\n  per-image cost vs batch 1:")
        for r in ok:
            print("    batch %-2d CPU %7.0f ms (%5.2fx)  GPU %7.0f ms (%5.2fx)"
                  % (r["batch"], r["cpu_per_image_s"] * 1e3,
                     base["cpu_per_image_s"] / r["cpu_per_image_s"],
                     r["gpu_per_image_s"] * 1e3,
                     base["gpu_per_image_s"] / r["gpu_per_image_s"]))
    C.write_json(os.path.join(args.outdir, "gate3_batch.json"), rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
