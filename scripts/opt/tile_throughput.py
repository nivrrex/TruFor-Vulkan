#!/usr/bin/env python3
"""CPU + GPU tile throughput on the TruFor model. T6.

Runs the same ONNX model on two backends at once, consuming a shared queue of
image tiles, and reports tiles per second for each configuration:

  * CPU only, ONNX Runtime with a chosen intra-op thread count
  * GPU only, TVM Vulkan
  * both at once, one worker thread per backend

The point is throughput, not single-image latency. The whole graph here is one
image, so a single run is latency-bound and the two backends cannot overlap
anything; a queue of tiles lets the CPU work on tile N+1 while the GPU works on
tile N. Anything that shares a device, a process-global pool or a scratch
allocator between the two threads would show up immediately as a throughput
number that is worse than the better of the two singles, which is why the script
also runs the singles as a control in the same session.

This script does not modify the production path. It builds its own TVM module
and its own ORT session, and it writes nothing outside reports/.

Usage:
    python3 scripts/opt/tile_throughput.py [--tiles 8] [--cpu-threads 4] [--mode all]
"""

import argparse
import os
import statistics
import sys
import threading
import time
import queue

import numpy as np

TVM_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "tvm-venv")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODEL = os.path.join(ROOT, "reports", "onnx-dynamo", "trufor_512x512_dynamo_inline.onnx")
IMAGES = os.path.join(ROOT, "truFor-src", "test_docker", "images")
MESA_LIB = os.path.join(ROOT, "vendor", "mesa-ab", "root", "usr", "lib", "x86_64-linux-gnu")
MESA_ICD = os.path.join(ROOT, "vendor", "mesa-ab", "root", "usr", "share", "vulkan", "icd.d", "radeon_icd.json")


def load_tiles(n):
    from PIL import Image
    names = ["tampered2.png", "tampered1.png", "pristine1.jpg", "pristine2.jpg"]
    tiles = []
    for i in range(n):
        f = os.path.join(IMAGES, names[i % len(names)])
        img = Image.open(f).convert("RGB").resize((512, 512))
        tiles.append((np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.).astype(np.float32))
    return tiles


def make_cpu(k):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = k
    so.inter_op_num_threads = 1
    return ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])


def make_gpu():
    import tvm
    import onnx
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir import dlight as dl
    from tvm.relax.backend.gpu_generic import pipeline as GP
    import tvm.ir.transform as T
    from tvm.relax.transform import (LegalizeOps, AnnotateTIROpPattern,
                                     FoldConstant, FuseOps, FuseTIR)
    sys.path.insert(0, os.path.join(ROOT, "scripts", "opt"))
    import swap_conv

    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    dev = tvm.vulkan()
    ir = from_onnx(onnx.load(MODEL), shape_dict={"rgb": [1, 3, 512, 512]},
                   dtype_dict="float32")
    seq = T.Sequential(
        GP.library_dispatch_passes(target) +
        [LegalizeOps(), AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR(),
         dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                                 dl.gpu.GeneralReduction(), dl.gpu.Fallback())] +
        GP.dataflow_lower_passes(target) + GP.finalize_passes(target))
    with target:
        mod = seq(ir)
    # verify is not optional here. Without it swap_conv picks relu=True for every
    # depthwise kernel, and half of the fused depthwise convolutions end in a
    # bare residual add rather than a ReLU; substituting those silently changes
    # the output. Verification costs a few seconds once.
    n, _, log = swap_conv.swap(mod, use_depthwise=True, layout=True,
                               verify=(dev, target), verbose=False)
    sys.stderr.write("GPU substitutions: %d (all depthwise verified)\n" % n)
    e = relax.build(mod, target=target)
    return relax.VirtualMachine(e.jit(), dev), dev


def digest(out):
    """Order-sensitive summary of one tile's output, for the equality check.

    A scalar such as m.sum() is not enough: a ReLU applied where the graph has a
    bare residual add changes the distribution while leaving the total nearly
    fixed, which is exactly the substitution bug this digest is meant to catch.
    So it carries the full map checksum and the raw detector logit, not just
    aggregates.
    """
    import torch
    import torch.nn.functional as F
    m = F.softmax(torch.from_numpy(np.asarray(out[0])[0]), 0)[1].numpy().astype(np.float64)
    d = np.asarray(out[2]).reshape(-1).astype(np.float64)
    return (float(m.sum()), float(m.mean()), float(np.abs(m).max()),
            float(m[::37].sum()),          # sparse sample, catches local changes
            float(m.var()), float(d[0]), float(np.abs(d).max()))


def run_singles(tiles, cpu_k, gpu_vm, dev, reps):
    import tvm
    sess = make_cpu(cpu_k)
    for t in tiles[:2]:
        sess.run(None, {"rgb": t})
    gpu_vm["main"](tvm.runtime.tensor(tiles[0], dev)); dev.sync()
    cpu_t, gpu_t = [], []
    for i in range(reps):
        t0 = time.perf_counter()
        sess.run(None, {"rgb": tiles[i % len(tiles)]})
        cpu_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        gpu_vm["main"](tvm.runtime.tensor(tiles[i % len(tiles)], dev)); dev.sync()
        gpu_t.append(time.perf_counter() - t0)
    return statistics.median(cpu_t), statistics.median(gpu_t)


def run_pipelined(tiles, cpu_k, gpu_vm, dev, n_workers_gpu, spin):
    """One CPU worker and n_workers_gpu GPU workers over a shared queue.

    spin controls how the GPU workers wait: a busy loop on device completion
    versus a short sleep. On a busy machine the busy loop can starve the CPU
    worker of a core, which is exactly the effect worth measuring.
    """
    import tvm
    q = queue.Queue()
    for i, t in enumerate(tiles):
        q.put((i, t))
    for _ in range(len(tiles)):
        q.put(None)

    results = {}
    lock = threading.Lock()
    counter = {"done": 0}
    t_start = time.perf_counter()

    def cpu_worker():
        sess = make_cpu(cpu_k)
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            idx, t = item
            out = sess.run(None, {"rgb": t})
            with lock:
                results["cpu_%d" % idx] = digest(out)
                counter["done"] += 1
            q.task_done()

    def gpu_worker():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            idx, t = item
            tin = tvm.runtime.tensor(t, dev)
            o = gpu_vm["main"](tin)
            dev.sync()
            if not spin:
                time.sleep(0.0005)
            with lock:
                results["gpu_%d" % idx] = digest([o[j].numpy() for j in range(len(o))])
                counter["done"] += 1
            q.task_done()

    threads = [threading.Thread(target=cpu_worker, daemon=True)]
    for _ in range(n_workers_gpu):
        threads.append(threading.Thread(target=gpu_worker, daemon=True))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - t_start
    return elapsed, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--mode", default="all", choices=["all", "single", "pipe"])
    args = ap.parse_args()

    os.environ.setdefault("LD_LIBRARY_PATH", MESA_LIB)
    os.environ.setdefault("VK_DRIVER_FILES", MESA_ICD)
    # Only the source tree goes on sys.path. Inserting the venv's
    # site-packages ahead of it makes `import tvm` resolve to a pure-python
    # package with no libtvm_runtime behind it, and relax.build then fails with
    # "Cannot find global function target.build.vulkan". Run this with the venv
    # interpreter so its site-packages is already on the path, but do not
    # re-order it above the source tree.
    sys.path.insert(0, os.path.join(ROOT, "vendor", "tvm-src", "python"))

    tiles = load_tiles(args.tiles)
    gpu_vm, dev = make_gpu()
    print("tiles=%d  cpu_threads=%d  physical cores=%d"
          % (len(tiles), args.cpu_threads, os.cpu_count()))

    if args.mode in ("all", "single"):
        cpu_med, gpu_med = run_singles(tiles, args.cpu_threads, gpu_vm, dev, args.reps)
        print("\n单路中位数 (3 温 + %d 次):" % args.reps)
        print("  CPU  %7.1f ms  -> %.3f tile/s" % (cpu_med * 1e3, 1 / cpu_med))
        print("  GPU  %7.1f ms  -> %.3f tile/s" % (gpu_med * 1e3, 1 / gpu_med))

    if args.mode in ("all", "pipe"):
        print("\n流水线 (1 CPU worker + 1 GPU worker, 共享队列):")
        best = None
        for spin in (False, True):
            for k in (2, 3, 4):
                el, res = run_pipelined(tiles, k, gpu_vm, dev, 1, spin)
                tps = len(tiles) / el
                mism = [i for i in range(len(tiles))
                        if ("cpu_%d" % i) in res and ("gpu_%d" % i) in res
                        and res["cpu_%d" % i] != res["gpu_%d" % i]]
                print("  sync=%-5s cpu_threads=%d  %6.2f s  %.3f tile/s  %d/%d 结果不一致"
                      % ("spin" if spin else "sleep", k, el, tps, len(mism), len(tiles)))
                if best is None or tps > best[0]:
                    best = (tps, k, spin)
        print("\n最优: %.3f tile/s  (cpu_threads=%d, %s)"
              % (best[0], best[1], "spin" if best[2] else "sleep"))


if __name__ == "__main__":
    main()
