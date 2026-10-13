#!/usr/bin/env python3
"""Step 3: run the TVM Vulkan path under AddressSanitizer.

The heap corruption (`corrupted double-linked list`) only appeared with one
pipeline configuration in Phase 8B, so that is the default here: it gives ASan
something to catch. `--variant default|B|C|D|E` selects others.

ASan lives in the shared objects, but the process is a Python interpreter, so
libasan must be preloaded. `libasan_path` is resolved from gcc.

Usage:
    LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
    PYTHONPATH=vendor/tvm-src/python \
    TVM_LIBRARY_PATH=vendor/tvm-src/build-asan \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    ASAN_OPTIONS=... python3 scripts/phase8_asan_run.py --onnx <model.onnx>
"""

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", default="truFor-src/test_docker/images/tampered2.png")
    ap.add_argument("--variant", default="D", choices=["default", "B", "C", "D", "E"])
    ap.add_argument("--log", default="reports/phase8c/asan-run.log")
    ap.add_argument("--abort-on-error", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)

    import tvm
    from tvm import relax
    from tvm.relax.backend.gpu_generic import pipeline as GP
    from tvm.relax.frontend.onnx import from_onnx
    import numpy as np
    import onnx
    from PIL import Image

    dev = tvm.vulkan()
    vk = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("device: %s" % dev.device_name, flush=True)
    print("asan preloaded: %s" % os.environ.get("LD_PRELOAD", "<none>"), flush=True)

    img = Image.open(args.image).convert("RGB").resize((512, 512))
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)

    om = onnx.load(args.onnx, load_external_data=False)
    ish = [d.dim_value for d in om.graph.input[0].type.tensor_type.shape.dim]
    ir = from_onnx(om, shape_dict={"rgb": [1, 3, int(ish[2]), int(ish[3])]}, dtype_dict="float32")
    print("import ok", flush=True)

    drop_fuse = args.variant in ("C", "D", "E")
    drop_sched = args.variant in ("D", "E")

    def pipeline(target):
        @tvm.transform.module_pass(opt_level=0)
        def _p(mod, _ctx):
            leg = []
            for p in GP.legalize_passes(target):
                cls = type(p).__name__
                if drop_fuse and cls in ("FuseOps", "FuseTIR"):
                    continue
                if drop_sched and cls.startswith("ApplyDefaultSchedule"):
                    continue
                leg.append(p)
            seq = tvm.transform.Sequential(
                GP.library_dispatch_passes(target) + leg
                + GP.dataflow_lower_passes(target) + GP.finalize_passes(target))
            return seq(mod)
        return _p

    print("variant %s (drop_fuse=%s, drop_schedule=%s); building ..." % (args.variant, drop_fuse, drop_sched), flush=True)
    t0 = time.perf_counter()
    if args.variant == "default":
        exe = relax.build(ir, target=vk)
    else:
        exe = relax.build(ir, target=vk, relax_pipeline=pipeline(vk))
    print("build ok in %.1fs; running ..." % (time.perf_counter() - t0), flush=True)

    out = relax.VirtualMachine(exe.jit(), dev)["main"](tvm.runtime.tensor(x, dev))
    import tvm_ffi
    got = ([out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array)
           else ([o.numpy() for o in out] if isinstance(out, (list, tuple)) else [out.numpy()]))
    print("run completed. output shapes: %s" % [list(g.shape) for g in got], flush=True)
    for i, g in enumerate(got):
        print("  out%d sum=%.6f" % (i, float(np.asarray(g, np.float64).sum())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
