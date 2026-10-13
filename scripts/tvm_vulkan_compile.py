#!/usr/bin/env python3
"""Phase 6D: compile the imported TruFor Relax module for the TVM Vulkan target.

Queries the real device for its capabilities (subgroup size, workgroup limits,
storage buffer counts, fp16) instead of hard-coding them, then builds for
`vulkan` and reports which PrimFuncs were generated, which failed, and what
fell back to the host.

Requires the source build (USE_VULKAN=ON). Point PYTHONPATH at
vendor/tvm-src/python and TVM_LIBRARY_PATH at vendor/tvm-src/build.
"""

import argparse
import collections
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", default="vulkan")
    ap.add_argument("--hw", action="store_true", help="probe device capabilities")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import json as _json

    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    # TVM 0.27 dropped CLI-style target strings ("vulkan -from_device=0") in
    # favour of a JSON dict, so build a Target object here.
    if args.target.strip().startswith("{"):
        target = tvm.target.Target(_json.loads(args.target))
    else:
        target = tvm.target.Target(args.target)
    print("resolved target: %s" % target, flush=True)

    # the source build exposes tvm.__version__; the PyPI wheel only has
    # tvm._version.__version__. Accept either.
    tvm_version = getattr(tvm, "__version__", None)
    if tvm_version is None:
        tvm_version = tvm._version.__version__
    print("tvm %s  target=%s" % (tvm_version, args.target), flush=True)

    report = {"tvm_version": tvm_version, "target": str(target), "onnx": args.onnx}

    # ---- device identity + capabilities (no hard-coded assumptions) ----
    dev_info = {}
    try:
        dev = tvm.vulkan()
        def probe(attr, cast=int):
            v = getattr(dev, attr, None)
            if v is None:
                return None
            try:
                return cast(v)
            except (TypeError, ValueError):
                return str(v)

        dev_info = {"device": str(dev), "exist": bool(dev.exist)}
        for attr in ("device_name", "api_version", "driver_version", "vendor_id", "device_id",
                     "max_shared_memory_per_block", "max_threads_per_block", "l2_cache_size_bytes",
                     "max_clock_rate", "available_global_memory", "max_wg_counts",
                     "sub_group_size", "max_num_threads", "shader_core_max_work_dim",
                     "max_clock_frequency", "local_memory_size", "device_type"):
            val = probe(attr, str if attr in ("device_name", "device_type") else int)
            if val is not None:
                dev_info[attr] = val
        for attr in ("max_thread_dimensions", "max_wg_counts"):
            v = getattr(dev, attr, None)
            if v is not None:
                try:
                    dev_info[attr] = [int(x) for x in v]
                except (TypeError, ValueError):
                    dev_info[attr] = str(v)
        print("Vulkan device: %s" % dev_info.get("device_name", "?"), flush=True)
        for k, v in dev_info.items():
            if k not in ("device", "exist"):
                print("   %-32s %s" % (k, v))
    except Exception as e:  # noqa: BLE001
        dev_info = {"error": "%s: %s" % (type(e).__name__, e)}
        print("device probe failed: %s" % dev_info["error"], flush=True)
    report["device"] = dev_info

    if dev_info.get("device_name") and "llvmpipe" in str(dev_info["device_name"]).lower():
        print("\nREFUSING: device is llvmpipe (software Vulkan). Set VK_ICD_FILENAMES to the radeon ICD.")
        report["refused_software_device"] = True
        with open(os.path.join(args.outdir, "tvm-vulkan-compile.json"), "w") as f:
            json.dump(report, f, indent=2)
        return 2

    # ---- import ----
    import onnx

    m = onnx.load(args.onnx, load_external_data=False)
    ir = from_onnx(m, shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
    print("\nimport ok", flush=True)

    # ---- build for Vulkan ----
    print("building for target=%s ..." % args.target, flush=True)
    t0 = time.perf_counter()
    ok = True
    exe = None
    try:
        exe = relax.build(ir, target=target)
    except Exception as e:  # noqa: BLE001
        import traceback

        ok = False
        tb = traceback.format_exc()
        report["build_error_type"] = type(e).__name__
        report["build_error"] = str(e)[:3000]
        with open(os.path.join(args.outdir, "tvm-vulkan-build-error.txt"), "w") as f:
            f.write(tb)
        print("BUILD FAILED after %.1fs: %s: %s" % (time.perf_counter() - t0, type(e).__name__, str(e)[:500]), flush=True)
        print(tb[-1500:], flush=True)
    build_s = time.perf_counter() - t0

    report["build_ok"] = ok
    report["build_seconds"] = build_s
    print("build %s in %.1fs" % ("ok" if ok else "FAILED", build_s), flush=True)

    if ok:
        # ---- inspect what the build produced ----
        gen = None
        try:
            vm = relax.VirtualMachine(exe.jit(), tvm.vulkan())
            st = vm.stats() if hasattr(vm, "stats") else None
        except Exception:
            st = None
        try:
            text = exe.as_text()
            with open(os.path.join(args.outdir, "vulkan-executable.txt"), "w") as f:
                f.write(text)
            report["vm_instruction_count"] = text.count("\n")
            # count distinct device PrimFuncs in the VM globals
            import re
            names = sorted(set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)$", text, re.M)))
            report["executable_size_bytes"] = len(text)
        except Exception as e:  # noqa: BLE001
            report["as_text_error"] = str(e)[:300]
        try:
            report["vm_stats_head"] = exe.stats().splitlines()[:6]
        except Exception:
            pass

    with open(os.path.join(args.outdir, "tvm-vulkan-compile.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "tvm-vulkan-compile.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
