"""Shared plumbing for the GPU capability / scaling benchmark (Gate 0-7).

Kept deliberately separate from every correctness script: nothing here may
change how the model computes. The only thing this module is allowed to do is
time work and record environment facts.

Timing protocol, fixed for the whole study:
    warmup    = 3
    repeats   = 15   (10 for the long tile sweep)
Reported: min, p10, median, p90, max  -- not just the mean, because a single
dispatch hiccup otherwise dominates a 2.6 s measurement.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time

WARMUP = 3
REPEATS = 15


# --------------------------------------------------------------------- stats
def summarize(samples):
    s = sorted(samples)
    n = len(s)
    def pct(p):
        if n == 1:
            return s[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return s[f] + (s[c] - s[f]) * (k - f)
    return {
        "n": n,
        "min": s[0],
        "p10": pct(0.10),
        "median": statistics.median(s),
        "p90": pct(0.90),
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if n > 1 else 0.0,
    }


# --------------------------------------------------------------- environment
def vram_bytes():
    for p in ("/sys/class/drm/card0/mem_info_vram_total",
              "/sys/class/drm/card0/mem_info_total"):
        try:
            with open(p) as f:
                return int(f.read().strip())
        except OSError:
            continue
    return None


def vram_used():
    for p in ("/sys/class/drm/card0/mem_info_vram_used",
              "/sys/class/drm/card0/mem_info_used"):
        try:
            with open(p) as f:
                return int(f.read().strip())
        except OSError:
            continue
    return None


def vram_peak_sample(ms=25):
    """Background sampler for VRAM high-water mark across one benchmark region."""
    import threading

    peak = {"v": vram_used() or 0, "stop": False}

    def run():
        while not peak["stop"]:
            v = vram_used()
            if v and v > peak["v"]:
                peak["v"] = v
            time.sleep(ms / 1000.0)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return peak


def collect_env():
    import numpy

    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "cpu_count": os.cpu_count(),
        "vram_total_bytes": vram_bytes(),
        "env": {k: os.environ.get(k) for k in (
            "VK_DRIVER_FILES", "VK_ICD_FILENAMES", "LD_LIBRARY_PATH",
            "PYTHONPATH", "TVM_LIBRARY_PATH")},
    }
    try:
        import tvm
        env["tvm"] = getattr(tvm, "__version__", None) or tvm._version.__version__
        li = tvm.support.libinfo()
        env["tvm_use_vulkan"] = li.get("USE_VULKAN")
        env["tvm_use_llvm"] = li.get("USE_LLVM")
        dev = tvm.vulkan()
        env["vulkan"] = {
            "device": dev.device_name,
            "exists": bool(dev.exist),
            "api_version": str(dev.api_version),
            "driver_version": str(dev.driver_version),
        }
        for attr in ("max_threads_per_block", "max_shared_memory_per_block",
                     "thread_warp_size", "l2_cache_size_bytes",
                     "available_global_memory", "max_wg_counts",
                     "sub_group_size", "warp_size", "multi_processor_count"):
            if hasattr(dev, attr):
                try:
                    env["vulkan"][attr] = int(getattr(dev, attr))
                except (TypeError, ValueError):
                    env["vulkan"][attr] = str(getattr(dev, attr))
    except Exception as e:  # noqa: BLE001
        env["tvm_error"] = "%s: %s" % (type(e).__name__, e)
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True,
                             text=True, timeout=60).stdout
        for line in out.splitlines():
            s = line.strip()
            if s.startswith(("deviceName", "driverName", "driverInfo", "apiVersion",
                             "deviceType", "conformanceVersion")):
                env["vulkan"].setdefault("summary", []).append(s)
    except Exception as e:  # noqa: BLE001
        env["vulkaninfo_error"] = str(e)[:200]
    return env


def model_facts(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    shapes = {}
    for i in m.graph.input:
        shapes[i.name] = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
    for o in m.graph.output:
        shapes[o.name] = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
    st = os.stat(path)
    h = subprocess.run(["sha256sum", path], capture_output=True, text=True).stdout.split()[0]
    return {"path": path, "bytes": st.st_size, "sha256": h,
            "nodes": len(m.graph.node), "inputs": shapes}


# -------------------------------------------------------------------- runner
class Timer:
    def __init__(self, warmup=WARMUP, repeats=REPEATS):
        self.warmup = warmup
        self.repeats = repeats
        self.samples = []
        self.last_result = None

    def __call__(self, fn):
        """fn() must return the value to be drained (so D2H is inside the region)."""
        for _ in range(self.warmup):
            self.last_result = fn()
        self.samples = []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            self.last_result = fn()
            self.samples.append(time.perf_counter() - t0)
        return self.samples

    def report(self):
        return summarize(self.samples)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print("wrote %s" % path)
