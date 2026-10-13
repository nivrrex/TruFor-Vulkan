#!/usr/bin/env python3
"""A4: ncnn throughput + VRAM/RSS behaviour on Vulkan devices.

Usage: pnnx_a4_perf.py --device <vulkan index> [--sizes 512x512 1926x1264 ...]
One process = all listed sizes, but for OOM-risk sweeps run one size per process
from outside. Steady reps create a fresh extractor each time. VRAM is polled in
a background thread (AMD sysfs + nvidia-smi). conf md5 printed for cross-process
determinism checks (A4 x #44).
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys
import threading
import time

import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(HERE, "ncnn_a1")


def peak_rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return -1


def vram_snapshot_mib():
    out = {}
    for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try:
            out["card" + f.split("/")[4][4:]] = int(open(f).read()) / 1048576.0
        except Exception:
            pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        for i, line in enumerate(r.stdout.strip().splitlines()):
            out["nv%d" % i] = float(line)
    except Exception:
        pass
    return out


class VramPoller(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = {}
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            for k, v in vram_snapshot_mib().items():
                if v > self.peak.get(k, -1):
                    self.peak[k] = v
            self.stop_flag.wait(self.interval)

    def start_poll(self):
        self.peak = {}
        self.stop_flag.clear()

    def stop_poll(self):
        self.stop_flag.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, type=int)
    ap.add_argument("--sizes", nargs="*", default=["512x512", "1926x1264"])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    ncnn.create_gpu_instance()
    gi = ncnn.get_gpu_info(args.device)
    print("device %d: %s" % (args.device, gi.device_name()), flush=True)
    n = ncnn.Net()
    n.opt.use_vulkan_compute = True
    n.set_vulkan_device(args.device)
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    assert n.load_param(os.path.join(MODELDIR, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(MODELDIR, "trufor.ts.ncnn.bin")) == 0
    in_name = list(n.input_names())[0]
    out_names = list(n.output_names())
    base = vram_snapshot_mib()
    print("after-load VRAM(MiB): %s" % base, flush=True)

    poller = VramPoller()
    poller.start()
    for spec in args.sizes:
        h, w = (int(v) for v in spec.split("x"))
        blob = np.random.RandomState(0).rand(3, h, w).astype("float32")
        poller.start_poll()
        t0 = time.perf_counter()
        ex = n.create_extractor()
        ex.input(in_name, ncnn.Mat(blob))
        mats = []
        for nm in out_names:
            ret, m = ex.extract(nm)
            assert ret == 0, "cold %dx%d rc=%d" % (h, w, ret)
            mats.append(np.array(m))
        cold = time.perf_counter() - t0
        conf_md5 = hashlib.md5(mats[1].astype("float32").tobytes()).hexdigest()[:12]
        conf_sum = float(mats[1].astype("float64").sum())
        del mats
        stimes = []
        for _ in range(args.reps):
            ex = n.create_extractor()
            ex.input(in_name, ncnn.Mat(blob))
            t1 = time.perf_counter()
            for nm in out_names:
                ret, m = ex.extract(nm)
                assert ret == 0
            stimes.append(time.perf_counter() - t1)
        del blob
        poller.stop_poll()
        time.sleep(0.1)
        print("%4dx%-4d cold=%.2fs steady=%s min=%.2fs conf_md5=%s conf_sum=%.8e peakVRAM=%s peakRSS=%.0fMB"
              % (h, w, cold, " ".join("%.2f" % t for t in stimes), min(stimes),
                 conf_md5, conf_sum, poller.peak, peak_rss_mb()), flush=True)
    poller.stop_poll()
    print("A4_DONE device=%s" % args.device, flush=True)
    # destroy_gpu_instance while Net alive crashes at teardown (destruct order);
    # let process exit release the GPU instance.
    return 0


if __name__ == "__main__":
    sys.exit(main())
