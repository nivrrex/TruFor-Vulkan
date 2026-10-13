"""One timing config from the cached executable. TVM_VM_SKIP is read at library
load, so each configuration is its own process; the driver interleaves baselines.
"""
import os, sys, time, statistics
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/vendor/tvm-src/python")
import tvm
from tvm import relax
from PIL import Image
n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
t0 = time.time()
dev = tvm.vulkan()
mod = tvm.runtime.load_module("/tmp/vmexe.so")
vm = relax.VirtualMachine(mod, dev)
img = Image.open(ROOT + "/truFor-src/test_docker/images/tampered2.png")
x = (np.asarray(img.convert("RGB").resize((512, 512)), np.float32)
     .transpose(2, 0, 1)[None] / 255.).astype(np.float32)
tin = tvm.runtime.tensor(x, dev)
for _ in range(3):
    vm["main"](tin); dev.sync()
ts = []
for _ in range(n):
    t0 = time.perf_counter(); vm["main"](tin); dev.sync()
    ts.append(time.perf_counter() - t0)
v = sorted(ts)
print("RESULT %.2f %.2f %.2f %.2f" % (statistics.median(v) * 1e3,
      (v[3 * n // 4] - v[n // 4]) * 1e3, v[0] * 1e3, v[-1] * 1e3))
