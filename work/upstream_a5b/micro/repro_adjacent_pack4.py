#!/usr/bin/env python3
"""P0-b ④ 相邻层迷你网格：Split / Crop（全网格）+ Slice / Concat / ReLU（探针）

背景（P0-b ④ 执行中发现，[已验证事实]）：
  迷你网中直接抽取 **Slice / Concat / ReLU** 的 GPU 输出必 SIGSEGV(139)，
  与 (H·W)%4 无关、与 blob 命名无关、与输入 Mat 是否零拷贝无关，且 **stock 轮
  同样崩**——即已知的"第二缺陷"（`work/STATE.md:280-283`、C6/C7），非本补丁引入。
  纯净搬运层 Split / Crop 可正常抽取。

因此本脚本：
  1) 对 Split / Crop 跑完整 c×size 网格，逐例比对 GPU 回读 == CPU 回读（位级）；
  2) 对 Slice / Concat / ReLU 用**子进程逐例**探测，记录 exit code（139=第二缺陷），
     不重试、不掩盖；
  3) Slice/Concat 在**真实全网**中的覆盖由 ② 的逐 blob rel_L2 提供（见 REPORT_relL2.md）。

Run: python repro_adjacent_pack4.py [device]      （默认 1 = Tesla P4）
Marker: REPRO_ADJACENT_PACK4_DONE
"""
import os
import subprocess
import sys

import numpy as np
import ncnn

DEVICE = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1] != "--case") else 1
SMALL = [(1, 1), (2, 1), (1, 2), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2),
         (3, 3), (4, 4), (5, 3), (3, 5), (6, 2)]
LARGE = [(967, 714), (967, 712), (715, 967), (714, 968)]


def param_split(c, h, w):
    return ("7767517\n2 3\nInput  in0 0 1 in0 0=%d 1=%d 2=%d\n"
            "Split  sp  1 2 in0 a b\n" % (w, h, c))


def param_crop(c, h, w):
    return ("7767517\n2 2\nInput  in0 0 1 in0 0=%d 1=%d 2=%d\n"
            "Crop   cr  1 1 in0 out 0=0 1=0 2=0 3=%d 4=%d 5=%d\n" % (w, h, c, w, h, c))


def param_slice(c, h, w):
    k = max(1, c // 2)
    return ("7767517\n2 3\nInput  in0 0 1 in0 0=%d 1=%d 2=%d\n"
            "Slice  sl  1 2 in0 a b -23300=2,%d,-233 1=0\n" % (w, h, c, k))


def param_concat(c, h, w):
    k = max(1, c // 2)
    return ("7767517\n3 4\nInput  in0 0 1 in0 0=%d 1=%d 2=%d\n"
            "Slice  sl  1 2 in0 a b -23300=2,%d,-233 1=0\n"
            "Concat cc  2 1 a b out 0=0\n" % (w, h, c, k))


def param_relu(c, h, w):
    return ("7767517\n2 2\nInput  in0 0 1 in0 0=%d 1=%d 2=%d\n"
            "ReLU   r   1 1 in0 out\n" % (w, h, c))


def make_net(param, gpu):
    n = ncnn.Net()
    n.opt.use_vulkan_compute = gpu
    for a in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic"):
        setattr(n.opt, a, False)
    assert n.load_param_mem(param) == 0, "load_param_mem failed"
    return n


def extract(net, blob, x):
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    ret, m = ex.extract(blob)
    if ret != 0 or m.empty():
        return None
    return np.array(m)


def child_case(kind, c, h, w, blob):
    param = {"split": param_split, "crop": param_crop, "slice": param_slice,
             "concat": param_concat, "relu": param_relu}[kind](c, h, w)
    x = np.ascontiguousarray(
        np.arange(c * h * w, dtype="float32").reshape(c, h, w) + 1.0)
    nc = make_net(param, False)
    ng = make_net(param, True)
    ncnn.create_gpu_instance()
    ng.set_vulkan_device(DEVICE)
    a = extract(nc, blob, x)
    b = extract(ng, blob, x)
    if a is None or b is None:
        return 3
    return 0 if np.array_equal(a, b) else 4


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--case":
        kind, c, h, w, blob = (sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                               int(sys.argv[5]), sys.argv[6])
        return child_case(kind, c, h, w, blob)

    print("ncnn version:", ncnn.__version__, flush=True)
    rs = np.random.RandomState(0)

    # --- 1) Split / Crop 全网格（in-process） ---
    total = bad = 0
    for kind, fn, blob in (("Split", param_split, "a"), ("Crop", param_crop, "out")):
        print("\n=== %s (in-process) ===" % kind, flush=True)
        for (h, w) in SMALL + LARGE:
            cs = range(1, 9) if (h, w) in SMALL else (2, 3, 4)
            row = []
            for c in cs:
                x = np.ascontiguousarray(rs.rand(c, h, w).astype("float32"))
                param = fn(c, h, w)
                a = extract(make_net(param, False), blob, x)
                ng = make_net(param, True)
                ng.set_vulkan_device(DEVICE)
                b = extract(ng, blob, x)
                total += 1
                ok = a is not None and b is not None and np.array_equal(a, b)
                if not ok:
                    bad += 1
                row.append("ok" if ok else "BAD")
            print("  %4dx%-4d(hw%%4=%d): %s" %
                  (h, w, (h * w) % 4, " ".join("%-3s" % v for v in row)), flush=True)
    print("\nSplit/Crop extractions: %d, mismatches: %d" % (total, bad), flush=True)

    # --- 2) Slice / Concat / ReLU 子进程探针 ---
    print("\n=== Slice / Concat / ReLU 子进程探针（139=第二缺陷） ===", flush=True)
    probe = [(2, 3, 3), (3, 3, 3), (2, 5, 3), (2, 967, 714), (2, 714, 968)]
    crashes = {}
    for kind, blob in (("slice", "a"), ("concat", "out"), ("relu", "out")):
        row = []
        for (c, h, w) in probe:
            r = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--case", kind, str(c), str(h), str(w), blob],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rc = r.returncode if r.returncode >= 0 else 128 - r.returncode
            row.append("%dx%d:%s" % (c, (h * w) % 4, rc))
            if rc != 0:
                crashes[kind] = crashes.get(kind, 0) + 1
        print("  %-7s %s" % (kind, " ".join(row)), flush=True)
    print("\n探针非零退出（第二缺陷）计数: %s" % crashes, flush=True)
    print("REPRO_ADJACENT_PACK4_DONE", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
