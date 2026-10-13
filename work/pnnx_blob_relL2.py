#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-b ② 判定性仪器：ncnn-Vulkan(补丁轮) vs ncnn-CPU 逐 blob 逐元素 rel_L2。

判据见 work/p0b_CRITERIA.md（运行前已固定提交）。本脚本只产生每 blob 的数值
与形态标记，不保存原始张量。

协议：
  - 同一进程内建 CPU 网（use_vulkan_compute=False）与 GPU 网（True, device）；
  - 输入同一份 RandomState(0) 张量（或 --png）；
  - **每个 blob 各用 fresh extractor**，规避"多输出同 extractor"已知 SIGSEGV；
  - 逐 blob 流式计算：rel_L2 = ||gpu-cpu||_2 / max(||cpu||_2,1e-30)、max_abs；
  - rel_L2 > --shift-cap（默认 1e-4）时做**逐平面整数平移检测**：
      把扁平数组按 c 个平面切分，对每个平面在整数位移 s∈[-8,8] 内找
      ||b[s:]-a[:-s]|| 最小者；若最小相对残差 < 1e-5 且 s≠0 → 标记形态错位。
    该形态标记与数值大小无关。
  - extract 返回非零/空/维度<2 的 blob 记 status=skip，列清单，不计入通过。

用法：
  python3 work/pnnx_blob_relL2.py --size HxW --device 1 --range 1-550 --out X.tsv
分块（每块 fresh 进程）是必需的：GPU 侧单进程约 600 blob 处会 OOM(137)。
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM = os.path.join(HERE, "ncnn_a1", "trufor.ts.ncnn.param")
MBIN = os.path.join(HERE, "ncnn_a1", "trufor.ts.ncnn.bin")
# 2026-10-14：允许命令行覆盖资产（默认仍为上面的上游原始模型，向后兼容）。
# 原实现把 param/bin 写死，导致无法用本仪器检验**改写过的** param（如 SDPA 改写版
# 会改变 blob 编号与层数），只能测上游原始模型。
PARAM_OVERRIDE = None
BIN_OVERRIDE = None
SHIFT_SMAX = 8
SHIFT_RESID = 1e-5


def make_net(side, device, lightmode=False):
    import ncnn

    n = ncnn.Net()
    n.opt.use_vulkan_compute = (side == "gpu")
    if lightmode:
        n.opt.lightmode = True
    for a in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic"):
        setattr(n.opt, a, False)
    for a in ("use_bf16_packed", "use_bf16_storage", "use_bf16_arithmetic"):
        if hasattr(n.opt, a):
            setattr(n.opt, a, False)
    if side == "gpu":
        ncnn.create_gpu_instance()
        n.set_vulkan_device(device)
    if n.load_param(PARAM_OVERRIDE or PARAM) != 0:
        raise RuntimeError("load_param失败")
    if n.load_model(BIN_OVERRIDE or MBIN) != 0:
        raise RuntimeError("load_model失败")
    return n, ncnn


def extract_one(net, ncnn, x, blob):
    """fresh extractor 单取一个 blob；失败返回 None。"""
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    ret, m = ex.extract(blob)
    a = None
    if ret == 0 and not m.empty() and m.dims >= 2:
        a = np.array(m)
        if a.size == 0:
            a = None
    del ex, m
    return a


def shift_flag(a, b, c, smax=SHIFT_SMAX):
    """逐平面整数平移检测。返回 (flag, s, resid, tailzero)。

    跳过退化平面（范数或标准差为 0），否则全零/常量平面会与任意位移平凡匹配。
    flag = 存在非退化平面且最佳位移残差 < SHIFT_RESID。tailzero 表示该位移下
    b 的尾接 |s| 个元素是否恰为零（上传缺陷"末平面尾部未写入=新鲜零"的特征）。
    """
    if c < 2:
        return False, 0, 0.0, False
    fa = a.astype(np.float64).ravel()
    fb = b.astype(np.float64).ravel()
    if fa.size != fb.size or fa.size % c != 0:
        return False, 0, 0.0, False
    n = fa.size // c
    for p in range(c):
        pa = fa[p * n:(p + 1) * n]
        pb = fb[p * n:(p + 1) * n]
        na = np.linalg.norm(pa)
        if na < 1e-9 or pa.std() == 0.0:
            continue
        best = (1e30, 0)
        for s in range(-smax, smax + 1):
            if s == 0 or abs(s) >= pa.size:
                continue
            d = (pb[s:] - pa[:-s]) if s > 0 else (pb[:s] - pa[-s:])
            r = np.linalg.norm(d) / na
            if r < best[0]:
                best = (r, s)
        if best[0] < SHIFT_RESID:
            k = abs(best[1])
            tailzero = bool(np.all(pb[-k:] == 0.0))
            return True, best[1], best[0], tailzero
    return False, 0, 0.0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default=None,
                    help="覆盖 param 路径（默认上游 work/ncnn_a1/trufor.ts.ncnn.param）；"
                         "注意：包络判据是特定 param 口径，换 param 后包络不可直接比较")
    ap.add_argument("--bin", default=None, help="覆盖 bin 路径（默认与 param 同目录的原模型）")
    ap.add_argument("--size", default="714x967")
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--range", default=None, help="lo-hi，fresh 进程分块")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default=None)
    ap.add_argument("--shift-cap", type=float, default=1e-4,
                    help="rel_L2 超此值才做平移形态检测")
    ap.add_argument("--lightmode", action="store_true",
                    help="开启 ncnn lightmode 降低中间 blob 驻留内存")
    args = ap.parse_args()
    global PARAM_OVERRIDE, BIN_OVERRIDE
    if args.param:
        PARAM_OVERRIDE = os.path.abspath(args.param)
    if args.bin:
        BIN_OVERRIDE = os.path.abspath(args.bin)

    h, w = (int(v) for v in args.size.split("x"))
    lo, hi = (int(v) for v in (args.range.split("-") if args.range else (1, 10 ** 9)))
    if args.png:
        import cv2
        img = cv2.cvtColor(cv2.imread(args.png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = np.ascontiguousarray(img.transpose(2, 0, 1).astype(np.float32) / np.float32(256.0))
    else:
        x = np.random.RandomState(0).rand(3, h, w).astype("float32")

    ncpu, ncnn = make_net("cpu", -1, args.lightmode)
    ngpu, _ = make_net("gpu", args.device, args.lightmode)
    nb = len(list(ncpu.blobs()))
    hi = min(hi, nb)

    # 每侧一个持久 extractor（与 pnnx_gpu_blob_diff.sweep 同协议）：fresh extractor
    # 逐 blob 会使 ncnn 每个 blob 重跑整图，O(n^2)，实测 15s/blob 超时。
    ex_cpu = ncpu.create_extractor()
    ex_cpu.input("in0", ncnn.Mat(x))
    ex_gpu = ngpu.create_extractor()
    ex_gpu.input("in0", ncnn.Mat(x))

    def _get(ex, blob):
        ret, m = ex.extract(blob)
        a = None
        if ret == 0 and not m.empty() and m.dims >= 2:
            a = np.array(m)
            if a.size == 0:
                a = None
        return a

    n_skip = 0
    n_shift = 0
    worst = (0.0, 0)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("#blob\tdims\tc\th\tw\tcpu_norm\trel_l2\tmax_abs\tstatus\tshift_flag\tshift_s\tshift_resid\ttail_zero\n")
        for i in range(lo, hi + 1):
            if i % 200 == 0:
                print("relL2 %d/%d" % (i, nb), flush=True)
            a = _get(ex_cpu, str(i))
            b = _get(ex_gpu, str(i))
            if a is None or b is None or a.shape != b.shape:
                n_skip += 1
                f.write("%d\t-\t-\t-\t-\t-\t-\t-\tskip\t0\t0\t0\t0\n" % i)
                continue
            diff = b.astype(np.float64) - a.astype(np.float64)
            cn = float(np.linalg.norm(a.astype(np.float64)))
            rel = float(np.linalg.norm(diff) / max(cn, 1e-30))
            mx = float(np.abs(diff).max())
            flag, s, resid, tailzero = (False, 0, 0.0, False)
            if rel > args.shift_cap:
                flag, s, resid, tailzero = shift_flag(a, b, a.shape[0] if a.ndim >= 3 else 1)
            if flag:
                n_shift += 1
            if rel > worst[0]:
                worst = (rel, i)
            f.write("%d\t%d\t%d\t%d\t%d\t%.9e\t%.9e\t%.9e\tok\t%d\t%d\t%.3e\t%d\n"
                    % (i, a.ndim, a.shape[0], a.shape[1], a.shape[2] if a.ndim >= 3 else 1,
                       cn, rel, mx, 1 if flag else 0, s, resid, 1 if tailzero else 0))
    print("relL2 done range=%d-%d nb=%d skip=%d shift_flag=%d worst_rel=%.3e@blob%d"
          % (lo, hi, nb, n_skip, n_shift, worst[0], worst[1]), flush=True)
    print("BLOB_RELL2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
