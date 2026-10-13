# -*- coding: utf-8 -*-
"""A5b：ncnn-Vulkan vs ORT-CPU 任务级一致性（16 篡改样本，**原尺寸**，无 resize）。

阶段B S6 前置门（方案 dawn-ledge-asp）：pnnx_a5_tasklevel.py 的 stretch512 版
只证了两条后端在同一缩图输入上的一致；ncnn 进生产 auto 链前须补验原尺寸
（C2 结论：生产一律原尺寸推理）。判据同 A5：|ΔAUC|≤2e-3、prob 在工作阈值
（thr99≈FPR1% 与 0.5）上的翻转像素占比 <0.1%；recall/Δdet/conf max_abs 仅记录。

分相执行（2026-10-02 实测：两后端同进程在 b5=2.43MP 被 OOM-killer 击杀，
exit137；ORT-CPU 单进程峰值 RSS≈6.0GB、ncnn 单进程 A4 实测可控，分相即可）：
  1) python3 work/pnnx_a5b_native.py --backend ort    # ORT-CPU 逐样本落 npz
  2) python3 work/pnnx_a5b_native.py --backend ncnn --device 1
  3) python3 work/pnnx_a5b_native.py --backend metrics # 读两侧 npz 算任务级指标
npz 目录 work/ablation_results/a5b_dumps/{ort,ncnn}/<sample>.npz（prob/conf/det）。

注：b5 样本 1926×1264=2.43MP 略高于生产守卫档位 2.4MP——本实验直接跑原尺寸
（A4 已实测 P4 在 1926×1264 无 OOM），守卫拒绝属 下游应用侧策略层，不在此门内。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

INPUT_DIR = os.path.join(HERE, "ablation_inputs")
RESULT_DIR = os.path.join(HERE, "ablation_results")
DUMP_DIR = os.path.join(RESULT_DIR, "a5b_dumps")


def prob_from_out(out0):
    a = np.asarray(out0)
    o = a[0] if a.ndim == 4 else a
    e = np.exp(o - o.max(axis=0, keepdims=True))
    return (e / e.sum(axis=0, keepdims=True))[1]


def load_manifest(manifest_path, limit, only=None):
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    if only:
        manifest = [m for m in manifest if m["name"] == only]
    if limit:
        manifest = manifest[:limit]
    return manifest


def run_ort(args):
    import cv2
    import onnxruntime as ort

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    sess.set_providers(["CPUExecutionProvider"], [{"log_severity_level": 3}])
    in_name = sess.get_inputs()[0].name
    out_dir = os.path.join(DUMP_DIR, "ort")
    os.makedirs(out_dir, exist_ok=True)
    for m in load_manifest(args.manifest, args.limit, args.only):
        name = m["name"]
        if args.skip_done and os.path.isfile(os.path.join(out_dir, "%s.npz" % name)):
            print("ORT skip %s" % name, flush=True)
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(INPUT_DIR, "samples", "%s.png" % name),
                                       cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(256.0)
        t0 = time.perf_counter()
        out, conf, det = sess.run(None, {in_name: x})
        ms = (time.perf_counter() - t0) * 1000
        np.savez_compressed(
            os.path.join(out_dir, "%s.npz" % name),
            prob=prob_from_out(out).astype(np.float32),
            conf=np.asarray(conf, np.float32),
            det=np.asarray(det, np.float32).ravel()[0],
            ms=np.float64(ms))
        print("ORT %s %.0fms" % (name, ms), flush=True)
    print("A5B_ORT_DONE", flush=True)


def run_ncnn(args):
    import cv2
    import ncnn

    ncnn.create_gpu_instance()
    n = ncnn.Net()
    n.opt.use_vulkan_compute = True
    n.set_vulkan_device(args.device)
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    md = os.path.join(HERE, "ncnn_a1")
    assert n.load_param(os.path.join(md, "trufor.ts.ncnn.param")) == 0
    assert n.load_model(os.path.join(md, "trufor.ts.ncnn.bin")) == 0
    in_nm, out_nms = list(n.input_names())[0], list(n.output_names())
    out_dir = os.path.join(DUMP_DIR, "ncnn")
    os.makedirs(out_dir, exist_ok=True)
    for m in load_manifest(args.manifest, args.limit, args.only):
        name = m["name"]
        if args.skip_done and os.path.isfile(os.path.join(out_dir, "%s.npz" % name)):
            print("NCNN skip %s" % name, flush=True)
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(INPUT_DIR, "samples", "%s.png" % name),
                                       cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(256.0)
        ex = n.create_extractor()
        # ncnn.Mat 零拷贝+上传延迟至 extract：arr 必须存活到全部 extract 完成
        arr = np.ascontiguousarray(x[0])
        ex.input(in_nm, ncnn.Mat(arr))
        t0 = time.perf_counter()
        _, n_out = ex.extract(out_nms[0])
        _, n_conf = ex.extract(out_nms[1])
        _, n_det = ex.extract(out_nms[2])
        ms = (time.perf_counter() - t0) * 1000
        prob = prob_from_out(np.array(n_out).astype(np.float32))
        np.savez_compressed(
            os.path.join(out_dir, "%s.npz" % name),
            prob=prob.astype(np.float32),
            conf=np.array(n_conf).astype(np.float32),
            det=np.array(n_det).astype(np.float32).ravel()[0],
            ms=np.float64(ms))
        del arr, ex, n_out, n_conf, n_det
        print("NCNN %s %.0fms" % (name, ms), flush=True)
    print("A5B_NCNN_DONE", flush=True)


def run_metrics(args):
    from resize_ablation import compute_metrics
    import cv2

    rows = []
    tmp_mask = os.path.join(RESULT_DIR, "_mask_native_tmp.png")
    for m in load_manifest(args.manifest, args.limit, args.only):
        name = m["name"]
        o = np.load(os.path.join(DUMP_DIR, "ort", "%s.npz" % name))
        c = np.load(os.path.join(DUMP_DIR, "ncnn", "%s.npz" % name))
        po, pn = o["prob"].astype(np.float64), c["prob"].astype(np.float64)
        mp = os.path.join(INPUT_DIR, "samples", "%s_mask.png" % name)
        cv2.imwrite(tmp_mask, (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255)
        ro = compute_metrics(po, tmp_mask)
        rn = compute_metrics(c["prob"].astype(np.float64), tmp_mask)
        thr99 = ro["thr_out99"]
        H, W = po.shape
        conf_o = o["conf"].reshape(-1, H, W)[0].astype(np.float64)
        conf_n = c["conf"].reshape(-1, H, W)[0].astype(np.float64)

        def flip(a, b, t):
            return float(np.mean((a > t) != (b > t)))

        row = {
            "sample": name, "ratio": m["ratio"], "size": "%dx%d" % (W, H),
            "auc_ort": ro["auc"], "auc_ncnn": rn["auc"],
            "rec99_ort": ro["recall@out99"], "rec99_ncnn": rn["recall@out99"],
            "rec05_ort": ro["recall@0.5"], "rec05_ncnn": rn["recall@0.5"],
            "thr99_ort": thr99, "thr99_ncnn": rn["thr_out99"],
            "flip05_prob": flip(po, pn, 0.5),
            "flip99_prob": flip(po, pn, thr99),
            "flip05_conf": flip(conf_o, conf_n, 0.5),
            "conf_maxabs": float(np.abs(conf_o - conf_n).max()),
            "prob_maxabs": float(np.abs(po - pn).max()),
            "det_ort": float(1.0 / (1.0 + np.exp(-float(o["det"])))),
            "det_ncnn": float(1.0 / (1.0 + np.exp(-float(c["det"])))),
            "ort_ms": round(float(o["ms"]), 1), "ncnn_ms": round(float(c["ms"]), 1),
        }
        rows.append(row)
        print("ROW %s" % json.dumps(row, ensure_ascii=False), flush=True)
    if os.path.exists(tmp_mask):
        os.remove(tmp_mask)
    out = args.out or os.path.join(RESULT_DIR, "a5b_native.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def stat(k):
        return np.array([r[k] for r in rows], dtype=np.float64)
    d_auc = stat("auc_ncnn") - stat("auc_ort")
    d_rec = stat("rec99_ncnn") - stat("rec99_ort")
    d_rec05 = stat("rec05_ncnn") - stat("rec05_ort")
    f99, f05, fc = stat("flip99_prob"), stat("flip05_prob"), stat("flip05_conf")
    d_det = stat("det_ncnn") - stat("det_ort")
    lines = [
        "=== A5b 汇总（原尺寸，n=%d，Δ = ncnn - ORT） ===" % len(rows),
        "ΔAUC: mean %.5f  min %.5f  max %.5f" % (d_auc.mean(), d_auc.min(), d_auc.max()),
        "|ΔAUC| max: %.5f（门 2e-3）" % np.abs(d_auc).max(),
        "Δrecall@FPR1%%: mean %.4f  min %.4f  max %.4f" % (d_rec.mean(), d_rec.min(), d_rec.max()),
        "Δrecall@0.5: mean %.4f  min %.4f  max %.4f" % (d_rec05.mean(), d_rec05.min(), d_rec05.max()),
        "prob翻转占比@0.5: mean %.5f  max %.5f（门 0.001）" % (f05.mean(), f05.max()),
        "prob翻转占比@thr99: mean %.5f  max %.5f（门 0.001）" % (f99.mean(), f99.max()),
        "conf翻转占比@0.5: mean %.5f  max %.5f" % (fc.mean(), fc.max()),
        "Δdet score: mean %.5f  max|.| %.5f" % (d_det.mean(), np.abs(d_det).max()),
        "conf max_abs: min %.3e  max %.3e（记录项）" % (stat("conf_maxabs").min(), stat("conf_maxabs").max()),
        "耗时(ms): ORT mean %.0f | ncnn mean %.0f" % (stat("ort_ms").mean(), stat("ncnn_ms").mean()),
    ]
    s = "\n".join(lines)
    print(s, flush=True)
    with open(os.path.splitext(out)[0] + "_summary.txt", "w", encoding="utf-8") as f:
        f.write(s + "\n")
    gate = (np.abs(d_auc).max() <= 2e-3) and (f99.max() < 1e-3) and (f05.max() < 1e-3)
    print("A5B_GATE=%s" % ("PASS" if gate else "FAIL"), flush=True)


def main():
    global DUMP_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ort", "ncnn", "metrics"], required=True)
    ap.add_argument("--device", type=int, default=1, help="vulkan device index")
    ap.add_argument("--model", default=os.path.join(
        REPO_ROOT, "reports", "onnx-dynamo", "trufor_dynamic_dynamo.onnx"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=os.path.join(INPUT_DIR, "manifest.json"),
                    help="样本清单路径（默认既有16样本manifest）")
    ap.add_argument("--dump-dir", default=DUMP_DIR,
                    help="npz 落盘目录（默认 a5b_dumps；按设备分目录用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="只处理manifest中指定name的样本")
    ap.add_argument("--skip-done", action="store_true", help="已有npz的样本跳过（fresh进程逐样本续跑用）")
    args = ap.parse_args()
    DUMP_DIR = args.dump_dir
    if args.backend == "ort":
        run_ort(args)
    elif args.backend == "ncnn":
        run_ncnn(args)
    else:
        run_metrics(args)


if __name__ == "__main__":
    main()
