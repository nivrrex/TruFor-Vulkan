# -*- coding: utf-8 -*-
"""A5-lite：ncnn-Vulkan vs ORT-CPU 任务级一致性（16 篡改样本，stretch512）。

判据（2026-10-02 评审确定）：AUC/recall@FPR1% 差异在样本噪声内，且 conf 在
工作阈值上的翻转像素占比很小（<0.1%）；max_abs 仅记录不作门限。
指标复用 resize_ablation.compute_metrics（外区=腐蚀让带后的区外，thr99≈FPR1%）。

用法：
  python3 work/pnnx_a5_tasklevel.py --device 1 [--limit N]
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
from resize_ablation import compute_metrics  # noqa: E402

INPUT_DIR = os.path.join(HERE, "ablation_inputs")
RESULT_DIR = os.path.join(HERE, "ablation_results")


def prob_from_out(out0):
    a = np.asarray(out0)
    o = a[0] if a.ndim == 4 else a
    e = np.exp(o - o.max(axis=0, keepdims=True))
    return (e / e.sum(axis=0, keepdims=True))[1]


def main():
    import cv2
    import onnxruntime as ort
    import ncnn

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1, help="vulkan device index")
    ap.add_argument("--model", default=os.path.join(
        REPO_ROOT, "reports", "onnx-dynamo", "trufor_dynamic_dynamo.onnx"))
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "a5_tasklevel.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(INPUT_DIR, "manifest.json"), encoding="utf-8"))
    if args.limit:
        manifest = manifest[:args.limit]

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    sess.set_providers(["CPUExecutionProvider"], [{"log_severity_level": 3}])
    in_name = sess.get_inputs()[0].name

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
    ncnn_names = list(n.input_names()), list(n.output_names())

    os.makedirs(RESULT_DIR, exist_ok=True)
    rows = []
    for m in manifest:
        name = m["name"]
        sp = os.path.join(INPUT_DIR, "samples", "%s.png" % name)
        mp = os.path.join(INPUT_DIR, "samples", "%s_mask.png" % name)
        img = cv2.cvtColor(cv2.imread(sp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        feed = cv2.resize(img, (512, 512),
                          interpolation=cv2.INTER_AREA if max(W, H) > 512 else cv2.INTER_LINEAR)
        x = feed.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(256.0)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
        mask512 = cv2.resize(mask.astype(np.uint8), (512, 512),
                             interpolation=cv2.INTER_NEAREST).astype(bool)

        t0 = time.perf_counter()
        o_out, o_conf, o_det = sess.run(None, {in_name: x})
        ort_ms = (time.perf_counter() - t0) * 1000

        ex = n.create_extractor()
        # ncnn.Mat 为零拷贝包装且上传延迟至 extract：arr 必须存活到全部 extract 完成
        arr = np.ascontiguousarray(x[0])
        ex.input(ncnn_names[0][0], ncnn.Mat(arr))
        t0 = time.perf_counter()
        _, n_out = ex.extract(ncnn_names[1][0])
        _, n_conf = ex.extract(ncnn_names[1][1])
        _, n_det = ex.extract(ncnn_names[1][2])
        ncnn_ms = (time.perf_counter() - t0) * 1000
        p_ncnn = np.array(n_out).astype(np.float32)
        c_ncnn = np.array(n_conf).astype(np.float32)
        d_ncnn = np.array(n_det).astype(np.float32)
        del arr

        po, pn = prob_from_out(o_out), prob_from_out(p_ncnn)

        def metrics(pmap):
            import cv2 as _c
            tmp = os.path.join(RESULT_DIR, "_mask512_tmp.png")
            _c.imwrite(tmp, (mask512.astype(np.uint8)) * 255)
            r = compute_metrics(pmap.astype(np.float64), tmp)
            os.remove(tmp)
            return r

        ro, rn = metrics(po), metrics(pn)
        thr99 = ro["thr_out99"]
        conf_o = np.asarray(o_conf).astype(np.float64).reshape(512, 512)
        conf_n = c_ncnn.astype(np.float64).reshape(512, 512)

        def flip(a, b, t):
            return float(np.mean((a > t) != (b > t)))

        det_s_o = float(1.0 / (1.0 + np.exp(-float(np.asarray(o_det).ravel()[0]))))
        det_s_n = float(1.0 / (1.0 + np.exp(-float(d_ncnn.ravel()[0]))))

        row = {
            "sample": name, "ratio": m["ratio"],
            "auc_ort": ro["auc"], "auc_ncnn": rn["auc"],
            "rec99_ort": ro["recall@out99"], "rec99_ncnn": rn["recall@out99"],
            "rec05_ort": ro["recall@0.5"], "rec05_ncnn": rn["recall@0.5"],
            "thr99_ort": thr99, "thr99_ncnn": rn["thr_out99"],
            "flip05_prob": flip(po, pn, 0.5),
            "flip99_prob": flip(po, pn, thr99),
            "flip05_conf": flip(conf_o, conf_n, 0.5),
            "conf_maxabs": float(np.abs(conf_o - conf_n).max()),
            "prob_maxabs": float(np.abs(po - pn).max()),
            "det_ort": det_s_o, "det_ncnn": det_s_n,
            "ort_ms": round(ort_ms, 1), "ncnn_ms": round(ncnn_ms, 1),
        }
        rows.append(row)
        print("ROW %s" % json.dumps(row, ensure_ascii=False), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def stat(k):
        a = np.array([r[k] for r in rows], dtype=np.float64)
        return a
    d_auc = stat("auc_ncnn") - stat("auc_ort")
    d_rec = stat("rec99_ncnn") - stat("rec99_ort")
    d_rec05 = stat("rec05_ncnn") - stat("rec05_ort")
    f99 = stat("flip99_prob")
    f05 = stat("flip05_prob")
    fc = stat("flip05_conf")
    d_det = stat("det_ncnn") - stat("det_ort")
    lines = [
        "=== A5-lite 汇总（n=%d，Δ = ncnn - ORT） ===" % len(rows),
        "ΔAUC: mean %.5f  min %.5f  max %.5f" % (d_auc.mean(), d_auc.min(), d_auc.max()),
        "Δrecall@FPR1%%: mean %.4f  min %.4f  max %.4f" % (d_rec.mean(), d_rec.min(), d_rec.max()),
        "Δrecall@0.5: mean %.4f  min %.4f  max %.4f" % (d_rec05.mean(), d_rec05.min(), d_rec05.max()),
        "prob翻转占比@0.5: mean %.5f  max %.5f" % (f05.mean(), f05.max()),
        "prob翻转占比@thr99: mean %.5f  max %.5f" % (f99.mean(), f99.max()),
        "conf翻转占比@0.5: mean %.5f  max %.5f" % (fc.mean(), fc.max()),
        "Δdet score: mean %.5f  max|.| %.5f" % (d_det.mean(), np.abs(d_det).max()),
        "conf max_abs: min %.3e  max %.3e" % (stat("conf_maxabs").min(), stat("conf_maxabs").max()),
        "耗时(ms): ORT mean %.0f | ncnn mean %.0f" % (stat("ort_ms").mean(), stat("ncnn_ms").mean()),
    ]
    s = "\n".join(lines)
    print(s, flush=True)
    with open(os.path.join(RESULT_DIR, "a5_summary.txt"), "w", encoding="utf-8") as f:
        f.write(s + "\n")
    print("A5_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
