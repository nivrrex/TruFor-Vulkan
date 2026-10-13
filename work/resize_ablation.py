# -*- coding: utf-8 -*-
"""resize-to-512 取证语义消融（C2 门控实验，2026-10-02）。

目的：验证"把任意尺寸输入 resize 到 512x512 喂给 fixed-512 GPU 路径"是否会
实质破坏 TruFor 对小面积篡改的可检出性。若破坏成立，M2 嵌入的 C2 冲突被
锁死；若不成立，C2 有 resize 绕行解。不需要 GPU，只用 ORT-CPU + dynamic ONNX。

设计要点（来自外部评审对实验的三点补充要求）：
  1. 篡改区面积占比分层：0.1% / 1% / 10%；
  2. 区分下采样（大图→512）与上采样（小图→512）两类 case，外加 512x512
     原尺寸对照组（harness 自检：该组各路径指标应几乎相等）；
  3. 指标 = 对基线路径的 AUC / 区域召回 / 最大连通域 IoU / 区域内外均值差，
     不比相关系数。

路径集合（每张样本图各跑一遍）：
  native        —— 原尺寸直接推理（内存不足时由子进程隔离 OOM，记 skipped）
  cap768        —— 长边缩到 768（生产 CPU 路径在内存受限时的近似口径）
  stretch512    —— 拉伸到 512x512（GPU 路径唯一能吃的形状）
  letterbox512  —— 等比放进 512x512 + BORDER_REPLICATE 补边（GPU 形状的另一种喂法）
  map 一律双线性放回原尺寸后在原始坐标系上与 mask 比。

篡改构造：确定性（固定种子），从另一张真实文档扫描图 splice 同尺寸区域
（跨图 splice，贴近"换章/贴图"场景；自 copy-move 对 TruFor 更难，不在本轮范围）。

子进程模式：--single 只跑一个 (sample, pass) 并输出一行 JSON——单张图 OOM
（RLIMIT_AS 触发 MemoryError 或 C++ 侧 abort/137）只损失一个子进程。

用法：
  python3 work/resize_ablation.py                 # 构造样本+全量扫描+汇总
  python3 work/resize_ablation.py --limit-samples 2   # 冒烟
依赖：numpy opencv-python-headless pillow onnxruntime；模型默认取
  reports/onnx-dynamo/trufor_dynamic_dynamo.onnx（相对仓库根）。
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(REPO_ROOT, "work", "ablation_inputs")
RESULT_DIR = os.path.join(REPO_ROOT, "work", "ablation_results")

RATIOS = (0.001, 0.01, 0.10)
PASSES = ("native", "cap768", "stretch512", "letterbox512")
SEED = 20261002
MEM_LIMIT_GIB = 12  # 每子进程 RLIMIT_AS


def _resize(img, w, h, down):
    import cv2
    return cv2.resize(img, (w, h),
                      interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)


def build_samples():
    """确定性构造篡改样本。返回 manifest 列表并落盘 png + mask。"""
    import cv2
    from PIL import Image

    raw = os.path.join(INPUT_DIR, "raw")
    out = os.path.join(INPUT_DIR, "samples")
    os.makedirs(out, exist_ok=True)

    def load_rgb(name):
        return np.array(Image.open(os.path.join(raw, name)).convert("RGB"))

    bases = {
        "b5": load_rgb("图片5.png"),    # 1264x1926 RGB
        "b28": load_rgb("图片28.png"),  # 1190x1684 RGB
        "b3": load_rgb("图片3.png"),    # 1000x1721 RGB
        "b22": load_rgb("图片22.png"),  # 714x967 RGBA->RGB
        "b24": load_rgb("图片24.png"),  # 785x844，供小图上采样 crop
    }
    donors = {"b5": "b28", "b28": "b5", "b3": "b24", "b22": "b24",
              "small_a": "b22", "ctrl512": "b22"}

    manifest = []

    def place_rects(rnd, W, H, side):
        margin = 3
        if side * 2 + 2 * margin > min(W, H):
            side = int((min(W, H) - 2 * margin) / 2)
        lo_x, hi_x = margin, W - margin - side
        lo_y, hi_y = margin, H - margin - side
        for _ in range(64):
            tx = rnd.randint(lo_x, hi_x); ty = rnd.randint(lo_y, hi_y)
            sx = rnd.randint(lo_x, hi_x); sy = rnd.randint(lo_y, hi_y)
            if abs(sx - tx) >= side or abs(sy - ty) >= side:
                return sx, sy, tx, ty, side
        return None

    def splice(key, img, ratio, tag):
        H, W = img.shape[:2]
        rnd = random.Random(f"{SEED}-{tag}")
        side = max(8, int(round((W * H * ratio) ** 0.5)))
        pr = place_rects(rnd, W, H, side)
        if pr is None:
            return None
        sx, sy, tx, ty, side = pr
        donor = bases[donors[key]]
        dH, dW = donor.shape[:2]
        dx = int(sx / W * dW); dy = int(sy / H * dH)
        dside_w = max(8, int(side / W * dW)); dside_h = max(8, int(side / H * dH))
        dside_w = min(dside_w, dW - 1 - dx); dside_h = min(dside_h, dH - 1 - dy)
        patch = _resize(donor[dy:dy + dside_h, dx:dx + dside_w], side, side,
                        down=(dside_w < side or dside_h < side))
        out_img = img.copy()
        out_img[ty:ty + side, tx:tx + side] = patch
        mask = np.zeros((H, W), np.uint8)
        mask[ty:ty + side, tx:tx + side] = 255
        name = f"{tag}_r{ratio}"
        cv2.imwrite(os.path.join(out, f"{name}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out, f"{name}_mask.png"), mask)
        manifest.append({"name": name, "base": key, "ratio": ratio,
                         "rect": [tx, ty, side]})
        return True

    for key in ("b5", "b28", "b3", "b22"):
        for ratio in RATIOS:
            splice(key, bases[key], ratio, key)

    # 上采样 case：380x540 小 crop（来自真实扫描，不是再采样产物）
    img = bases["b24"]
    bases["small_a"] = img[150:150 + 540, 200:200 + 380].copy()
    for ratio in RATIOS:
        splice("small_a", bases["small_a"], ratio, "small_a")

    # 512x512 对照组
    img = bases["b28"]
    bases["ctrl512"] = img[500:1012, 300:812].copy()
    splice("ctrl512", bases["ctrl512"], 0.01, "ctrl512")

    with open(os.path.join(INPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def _set_mem_limit(gib):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (gib * 2**30, gib * 2**30))
    except Exception:
        pass


def run_single(model_path, sample_path, mask_path, path_name, as_json):
    import cv2
    import onnxruntime as ort

    img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不到样本: {sample_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    _set_mem_limit(MEM_LIMIT_GIB)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    sess.set_providers(["CPUExecutionProvider"],
                       [{"log_severity_level": 3}])
    in_name = sess.get_inputs()[0].name

    if path_name == "native":
        feed, meta = img, {"feed_wh": [W, H]}
    elif path_name == "cap768":
        s = 768 / max(W, H)
        if s >= 1.0:
            feed, meta = img, {"feed_wh": [W, H], "note": "no-op(<=768)"}
        else:
            fw, fh = int(round(W * s)), int(round(H * s))
            feed, meta = _resize(img, fw, fh, True), {"feed_wh": [fw, fh]}
    elif path_name == "stretch512":
        feed, meta = _resize(img, 512, 512, max(W, H) > 512), {"feed_wh": [512, 512]}
    elif path_name == "letterbox512":
        s = 512 / max(W, H)
        fw, fh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        inner = _resize(img, fw, fh, max(W, H) > 512)
        pad_top = (512 - fh) // 2
        pad_left = (512 - fw) // 2
        feed = cv2.copyMakeBorder(inner, pad_top, 512 - fh - pad_top,
                                  pad_left, 512 - fw - pad_left,
                                  cv2.BORDER_REPLICATE)
        meta = {"feed_wh": [512, 512], "inner": [pad_left, pad_top, fw, fh]}
    else:
        raise SystemExit(f"未知路径: {path_name}")

    x = feed.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(256.0)
    t0 = time.perf_counter()
    out, conf, det = sess.run(None, {in_name: x})
    infer_ms = (time.perf_counter() - t0) * 1000.0

    o = out[0]
    e = np.exp(o - o.max(axis=0, keepdims=True))
    prob = (e / e.sum(axis=0, keepdims=True))[1]  # (H',W')

    if path_name == "letterbox512":
        px, py, fw, fh = meta["inner"]
        prob = prob[py:py + fh, px:px + fw]
    map_up = cv2.resize(prob.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_LINEAR).astype(np.float64)

    score = float(1.0 / (1.0 + np.exp(-float(det.ravel()[0]))))
    metrics = compute_metrics(map_up, mask_path)
    metrics.update({"sample": os.path.basename(sample_path), "pass": path_name,
                    "orig_wh": [W, H], "feed_wh": meta["feed_wh"],
                    "infer_ms": round(infer_ms, 1), "score": score})
    print(json.dumps(metrics, ensure_ascii=False))


def compute_metrics(map64, mask_path):
    import cv2
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    H, W = mask.shape
    k = max(3, int(0.005 * min(H, W)) | 1)
    mask_er = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    # 外区：再向外让出一圈未腐蚀的同宽带，避免边界泄漏
    dil = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    inside = map64[mask_er]
    outside = map64[~dil]

    pos, neg = inside, outside
    s = np.concatenate([pos, neg])
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts + 1) / 2.0
    r = ranks[inv]
    npos, nneg = len(pos), len(neg)
    auc = (r[:npos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

    thr99 = float(np.quantile(neg, 0.99))
    def recall_at(thr):
        return float((pos > thr).mean())
    def iou_cc(thr):
        pred = (map64 > thr).astype(np.uint8)
        n, labels = cv2.connectedComponents(pred)
        best = 0.0
        areas = np.bincount(labels.ravel())
        for lab in range(1, n):
            comp = labels == lab
            i = np.count_nonzero(comp & mask)
            if i == 0:
                continue
            u = areas[lab] + mask.sum() - i
            best = max(best, i / u)
        return float(best)

    return {
        "auc": float(auc),
        "mean_in": float(pos.mean()), "mean_out": float(neg.mean()),
        "thr_out99": thr99,
        "recall@0.5": recall_at(0.5), "recall@out99": recall_at(thr99),
        "iou@0.5": iou_cc(0.5), "iou@out99": iou_cc(thr99),
        "n_in": int(npos), "n_out": int(nneg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        REPO_ROOT, "reports", "onnx-dynamo", "trufor_dynamic_dynamo.onnx"))
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--single", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--pass", dest="pass_name", default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.single:
        run_single(args.model, args.sample, args.mask, args.pass_name, True)
        return

    mpath = os.path.join(INPUT_DIR, "manifest.json")
    if args.rebuild or not os.path.exists(mpath):
        manifest = build_samples()
        print(f"[samples] {len(manifest)} 个样本已构造 -> {INPUT_DIR}/samples")
    else:
        manifest = json.load(open(mpath, encoding="utf-8"))
    if args.limit_samples:
        manifest = manifest[:args.limit_samples]

    os.makedirs(RESULT_DIR, exist_ok=True)
    runs_path = os.path.join(RESULT_DIR, "runs.jsonl")
    done = set()
    rows = []
    if os.path.exists(runs_path):
        with open(runs_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done.add((r["sample"].removesuffix(".png"), r["pass"]))
                rows.append(r)
            print(f"[resume] 已完成 {len(done)} 组合，跳过")
    jsonl = open(runs_path, "a", encoding="utf-8")
    sam_dir = os.path.join(INPUT_DIR, "samples")
    for s in manifest:
        for p in PASSES:
            if (s["name"], p) in done:
                continue
            sp = os.path.join(sam_dir, f"{s['name']}.png")
            mp = os.path.join(sam_dir, f"{s['name']}_mask.png")
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--single", "--model", args.model,
                   "--sample", sp, "--mask", mp, "--pass", p]
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=900)
                if r.returncode == 0 and r.stdout.strip():
                    row = json.loads(r.stdout.strip().splitlines()[-1])
                    row["sample"] = row["sample"].removesuffix(".png")
                else:
                    row = {"sample": s["name"], "pass": p,
                           "skipped": f"exit={r.returncode} "
                                      f"{r.stderr.strip()[-200:]}"}
            except subprocess.TimeoutExpired:
                row = {"sample": s["name"], "pass": p, "skipped": "timeout900"}
            row["ratio"] = s["ratio"]; row["base"] = s["base"]
            rows.append(row)
            jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl.flush()
            print(f"[{s['name']}|{p}] " +
                  (f"AUC={row['auc']:.4f} rec@out99={row['recall@out99']:.3f} "
                   f"iou@out99={row['iou@out99']:.3f} {row['infer_ms']:.0f}ms"
                   if "auc" in row else f"SKIPPED: {row['skipped']}") +
                  f" (wall {time.time()-t0:.0f}s)", flush=True)
    jsonl.close()

    # 汇总：按 (base类, ratio) x pass 的平均 AUC / recall
    good = [r for r in rows if "auc" in r]
    smalls = [r for r in good if r["base"] in ("small_a", "ctrl512")]
    bigs = [r for r in good if r["base"] not in ("small_a", "ctrl512")]

    def agg(group):
        table = {}
        for ratio in RATIOS:
            for p in PASSES:
                sel = [r for r in group if r["ratio"] == ratio and r["pass"] == p]
                if sel:
                    table[(ratio, p)] = (np.mean([r["auc"] for r in sel]),
                                         np.mean([r["recall@out99"] for r in sel]),
                                         len(sel))
        return table

    lines = ["", "=== 汇总（均值 AUC / recall@FPR1% / n） ==="]
    for tag, group in (("大图(下采样类)", bigs), ("小图/对照(上采样类)", smalls)):
        if not group:
            continue
        lines.append(f"-- {tag} --")
        t = agg(group)
        for ratio in RATIOS:
            cells = []
            for p in PASSES:
                if (ratio, p) in t:
                    a, rc, n = t[(ratio, p)]
                    cells.append(f"{p}: AUC {a:.4f} rec {rc:.3f} (n={n})")
            lines.append(f"  ratio={ratio}: " + " | ".join(cells))
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(RESULT_DIR, "summary.txt"), "a", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
