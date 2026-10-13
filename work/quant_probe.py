# -*- coding: utf-8 -*-
"""ORT-CPU 量化/调参探针（M2 后续调研 4，2026-10-02）。

背景：外部评审指出 int8 动态量化是候选遗漏项。本探针回答两件事：
  1. quantize_dynamic 对该 Conv 为主的 1979 节点图实际能量化多少节点
     （ORT 动态量化主要覆盖 MatMul/Gemm/LSTM，CNN 大概率近乎空转）；
  2. 若量化产物可用，与 fp32 基线比 512x512 CPU 延迟 + map 数值偏差 +
     消融样本指标（ctrl512/small_a/b5_r0.01）。
附加：intra_op 线程数 4/6/默认 的 512² 延迟对比（成本一行级）。

用法：python3 -u work/quant_probe.py [--model PATH]
依赖：onnx onnxruntime numpy opencv pillow。量化产物写 /tmp/ort_quant/。
"""
import argparse
import json
import os
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_arrays(model_path):
    import onnx
    m = onnx.load(model_path, load_external_data=False)
    return m


def node_ops(m):
    from collections import Counter
    return Counter(n.op_type for n in m.graph.node)


def bench(sess, in_name, x, n=8):
    sess.run(None, {in_name: x})
    t = time.perf_counter()
    for _ in range(n):
        sess.run(None, {in_name: x})
    return (time.perf_counter() - t) / n * 1000.0


def postprocess(out):
    o = out[0]
    e = np.exp(o - o.max(axis=0, keepdims=True))
    return (e / e.sum(axis=0, keepdims=True))[1].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        REPO_ROOT, "reports", "onnx-dynamo", "trufor_dynamic_dynamo.onnx"))
    args = ap.parse_args()

    import onnxruntime as ort
    import cv2

    print("== 1. 图结构 op 分布 ==")
    m = load_arrays(args.model)
    ops = node_ops(m)
    print(json.dumps(dict(ops.most_common(12)), ensure_ascii=False))
    n_mm = ops.get("MatMul", 0) + ops.get("Gemm", 0)
    print(f"MatMul+Gemm={n_mm} / Conv={ops.get('Conv',0)} 总节点={len(m.graph.node)}")
    del m

    print("== 2. fp32 基线（本进程）+ 线程对比 ==")
    img = cv2.cvtColor(cv2.imread(os.path.join(
        REPO_ROOT, "work", "ablation_inputs", "samples", "ctrl512_r0.01.png")),
        cv2.COLOR_BGR2RGB)
    x = img.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(256.0)

    def make_sess(threads=None, path=None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        return ort.InferenceSession(path or args.model, so,
                                    providers=["CPUExecutionProvider"])

    s0 = make_sess()
    name0 = s0.get_inputs()[0].name
    base_ms = bench(s0, name0, x)
    rel4 = bench(make_sess(4), name0, x)
    rel6 = bench(make_sess(6), name0, x)
    print(f"fp32 default threads: {base_ms:.0f} ms | intra4: {rel4:.0f} | intra6: {rel6:.0f}")
    ref_map = postprocess(s0.run(None, {name0: x})[0])
    ref_out = s0.run(None, {name0: x})
    ref_score = float(ref_out[2].ravel()[0])
    del s0

    print("== 3. quantize_dynamic (QUInt8, per-channel) ==")
    os.makedirs("/tmp/ort_quant", exist_ok=True)
    qpath = "/tmp/ort_quant/trufor_dyn_qdyn.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        t = time.time()
        quantize_dynamic(args.model, qpath, weight_type=QuantType.QUInt8,
                         per_channel=True, extra_options={"ReduceRange": False})
        print(f"量化完成 {time.time()-t:.0f}s，产物 {os.path.getsize(qpath)/1e6:.0f} MB"
              f"（原 {os.path.getsize(args.model)/1e6:.0f} MB）")
    except Exception as ex:
        print(f"FAIL quantize_dynamic: {type(ex).__name__}: {str(ex)[:300]}")
        return

    try:
        sq = ort.InferenceSession(qpath, providers=["CPUExecutionProvider"])
    except Exception as ex:
        print(f"FAIL load quantized: {type(ex).__name__}: {str(ex)[:300]}")
        return
    nq = sq.get_inputs()[0].name
    qms = bench(sq, nq, x)
    qmap = postprocess(sq.run(None, {nq: x})[0])
    rel = float(np.linalg.norm(qmap - ref_map) / (np.linalg.norm(ref_map) + 1e-12))
    qout = sq.run(None, {nq: x})
    print(f"int8 512²: {qms:.0f} ms（fp32 {base_ms:.0f} ms，比值 {base_ms/qms:.2f}×）；"
          f"map rel_L2 vs fp32 = {rel:.2e}；det logit Δ = "
          f"{abs(float(qout[2].ravel()[0]) - ref_score):.3f}")

    print("== 4. 量化图在消融样本上的指标 ==")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ra", os.path.join(
        REPO_ROOT, "work", "resize_ablation.py"))
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    for sname in ("ctrl512_r0.01", "small_a_r0.01", "b5_r0.01"):
        sp = os.path.join(REPO_ROOT, "work", "ablation_inputs", "samples",
                          f"{sname}.png")
        mp = sp.replace(".png", "_mask.png")
        im = cv2.cvtColor(cv2.imread(sp), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        xx = im.transpose(2, 0, 1)[None].astype(np.float32) / 256.0
        pm = postprocess(sq.run(None, {nq: xx})[0])
        mu = cv2.resize(pm, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        mm = ra.compute_metrics(mu, mp)
        print(f"{sname:16s} int8: AUC={mm['auc']:.4f} rec@out99={mm['recall@out99']:.3f} "
              f"iou@out99={mm['iou@out99']:.3f}")


if __name__ == "__main__":
    main()
