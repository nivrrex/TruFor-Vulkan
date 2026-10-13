#!/usr/bin/env python3
"""Phase 8A: rewrite dim-0 scalar-index Gather into Slice + Squeeze.

Phase 8 established that TVM 0.27.0's Vulkan backend miscomputes `t[k]`
(dim-0 integer index) when the tensor's second-to-last dimension is 1, while
`c[1:2].squeeze(0)` and `c.narrow(0,1,1).squeeze(0)` are bit-exact. The root
cause sits in the take/select kernel, so the rewrite has to remove every such
take from the graph -- not just the one that happens to dominate the error.

Scope is deliberately narrow, as the plan requires: **only** `Gather` nodes that
have `axis == 0` **and** a single-element constant index. All other `Gather`
(axis != 0, vector or negative indices), plus `Slice`, `Split`, `Squeeze`,
`Expand`, `Reshape`, `Transpose` are left untouched.

    Gather(X, [k], axis=0)      X:[d0, r...]      ->  [r...]
        |
        v
    Slice(X, starts=[k], ends=[k+1], axes=[0])   ->  [1, r...]
        |
        v
    Squeeze(axis=0)                             ->  [r...]

The graph is then validated twice: `onnx.checker`, and a full ONNX Runtime CPU
re-run compared against the untouched model (shape, max_abs, MAE, RMSE,
correlation, localization map, integrity score). Vulkan is not touched unless
that gate passes.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def load_const(name, inits, prod):
    t = inits.get(name)
    if t is not None:
        return numpy_helper.to_array(t)
    p = prod.get(name)
    if p is not None and p.op_type == "Constant":
        for a in p.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default="trufor-512x512-slicefix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model = onnx.load(args.onnx)
    g = model.graph

    img = Image.open(args.image).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    in_name = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    ref = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"]).run(None, {in_name: x})

    inits = {t.name: t for t in g.initializer}
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n

    inferred = onnx.shape_inference.infer_shapes(model)
    shape_of = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.type.HasField("tensor_type"):
            d = [x.dim_value for x in vi.type.tensor_type.shape.dim]
            if d and all(v > 0 for v in d):
                shape_of[vi.name] = d

    # ---- Phase 8C census: candidates, how many we patch, how many risk being left ----
    total_gather = sum(1 for n in g.node if n.op_type == "Gather")
    candidates = []
    for n in g.node:
        if n.op_type != "Gather" or len(n.input) < 2:
            continue
        axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
        idx = load_const(n.input[1], inits, prod)
        if axis != 0 or idx is None or idx.size != 1:
            continue
        k = int(idx.reshape(-1)[0])
        src = shape_of.get(n.input[0])
        # Phase 8 trigger: second-to-last dim == 1
        risky = bool(src) and len(src) >= 2 and src[-2] == 1
        candidates.append({"node": n.name, "src": n.input[0], "src_shape": src,
                           "index": k, "phase8_risky": risky})

    n_risky = sum(1 for c in candidates if c["phase8_risky"])
    print("Gather nodes total            : %d" % total_gather)
    print("dim-0 scalar-index candidates : %d" % len(candidates))
    print("  of which phase-8 risky (src[-2]==1): %d" % n_risky)
    print("  of which other shapes             : %d" % (len(candidates) - n_risky))

    if args.dry_run:
        print("\n--dry-run: no file written")
        return 0

    # ---- rewrite every candidate (all are semantically Slice+Squeeze) ----
    used_names = {n.name for n in g.node if n.name}
    patched, new_nodes, new_inits = [], [], []
    for n in g.node:
        cand = next((c for c in candidates if c["node"] == n.name), None)
        if cand is None:
            new_nodes.append(n)
            continue
        k = cand["index"]
        slice_out = n.output[0] + "_slice"
        base = (n.name or n.output[0]).replace("/", "_")
        starts = base + "_starts"
        ends = base + "_ends"
        axes = base + "_axes"
        new_inits.append(numpy_helper.from_array(np.array([k], dtype=np.int64), starts))
        new_inits.append(numpy_helper.from_array(np.array([k + 1], dtype=np.int64), ends))
        new_inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), axes))
        new_nodes.append(helper.make_node(
            "Slice", [n.input[0], starts, ends, axes], [slice_out], name=n.name + "_to_slice"))
        new_nodes.append(helper.make_node(
            "Squeeze", [slice_out, axes], list(n.output), name=n.name + "_to_squeeze"))
        patched.append(cand)

    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)

    out_path = os.path.join(args.outdir, args.name + ".onnx")
    onnx.save(model, out_path)

    # ---- remaining candidates after the rewrite ----
    m2 = onnx.load(out_path, load_external_data=False)
    inits2 = {t.name: t for t in m2.graph.initializer}
    prod2 = {}
    for n in m2.graph.node:
        for o in n.output:
            prod2[o] = n
    remaining = 0
    for n in m2.graph.node:
        if n.op_type != "Gather" or len(n.input) < 2:
            continue
        axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
        idx = load_const(n.input[1], inits2, prod2)
        if axis == 0 and idx is not None and idx.size == 1:
            remaining += 1
    print("\npatched Gather nodes           : %d" % len(patched))
    print("remaining dim-0 scalar Gather  : %d" % remaining)

    # ---- onnx.checker ----
    checker_ok, checker_msg = True, ""
    try:
        onnx.checker.check_model(out_path, full_check=True)
    except Exception as e:  # noqa: BLE001
        checker_ok = False
        checker_msg = "%s: %s" % (type(e).__name__, str(e)[:400])
    print("onnx.checker                   : %s%s" % ("OK" if checker_ok else "FAILED",
                                                       "" if checker_ok else " -- " + checker_msg))

    # ---- CPU equivalence gate ----
    got = ort.InferenceSession(out_path, so, providers=["CPUExecutionProvider"]).run(None, {in_name: x})

    def post(res):
        out, conf, det = res
        o = F.softmax(torch.from_numpy(out[0]), dim=0)[1].numpy()
        c = torch.sigmoid(torch.from_numpy(conf[0]))[0].numpy()
        d = float(np.asarray(det).reshape(-1)[0])
        return o, c, 1.0 / (1.0 + np.exp(-d))

    def stats(a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        d = np.abs(a - b)
        return {"shape_match": True, "shape": list(a.shape),
                "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
                "rmse": float(np.sqrt((d ** 2).mean())),
                "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else None}

    r_map, r_conf, r_det = post(ref)
    g_map, g_conf, g_det = post(got)
    eq = {"map": stats(r_map, g_map), "conf": stats(r_conf, g_conf),
          "integrity_score": {"before": r_det, "after": g_det, "abs_err": abs(r_det - g_det)}}
    worst = max(eq["map"].get("max_abs_err", 1.0), eq["conf"].get("max_abs_err", 1.0))
    gate = bool(checker_ok and worst == 0.0 and eq["integrity_score"]["abs_err"] == 0.0)

    print("\n=== ORT CPU equivalence (original vs Slice+Squeeze rewrite) ===")
    for k in ("map", "conf"):
        s = eq[k]
        print("  %-5s max_abs=%.3e mean_abs=%.3e rmse=%.3e corr=%.12f"
              % (k, s.get("max_abs_err", float("nan")), s.get("mean_abs_err", float("nan")),
                 s.get("rmse", float("nan")), s.get("corr") if s.get("corr") is not None else float("nan")))
    print("  det   before=%.10f after=%.10f err=%.3e"
          % (r_det, g_det, eq["integrity_score"]["abs_err"]))
    print("  GATE (bit-identical + checker): %s" % ("PASS" if gate else "FAIL"))

    report = {
        "source": args.onnx, "output": out_path,
        "gather_total": total_gather,
        "candidates_dim0_scalar": len(candidates),
        "candidates_phase8_risky": n_risky,
        "candidates_other_shapes": len(candidates) - n_risky,
        "patched": len(patched),
        "remaining_dim0_scalar_gather": remaining,
        "patched_list": patched,
        "checker_ok": checker_ok, "checker_msg": checker_msg,
        "equivalence": eq, "cpu_gate_pass": gate,
    }
    with open(os.path.join(args.outdir, "slicefix-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % out_path)
    print("wrote %s" % os.path.join(args.outdir, "slicefix-report.json"))
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
