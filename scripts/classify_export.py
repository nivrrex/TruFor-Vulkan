#!/usr/bin/env python3
"""Phase 2b: classify every operator in the torch.export graph against the
ExecuTorch Vulkan operator registry, with an explicit equivalence table.

Two verdict levels, because a raw name lookup is misleading:

  L0  SUPPORTED       the exact aten name is in the Vulkan registry.
  L1  EQUIVALENT      not listed under that name, but ExecuTorch's lowering
                      passes rewrite it to a registered form
                      (dim-order normalisation, decompositions). Each mapping
                      below cites the registry entry it lands on.
  L2  BLOCKING        no registered form and no decomposition available.
                      These must stay on the CPU partition, or the model needs
                      a local rewrite.

Counts are per *graph node* of the exported program, not per runtime call,
so they map 1:1 onto what a partitioner would see.
"""

import argparse
import collections
import json
import os
import re
import sys

# registry-entry -> what the lowering pass produces it from
# key: aten op as it appears in the exported graph
EQUIVALENCE = {
    # --- exported overload renamed to the registry's canonical name ---
    "aten::conv2d": ("aten::convolution", "registry lists the generic convolution "
                      "(notes: batch=1 for 2D conv, which matches batch_size=1 inference)"),
    "aten::softmax": ("aten::_softmax", "renamed overload"),
    "aten::log_softmax": ("aten::_log_softmax", "renamed overload"),
    # --- dim-order / layout normalisation (registry has *_copy variants) ---
    "aten::view": ("aten::view_copy", "dim_order pass"),
    "aten::reshape": ("aten::view_copy", "dim_order pass (reshape -> view_copy)"),
    "aten::transpose": ("aten::permute_copy", "dim_order pass"),
    "aten::permute": ("aten::permute_copy", "dim_order pass (aten::permute is not registered itself)"),
    "aten::select": ("aten::select_copy", "dim_order pass"),
    "aten::slice": ("aten::slice_copy", "dim_order pass"),
    "aten::unsqueeze": ("aten::unsqueeze_copy", "dim_order pass"),
    "aten::contiguous": ("aten::clone", "dim_order pass"),
    "aten::flatten": ("aten::view_copy", "reshape decomposition"),
    "aten::to": ("aten::_to_copy", "dim_order pass"),
    # --- decompositions into registered primitives ---
    "aten::linear": ("aten::linear", "direct"),
    "aten::matmul": ("aten::mm / aten::bmm", "matmul decomposition"),
    "aten::layer_norm": ("aten::native_layer_norm", "layer_norm decomposition"),
    "aten::batch_norm": ("aten::_native_batch_norm_legit_no_training", "eval-mode batch_norm decomposition"),
    "aten::relu_": ("aten::relu", "in-place functionalisation"),
    "aten::chunk": ("aten::split", "chunk -> split"),
    "aten::unbind": ("aten::split", "unbind -> split"),
    "aten::upsample_bilinear2d": ("aten::upsample_bilinear2d", "direct"),
    "aten::adaptive_avg_pool2d": ("aten::avg_pool2d", "static-shape -> avg_pool2d (only at fixed input size)"),
    "aten::adaptive_max_pool2d": ("aten::max_pool2d_with_indices", "static-shape -> max_pool2d_with_indices"),
    "aten::tile": ("aten::repeat", "tile -> repeat (noiseprint 1ch -> 3ch); expand_copy is not registered in this ref"),    # --- no-op in eval mode, removed by a graph pass ---
    "aten::dropout": (None, "eval-mode no-op, removed by pass"),
    "aten::feature_dropout": (None, "eval-mode no-op, removed by pass"),
    "aten::lift_fresh_copy": (None, "meta op, removed by pass"),
    "aten::zeros": ("aten::zeros", "direct (constant folding usually removes it)"),
}

# operators that have NO registered form under any lowering
BLOCKING = {
    "aten::logsumexp": "not registered. Only in detection head "
    "weighted_statistics_pooling (4 nodes, tiny [B,C,H,W] -> [B,C,2] tensors).",
    "aten::log_sigmoid": "not registered (only _log_softmax is, and log_sigmoid is "
    "not a special case of it). 1 node, detection head only.",
    "aten::upsample_bilinear2d": "not registered in this ref (only upsample_nearest2d "
    "is). 8 nodes -- the final F.interpolate that resizes the loc/conf maps back to "
    "the input size. Needs either nearest-neighbour (numerically different output) "
    "or a custom kernel.",
    "aten::sum": "registered, but verify the keepdim/max-2D-reduction constraints the "
    "reduce-op registration imposes. Detection head sums are keepdim=False over the "
    "last dim -- a caller-side fix if the constraint bites, not a new kernel.",
}


def load_registry(path):
    reg = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            reg["%s::%s" % (row["Namespace"].strip(), row["Operator"].strip())] = (row["Notes"] or "").strip()
    return reg


import csv  # noqa: E402


def parse_export_graph(path):
    txt = open(path).read()
    if "========== CODE ==========" not in txt:
        sys.exit("export-graph.txt has no CODE section; re-run operator_inventory.py")
    code = txt.split("========== CODE ==========")[1]
    c = collections.Counter()
    for line in code.splitlines():
        m = re.search(r"torch\.ops\.([A-Za-z0-9_]+)\.([A-Za-z0-9_.]+)\(", line)
        if m:
            # strip the overload suffix (.default / .int / .Tensor / .vec ...)
            head, _, overload = m.group(2).rpartition(".")
            c["%s::%s" % (m.group(1), head if overload else m.group(2))] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    reg = load_registry(args.registry)
    counts = parse_export_graph(args.graph)
    total = sum(counts.values())

    rows = []
    for op, n in counts.most_common():
        target, note = None, ""
        if op in reg:
            verdict, target, note, why = "L0", op, reg[op], "in registry"
        elif op in BLOCKING:
            verdict, target, note, why = "L2", None, BLOCKING[op], "no registered form"
        elif op in EQUIVALENCE:
            t, how = EQUIVALENCE[op]
            if t is None:
                verdict, why = "REMOVED", "eliminated by a pass before lowering"
            elif any(x.strip() in reg for x in t.split("/")):
                verdict, why = "L1", "lowered to %s (%s)" % (t, how)
            else:
                verdict, why = "L2", "lowering target %s NOT in registry" % t
        else:
            verdict, target, why = "UNKNOWN", None, "no mapping defined; needs manual check"
        rows.append(
            {"op": op, "nodes": n, "verdict": verdict, "registry_target": target, "registry_note": note, "reason": why}
        )

    by_v = collections.Counter()
    nodes_by_v = collections.Counter()
    for r in rows:
        by_v[r["verdict"]] += 1
        nodes_by_v[r["verdict"]] += r["nodes"]

    payload = {
        "total_graph_nodes": total,
        "distinct_ops": len(rows),
        "ops_by_verdict": dict(by_v),
        "nodes_by_verdict": dict(nodes_by_v),
        "l2_blocking_nodes": sum(r["nodes"] for r in rows if r["verdict"] == "L2"),
        "l2_coverage_pct": round(100.0 * (total - nodes_by_v["L2"]) / total, 2),
        "rows": rows,
    }
    with open(os.path.join(args.outdir, "export-classification.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print("total graph nodes (excluding placeholders): %d, distinct ops: %d" % (total, len(rows)))
    for v in ("L0", "L1", "REMOVED", "L2", "UNKNOWN"):
        if v in by_v:
            print("  %-8s %2d ops  %5d nodes" % (v, by_v[v], nodes_by_v[v]))
    print("\nL2 / UNKNOWN detail:")
    for r in rows:
        if r["verdict"] in ("L2", "UNKNOWN"):
            print("  %-40s %4d nodes  %s" % (r["op"], r["nodes"], r["reason"]))


if __name__ == "__main__":
    main()
