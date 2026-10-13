#!/usr/bin/env python3
"""Phase 2: TruFor operator inventory.

Two independent sources of truth, cross-checked:

  A) static  : torch.export of the full model at a fixed 512x512 input
  B) dynamic : TorchDispatchMode capturing every aten call actually executed,
               on a real test image (arbitrary size)

For each collected operator we look up the ExecuTorch Vulkan backend operator
registry (vulkan-op-support-table.csv) and report support + caveats.

Outputs:
  <outdir>/export-graph.txt       readable listing of the exported graph
  <outdir>/runtime-ops-<image>.json   raw runtime operator census (with shapes)
  <outdir>/operator-inventory.json    merged inventory with Vulkan verdicts
  <outdir>/OPERATOR-INVENTORY.md      report
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRUFOR_ROOT = os.environ.get("TRUFOR_ROOT", os.path.join(ROOT, "truFor-src"))
TRAIN_DIR = os.path.join(TRUFOR_ROOT, "TruFor_train_test")
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from lib.config import config, update_config  # noqa: E402
from lib.utils import get_model  # noqa: E402
from dataset.dataset_test import TestDataset  # noqa: E402

# Modules that make up TruFor, used to attribute operators to a component.
COMPONENTS = OrderedDict(
    [
        ("npp", "Noiseprint++ (DnCNN, 17 conv levels)"),
        ("backbone", "SegFormer-B2 dual-modality encoder (CMX)"),
        ("loc_head", "MLP localization decoder"),
        ("conf_head", "MLP confidence decoder"),
        ("det_head", "detection MLP on pooled statistics"),
        ("other", "pre/post processing, shape ops, glue"),
    ]
)


def build_model(model_file, experiment="trufor_ph3"):
    class _Args:
        pass

    prev = os.getcwd()
    os.chdir(TRAIN_DIR)
    try:
        a = _Args()
        a.experiment = experiment
        a.opts = []
        update_config(config, a)
        model = get_model(config)
    finally:
        os.chdir(prev)
    ckpt = torch.load(model_file, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------- A) export
def run_export(model, size, outdir):
    print("=== [A] torch.export @ %dx%d ===" % (size, size), flush=True)
    dummy = torch.rand(1, 3, size, size)
    result = {"ok": False}
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            ep = torch.export.export(model, (dummy,), strict=False)
        result["ok"] = True
        result["seconds"] = time.perf_counter() - t0
        result["node_count"] = sum(1 for _ in ep.graph.nodes)
        result["op_count"] = sum(1 for _ in ep.graph_module.code.splitlines() if " = " in _)
        with open(os.path.join(outdir, "export-graph.txt"), "w") as f:
            f.write(str(ep.graph))
            f.write("\n\n========== CODE ==========\n")
            f.write(ep.graph_module.code)
        result["graph"] = str(ep)
    except Exception as e:  # noqa: BLE001
        import traceback

        result["error_type"] = type(e).__name__
        result["error"] = str(e)[:4000]
        result["traceback"] = traceback.format_exc()[-4000:]
        print("  export FAILED: %s: %s" % (type(e).__name__, str(e)[:300]), flush=True)
    return result


def collect_export_targets(export_txt):
    """Extract 'namespace::op' targets from the printed exported graph."""
    if export_txt is None:
        return []
    seen = []
    for line in export_txt.splitlines():
        if " = " not in line:
            continue
        lhs = line.split(" = ")[0]
        if ":" not in lhs:
            continue
        tgt = lhs.split(":", 1)[1].strip()
        if tgt in seen:
            continue
        seen.append(tgt)
    return seen


# --------------------------------------------------------------- B) runtime
class OpCensus(TorchDispatchMode):
    def __init__(self):
        self.counts = defaultdict(int)
        self.samples = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.counts[name] += 1
        if name not in self.samples:
            self.samples[name] = _shapes(args)
        return func(*args, **(kwargs or {}))


def _shapes(args):
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append([str(a.dtype), list(a.shape)])
        elif isinstance(a, (int, float, bool)):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.append("<%s len=%d>" % (type(a).__name__, len(a)))
        else:
            out.append(type(a).__name__)
    return out


def run_runtime(model, image_path, outdir):
    print("=== [B] runtime dispatch census @ %s ===" % image_path, flush=True)
    dataset = TestDataset(list_img=[image_path])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    rgb, paths = next(iter(loader))
    census = OpCensus()
    with torch.no_grad(), census:
        model(rgb, save_np=False)
    payload = {
        "image": paths[0],
        "shape": list(rgb.shape),
        "total_calls": sum(census.counts.values()),
        "counts": dict(census.counts),
        "sample_args": census.samples,
    }
    with open(os.path.join(outdir, "runtime-ops-%s.json" % os.path.basename(image_path).rsplit(".", 1)[0]), "w") as f:
        json.dump(payload, f, indent=2)
    print("  %d aten calls, %d distinct ops" % (payload["total_calls"], len(payload["counts"])), flush=True)
    return payload


def normalize_target(name):
    """'aten.conv2d.default' -> 'aten::conv2d'."""
    ns, _, rest = name.partition(".")
    op = rest.rsplit(".", 1)[0] if "." in rest else rest
    return "%s::%s" % (ns, op)


# ----------------------------------------------------------------- registry
def load_registry(csv_path):
    reg = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = "%s::%s" % (row["Namespace"].strip(), row["Operator"].strip())
            reg[key] = (row["Notes"] or "").strip()
    return reg


# ------------------------------------------------------------------ report
def build_inventory(runtime, export, registry):
    inv = OrderedDict()
    for raw, cnt in runtime["counts"].items():
        key = normalize_target(raw)
        e = inv.setdefault(
            key,
            {
                "target": key,
                "runtime_count": 0,
                "static_count": 0,
                "in_export_graph": False,
                "example_args": None,
                "vulkan": "UNKNOWN",
                "vulkan_note": "",
            },
        )
        e["runtime_count"] += cnt
        if e["example_args"] is None:
            e["example_args"] = runtime["sample_args"][raw]

    for t in export.get("targets", []):
        e = inv.setdefault(
            t,
            {
                "target": t,
                "runtime_count": 0,
                "static_count": 0,
                "in_export_graph": True,
                "example_args": None,
                "vulkan": "UNKNOWN",
                "vulkan_note": "",
            },
        )
        e["in_export_graph"] = True

    for e in inv.values():
        if e["target"] in registry:
            e["vulkan"] = "SUPPORTED"
            e["vulkan_note"] = registry[e["target"]]
        else:
            e["vulkan"] = "NOT_LISTED"
    return OrderedDict(sorted(inv.items(), key=lambda kv: -kv[1]["runtime_count"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "reports/operator-inventory/vulkan-op-support-table.csv"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--exp", default="trufor_ph3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model = build_model(args.model, args.exp)
    print("model built: %.2f M params" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    runtime = run_runtime(model, args.image, args.outdir)
    export = run_export(model, args.size, args.outdir)
    if export.get("ok"):
        export["targets"] = collect_export_targets(os.path.join(args.outdir, "export-graph.txt"))

    registry = load_registry(args.registry)
    inv = build_inventory(runtime, export, registry)

    total_calls = runtime["total_calls"]
    supported_calls = sum(e["runtime_count"] for e in inv.values() if e["vulkan"] == "SUPPORTED")
    blocked = [e for e in inv.values() if e["vulkan"] != "SUPPORTED" and e["runtime_count"] > 0]

    summary = {
        "image": runtime["image"],
        "runtime_total_aten_calls": total_calls,
        "distinct_ops": len(inv),
        "supported_op_count": sum(1 for e in inv.values() if e["vulkan"] == "SUPPORTED"),
        "unsupported_op_count": len(blocked),
        "aten_call_coverage_pct": round(100.0 * supported_calls / total_calls, 2),
        "export_ok": export.get("ok", False),
        "export_error_type": export.get("error_type"),
        "export_node_count": export.get("node_count"),
        "registry_size": len(registry),
    }

    with open(os.path.join(args.outdir, "operator-inventory.json"), "w") as f:
        json.dump({"summary": summary, "operators": inv}, f, indent=2)
    with open(os.path.join(args.outdir, "export-summary.json"), "w") as f:
        json.dump(export, f, indent=2)

    # ---- markdown report ----
    lines = []
    lines.append("# TruFor Operator Inventory (Phase 2)\n")
    lines.append("- TruFor commit: see `git -C truFor-src rev-parse HEAD`")
    lines.append("- Model: `%s` (experiment `%s`)" % (args.model, args.exp))
    lines.append("- torch: `%s`" % torch.__version__)
    lines.append("- Runtime census image: `%s` %s" % (runtime["image"], runtime["shape"]))
    lines.append("- ExecuTorch Vulkan registry: `%s` (%d operators)" % (os.path.basename(args.registry), len(registry)))
    lines.append("")
    lines.append("## Summary\n")
    for k, v in summary.items():
        lines.append("- **%s**: %s" % (k, v))
    lines.append("")
    lines.append("## Operators actually executed (sorted by call count)\n")
    lines.append("| target | calls | in export graph | Vulkan | note |")
    lines.append("|---|---:|:---:|---|---|")
    for e in inv.values():
        if e["runtime_count"] == 0:
            continue
        lines.append(
            "| `%s` | %d | %s | %s | %s |"
            % (
                e["target"],
                e["runtime_count"],
                "Y" if e["in_export_graph"] else "n",
                e["vulkan"],
                e["vulkan_note"].replace("|", "\\|"),
            )
        )
    lines.append("")
    if export.get("ok"):
        lines.append("## Export graph\n")
        lines.append("- full graph listing: `export-graph.txt`")
        lines.append("- node count: %s" % export.get("node_count"))
    else:
        lines.append("## Export graph\n")
        lines.append("`torch.export` **failed**.\n")
        lines.append("- error type: `%s`" % export.get("error_type"))
        lines.append("```\n%s\n```" % export.get("error", ""))
    lines.append("")
    lines.append("## Operators with no Vulkan support (blocking candidates)\n")
    if blocked:
        lines.append("| target | calls | example args |")
        lines.append("|---|---:|---|")
        for e in sorted(blocked, key=lambda x: -x["runtime_count"]):
            lines.append("| `%s` | %d | `%s` |" % (e["target"], e["runtime_count"], e["example_args"]))
    else:
        lines.append("None — every executed operator appears in the Vulkan registry.")
    lines.append("")
    with open(os.path.join(args.outdir, "OPERATOR-INVENTORY.md"), "w") as f:
        f.write("\n".join(lines))

    print("\n" + json.dumps(summary, indent=2))
    if blocked:
        print("\nBLOCKING CANDIDATES (no Vulkan support):")
        for e in sorted(blocked, key=lambda x: -x["runtime_count"]):
            print("  %-45s calls=%d" % (e["target"], e["runtime_count"]))


if __name__ == "__main__":
    main()
