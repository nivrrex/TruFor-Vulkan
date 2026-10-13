#!/usr/bin/env python3
"""Probe 3: can TruFor be exported to ONNX at a fixed 512x512 input?

Two exporters are tried, because torch 2.7's default (dynamo) exporter and the
legacy TorchScript exporter fail on different models:

  dynamo  : torch.onnx.export(..., dynamo=True)   -- opset 18, torch.export based
  legacy  : torch.onnx.export(..., dynamo=False)  -- TorchScript tracer

Reports per-op census of the produced ONNX graph, because op coverage -- not the
exporter -- is what decides whether any downstream GPU runtime can execute it.
No model modification, no numerics change.
"""

import argparse
import collections
import json
import os
import sys
import time

import torch

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


def build_model(model_file, exp="trufor_ph3"):
    class _Args:
        pass

    prev = os.getcwd()
    os.chdir(TRAIN_DIR)
    try:
        a = _Args()
        a.experiment = exp
        a.opts = []
        update_config(config, a)
        model = get_model(config)
    finally:
        os.chdir(prev)
    ckpt = torch.load(model_file, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def census(path):
    import onnx

    m = onnx.load(path)
    c = collections.Counter(n.op_type for n in m.graph.node)
    return c, len(m.graph.node), len(m.graph.initializer)


def try_export(model, dummy, path, mode, size):
    t0 = time.perf_counter()
    if mode == "dynamo":
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["rgb"],
            output_names=["out", "conf", "det"],
            opset_version=18,
            dynamo=True,
            verbose=False,
        )
    else:
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["rgb"],
            output_names=["out", "conf", "det"],
            opset_version=17,
            dynamo=False,
            verbose=False,
        )
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--exp", default="trufor_ph3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model = build_model(args.model, args.exp)
    dummy = torch.rand(1, 3, args.size, args.size)

    results = {}
    for mode in ("dynamo", "legacy"):
        path = os.path.join(args.outdir, "trufor_%dx%d_%s.onnx" % (args.size, args.size, mode))
        print("=== export via %s ===" % mode, flush=True)
        rec = {"ok": False}
        try:
            rec["seconds"] = try_export(model, dummy, path, mode, args.size)
            rec["ok"] = True
            rec["file"] = path
            rec["bytes"] = os.path.getsize(path)
            c, nodes, inits = census(path)
            rec["node_count"] = nodes
            rec["initializer_count"] = inits
            rec["op_census"] = dict(c.most_common())
            print("  exported in %.1fs, %d nodes, %.1f MB" % (rec["seconds"], nodes, rec["bytes"] / 1e6))
        except Exception as e:  # noqa: BLE001
            import traceback

            rec["error_type"] = type(e).__name__
            rec["error"] = str(e)[:2000]
            rec["traceback"] = traceback.format_exc()[-2500:]
            print("  FAILED: %s: %s" % (type(e).__name__, str(e)[:300]))
            if os.path.exists(path):
                os.remove(path)
        results[mode] = rec

    with open(os.path.join(args.outdir, "onnx-export.json"), "w") as f:
        json.dump(results, f, indent=2)

    ok = [m for m in results if results[m]["ok"]]
    print("\n=== summary ===")
    for m, r in results.items():
        print("%-8s %s" % (m, ("ok (%.1fs, %d nodes)" % (r["seconds"], r["node_count"])) if r["ok"] else "FAILED: " + r["error_type"]))
    if ok:
        best = min(ok, key=lambda m: results[m]["node_count"])
        print("\nop census (%s):" % best)
        for op, n in sorted(results[best]["op_census"].items(), key=lambda kv: -kv[1]):
            print("  %5d  %s" % (n, op))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
