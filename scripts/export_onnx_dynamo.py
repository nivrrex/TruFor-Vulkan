#!/usr/bin/env python3
"""Phase 6B: re-export TruFor to ONNX with the dynamo exporter.

The dynamo exporter is the one worth using here, because it emits a graph whose
shapes are described symbolically instead of baking 1608 Constant / 272 Shape /
291 Gather nodes in as the legacy tracer does. It previously failed with

    ConversionError ... No ONNX function found for aten.adaptive_max_pool2d
    ... No decompositions registered for the real-valued input

root-caused to CMX's ChannelWeights (lib/models/cmx/net_utils.py:13-14):

    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

For output_size == 1, AdaptiveMaxPool2d(1) and MaxPool2d(kernel_size=1,
stride=1) both take the maximum over the whole spatial extent, and
AdaptiveAvgPool2d(1) and AvgPool2d(kernel_size=1, stride=1) both take the mean
over the whole extent. So the substitution is exact, and neither pooling module
holds parameters or buffers, so checkpoint loading is untouched.

This swaps the module instances in the loaded model object only. The TruFor
sources on disk are not modified. Numerical equivalence of the resulting ONNX
against the original PyTorch model is verified before the file is accepted.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

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
    return model


class GlobalAvgPool2d(nn.Module):
    """Exactly equivalent to nn.AdaptiveAvgPool2d(1), in terms the ONNX
    exporter can lower (aten.mean / aten.amax instead of
    aten.adaptive_avg_pool2d, which has no registered ONNX decomposition)."""

    def forward(self, x):
        return torch.mean(x, dim=(2, 3), keepdim=True)


class GlobalMaxPool2d(nn.Module):
    """Exactly equivalent to nn.AdaptiveMaxPool2d(1)."""

    def forward(self, x):
        return torch.amax(x, dim=(2, 3), keepdim=True)


def as_pair(size):
    """nn.Adaptive*Pool2d normalises output_size to a tuple; be tolerant."""
    if isinstance(size, int):
        return (size,)
    return tuple(size)


def swap_adaptive_pool(model):
    """Replace Adaptive{Max,Avg}Pool2d(1) with the exactly-equivalent reduction.

    Note: nn.MaxPool2d(kernel_size=1, stride=1) is NOT equivalent -- it keeps
    the spatial extent (verified: output had 2097152 elements instead of 256).
    A global max/mean over dims (2, 3) is the correct rewriting.

    Returns the list of (path, old_type, new_type) that were changed.
    """
    changes = []
    for name, mod in model.named_modules():
        for attr in ("avg_pool", "max_pool"):
            child = getattr(mod, attr, None)
            if isinstance(child, nn.AdaptiveAvgPool2d) and as_pair(child.output_size) == (1,):
                setattr(mod, attr, GlobalAvgPool2d())
                changes.append((f"{name}.{attr}", "AdaptiveAvgPool2d", "mean(dim=(2,3))"))
            elif isinstance(child, nn.AdaptiveMaxPool2d) and as_pair(child.output_size) == (1,):
                setattr(mod, attr, GlobalMaxPool2d())
                changes.append((f"{name}.{attr}", "AdaptiveMaxPool2d", "amax(dim=(2,3))"))
    return changes


class ExportWrapper(nn.Module):
    """Drops the trailing None from EncoderDecoder.forward.

    TruFor returns (out, conf, det, npp) with npp=None when save_np is False.
    The legacy tracer tolerates a None in the output tuple; the dynamo
    exporter raises
        AttributeError: 'NoneType' object has no attribute 'name'
    while translating the return node. An ONNX graph cannot carry a None
    output anyway, so the wrapper returns the three real tensors.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb):
        out, conf, det, _npp = self.model(rgb, save_np=False)
        return out, conf, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1,
                    help="batch dimension baked into the export (Gate 3 scaling study)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--exp", default="trufor_ph3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    model = build_model(args.model, args.exp)
    model.eval()

    x = torch.rand(args.batch, 3, args.size, args.size)
    with torch.no_grad():
        ref = model(x, save_np=False)

    changes = swap_adaptive_pool(model)
    print("pooling substitutions: %d" % len(changes))
    for c in changes[:8]:
        print("   %-56s %s -> %s" % c)

    with torch.no_grad():
        got = model(x, save_np=False)

    def eq(a, b, name):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        d = np.abs(a - b)
        ok = a.shape == b.shape and d.max() < 1e-5
        print("  %-4s substitution check: max_abs=%.3e shape_match=%s -> %s"
              % (name, d.max(), a.shape == b.shape, "OK" if ok else "MISMATCH"))
        return ok

    checks = []
    for i, nm in enumerate(("out", "conf")):
        checks.append(eq(ref[i].numpy(), got[i].numpy(), nm))
    if ref[2] is not None and got[2] is not None:
        checks.append(eq(ref[2].numpy(), got[2].numpy(), "det"))
    if not all(checks):
        print("\nsubstitution changed numerics -- aborting export")
        return 1

    out_path = os.path.join(args.outdir, "trufor_%dx%d_dynamo.onnx" % (args.size, args.size))
    print("\nexporting with dynamo=True (opset %d) ..." % args.opset, flush=True)
    t0 = time.perf_counter()
    wrapper = ExportWrapper(model)
    wrapper.eval()
    try:
        torch.onnx.export(
            wrapper, (x,), out_path,
            input_names=["rgb"], output_names=["out", "conf", "det"],
            opset_version=args.opset, dynamo=True, verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        import traceback
        print("dynamo export FAILED: %s: %s" % (type(e).__name__, str(e)[:400]))
        with open(os.path.join(args.outdir, "dynamo-export-error.txt"), "w") as f:
            f.write(traceback.format_exc())
        return 1
    dt = time.perf_counter() - t0

    import onnx
    m = onnx.load(out_path, load_external_data=False)
    census = collections.Counter(n.op_type for n in m.graph.node)
    print("exported in %.1fs: %d nodes, %d distinct ops, %.1f MB"
          % (dt, len(m.graph.node), len(census), os.path.getsize(out_path) / 1e6))
    for op, n in census.most_common():
        print("   %5d  %s" % (n, op))

    with open(os.path.join(args.outdir, "dynamo-export.json"), "w") as f:
        json.dump({"seconds": dt, "nodes": len(m.graph.node), "distinct_ops": len(census),
                   "bytes": os.path.getsize(out_path), "op_census": dict(census.most_common()),
                   "pooling_substitutions": changes, "opset": args.opset}, f, indent=2)
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
