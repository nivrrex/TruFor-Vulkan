#!/usr/bin/env python3
"""Export TruFor to ONNX with dynamic spatial axes (H, W).

Sibling of export_onnx_dynamo.py. That one bakes a fixed size into the graph
(`rgb` is 1x3xSxS) because the production TVM/Vulkan pipeline matches its
hand-written kernels against static buffer signatures. This one is for the
ONNX Runtime path, where arbitrary input sizes must work: the model already
resizes `out`/`conf` back to the input spatial extent inside the graph

    builder_np_conf.encode_decode:
        orisize = rgb.shape
        out  = F.interpolate(out,  size=orisize[2:], mode='bilinear', align_corners=False)
        conf = F.interpolate(conf, size=orisize[2:], mode='bilinear', align_corners=False)

so as long as H and W stay symbolic through the export, the map comes back at
exactly the input size for any size. That is the contract the caller relies on.

The pooling substitutions and the None-output wrapper are reused verbatim from
export_onnx_dynamo (see that module for why the swap is exact); this file only
changes the sample tensor and adds the dynamic axes.

Self-contained: uses only this project's truFor-src model. Does not touch or
import anything outside this repository.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from export_onnx_dynamo import (  # noqa: E402
    build_model, swap_adaptive_pool, ExportWrapper,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    # Non-square sample: square risks the tracer collapsing H and W onto the
    # same symbol. Both dims divisible by 32 (the encoder's max downsample).
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--exp", default="trufor_ph3")
    ap.add_argument("--out", default="trufor_dynamic_dynamo.onnx")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.set_grad_enabled(False)

    model = build_model(args.model, args.exp)
    model.eval()

    x = torch.rand(1, 3, args.h, args.w)
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

    checks = [eq(ref[i].numpy(), got[i].numpy(), nm) for i, nm in enumerate(("out", "conf"))]
    if ref[2] is not None and got[2] is not None:
        checks.append(eq(ref[2].numpy(), got[2].numpy(), "det"))
    if not all(checks):
        print("\nsubstitution changed numerics -- aborting export")
        return 1

    out_path = os.path.join(args.outdir, args.out)
    print("\nexporting with dynamo=True (opset %d), dynamic H/W ..." % args.opset, flush=True)
    t0 = time.perf_counter()
    wrapper = ExportWrapper(model)
    wrapper.eval()
    dynamic_axes = {
        "rgb":  {2: "height", 3: "width"},
        "out":  {2: "height", 3: "width"},
        "conf": {2: "height", 3: "width"},
        # det collapses the spatial extent (weighted_statistics_pooling),
        # so its shape is input-size independent: left static on purpose.
    }
    try:
        torch.onnx.export(
            wrapper, (x,), out_path,
            input_names=["rgb"], output_names=["out", "conf", "det"],
            opset_version=args.opset, dynamic_axes=dynamic_axes,
            dynamo=True, verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        import traceback
        print("dynamo export FAILED: %s: %s" % (type(e).__name__, str(e)[:400]))
        with open(os.path.join(args.outdir, "dynamic-export-error.txt"), "w") as f:
            f.write(traceback.format_exc())
        return 1
    dt = time.perf_counter() - t0

    import onnx
    # torch>=2.13's dynamo exporter writes initializers to a sidecar
    # .onnx.data file. Re-save inline so the artifact is a single self-contained
    # .onnx (matches the fixed 512 export; keeps the eventual embed to one file).
    sidecar = out_path + ".data"
    m = onnx.load(out_path, load_external_data=True)
    for init in m.graph.initializer:
        init.data_location = onnx.TensorProto.DEFAULT
    onnx.save_model(m, out_path, save_as_external_data=False)
    if os.path.exists(sidecar):
        os.remove(sidecar)

    m = onnx.load(out_path, load_external_data=False)
    census = collections.Counter(n.op_type for n in m.graph.node)

    # Confirm the axes actually stayed symbolic: the input tensor's dim 2/3
    # should be a param/dynamic dim, not a fixed integer.
    inp = m.graph.input[0].type.tensor_type.shape
    dims = []
    for d in inp.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(str(d.dim_value))
        else:
            dims.append(d.dim_param or "?")
    print("rgb dims: %s" % " x ".join(dims))

    print("exported in %.1fs: %d nodes, %d distinct ops, %.1f MB"
          % (dt, len(m.graph.node), len(census), os.path.getsize(out_path) / 1e6))
    for op, n in census.most_common(12):
        print("   %5d  %s" % (n, op))

    with open(os.path.join(args.outdir, "dynamic-export.json"), "w") as f:
        json.dump({"seconds": dt, "nodes": len(m.graph.node),
                   "distinct_ops": len(census), "bytes": os.path.getsize(out_path),
                   "rgb_dims": dims, "dynamic_axes": dynamic_axes,
                   "op_census_top": dict(census.most_common(30)),
                   "pooling_substitutions": changes, "opset": args.opset}, f, indent=2)
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
