#!/usr/bin/env python3
"""Phase 8D: minimal reproducer, independent of TruFor.

The stage-1 bisect pinned the fault to a single construct:

    channel_weights : [2, 1, 64,  1,  1]   -> cw[0], cw[1]   WRONG on Vulkan
    spatial_weights  : [2, 1,  1, 128,128] -> sw[0], sw[1]   correct on Vulkan

Both are 5-D selects on dim 0. The shapes that fail have their last two
dimensions equal to 1, so almost the whole tensor is degenerate; the one that
works keeps two full 128-wide dimensions. Nothing else about them differs, and
every other operation tested on the same values (mul, add, broadcast mul, scalar
scale) is bit-exact.

This reproduces the bug from synthetic data with no model, no ONNX export of a
network, and no weights, so it can be filed upstream as-is.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


class Sel0(nn.Module):
    def forward(self, c):
        return c[0]


class Sel1(nn.Module):
    def forward(self, c):
        return c[1]


class SelBoth(nn.Module):
    def forward(self, c):
        return c[0], c[1]


class SelLast(nn.Module):
    def forward(self, c):
        return c[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    dev = tvm.vulkan()
    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    llvm = tvm.target.Target({"kind": "llvm"})
    print("device: %s\n" % dev.device_name, flush=True)

    def synth(shape):
        return torch.randn(*shape, dtype=torch.float32)

    shapes = [
        [2, 1, 64, 1, 1],        # channel_weights  (fails in the model)
        [2, 1, 1, 128, 128],     # spatial_weights   (works in the model)
        [2, 1, 64, 1, 1],        # control: identical
        [2, 1, 32, 1, 1],        # control: smaller degenerate tail
        [2, 1, 64, 2, 1],        # control: one non-unit trailing dim
        [2, 1, 64, 1, 2],        # control: the other trailing dim
        [2, 1, 64, 128, 1],      # control: one wide dim
        [2, 1, 64, 1],           # control: 4-D, degenerate tail
        [2, 3, 64, 1, 1],        # control: non-unit dim1
        [8, 1, 64, 1, 1],        # control: longer dim0
    ]
    ops = [("sel0", Sel0()), ("sel1", Sel1()), ("sel_both", SelBoth())]

    def stats(a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        d = np.abs(a - b)
        denom = max(float(np.abs(a).max()), 1e-9)
        return {"shape_match": True, "max_abs_err": float(d.max()),
                "rel_to_max": float(d.max() / denom),
                "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else None}

    def run(mod, inp, tgt, dev_obj):
        onx = os.path.join(args.outdir, "_tmp.onnx")
        torch.onnx.export(mod, (inp,), onx, input_names=["i0"], output_names=["o0"],
                          opset_version=18, dynamo=True, verbose=False)
        import onnx
        om = onnx.load(onx, load_external_data=True)
        ref = mod(inp)
        ref_list = ref if isinstance(ref, (list, tuple)) else (ref,)
        ref_arr = [t.numpy() for t in ref_list if torch.is_tensor(t)]
        names = ["o%d" % i for i in range(len(ref_arr))]
        torch.onnx.export(mod, (inp,), onx, input_names=["i0"], output_names=names,
                          opset_version=18, dynamo=True, verbose=False)
        om = onnx.load(onx, load_external_data=True)
        ir = from_onnx(om, shape_dict={"i0": list(inp.shape)}, dtype_dict="float32")
        exe = relax.build(ir, target=tgt)
        vm = relax.VirtualMachine(exe.jit(), dev_obj)
        import tvm_ffi
        out = vm["main"](tvm.runtime.tensor(inp.numpy().astype("float32"), dev_obj))
        got = ([out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array)
               else ([o.numpy() for o in out] if isinstance(out, (list, tuple)) else [out.numpy()]))
        return [stats(r, g) for r, g in zip(ref_arr, got)]

    rows = []
    print("%-22s %-9s %-12s %-12s %-12s %s"
          % ("input shape", "op", "vk_max_abs", "vk_rel", "vk_corr", "verdict"), flush=True)
    for shape in shapes:
        inp = synth(shape)
        for opname, mod in ops:
            try:
                vk = run(mod, inp, target, dev)
                # LLVM as the control: same graph, CPU backend
                cpu = run(mod, inp, llvm, tvm.cpu())
                vk_rel = max((s.get("rel_to_max", 1.0) if s.get("shape_match") else 1.0) for s in vk)
                cpu_rel = max((s.get("rel_to_max", 1.0) if s.get("shape_match") else 1.0) for s in cpu)
                ok = vk_rel < args.tolerance
                row = {"shape": shape, "op": opname,
                       "vulkan": vk, "llvm": cpu, "verdict": "OK" if ok else "WRONG",
                       "vulkan_rel": vk_rel, "llvm_rel": cpu_rel}
                print("%-22s %-9s %-12.3e %-12.3e %-12s %s"
                      % (str(shape), opname, vk[0].get("max_abs_err", float("nan")),
                         vk[0].get("rel_to_max", float("nan")),
                         ("%.8f" % vk[0]["corr"]) if vk[0].get("corr") is not None else "n/a",
                         row["verdict"]), flush=True)
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                print("%-22s %-9s ERROR %s" % (str(shape), opname, str(e)[:70]), flush=True)
                rows.append({"shape": shape, "op": opname, "verdict": "ERROR", "error": str(e)[:200]})

    bad = [r for r in rows if r.get("verdict") == "WRONG"]
    print("\n%d/%d shape+op combinations wrong on Vulkan" % (len(bad), len(rows)))
    if bad:
        print("\nFailing shapes (op=sel1, sel_both):")
        for r in bad:
            print("   %-22s %-9s vk_rel=%.3e   llvm_rel=%.3e" % (str(r["shape"]), r["op"], r["vulkan_rel"], r["llvm_rel"]))
    with open(os.path.join(args.outdir, "minimal-repro.json"), "w") as f:
        json.dump({"device": dev.device_name, "tolerance": args.tolerance, "rows": rows}, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "minimal-repro.json"))


if __name__ == "__main__":
    sys.exit(main())
