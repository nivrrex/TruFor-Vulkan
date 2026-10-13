#!/usr/bin/env python3
"""Phase 8A + 8C: find the first wrong tensor inside SegFormer stage 1.

Context. Everything below TVM is verified correct:
    PyTorch CPU  ==  ONNX Runtime CPU  ==  TVM llvm
and on the Vulkan device every individual operator (including large tensors and
the 5-D CMX attention layout) matches eager. Yet the backbone's stage-1 output
f0 already carries ~20% relative error.

So this walks stage 1 in execution order, and for each sub-module builds a small
ONNX from the *real* captured inputs, runs it on TVM-Vulkan, and compares against
PyTorch. The first sub-module whose output diverges is the first wrong tensor.

Stage 1 order (lib/models/cmx/encoders/dual_segformer.py: forward_features):
    patch_embed1 -> extra_patch_embed1
    -> block1[0..2] -> extra_block1[0..2]
    -> norm1 -> extra_norm1
    -> reshape/permute
    -> FRMs[0]  (CrossAttention: the only 5-D producer)
    -> FFMs[0]

TruFor sources are untouched: sub-modules are the model's own, wrapped only for
export, and fed with values PyTorch actually produced.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_REAL_LOAD = None


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _REAL_LOAD(*a, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tolerance", type=float, default=5e-2)
    ap.add_argument("--stop-on-first", action="store_true", default=True)
    ap.add_argument("--all", dest="stop_on_first", action="store_false")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    global _REAL_LOAD
    _REAL_LOAD = torch.load
    torch.load = _patched_load
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    from PIL import Image

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    TRAIN_DIR = os.path.join(os.environ.get("TRUFOR_ROOT", os.path.join(ROOT, "truFor-src")), "TruFor_train_test")
    if TRAIN_DIR not in sys.path:
        sys.path.insert(0, TRAIN_DIR)
    from lib.config import config, update_config
    from lib.utils import get_model

    class _A:
        pass

    prev = os.getcwd()
    os.chdir(TRAIN_DIR)
    try:
        a = _A()
        a.experiment = "trufor_ph3"
        a.opts = []
        update_config(config, a)
        model = get_model(config)
    finally:
        os.chdir(prev)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["state_dict"], strict=True)
    model.eval()

    # same global max/avg pooling rewrite used for every other export
    class GlobalAvgPool2d(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

    class GlobalMaxPool2d(nn.Module):
        def forward(self, x):
            return x.amax(dim=(2, 3), keepdim=True)

    n_swap = 0
    for mod_name, m in model.named_modules():
        for attr in ("avg_pool", "max_pool"):
            child = getattr(m, attr, None)
            osz = getattr(child, "output_size", None)
            osz = (osz,) if isinstance(osz, int) else (tuple(osz) if osz is not None else None)
            if osz == (1,):
                if isinstance(child, nn.AdaptiveAvgPool2d):
                    setattr(m, attr, GlobalAvgPool2d())
                elif isinstance(child, nn.AdaptiveMaxPool2d):
                    setattr(m, attr, GlobalMaxPool2d())
                else:
                    continue
                n_swap += 1
    print("pooling substitutions: %d" % n_swap, flush=True)

    bb = model.backbone
    dev = tvm.vulkan()
    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("device: %s\n" % dev.device_name, flush=True)

    # ---- real stage-1 inputs ----
    img = Image.open(args.image).convert("RGB")
    if img.size != (args.size, args.size):
        img = img.resize((args.size, args.size), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    rgb = torch.from_numpy(x)
    with torch.no_grad():
        np1 = model.dncnn(rgb)
        np3 = torch.tile(np1, (3, 1, 1))
        mu = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
        sd = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
        rgbin = torch.from_numpy(((x - mu) / sd).astype(np.float32))

    # ---- capture the real inputs of every stage-1 sub-module ----
    captured = {}
    handles = []

    def grab(name):
        def hook(_m, inputs, output):
            ins = [t.detach().clone() for t in inputs if torch.is_tensor(t)]
            # SegFormer Block.forward(x, H, W) also takes the spatial size as
            # plain ints; they must be replayed or the export fails.
            extra = [v for v in inputs if not torch.is_tensor(v)]
            outs = output if isinstance(output, (list, tuple)) else (output,)
            outs = [t.detach().clone() for t in outs if torch.is_tensor(t)]
            if ins and outs and name not in captured:
                captured[name] = (ins, outs, extra)
        return hook

    targets = [
        ("patch_embed1", bb.patch_embed1),
        ("extra_patch_embed1", bb.extra_patch_embed1),
        ("block1[0]", bb.block1[0]),
        ("block1[1]", bb.block1[1]),
        ("block1[2]", bb.block1[2]),
        ("extra_block1[0]", bb.extra_block1[0]),
        ("extra_block1[1]", bb.extra_block1[1]),
        ("extra_block1[2]", bb.extra_block1[2]),
        ("norm1", bb.norm1),
        ("extra_norm1", bb.extra_norm1),
        ("FRMs[0]", bb.FRMs[0]),
        ("FFMs[0]", bb.FFMs[0]),
    ]
    for name, m in targets:
        handles.append(m.register_forward_hook(grab(name)))

    with torch.no_grad():
        bb.forward_features(rgbin, np3)   # runs all 4 stages; we only need stage-1 hooks
    for h in handles:
        h.remove()

    print("captured %d/%d stage-1 sub-modules\n" % (len(captured), len(targets)), flush=True)

    def npify(o):
        return o.numpy() if torch.is_tensor(o) else np.asarray(o)

    def stats(a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        d = np.abs(a - b)
        denom = max(float(np.abs(a).max()), 1e-9)
        return {"shape_match": True, "max_abs_err": float(d.max()),
                "mean_abs_err": float(d.mean()), "rel_to_max": float(d.max() / denom),
                "rmse": float(np.sqrt((d ** 2).mean())),
                "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else None,
                "ref_absmax": float(np.abs(a).max())}

    rows = []
    print("%-18s %-8s %-11s %-11s %-11s %s" % ("sub-module", "build_s", "max_abs", "rel_max", "corr", "verdict"))
    for name, _m in targets:
        if name not in captured:
            rows.append({"submodule": name, "verdict": "NOT-CAPTURED"})
            print("%-18s %s" % (name, "NOT-CAPTURED"))
            continue
        ins, ref_outs, extra_args = captured[name]
        row = {"submodule": name, "n_inputs": len(ins), "n_outputs": len(ref_outs),
               "extra_args": [str(v) for v in extra_args],
               "input_shapes": [list(t.shape) for t in ins],
               "output_shapes": [list(t.shape) for t in ref_outs]}
        try:
            t0 = time.perf_counter()
            onx = os.path.join(args.outdir, "s1-%s.onnx" % name.replace("[", "_").replace("]", ""))
            in_names = ["i%d" % i for i in range(len(ins))]
            out_names = ["o%d" % i for i in range(len(ref_outs))]

            mod = dict(targets)[name]
            extra_t = tuple(v for v in extra_args)

            class Wrap(nn.Module):
                def __init__(self, mod, nout):
                    super().__init__()
                    self.mod = mod
                    self.nout = nout

                def forward(self, *t):
                    o = self.mod(*t, *extra_t) if extra_t else self.mod(*t)
                    o = o if isinstance(o, (list, tuple)) else (o,)
                    o = [x for x in o if torch.is_tensor(x)]
                    return tuple(o) if len(o) > 1 else o[0]

            wrap = Wrap(mod, len(ref_outs)).eval()
            torch.onnx.export(wrap, tuple(ins), onx, input_names=in_names,
                              output_names=out_names, opset_version=18, dynamo=True, verbose=False)
            import onnx
            om = onnx.load(onx, load_external_data=True)
            ir = from_onnx(om, shape_dict={n: list(t.shape) for n, t in zip(in_names, ins)},
                           dtype_dict="float32")
            build_s = time.perf_counter() - t0
            exe = relax.build(ir, target=target)
            vm = relax.VirtualMachine(exe.jit(), dev)
            import tvm_ffi
            out = vm["main"](*[tvm.runtime.tensor(t.numpy().astype("float32"), dev) for t in ins])
            got = ([out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array)
                   else ([o.numpy() for o in out] if isinstance(out, (list, tuple)) else [out.numpy()]))

            per_out = []
            worst_rel = 0.0
            for r, g in zip(ref_outs, got):
                st = stats(npify(r), g)
                per_out.append(st)
                if st.get("shape_match") and st.get("rel_to_max") is not None:
                    worst_rel = max(worst_rel, st["rel_to_max"])
                elif not st.get("shape_match"):
                    worst_rel = float("inf")
            row["build_s"] = build_s
            row["outputs"] = per_out
            row["worst_rel_to_max"] = worst_rel
            ok = worst_rel < args.tolerance
            row["verdict"] = "OK" if ok else "DIVERGES"
            w = per_out[0] if per_out else {}
            print("%-18s %-8.2f %-11.3e %-11.3e %-11s %s"
                  % (name, build_s, w.get("max_abs_err", float("nan")),
                     w.get("rel_to_max", float("nan")),
                     ("%.8f" % w["corr"]) if w.get("corr") is not None else "n/a", row["verdict"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["verdict"] = "ERROR"
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            with open(os.path.join(args.outdir, "s1-%s-error.txt" % name.replace("[", "_").replace("]", "")), "w") as f:
                f.write(traceback.format_exc())
            print("%-18s %-8s %-11s %-11s %-11s ERROR %s" % (name, "-", "-", "-", "-", row["error"][:45]), flush=True)
        rows.append(row)
        if row["verdict"] == "DIVERGES" and args.stop_on_first:
            print("\n  -> first divergence at '%s'; stopping." % name, flush=True)
            break

    with open(os.path.join(args.outdir, "stage1-bisect.json"), "w") as f:
        json.dump({"device": dev.device_name, "tolerance": args.tolerance, "rows": rows}, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "stage1-bisect.json"))


if __name__ == "__main__":
    sys.exit(main())
