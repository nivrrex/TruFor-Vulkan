#!/usr/bin/env python3
"""Phase 6G: find the first stage where TVM-vulkan diverges from the reference.

The op-isolation suite showed every individual operator computes correctly on the
RX 560, including large tensors, yet the full 1918-node graph returns a
localization map with correlation -0.29 against the reference. So the fault is
in a stage-level combination, not in a single kernel.

This splits the *unmodified* TruFor model along its own structure
(Noiseprint++ -> preprocess -> SegFormer backbone -> loc head -> conf head ->
interpolate -> detection head), exports each stage as its own small ONNX from the
real stage inputs, and compares TVM-vulkan against PyTorch for each stage. The
first stage that diverges is the culprit; everything before it is exonerated.

TruFor sources are not modified: the stages are the model's own submodules,
wrapped only for export.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_orig_load = None


_REAL_LOAD = None


def _patched_load(*a, **kw):
    # Must not call torch.load through the module-level alias, or this recurses.
    kw.setdefault("weights_only", False)
    return _REAL_LOAD(*a, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tolerance", type=float, default=2e-2)
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
    ckpt = torch.load(args.model, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # The dynamo exporter has no ONNX decomposition for aten.adaptive_max_pool2d,
    # which CMX's ChannelWeights uses (net_utils.py:13-14). For output_size==1 a
    # global max/mean over dims (2,3) is exactly equivalent, and neither pooling
    # module holds parameters, so the checkpoint is unaffected. This mirrors what
    # export_onnx_dynamo.py does for the full model.
    class GlobalAvgPool2d(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

    class GlobalMaxPool2d(nn.Module):
        def forward(self, x):
            return x.amax(dim=(2, 3), keepdim=True)

    n_swapped = 0
    for mod_name, mod in model.named_modules():
        for attr in ("avg_pool", "max_pool"):
            child = getattr(mod, attr, None)
            osz = getattr(child, "output_size", None)
            osz = (osz,) if isinstance(osz, int) else (tuple(osz) if osz is not None else None)
            if osz == (1,):
                if isinstance(child, nn.AdaptiveAvgPool2d):
                    setattr(mod, attr, GlobalAvgPool2d())
                elif isinstance(child, nn.AdaptiveMaxPool2d):
                    setattr(mod, attr, GlobalMaxPool2d())
                else:
                    continue
                n_swapped += 1
    print("pooling substitutions: %d" % n_swapped, flush=True)

    class PackedDecoder(nn.Module):
        """DecoderHead.forward takes the feature *list*; give the exporter
        explicit positional inputs instead of a list constant."""

        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, a, b, c, d):
            return self.dec([a, b, c, d])

    dev = tvm.vulkan()
    print("device: %s\n" % dev.device_name, flush=True)
    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})

    # ---- real input ----
    img = Image.open(args.image).convert("RGB")
    if img.size != (args.size, args.size):
        img = img.resize((args.size, args.size), Image.BILINEAR)
    rgb01 = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    rgb = torch.from_numpy(rgb01)

    # ---- stage inputs, computed in PyTorch (the reference path) ----
    modal_x = model.dncnn(rgb)                       # [1,1,H,W]
    modal_x3 = torch.tile(modal_x, (3, 1, 1))        # [1,3,H,W]
    rgb_in = model.prepro(rgb01) if False else (rgb01 - np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)) / \
        np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    rgb_in = torch.from_numpy(rgb_in.astype(np.float32))
    feats = model.backbone(rgb_in, modal_x3)         # list/conv of features
    feats = [f if torch.is_tensor(f) else f for f in (feats if isinstance(feats, (list, tuple)) else [feats])]
    print("backbone outputs: %s" % [list(f.shape) for f in feats], flush=True)
    out_head = model.decode_head(feats)
    conf_head = model.decode_head_conf(feats)
    out_up = F.interpolate(out_head, size=(args.size, args.size), mode="bilinear", align_corners=False)
    conf_up = F.interpolate(conf_head, size=(args.size, args.size), mode="bilinear", align_corners=False)

    class Multi(nn.Module):
        def __init__(self, mods):
            super().__init__()
            self.mods = nn.ModuleList(mods)

        def forward(self, *ins):
            out = ins
            for m in self.mods:
                out = m(*out) if isinstance(out, (tuple, list)) else m(out)
                out = out if isinstance(out, (tuple, list)) else (out,)
            return out[0] if len(out) == 1 else tuple(out)

    def seq(*mods):
        class S(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.ModuleList(mods)

            def forward(self, x):
                for mm in self.m:
                    x = mm(x)
                return x
        return S()

    up = nn.Upsample(size=(args.size, args.size), mode="bilinear", align_corners=False)

    stages = [
        ("noiseprint++", model.dncnn, [rgb]),
        ("backbone", Multi([model.backbone]), [rgb_in, modal_x3]),
        ("loc_head", PackedDecoder(model.decode_head), list(feats)),
        ("conf_head", PackedDecoder(model.decode_head_conf), list(feats)),
        ("loc_up", up, [out_head]),
        ("conf_up", up, [conf_head]),
    ]

    def npify(x):
        if torch.is_tensor(x):
            return x.numpy()
        if isinstance(x, (list, tuple)):
            return [npify(i) for i in x]
        return x

    rows = []
    print("%-14s %-8s %-12s %-12s %s" % ("stage", "build_s", "max_abs", "corr", "verdict"))
    for name, mod, inputs in stages:
        row = {"stage": name, "n_inputs": len(inputs)}
        try:
            t0 = time.perf_counter()
            onx = os.path.join(args.outdir, "stage-%s.onnx" % name.replace("+", "p"))
            names = ["in%d" % i for i in range(len(inputs))]
            outs = mod(*inputs)
            ref_list = outs if isinstance(outs, (list, tuple)) else [outs]
            ref_arr = [o.numpy() for o in ref_list if torch.is_tensor(o)]
            torch.onnx.export(mod, tuple(inputs), onx, input_names=names,
                              output_names=["o%d" % i for i in range(len(ref_arr))],
                              opset_version=18, dynamo=True, verbose=False)
            build_export = time.perf_counter() - t0

            import onnx
            om = onnx.load(onx, load_external_data=True)
            shape_dict = {n: list(i.shape) for n, i in zip(names, inputs)}  # inputs are tensors here
            ir = from_onnx(om, shape_dict=shape_dict, dtype_dict="float32")

            t1 = time.perf_counter()
            exe = relax.build(ir, target=target)
            build_s = time.perf_counter() - t1
            vm = relax.VirtualMachine(exe.jit(), dev)
            tin = [tvm.runtime.tensor(i, dev) for i in inputs]
            out = vm["main"](*tin)
            import tvm_ffi
            outs_np = [out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array) else (
                [o.numpy() for o in out] if isinstance(out, (list, tuple)) else [out.numpy()])

            max_abs = 0.0
            corrs = []
            for r, g in zip(ref_arr, outs_np):
                r = r.astype(np.float64)
                g = np.asarray(g, np.float64)
                if r.shape != g.shape:
                    row["verdict"] = "SHAPE"
                    row["detail"] = "%s vs %s" % (r.shape, g.shape)
                    break
                d = np.abs(r - g)
                max_abs = max(max_abs, float(d.max()))
                if r.size > 1 and r.std() > 0 and g.std() > 0:
                    corrs.append(float(np.corrcoef(r.ravel(), g.ravel())[0, 1]))
            row.update({"export_s": build_export, "build_s": build_s,
                        "max_abs_err": max_abs,
                        "corr": min(corrs) if corrs else None,
                        "n_outputs": len(ref_arr)})
            ok = row["max_abs_err"] < args.tolerance
            row["verdict"] = "OK" if ok else "DIVERGES"
            print("%-14s %-8.2f %-12.3e %-12s %s"
                  % (name, build_s, max_abs, ("%.8f" % row["corr"]) if row["corr"] is not None else "n/a",
                     row["verdict"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["verdict"] = "ERROR"
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            with open(os.path.join(args.outdir, "stage-%s-error.txt" % name.replace("+", "p")), "w") as f:
                f.write(traceback.format_exc())
            print("%-14s %-8s %-12s %-12s ERROR %s" % (name, "-", "-", "-", row["error"][:55]), flush=True)
        rows.append(row)
        # stop at the first divergence: everything after it is uninterpretable
        if row.get("verdict") in ("DIVERGES", "SHAPE"):
            print("\n  -> first divergence at stage '%s'; stopping here." % name, flush=True)
            break

    with open(os.path.join(args.outdir, "stage-isolation.json"), "w") as f:
        json.dump({"device": dev.device_name, "tolerance": args.tolerance, "stages": rows}, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "stage-isolation.json"))


if __name__ == "__main__":
    sys.exit(main())
