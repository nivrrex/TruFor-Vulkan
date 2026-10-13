#!/usr/bin/env python3
"""Phase 8B/8C: dissect FeatureRectifyModule (FRMs[0]) on TVM-Vulkan.

Stage-1 bisect result: every sub-module up to and including `extra_norm1` is
exact on the RX 560 (rel_max ~1e-7, corr 1.00000000). The first divergence is

    FRMs[0]  --  rel_max 1.66e-01, corr 0.98026444

`FeatureRectifyModule.forward` (lib/models/cmx/net_utils.py) is:

    channel_weights = self.channel_weights(x1, x2)   # -> [2, B, C, 1, 1]  (5-D)
    spatial_weights  = self.spatial_weights(x1, x2)    # -> [2, B, 1, H, W]  (5-D)
    out_x1 = x1 + lambda_c * channel_weights[1] * x2 + lambda_s * spatial_weights[1] * x2
    out_x2 = x2 + lambda_c * channel_weights[0] * x1 + lambda_s * spatial_weights[0] * x1

The distinguishing shape pattern is a **5-D weight tensor multiplied against a
4-D activation** (implicit broadcast), which the earlier per-operator suite
never exercised -- `BIG_attn_5d` multiplied 5-D by 5-D.

This splits the module into four independently testable pieces, each fed the
real captured tensors:

    A  channel_weights(x1, x2)                  # the attention itself
    B  spatial_weights(x1, x2)                  # 1x1 conv stack
    C  the broadcast arithmetic only            # weights supplied as inputs
    D  the two composed, using the captured weights

If A and B are exact and C is not, the fault is the 5-D x 4-D broadcast kernel.
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
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    import torch.nn as nn

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

    class GlobalAvgPool2d(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

    class GlobalMaxPool2d(nn.Module):
        def forward(self, x):
            return x.amax(dim=(2, 3), keepdim=True)

    n_swap = 0
    for _n, m in model.named_modules():
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

    bb = model.backbone
    frm = bb.FRMs[0]
    dev = tvm.vulkan()
    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("device: %s   lambda_c=%s lambda_s=%s"
          % (dev.device_name, frm.lambda_c, frm.lambda_s), flush=True)

    img = Image.open(args.image).convert("RGB")
    if img.size != (args.size, args.size):
        img = img.resize((args.size, args.size), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    rgb = torch.from_numpy(x)
    with torch.no_grad():
        np3 = torch.tile(model.dncnn(rgb), (3, 1, 1))
        mu = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
        sd = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
        rgbin = torch.from_numpy(((x - mu) / sd).astype(np.float32))

    cap = {}
    hs = []
    for name, m in (("frm", frm), ("channel_weights", frm.channel_weights),
                    ("spatial_weights", frm.spatial_weights)):
        def mk(n):
            def hook(_m, inputs, output):
                ins = [t.detach().clone() for t in inputs if torch.is_tensor(t)]
                outs = output if isinstance(output, (list, tuple)) else (output,)
                outs = [t.detach().clone() for t in outs if torch.is_tensor(t)]
                if ins and outs and n not in cap:
                    cap[n] = (ins, outs)
            return hook
        hs.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        bb.forward_features(rgbin, np3)
    for h in hs:
        h.remove()

    frm_in, frm_out = cap["frm"]
    cw_in, cw_out = cap["channel_weights"]
    sw_in, sw_out = cap["spatial_weights"]
    x1, x2 = frm_in
    cw = cw_out[0]
    sw = sw_out[0]
    print("x1 %s  x2 %s  channel_weights %s  spatial_weights %s"
          % (list(x1.shape), list(x2.shape), list(cw.shape), list(sw.shape)), flush=True)

    class BroadcastOnly(nn.Module):
        """Exactly the FRM arithmetic, with the weights supplied as inputs."""

        def __init__(self, lc, ls):
            super().__init__()
            self.lc = lc
            self.ls = ls

        def forward(self, a, b, cwv, swv):
            o1 = a + self.lc * cwv[1] * b + self.ls * swv[1] * b
            o2 = b + self.lc * cwv[0] * a + self.ls * swv[0] * a
            return o1, o2

    cases = [
        ("A_channel_weights", frm.channel_weights, [x1, x2], 1),
        ("B_spatial_weights", frm.spatial_weights, [x1, x2], 1),
        ("C_broadcast_only", BroadcastOnly(frm.lambda_c, frm.lambda_s), [x1, x2, cw, sw], 2),
    ]

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
    print("\n%-22s %-8s %-11s %-11s %-11s %s" % ("case", "build_s", "max_abs", "rel_max", "corr", "verdict"))
    for label, mod, ins, nout in cases:
        row = {"case": label, "n_inputs": len(ins), "n_outputs": nout,
               "input_shapes": [list(t.shape) for t in ins]}
        try:
            with torch.no_grad():
                ref = mod(*ins)
            ref_list = ref if isinstance(ref, (list, tuple)) else (ref,)
            ref_arr = [t.numpy() for t in ref_list if torch.is_tensor(t)]

            t0 = time.perf_counter()
            onx = os.path.join(args.outdir, "frm-%s.onnx" % label)
            in_names = ["i%d" % i for i in range(len(ins))]
            out_names = ["o%d" % i for i in range(len(ref_arr))]
            torch.onnx.export(mod, tuple(ins), onx, input_names=in_names,
                              output_names=out_names, opset_version=18, dynamo=True, verbose=False)
            import onnx
            om = onnx.load(onx, load_external_data=True)
            ir = from_onnx(om, shape_dict={n: list(t.shape) for n, t in zip(in_names, ins)},
                           dtype_dict="float32")
            exe = relax.build(ir, target=target)
            build_s = time.perf_counter() - t0
            vm = relax.VirtualMachine(exe.jit(), dev)
            import tvm_ffi
            out = vm["main"](*[tvm.runtime.tensor(t.numpy().astype("float32"), dev) for t in ins])
            got = ([out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array)
                   else ([o.numpy() for o in out] if isinstance(out, (list, tuple)) else [out.numpy()]))

            per, worst = [], 0.0
            for r, g in zip(ref_arr, got):
                st = stats(r, g)
                per.append(st)
                worst = max(worst, st.get("rel_to_max", float("inf")) if st.get("shape_match") else float("inf"))
            row.update({"build_s": build_s, "outputs": per, "worst_rel": worst,
                        "verdict": "OK" if worst < args.tolerance else "DIVERGES"})
            w = per[0]
            print("%-22s %-8.2f %-11.3e %-11.3e %-11s %s"
                  % (label, build_s, w.get("max_abs_err", float("nan")),
                     w.get("rel_to_max", float("nan")),
                     ("%.8f" % w["corr"]) if w.get("corr") is not None else "n/a", row["verdict"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["verdict"] = "ERROR"
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            with open(os.path.join(args.outdir, "frm-%s-error.txt" % label), "w") as f:
                f.write(traceback.format_exc())
            print("%-22s %-8s %-11s %-11s %-11s ERROR %s" % (label, "-", "-", "-", "-", row["error"][:45]), flush=True)
        rows.append(row)

    with open(os.path.join(args.outdir, "frm-bisect.json"), "w") as f:
        json.dump({"device": dev.device_name, "lambda_c": float(frm.lambda_c),
                   "lambda_s": float(frm.lambda_s), "tolerance": args.tolerance,
                   "x1": list(x1.shape), "x2": list(x2.shape),
                   "channel_weights": list(cw.shape), "spatial_weights": list(sw.shape),
                   "rows": rows}, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "frm-bisect.json"))


if __name__ == "__main__":
    sys.exit(main())
