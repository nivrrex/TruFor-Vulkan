#!/usr/bin/env python3
"""Phase 8C, final level: which broadcast shape combination is wrong?

Established so far, all on the RX 560 against eager PyTorch:

    patch_embed1, extra_patch_embed1, block1[0..2], extra_block1[0..2],
    norm1, extra_norm1                    exact (rel_max ~1e-7, corr 1.0)
    FRMs[0].channel_weights                exact (rel_max 1.2e-7)
    FRMs[0].spatial_weights                exact (rel_max 2.1e-7)
    FRMs[0] broadcast arithmetic           WRONG (rel_max 1.57e-01)

So it is not fusion, not attention, not the 1x1 conv stack. It is the arithmetic
that mixes the 5-D weight tensors with the 4-D activations:

    x1 [1, 64,128,128] + 0.5 * channel_weights[1] [1,64,1,1] * x2
                       + 0.5 * spatial_weights[1]  [1,1,128,128] * x2

This isolates each shape combination separately, plus same-shape and scalar
multiplication as controls, so the failing one is unambiguous.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch.nn as nn

_REAL_LOAD = None


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _REAL_LOAD(*a, **kw)


class Mul(nn.Module):
    """a * b, or one of the controls. Inputs arrive already sliced."""

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, a, b):
        if self.op == "mul":
            return a * b
        if self.op == "add":
            return a + b
        if self.op == "scalar_mul":
            return 0.5 * b
        if self.op == "scale_mul":
            return 0.5 * a * b
        raise ValueError(self.op)


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

    class GAP(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

    class GMP(nn.Module):
        def forward(self, x):
            return x.amax(dim=(2, 3), keepdim=True)

    for _n, m in model.named_modules():
        for attr in ("avg_pool", "max_pool"):
            child = getattr(m, attr, None)
            osz = getattr(child, "output_size", None)
            osz = (osz,) if isinstance(osz, int) else (tuple(osz) if osz is not None else None)
            if osz == (1,):
                setattr(m, attr, GAP() if isinstance(child, nn.AdaptiveAvgPool2d) else GMP())

    bb = model.backbone
    frm = bb.FRMs[0]
    dev = tvm.vulkan()
    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    print("device: %s" % dev.device_name, flush=True)

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
    for name, m in (("frm", frm), ("cw", frm.channel_weights), ("sw", frm.spatial_weights)):
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

    x1, x2 = cap["frm"][0]
    cw = cap["cw"][1][0]
    sw = cap["sw"][1][0]
    cw1, cw0 = cw[1], cw[0]
    sw1, sw0 = sw[1], sw[0]
    print("x1=%s x2=%s  cw=%s sw=%s  cw1=%s sw1=%s"
          % (list(x1.shape), list(x2.shape), list(cw.shape), list(sw.shape),
             list(cw1.shape), list(sw1.shape)), flush=True)

    cases = [
        # label,                    op,            a,        b
        ("C1_chan_mul  [1,64,1,1]x[1,64,128,128]", "mul", cw1, x2),
        ("C2_spat_mul  [1,1,128,128]x[1,64,128,128]", "mul", sw1, x2),
        ("C3_chan_samemul [1,64,1,1]x[1,64,1,1]", "mul", cw1, cw1),
        ("C4_spat_samemul [1,1,128,128]x[1,1,128,128]", "mul", sw1, sw1),
        ("C5_spat_chan   [1,1,128,128]x[1,64,1,1]", "mul", sw1, cw1),
        ("C6_scalar_mul  0.5*x2", "scalar_mul", cw1, x2),
        ("C7_scale_chan  0.5*cw1*x2", "scale_mul", cw1, x2),
        ("C8_chan_add    x1+sw1 (broadcast add)", "add", x1, sw1),
        ("C9_full_samemul x2*x2", "mul", x2, x2),
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
    print("\n%-46s %-8s %-11s %-11s %-11s %s"
          % ("case", "build_s", "max_abs", "rel_max", "corr", "verdict"), flush=True)
    for label, op, a, b in cases:
        mod = Mul(op).eval()
        row = {"case": label, "op": op, "a_shape": list(a.shape), "b_shape": list(b.shape)}
        try:
            with torch.no_grad():
                ref = mod(a, b).numpy()
            t0 = time.perf_counter()
            onx = os.path.join(args.outdir, "bcast-%s.onnx" % label.split()[0])
            torch.onnx.export(mod, (a, b), onx, input_names=["a", "b"], output_names=["o"],
                              opset_version=18, dynamo=True, verbose=False)
            import onnx
            om = onnx.load(onx, load_external_data=True)
            ir = from_onnx(om, shape_dict={"a": list(a.shape), "b": list(b.shape)}, dtype_dict="float32")
            exe = relax.build(ir, target=target)
            build_s = time.perf_counter() - t0
            vm = relax.VirtualMachine(exe.jit(), dev)
            import tvm_ffi
            out = vm["main"](tvm.runtime.tensor(a.numpy().astype("float32"), dev),
                             tvm.runtime.tensor(b.numpy().astype("float32"), dev))
            got = ([out[i].numpy() for i in range(len(out))] if isinstance(out, tvm_ffi.Array)
                   else ([o.numpy() for o in out] if isinstance(out, (list, tuple)) else [out.numpy()]))
            st = stats(ref, got[0])
            row.update({"build_s": build_s, **st,
                        "verdict": "OK" if st.get("shape_match") and st.get("rel_to_max", 1) < args.tolerance else "DIVERGES"})
            print("%-46s %-8.2f %-11.3e %-11.3e %-11s %s"
                  % (label, build_s, st.get("max_abs_err", float("nan")), st.get("rel_to_max", float("nan")),
                     ("%.8f" % st["corr"]) if st.get("corr") is not None else "n/a", row["verdict"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["verdict"] = "ERROR"
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
            with open(os.path.join(args.outdir, "bcast-%s-error.txt" % label.split()[0]), "w") as f:
                f.write(traceback.format_exc())
            print("%-46s %-8s %-11s %-11s %-11s ERROR %s"
                  % (label, "-", "-", "-", "-", row["error"][:35]), flush=True)
        rows.append(row)

    with open(os.path.join(args.outdir, "broadcast-bisect.json"), "w") as f:
        json.dump({"device": dev.device_name, "tolerance": args.tolerance, "rows": rows}, f, indent=2)
    bad = [r["case"] for r in rows if r.get("verdict") != "OK"]
    print("\n%d/%d shape combinations correct" % (len(rows) - len(bad), len(rows)))
    if bad:
        print("DIVERGING: %s" % "; ".join(bad))
    print("wrote %s" % os.path.join(args.outdir, "broadcast-bisect.json"))


if __name__ == "__main__":
    sys.exit(main())
