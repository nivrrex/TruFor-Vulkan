#!/usr/bin/env python3
"""Phase 8C, decisive test: is the 5-D index (select) the culprit?

All 9 single broadcast shape combinations tested so far are bit-exact
(max_abs 0.0), yet the FRM arithmetic assembled in one graph is wrong
(rel_max 1.57e-01). The one thing the passing cases do *not* exercise is
slicing the 5-D weight tensors **inside** the graph:

    channel_weights [2,1,64,1,1]   -> channel_weights[1] -> [1,64,1,1]
    spatial_weights  [2,1,1,128,128]-> spatial_weights[1]  -> [1,1,128,128]

So this compares four variants of the same arithmetic, differing only in where
the slicing happens:

    S1  select only            cw5d -> cw5d[1]                 (isolates select)
    S2  select, both halves    cw5d -> (cw5d[0], cw5d[1])
    S3  full FRM, slice inside x1,x2,cw5d,sw5d -> (o1, o2)
    S4  full FRM, slice outside  x1,x2,cw1,sw1    -> (o1, o2)   (known WRONG)

S3 and S4 are mathematically identical. If S3 is exact and S4 is wrong, the
fault is in slicing-and-then-using inside a single graph, not in the broadcast
kernel and not in the select kernel alone.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_REAL_LOAD = None


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _REAL_LOAD(*a, **kw)


class Select(nn.Module):
    def forward(self, c):
        return c[1]


class SelectBoth(nn.Module):
    def forward(self, c):
        return c[0], c[1]


class FrmInside(nn.Module):
    """FRM arithmetic with the 5-D -> 4-D slice performed inside the graph."""

    def __init__(self, lc, ls):
        super().__init__()
        self.lc = lc
        self.ls = ls

    def forward(self, a, b, cwv, swv):
        o1 = a + self.lc * cwv[1] * b + self.ls * swv[1] * b
        o2 = b + self.lc * cwv[0] * a + self.ls * swv[0] * a
        return o1, o2


class FrmOutside(nn.Module):
    def __init__(self, lc, ls):
        super().__init__()
        self.lc = lc
        self.ls = ls

    def forward(self, a, b, cw1, sw1):
        o1 = a + self.lc * cw1 * b + self.ls * sw1 * b
        o2 = b + self.lc * cw1 * a + self.ls * sw1 * a
        return o1, o2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tolerance", type=float, default=5e-2)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

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
    print("device: %s  lambda_c=%s lambda_s=%s" % (dev.device_name, frm.lambda_c, frm.lambda_s), flush=True)

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
    print("x1=%s x2=%s cw=%s sw=%s" % (list(x1.shape), list(x2.shape), list(cw.shape), list(sw.shape)), flush=True)

    cases = [
        ("S1_select_only        cw5d->cw5d[1]", Select().eval(), [cw]),
        ("S2_select_both        cw5d->(cw5d[0],cw5d[1])", SelectBoth().eval(), [cw]),
        ("S2b_select_both_sw    sw5d->(sw5d[0],sw5d[1])", SelectBoth().eval(), [sw]),
        ("S3_frm_slice_inside   x1,x2,cw5d,sw5d->(o1,o2)", FrmInside(frm.lambda_c, frm.lambda_s).eval(), [x1, x2, cw, sw]),
        ("S4_frm_slice_outside  x1,x2,cw1,sw1->(o1,o2)", FrmOutside(frm.lambda_c, frm.lambda_s).eval(), [x1, x2, cw[1], sw[1]]),
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
    for label, mod, ins in cases:
        row = {"case": label, "input_shapes": [list(t.shape) for t in ins]}
        try:
            with torch.no_grad():
                ref = mod(*ins)
            ref_list = ref if isinstance(ref, (list, tuple)) else (ref,)
            ref_arr = [t.numpy() for t in ref_list if torch.is_tensor(t)]

            t0 = time.perf_counter()
            tag = label.split()[0]
            onx = os.path.join(args.outdir, "sel-%s.onnx" % tag)
            in_names = ["i%d" % i for i in range(len(ins))]
            torch.onnx.export(mod, tuple(ins), onx, input_names=in_names,
                              output_names=["o%d" % i for i in range(len(ref_arr))],
                              opset_version=18, dynamo=True, verbose=False)
            import onnx
            om = onnx.load(onx, load_external_data=True)
            ir = from_onnx(om, shape_dict={n: list(t.shape) for n, t in zip(in_names, ins)}, dtype_dict="float32")
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
            print("%-46s %-8.2f %-11.3e %-11.3e %-11s %s"
                  % (label, build_s, w.get("max_abs_err", float("nan")), w.get("rel_to_max", float("nan")),
                     ("%.8f" % w["corr"]) if w.get("corr") is not None else "n/a", row["verdict"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["verdict"] = "ERROR"
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
            with open(os.path.join(args.outdir, "sel-%s-error.txt" % label.split()[0]), "w") as f:
                f.write(traceback.format_exc())
            print("%-46s %-8s %-11s %-11s %-11s ERROR %s" % (label, "-", "-", "-", "-", row["error"][:35]), flush=True)
        rows.append(row)

    with open(os.path.join(args.outdir, "select-bisect.json"), "w") as f:
        json.dump({"device": dev.device_name, "tolerance": args.tolerance, "rows": rows}, f, indent=2)
    bad = [r["case"] for r in rows if r.get("verdict") != "OK"]
    print("\n%d/%d cases correct" % (len(rows) - len(bad), len(rows)))
    if bad:
        print("DIVERGING: %s" % "; ".join(b.split("  ")[0] for b in bad))
    print("wrote %s" % os.path.join(args.outdir, "select-bisect.json"))


if __name__ == "__main__":
    sys.exit(main())
