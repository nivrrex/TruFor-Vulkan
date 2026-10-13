#!/usr/bin/env python3
"""Locate which operator TVM's Vulkan backend computes incorrectly.

The full TruFor graph runs to completion on the RX 560 and is bit-stable across
runs (so it is not a race or an unsync), but its output is essentially
uncorrelated with the reference (map corr = -0.29). That points at one or more
wrong kernels rather than at scheduling or memory.

This builds one tiny ONNX model per operator kind that TruFor actually uses and
compares TVM-vulkan against eager PyTorch for each, so the first bad kernel is
isolated. Layers are kept small (32x32) so a wrong result is a wrong result, not
an accumulation artefact.

Run with the source build on PYTHONPATH / TVM_LIBRARY_PATH and the radeon ICD.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

OPS = collections.OrderedDict()


def op(name):
    def deco(fn):
        OPS[name] = fn
        return fn
    return deco


@op("conv2d")
def _conv():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())


@op("conv2d_nopad")
def _conv_np():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())


@op("batchnorm")
def _bn():
    import torch.nn as nn
    m = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
    m.train()  # export a real BatchNormalization node with non-trivial affine
    torch.manual_seed(3)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.uniform_(0.5, 1.5)
            mod.bias.uniform_(-0.5, 0.5)
    return m


@op("gelu")
def _gelu():
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64), nn.GELU())


@op("linear")
def _linear():
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64), nn.ReLU())


@op("softmax")
def _softmax():
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64), nn.Softmax(dim=-1))


@op("log_softmax")
def _logsoftmax():
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64), nn.LogSoftmax(dim=-1))


@op("layer_norm_4d")
def _ln4d():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.LayerNorm([16, 32, 32]))


@op("layer_norm_2d")
def _ln2d():
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.LayerNorm(3 * 32 * 32))


@op("sigmoid")
def _sigmoid():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.Sigmoid())


@op("resize2d_bilinear")
def _resize():
    import torch.nn as nn
    import torch.nn.functional as F

    class M(nn.Module):
        def forward(self, x):
            return F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

    return M()


@op("matmul_4d")
def _matmul():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4, 64, 64))
            self.b = nn.Parameter(torch.randn(4, 64, 64))

        def forward(self, x):
            return torch.matmul(self.a + x.mean(), self.b).sum(dim=(-1, -2), keepdim=True)

    return M()


@op("attention_4d")
def _attn():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Parameter(torch.randn(1, 4, 32, 64))
            self.k = nn.Parameter(torch.randn(1, 4, 32, 64))

        def forward(self, x):
            a = torch.softmax(torch.matmul(self.q, self.k.transpose(-2, -1)) / 8.0, dim=-1)
            return a.mean(dim=(-1, -2))

    return M()


@op("reduce_logsumexp")
def _rlse():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def forward(self, x):
            return torch.logsumexp(x.flatten(1)[:, None, :], dim=-1)

    return M()


@op("erf_gelu_chain")
def _erf():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(3 * 32 * 32, 64)

        def forward(self, x):
            h = self.l(x.flatten(1))
            return torch.nn.functional.gelu(h).mean(dim=1, keepdim=True)

    return M()


@op("mean_global")
def _mean():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

    return M()


@op("amax_global")
def _amax():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def forward(self, x):
            return x.amax(dim=(2, 3), keepdim=True)

    return M()


@op("BIG_conv_512")
def _big_conv():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())


@op("BIG_linear_16384")
def _big_linear():
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.Linear(64 * 16 * 16, 1280), nn.ReLU())


@op("BIG_matmul_2048")
def _big_matmul():
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(1, 5, 2048, 128))
            self.b = nn.Parameter(torch.randn(1, 5, 128, 2048))

        def forward(self, x):
            return torch.matmul(self.a, self.b).mean(dim=(1, 2, 3), keepdim=True)

    return M()


@op("BIG_attn_5d")
def _big_attn5d():
    """The [2, B, heads, N, D] layout CMX's cross-attention actually produces."""
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv = nn.Linear(64 * 16 * 16, 2 * 5 * 64)

        def forward(self, x):
            B = x.shape[0]
            k, v = self.kv(x.flatten(1)).reshape(B, -1, 2, 5, 64).permute(2, 0, 3, 1, 4)
            a = torch.softmax(torch.matmul(k, v.transpose(-2, -1)) / 8.0, dim=-1)
            return a.mean(dim=(-1, -2))

    return M()


@op("BIG_layernorm_1280x4096")
def _big_ln():
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.LayerNorm(64 * 16 * 16))


@op("BIG_resize_128x128")
def _big_resize():
    import torch.nn as nn
    import torch.nn.functional as F

    class M(nn.Module):
        def forward(self, x):
            return F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)

    return M()


def stats(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = np.abs(a - b)
    return {
        "shape_match": True, "shape": list(a.shape),
        "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import onnx
    import torch
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    dev = tvm.vulkan()
    print("device: %s\n" % dev.device_name, flush=True)
    target = tvm.target.Target({"kind": "vulkan", "from_device": 0})

    rows = []
    print("%-22s %-9s %-11s %-11s %s" % ("op", "build", "max_abs", "corr", "verdict"))
    BIG = {"BIG": True}
    for name, factory in OPS.items():
        row = {"op": name}
        big = name.startswith("BIG_")
        hw = 512 if name == "BIG_conv_512" else (128 if name == "BIG_resize_128x128" else 16)
        ch = 64
        try:
            torch.manual_seed(7 if big else 0)
            m = factory()
            if name == "BIG_conv_512":
                x = torch.rand(1, ch, hw, hw)
            elif name == "BIG_resize_128x128":
                x = torch.rand(1, ch, hw, hw)
            elif big:
                x = torch.rand(1, ch, 16, 16)
            else:
                x = torch.rand(1, 3, args.size, args.size)
            m.eval()
            with torch.no_grad():
                ref = m(x)
            onx = os.path.join(args.outdir, "op-%s.onnx" % name)
            torch.onnx.export(m, x, onx, input_names=["x"], output_names=["y"],
                              opset_version=18, dynamo=True, verbose=False)
            om = onnx.load(onx, load_external_data=True)
            ir = from_onnx(om, shape_dict={"x": list(x.shape)}, dtype_dict="float32")

            t0 = time.perf_counter()
            exe = relax.build(ir, target=target)
            build_s = time.perf_counter() - t0
            vm = relax.VirtualMachine(exe.jit(), dev)
            out = vm["main"](tvm.runtime.tensor(x.numpy().astype("float32"), dev))
            got = out[0].numpy() if hasattr(out, "__len__") and not isinstance(out, np.ndarray) else (
                out.numpy() if hasattr(out, "numpy") else np.asarray(out))
            st = stats(ref.numpy(), got)
            row.update({"build_s": build_s, **st})
            ok = st.get("shape_match") and st["max_abs_err"] < args.tol and st["corr"] > 0.99
            row["verdict"] = "OK" if ok else "MISMATCH"
            print("%-22s %-9.2f %-11.3e %-11.8f %s"
                  % (name, build_s, st.get("max_abs_err", float("nan")),
                     st.get("corr", float("nan")), row["verdict"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            row["verdict"] = "ERROR"
            row["input_shape"] = list(x.shape) if "x" in dir() else None
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            with open(os.path.join(args.outdir, "op-%s-error.txt" % name), "w") as f:
                f.write(traceback.format_exc())
            print("%-22s %-9s %-11s %-11s ERROR %s" % (name, "-", "-", "-", row["error"][:60]), flush=True)
        rows.append(row)

    with open(os.path.join(args.outdir, "op-isolation.json"), "w") as f:
        json.dump({"device": dev.device_name, "tol": args.tol, "size": args.size, "ops": rows}, f, indent=2)

    bad = [r["op"] for r in rows if r.get("verdict") != "OK"]
    print("\n%d/%d operators match eager PyTorch on Vulkan" % (len(rows) - len(bad), len(rows)))
    if bad:
        print("MISMATCH/ERROR: %s" % ", ".join(bad))


if __name__ == "__main__":
    sys.exit(main())
