#!/usr/bin/env python3
"""Minimal reproducer: TVM Vulkan backend miscomputes dim-0 integer indexing
(take/select) when the source tensor's second-to-last dimension is 1.

No model, no weights, no framework dependency -- pure torch.randn input.

Expected: every case is bit-exact on the LLVM backend, and exact on Vulkan too.
Actual (TVM 0.27.0, Vulkan backend, AMD RADV): the t[k] cases below are wrong.

Run with a TVM that has the Vulkan backend enabled:
    python tvm-vulkan-take-select-repro.py
"""

import numpy as np
import torch
import torch.nn as nn
import tvm
from tvm import relax


class Sel0(nn.Module):
    def forward(self, c):
        return c[0]


class Sel1(nn.Module):
    def forward(self, c):
        return c[1]


class SelBoth(nn.Module):
    def forward(self, c):
        return c[0], c[1]


def run_vulkan(mod, inp, target, dev):
    import onnx
    path = "/tmp/_take_repro.onnx"
    torch.onnx.export(mod, (inp,), path, input_names=["i0"], output_names=["o0"],
                      opset_version=18, dynamo=True, verbose=False)
    from tvm.relax.frontend.onnx import from_onnx
    ir = from_onnx(onnx.load(path, load_external_data=True),
                   shape_dict={"i0": list(inp.shape)}, dtype_dict="float32")
    exe = relax.build(ir, target=target)
    out = relax.VirtualMachine(exe.jit(), dev)["main"](
        tvm.runtime.tensor(inp.numpy().astype("float32"), dev))
    import tvm_ffi
    if isinstance(out, tvm_ffi.Array):
        return [out[i].numpy() for i in range(len(out))]
    return [out.numpy()]


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    dev = tvm.vulkan()
    print("device: %s (exist=%s)" % (dev.device_name, dev.exist))
    vk = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    cpu = tvm.target.Target({"kind": "llvm"})

    shapes = [
        [2, 1, 64, 1, 1],      # WRONG on Vulkan
        [2, 1, 1, 128, 128],  # correct on Vulkan
        [2, 1, 64, 128, 1],   # correct on Vulkan  <- differs only in dim3
        [2, 1, 64, 1, 2],     # WRONG
        [2, 1, 64, 1],        # WRONG (4-D)
    ]
    ops = [("t[0]", Sel0()), ("t[1]", Sel1()), ("t[0],t[1]", SelBoth())]

    bad = 0
    print("\n%-18s %-10s %-13s %-13s %s" % ("shape", "op", "vulkan_rel", "llvm_rel", "status"))
    for shape in shapes:
        inp = torch.randn(*shape)
        for oname, mod in ops:
            mod = mod.eval()
            ref = mod(inp)
            ref = ref if isinstance(ref, (list, tuple)) else (ref,)
            ref = [t.numpy() for t in ref]
            gv = run_vulkan(mod, inp, vk, dev)
            gc = run_vulkan(mod, inp, cpu, tvm.cpu())
            vr = max(float(np.abs(r - g).max()) / max(float(np.abs(r).max()), 1e-9)
                     for r, g in zip(ref, gv))
            cr = max(float(np.abs(r - g).max()) / max(float(np.abs(r).max()), 1e-9)
                     for r, g in zip(ref, gc))
            status = "OK" if vr < 1e-5 else "WRONG"
            if vr >= 1e-5:
                bad += 1
            print("%-18s %-10s %-13.3e %-13.3e %s" % (str(shape), oname, vr, cr, status))
    print("\n%d wrong on Vulkan, 0 expected" % bad)


if __name__ == "__main__":
    main()
