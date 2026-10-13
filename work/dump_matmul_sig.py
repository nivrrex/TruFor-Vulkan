"""Dump signatures of all matmul-like PrimFuncs in the field-built graph."""
import os, sys, re
sys.path.insert(0, os.path.join(os.environ["ROOT"], "vendor", "tvm-src", "python"))
sys.path.insert(0, os.path.join(os.environ["ROOT"], "scripts", "opt"))
import tvm, onnx
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir import dlight as dl
from tvm.relax.backend.gpu_generic import pipeline as GP
import tvm.ir.transform as T
from tvm.relax.transform import (LegalizeOps, AnnotateTIROpPattern,
                                 FoldConstant, FuseOps, FuseTIR)
import swap_conv

MODEL = os.path.join(os.environ["ROOT"], "reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx")
target = tvm.target.Target({"kind": "vulkan", "from_device": 0})
ir = from_onnx(onnx.load(MODEL), shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
seq = T.Sequential(
    GP.library_dispatch_passes(target) +
    [LegalizeOps(), AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR(),
     dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                             dl.gpu.GeneralReduction(), dl.gpu.Fallback())] +
    GP.dataflow_lower_passes(target) + GP.finalize_passes(target))
with target:
    mod = seq(ir)

for gv in list(mod.functions.keys()):
    fn = mod.functions[gv]
    if not hasattr(fn, "attrs"): continue
    s = fn.script()
    if "T.prim_func" not in s: continue
    name = str(gv).split('"')[1]
    if "matmul" not in name and "dense" not in name and "nn.matmul" not in name: continue
    sig = swap_conv.parse_sig(s)
    print(name, "sig=", sig)
