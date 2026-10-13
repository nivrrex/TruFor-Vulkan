"""Enumerate every PrimFunc the current graph calls, with call count and traffic.

Cheap: no device work, just main's binding list, so it can rank candidates for
the SKIP runs before spending 20 s per measurement on each.
"""
import os, sys, re
import numpy as np
from collections import Counter
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vendor", "tvm-src", "python"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "opt"))
import tvm, onnx
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir import dlight as dl
from tvm.relax.backend.gpu_generic import pipeline as GP
import tvm.ir.transform as T
from tvm.relax.transform import LegalizeOps, AnnotateTIROpPattern, FoldConstant, FuseOps, FuseTIR
import swap_conv

def main():
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    M = os.path.join(ROOT, "reports", "onnx-dynamo", "trufor_512x512_dynamo_inline.onnx")
    ir = from_onnx(onnx.load(M), shape_dict={"rgb": [1, 3, 512, 512]}, dtype_dict="float32")
    seq = T.Sequential(
        GP.library_dispatch_passes(tgt) +
        [LegalizeOps(), AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR(),
         dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                                 dl.gpu.GeneralReduction(), dl.gpu.Fallback())] +
        GP.dataflow_lower_passes(tgt) + GP.finalize_passes(target := tgt))
    with tgt:
        mod = seq(ir)
    dev = tvm.vulkan()
    n, names, log = swap_conv.swap(mod, use_depthwise=True, layout=True, verify=(dev, tgt), verbose=False)
    main = mod["main"]
    blk = main.body.blocks[0] if hasattr(main.body, "blocks") else None
    # after lowering main is a lowered lambda, so read the call names from the
    # PrimFunc signatures in the module plus the pre-lowering dump instead
    pre = mod.script()
    # count call_tir occurrences per function name in main's pre-lowered form
    info = {}
    # The attribute list varies: "@T.prim_func", "@T.prim_func(s_tir=True)",
    # "@T.prim_func(private=True, s_tir=True)". Match the whole line.
    for m in re.finditer(r"@T\.prim_func[^\n]*\n\s*def (\w+)\((.*?)\):", pre, re.S):
        nm, params = m.group(1), m.group(2)
        # Non-greedy up to the dtype: the dims text itself contains parens, so a
        # paren-counting regex on the shape fails.
        shapes = []
        for dims, dt in re.findall(r'T\.Buffer\(\((.*?)\), "(\w+)"\)', params, re.S):
            shapes.append([int(x) for x in re.findall(r"T\.int64\((\d+)\)", dims)])
        info[nm] = shapes
    # call counts: rebuild them from a pre-lowering pass so counts are exact
    seq_pre = T.Sequential(
        GP.library_dispatch_passes(tgt) +
        [LegalizeOps(), AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR(),
         dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                                 dl.gpu.GeneralReduction(), dl.gpu.Fallback())])
    with tgt:
        mod_pre = seq_pre(ir)
    swap_conv.swap(mod_pre, use_depthwise=True, layout=True, verify=(dev, tgt), verbose=False)
    sc_pre = mod_pre.script()
    # Count call sites by name, which is what matters for SKIP: each occurrence
    # of "name(" in the pre-lowered graph is one call.
    allnames = set(info)
    calls = Counter()
    for nm in allnames:
        # exclude the definition itself: "def name(" also matches "name("
        calls[nm] = len(re.findall(r"(?<!def )\b" + re.escape(nm) + r"\(", sc_pre))
    rows = []
    for nm, shapes in info.items():
        c = calls.get(nm, 0)
        if c == 0 or not shapes:
            continue
        els = [int(np.prod(sh)) for sh in shapes if sh]
        byts = (sum(sorted(els)[-2:]) if len(els) >= 2 else (els[0] if els else 0)) * 4
        rows.append((c * byts, c, byts, nm, shapes))
    rows.sort(reverse=True)
    print("substitutions applied: %d ; distinct called PrimFuncs: %d" % (n, len(rows)))
    print("\n%-46s %5s %10s %8s" % ("name", "calls", "MB x calls", "args"))
    for tot, c, byts, nm, shapes in rows[:30]:
        argstr = " ".join("x".join(map(str, s)) for s in shapes[:3])
        print("%-46s %5d %10.1f %8s" % (nm[:46], c, tot / 1e6, argstr[:60]))
    # dump machine-readable
    with open("/tmp/cands.txt", "w") as f:
        for tot, c, byts, nm, shapes in rows:
            f.write("%s\t%d\t%.1f\n" % (nm, c, tot / 1e6))

main()
