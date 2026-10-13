#!/usr/bin/env python3
"""Phase 6B: ONNX -> TVM Relax import of the verified TruFor model.

Run with the venv interpreter (vendor/tvm-venv/bin/python) so the installed
apache-tvm wheel is used. The wheel has USE_LLVM=ON but USE_VULKAN=OFF, so this
phase answers only "can TVM's Relax ONNX frontend consume this graph at all" --
a question that costs nothing to answer and that decides whether building TVM
with Vulkan is worth doing.

Reports: operator census of the ONNX file, import success/failure, the first
unsupported operator if it fails, and the imported Relax function census.
"""

import argparse
import collections
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import tvm
    from tvm import relax

    print("tvm %s  (libinfo Vulkan=%s LLVM=%s)"
          % (tvm._version.__version__,
             tvm.support.libinfo().get("USE_VULKAN"),
             tvm.support.libinfo().get("USE_LLVM")), flush=True)

    # ---- Phase 6B input census: what the ONNX file actually contains ----
    import onnx

    t0 = time.perf_counter()
    model = onnx.load(args.onnx, load_external_data=False)
    census = collections.Counter(n.op_type for n in model.graph.node)
    graph_inputs = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
    graph_outputs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]
    print("onnx loaded in %.1fs: %d nodes, %d distinct ops, %d initializers"
          % (time.perf_counter() - t0, len(model.graph.node), len(census), len(model.graph.initializer)), flush=True)
    print("inputs : %s" % graph_inputs, flush=True)
    print("outputs: %s" % graph_outputs, flush=True)

    report = {
        "tvm_version": tvm._version.__version__,
        "tvm_use_vulkan": tvm.support.libinfo().get("USE_VULKAN"),
        "tvm_use_llvm": tvm.support.libinfo().get("USE_LLVM"),
        "onnx_path": args.onnx,
        "onnx_nodes": len(model.graph.node),
        "onnx_distinct_ops": len(census),
        "onnx_op_census": dict(census.most_common()),
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
    }

    # ---- the import itself ----
    from tvm.relax.frontend.onnx import from_onnx

    print("calling tvm.relax.frontend.onnx.from_onnx ...", flush=True)
    shape_dict = {name: list(shape) for name, shape in graph_inputs}
    t0 = time.perf_counter()
    try:
        ir_mod = from_onnx(model, shape_dict=shape_dict, dtype_dict="float32")
        ok = True
        report["import_ok"] = True
        report["import_seconds"] = time.perf_counter() - t0
        print("import OK in %.1fs" % report["import_seconds"], flush=True)

        # ir["main"].body is a SeqExpr; blocks[0] is a DataflowBlock whose
        # .bindings each carry a DataflowVar/Var. The operator of a call lives
        # on the var's .op.
        f = ir_mod["main"]
        seq = f.body
        blocks = list(getattr(seq, "blocks", []) or [])
        relax_ops = collections.Counter()
        n_bind = 0
        for blk in blocks:
            for bind in getattr(blk, "bindings", []) or []:
                n_bind += 1
                v = bind.var
                op = getattr(v, "op", None)
                relax_ops[type(op).__name__ if op is not None else type(v).__name__] += 1
        # Relax DataflowVars do not expose .op, so read the operators off the
        # printed script. Cheap and stable across TVM's internal layout.
        text = ir_mod.script()
        with open(os.path.join(args.outdir, "relax-main.txt"), "w") as fh:
            fh.write(text)
        import re
        relax_ops = collections.Counter(re.findall(r"R\.(relax\.[A-Za-z_0-9\.]+)\(", text))
        report["relax_block_count"] = len(blocks)
        report["relax_binding_count"] = n_bind
        report["relax_op_census"] = dict(relax_ops.most_common())
        report["relax_func_names"] = [g.name_hint for g in ir_mod.functions]
        report["relax_script_bytes"] = len(text)
        print("relax funcs=%s blocks=%d bindings=%d script=%d bytes"
              % (report["relax_func_names"], report["relax_block_count"], n_bind, len(text)), flush=True)
        for k, v in list(report["relax_op_census"].items())[:30]:
            print("   %-40s %d" % (k, v))
    except Exception as e:  # noqa: BLE001
        import traceback

        ok = False
        report["import_ok"] = False
        report["import_seconds"] = time.perf_counter() - t0
        report["import_error_type"] = type(e).__name__
        report["import_error"] = str(e)[:3000]
        report["import_traceback"] = traceback.format_exc()[-3000:]
        print("import FAILED after %.1fs: %s: %s" % (report["import_seconds"], type(e).__name__, str(e)[:400]), flush=True)

    with open(os.path.join(args.outdir, "tvm-import.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "tvm-import.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
