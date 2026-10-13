#!/usr/bin/env python3
"""Phase 3c: THE decisive experiment.

Runs the official ExecuTorch VulkanPartitioner over the real TruFor graph and
reports how many nodes actually get delegated to Vulkan vs left on the default
(CPU) backend. This is the number that decides whether the project is worth
continuing, and it can only be obtained by running the partitioner -- the
operator registry alone is not authoritative because the partitioner also
applies dim-order passes and operator-support constraints (texture limits,
buffer limits, dynamic-shape rules).

It does NOT execute the resulting program. That is Phase 5.
"""

import argparse
import collections
import json
import os
import sys
import time

import torch

_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRUFOR_ROOT = os.environ.get("TRUFOR_ROOT", os.path.join(ROOT, "truFor-src"))
TRAIN_DIR = os.path.join(TRUFOR_ROOT, "TruFor_train_test")
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from lib.config import config, update_config  # noqa: E402
from lib.utils import get_model  # noqa: E402

from executorch.exir.pass_base import ExportPass  # noqa: E402
from torch.fx.passes.infra.pass_base import PassResult  # noqa: E402


class FillOmittedKeepdimPass(ExportPass):
    """Make keepdim explicit on aten.sum / aten.mean nodes.

    ExecuTorch 0.6.0's Vulkan reduce support check (op_registry.py,
    check_reduce_node) reads node.args[2] unconditionally:

        keepdim = node.args[2]

    TruFor's weighted_statistics_pooling calls torch.sum(x, dim=-1), so
    keepdim takes its default and the exported node carries only two args.
    The partitioner then raises IndexError instead of falling back to CPU.

    This pass appends the omitted default. keepdim=False is exactly what the
    op would have used, so numerics are unchanged; the node is still rejected
    for Vulkan (the check requires keepdim=True) and lands on the CPU
    partition, which is the intended outcome.

    partitioner itself stays untouched.
    """

    def call(self, graph_module):
        patched = 0
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            t = node.target
            name = getattr(t, "_opname", None) or (t if isinstance(t, str) else "")
            if name not in ("sum", "mean"):
                continue
            if len(node.args) < 3:
                node.args = tuple(node.args) + (False,)
                patched += 1
        self.patched_nodes = patched
        return PassResult(graph_module, bool(patched))


def build_model(model_file, exp="trufor_ph3"):
    class _Args:
        pass

    prev = os.getcwd()
    os.chdir(TRAIN_DIR)
    try:
        a = _Args()
        a.experiment = exp
        a.opts = []
        update_config(config, a)
        model = get_model(config)
    finally:
        os.chdir(prev)
    ckpt = torch.load(model_file, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def node_key(n):
    t = n.target
    if isinstance(t, str):
        return t
    try:
        return "%s::%s" % (t._namespace, t._opname)
    except Exception:
        return str(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--exp", default="trufor_ph3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    from executorch.backends.transforms.convert_dtype_pass import I64toI32
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    globals()["ExportPass"] = ExportPass

    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    print("executorch: %s" % __import__("executorch").__path__[0], flush=True)

    model = build_model(args.model, args.exp)
    sample = (torch.rand(1, 3, args.size, args.size),)

    from torch.export import export

    # strict=True fails: EncoderDecoder.forward() calls self.dncnn.eval() inline,
    # which dynamo cannot trace. The call is redundant (the module is already in
    # eval mode) and model.eval() has been called, so strict=False is exact here
    # and leaves the upstream source untouched.
    t0 = time.perf_counter()
    program = export(model, sample, dynamic_shapes=None, strict=False)
    t_export = time.perf_counter() - t0
    print("torch.export ok (%.1fs)" % t_export, flush=True)

    # ExecuTorch's EXIR verifier rejects operators outside the core ATen opset.
    # TruFor's weighted_statistics_pooling emits adaptive_max_pool2d, which the
    # Vulkan backend handles via the max_pool2d_with_indices kernel, so it goes
    # into the documented exception list rather than being rewritten.
    _exceptions = [torch.ops.aten.adaptive_max_pool2d.default]
    edge_config = EdgeCompileConfig(_skip_dim_order=False, _core_aten_ops_exception_list=_exceptions)
    keepdim_pass = FillOmittedKeepdimPass()
    t0 = time.perf_counter()
    edge_program = to_edge_transform_and_lower(
        program,
        compile_config=edge_config,
        transform_passes=[I64toI32(edge_config._skip_dim_order), keepdim_pass],
        partitioner=[VulkanPartitioner({})],
    )
    t_partition = time.perf_counter() - t0
    print("partition ok (%.1fs); keepdim args filled in on %d node(s)" % (t_partition, keepdim_pass.patched_nodes), flush=True)

    et_program = edge_program.to_executorch()
    ep = et_program.executorch_program
    plan = ep.execution_plan[0]
    pte = os.path.join(args.outdir, "partitioned.pte")
    with open(pte, "wb") as f:
        f.write(et_program.buffer)

    # ---- delegation census over the partitioned edge graph ----
    # After to_backend(), delegated regions are collapsed into
    # executorch_call_delegate nodes whose args[1] is a Subgraph node; every
    # other call_function is still destined for the default (CPU) backend.
    ep_fwd = edge_program._edge_programs["forward"]
    gm = ep_fwd.graph_module

    from executorch.exir.delegate import executorch_call_delegate

    gpu_nodes, cpu_nodes = [], []
    delegate_subgraphs = 0
    delegate_arg_shapes = {}
    node_kinds = collections.Counter()
    for n in gm.graph.nodes:
        tgt = n.target
        node_kinds["%s:%s" % (n.op, type(tgt).__name__)] += 1
        if n.op == "call_function" and tgt is executorch_call_delegate:
            if not node_kinds:
                pass
            found = False
            for a in n.args:
                # NOTE: torch.fx.Node also exposes a .graph property, so an
                # isinstance/hasattr test on "graph" would classify every node
                # argument as a subgraph. Match the Subgraph type explicitly.
                if type(a).__name__ == "Subgraph":
                    delegate_subgraphs += 1
                    gpu_nodes.extend([x for x in a.graph.nodes])
                    found = True
            if not found:
                node_kinds["delegate_without_subgraph"] += 1
                if not delegate_arg_shapes:
                    delegate_arg_shapes["args"] = [type(a).__name__ for a in n.args]
                    delegate_arg_shapes["kwargs"] = sorted(n.kwargs.keys())
                    delegate_arg_shapes["delegate_obj_type"] = type(plan.delegates[0]).__name__ if plan.delegates else None
                    try:
                        d0 = plan.delegates[0]
                        delegate_arg_shapes["delegate_obj_attrs"] = [x for x in dir(d0) if not x.startswith("_")]
                    except Exception as e:  # noqa: BLE001
                        delegate_arg_shapes["delegate_obj_attrs"] = "err: %s" % e
        elif n.op in ("call_function", "call_method", "get_attr", "call_module"):
            cpu_nodes.append(n)

    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    total = len(gpu_nodes) + len(cpu_nodes)

    def census(nodes):
        c = collections.Counter()
        for n in nodes:
            c[node_key(n)] += 1
        return c

    gpu_census = census(gpu_nodes)
    cpu_census = census(cpu_nodes)
    all_census = census(gpu_nodes + cpu_nodes)

    total = len(gpu_nodes) + len(cpu_nodes)
    delegated = len(gpu_nodes)

    # Which operators ended up on the CPU, and how many of them.
    cpu_only = collections.Counter()
    for op, n in cpu_census.items():
        if all_census.get(op, 0):
            cpu_only[op] = n
    # For each CPU operator, how many of its total occurrences fell back.
    fallback_rows = []
    for op, n in sorted(cpu_only.items(), key=lambda kv: -kv[1]):
        fallback_rows.append(
            {"op": op, "on_cpu": n, "in_graph_total": all_census.get(op, 0),
             "on_gpu": all_census.get(op, 0) - n}
        )

    report = {
        "size": args.size,
        "torch_version": torch.__version__,
        "executorch_version": getattr(__import__("executorch"), "__version__", "0.6.0a0+e67ef3b"),
        "export_seconds": t_export,
        "partition_seconds": t_partition,
        "delegates": [d.id for d in plan.delegates],
        "delegate_subgraphs": delegate_subgraphs,
        "nodes_total_excl_placeholder": total,
        "nodes_placeholder": len(placeholder_nodes),
        "nodes_delegated_to_vulkan": delegated,
        "nodes_on_default_backend": len(cpu_nodes),
        "delegation_pct": round(100.0 * delegated / total, 2) if total else 0.0,
        "fallback": fallback_rows,
        "gpu_op_census": dict(gpu_census.most_common()),
        "cpu_op_census": dict(cpu_census.most_common()),
        "node_kinds": dict(node_kinds),
        "delegate_debug": delegate_arg_shapes,
    }

    with open(os.path.join(args.outdir, "partition-report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Vulkan partition result (TruFor @%dx%d) ===" % (args.size, args.size))
    print("delegates                       : %d x %s" % (len(report["delegates"]), set(report["delegates"])))
    print("delegate subgraphs found        : %d" % delegate_subgraphs)
    print("graph nodes (excl. placeholder) : %d" % total)
    print("delegated to Vulkan             : %d  (%.2f%%)" % (delegated, report["delegation_pct"]))
    print("left on default backend         : %d" % len(cpu_nodes))
    print("\n--- operators that fell back to the default backend ---")
    if fallback_rows:
        print("%-42s %8s %10s %8s" % ("operator", "on_cpu", "in_graph", "on_gpu"))
        for r in fallback_rows:
            print("%-42s %8d %10d %8d" % (r["op"], r["on_cpu"], r["in_graph_total"], r["on_gpu"]))
    else:
        print("(none -- the entire graph was delegated to Vulkan)")
    print("\nwrote %s" % pte)
    return 0 if delegated else 1


if __name__ == "__main__":
    sys.exit(main())
