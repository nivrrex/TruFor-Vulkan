"""B3: measure what each class of PrimFunc actually costs, by ablation.

The per-instruction profile is not usable for this: it syncs on both sides of
every instruction, so the time attributed to a kernel includes waiting for its
neighbours, and two attempts to use it gave numbers that turned out to be
artefacts.

Ablation avoids instrumentation entirely. Replace the body of every PrimFunc in
a class with a no-op that still writes something to every output buffer, so the
signature and the memory traffic stay valid and the graph still runs to
completion. The drop in whole-graph median is that class's upper bound on what
optimising it could ever save. Outputs become meaningless, so this measures time
only and the run is never checked for correctness.
"""

import re

import tvm


def _out_buffers(script):
    """Buffer names in the parameter list, in order."""
    m = re.search(r"^def\s+\w+\((.*)$", script, re.M)
    if not m:
        return [], []
    head = script[m.start():].split("\n", 1)[0]
    params = re.findall(r"(\w+):\s*T\.Buffer\((.*?),\s*\"(\w+)\"\)", head)
    return [(n, d, t) for n, d, t in params], params


def make_noop(name, params):
    """A PrimFunc with the same signature that writes a constant to each output."""
    from tvm.script import tirx as T

    # Determine output buffers by the dim declaration, e.g. "T_add" or
    # "compute_intermediate"; the pipeline names them predictably, so fall back
    # to "everything after the first two parameters" which held for every
    # signature inspected.
    body_args = ", ".join("%s: T.Buffer((%s), %s)" % (n, d, '"%s"' % t) for n, d, t in params)
    lines = ["        T.func_attr({\"tirx.is_scheduled\": True, \"tirx.noalias\": True})"]
    for n, d, t in params[2:]:
        lines.append("        %s[...] = T.float32(0)" % n)
    lines.insert(0, "    def %s(%s):" % (name, body_args))
    src = "from tvm.script import tirx as T\n\n@T.prim_func\n" + "\n".join(lines) + "\n"
    return src


def ablate(mod, predicate, label, make_src, compile_fn):
    """Replace every PrimFunc matching `predicate` with a no-op. Returns ms delta."""
    import tvm.ir.transform as T

    with T.Sequential([]):
        pass
    saved = {}
    n = 0
    for gv in list(mod.functions.keys()):
        fn = mod.functions[gv]
        if not hasattr(fn, "attrs") or "global_symbol" not in fn.attrs:
            continue
        script = fn.script()
        if "T.prim_func" not in script:
            continue
        params, _ = _out_buffers(script)
        if not predicate(script, str(gv), params):
            continue
        try:
            src = make_src(fn.attrs["global_symbol"][0], params)
            newfn = compile_fn(src)
        except Exception as e:
            print("  %s: 生成 noop 失败 %s" % (label, str(e)[:70]))
            continue
        saved[gv] = fn
        mod[gv] = newfn.with_attr("global_symbol", fn.attrs["global_symbol"][0])
        n += 1
    return n, saved
