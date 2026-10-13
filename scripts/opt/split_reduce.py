"""Stage A3: split single-output reductions so the GPU schedule can parallelise them.

Rewrites ReduceMax / ReduceMin whose output has very few elements but whose input
is large:

    ReduceMax(X:[1,1,262144], axes=[2])  ->  (1,1,1)
becomes
    Reshape(X, [1,1,512,512])
    ReduceMax(., axes=[2], keepdims=1)   ->  (1,1,1,512)
    ReduceMax(., axes=[3], keepdims=1)   ->  (1,1,1,1)

The rewrite is exact for max/min: both are idempotent and order-independent over a
finite, NaN-free input, so the result is bit-identical. (Sum/Mean are not order
independent and are deliberately NOT rewritten here.)

The model is modified **in memory only**; the stored .onnx file is never touched, and
the whole rewrite is behind a switch so the default production path is unchanged.
"""

import math
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_INPUT_ELEMS = int(os.environ.get("TRUFOR_SPLIT_MIN_ELEMS", 65536))
MAX_OUTPUT_ELEMS = int(os.environ.get("TRUFOR_SPLIT_MAX_OUT", 64))


def enabled():
    return os.environ.get("TRUFOR_SPLIT_REDUCE", "1") not in ("0", "off", "false")


def _elems(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _factor_pair(n):
    a = int(math.isqrt(n))
    while a > 1 and n % a:
        a -= 1
    return a, n // a


def _shapes(model):
    out = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            out[vi.name] = dims
    return out


def split(model, min_in=MIN_INPUT_ELEMS, max_out=MAX_OUTPUT_ELEMS):
    """Return (new_model, records). Operates on the model object in memory."""
    if not enabled():
        return model, []

    g = model.graph
    sh = _shapes(model)
    inits = {t.name: t for t in g.initializer}
    new_inits, records = [], []
    keep, i = [], 0
    for n in g.node:
        hit = None
        if n.op_type in ("ReduceMax", "ReduceMin") and len(n.input) == 1:
            src = sh.get(n.input[0])
            dst = sh.get(n.output[0])
            if src and dst and len(src) == len(dst):
                axes = None
                for a in n.attribute:
                    if a.name == "axes":
                        axes = list(a.ints)
                    if a.name == "keepdims":
                        keepdims = a.i
                if axes and len(axes) == 1:
                    r = axes[0] % len(src)
                    red = src[r]
                    if _elems(src) >= min_in and _elems(dst) <= max_out and red > 1:
                        a_, b_ = _factor_pair(red)
                        if a_ > 1 and b_ > 1:
                            hit = (r, a_, b_, keepdims if "keepdims" in
                                   {at.name: at.i for at in n.attribute} else 1)
        if hit is None:
            keep.append(n)
            continue
        r, a_, b_, keepdims = hit
        new_shape = list(src)
        new_shape[r] = a_
        sname = n.name + "_split4d" if n.name else "split4d_%d" % i
        p1 = n.name + "_part1" if n.name else "part1_%d" % i
        p2 = n.name + "_part2" if n.name else "part2_%d" % i
        new_inits.append(numpy_helper.from_array(np.array(new_shape, np.int64), sname))
        keep.append(helper.make_node("Reshape", [n.input[0], sname], [p1], name=(n.name or "n%d" % i) + "_rs"))
        keep.append(helper.make_node(n.op_type, [p1], [p2], name=(n.name or "n%d" % i) + "_r1",
                                     axes=[r], keepdims=1))
        keep.append(helper.make_node(n.op_type, [p2], list(n.output), name=(n.name or "n%d" % i) + "_r2",
                                     axes=[r], keepdims=1))
        records.append({"node": n.name, "op": n.op_type, "input_shape": src, "output_shape": dst,
                        "split_into": [a_, b_], "reduced_axis": r})
        i += 1

    if not records:
        return model, []
    del g.node[:]
    g.node.extend(keep)
    g.initializer.extend(new_inits)
    return model, records


def load_and_split(path):
    m = onnx.load(path)
    m, recs = split(m)
    return m, recs
