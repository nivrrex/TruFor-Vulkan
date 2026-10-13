"""Stage A1: rewrite ONNX LogSoftmax into a two-stage reduction (ONNX level).

max4 is the numerically-stabilising max that legalisation of
`R.nn.log_softmax` emits, and it degenerates because the Vulkan reduction schedule
derives its grid from the OUTPUT extents: a scalar output means grid = 1, 255 of
256 threads masked off, one thread walking 262144 elements (1.0 MB in 84 ms =
0.012 GB/s against 112 GB/s peak). TVM 0.27 has no switch to disable the
decomposition, and the operator is absent from both the ONNX graph and the Relax
IR, so this is the only level where it can be replaced.

    LogSoftmax(x, axis=-1), x:[1,1,262144]
      -> Reshape to (1, R, C) with R*C = 262144, R ~= C
         ReduceMax axis=-1 -> (R,1)      # R outputs: the first stage has a grid
         ReduceMax axis=-2 -> (1,1)
         Sub, Exp
         ReduceSum axis=-1 -> (R,1)
         ReduceSum axis=-2 -> (1,1)
         Log, Sub
      -> Reshape back to the input shape

The reduced element set is unchanged, so max is exact; only the summation order
changes, hence a relative tolerance on the output. In-memory only, behind
TRUFOR_SPLIT_LOGSOFTMAX.
"""

import math
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMS = int(os.environ.get("TRUFOR_LS_MIN_ELEMS", 65536))


def enabled():
    return os.environ.get("TRUFOR_SPLIT_LOGSOFTMAX", "1") not in ("0", "off", "false")


def _pair(n):
    a = int(math.isqrt(n))
    while a > 1 and n % a:
        a -= 1
    return a, (n // a if a > 1 else n)


def _shapes(model):
    out = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        d = [x.dim_value for x in vi.type.tensor_type.shape.dim]
        if d and all(v > 0 for v in d):
            out[vi.name] = d
    return out


def _reduce(op_type, inputs, out, name, axes):
    """Reduce node. ONNX wants `axes` as a typed int64 attribute; make_node cannot
    infer the element type from a plain Python list."""
    node = helper.make_node(op_type, list(inputs), [out], keepdims=1, name=name)
    if axes is not None:
        node.attribute.append(helper.make_attribute("axes", np.array(axes, dtype=np.int64)))
    return node


def split_log_softmax(model, min_elems=MIN_ELEMS):
    """Edit `model` in place. Returns the list of rewrites performed."""
    if not enabled():
        return []
    g = model.graph
    opset = max([o.version for o in model.opset_import if o.domain in ("", "ai.onnx")] or [13])
    # ReduceMax takes axes as an attribute up to opset 17 and as an input from 18;
    # ReduceSum takes it as an input from 13 on. (Verified with ORT: at opset 17
    # ReduceSum rejects an `axes` attribute and ReduceMax rejects an axes input.)
    max_axes_input = opset >= 18
    sum_axes_input = opset >= 13
    sh = _shapes(model)

    keep, new_inits, records = [], [], []
    for n in g.node:
        if n.op_type != "LogSoftmax" or len(n.input) != 1:
            keep.append(n)
            continue
        src = sh.get(n.input[0])
        axis = next((a.i for a in n.attribute if a.name == "axis"), None)
        if src is None or axis is None:
            keep.append(n)
            continue
        r = axis % len(src)
        red = src[r]
        if red < min_elems or r != len(src) - 1:
            keep.append(n)
            continue
        R, C = _pair(red)
        if R < 2 or C < 2:
            keep.append(n)
            continue

        nm = n.name or ("ls%d" % len(records))
        # keep only the batch extent and drop unit non-reduced extents, so the two
        # reduce stages land on the last two axes. Keeping the unit extents would
        # put them on dimensions of extent 1 and reproduce the original defect.
        keep_dim = [v for k, v in enumerate(src) if k != r]
        batch = keep_dim[0] if keep_dim else 1
        rest = [v for v in keep_dim[1:] if v != 1]
        flat = [batch] + rest + [R, C]
        nd = len(flat)
        A1, A0 = [nd - 1], [nd - 2]

        sname, oname = nm + "_2d", nm + "_2dout"
        t, m1, mm, t1 = [nm + "_" + k for k in ("t", "m1", "m", "t1")]
        e, s1, ss, ls, y2 = [nm + "_" + k for k in ("e", "s1", "s", "ls", "y2")]
        new_inits.append(numpy_helper.from_array(np.array(flat, np.int64), sname))
        new_inits.append(numpy_helper.from_array(np.array(list(src), np.int64), oname))
        if max_axes_input:
            new_inits.append(numpy_helper.from_array(np.array(A1, np.int64), nm + "_ax1"))
            new_inits.append(numpy_helper.from_array(np.array(A0, np.int64), nm + "_ax0"))
        if sum_axes_input:
            new_inits.append(numpy_helper.from_array(np.array(A1, np.int64), nm + "_sax1"))
            new_inits.append(numpy_helper.from_array(np.array(A0, np.int64), nm + "_sax0"))

        max1 = [t, nm + "_ax1"] if max_axes_input else [t]
        max0 = [m1, nm + "_ax0"] if max_axes_input else [m1]
        sum1 = [e, nm + "_sax1"] if sum_axes_input else [e]
        sum0 = [s1, nm + "_sax0"] if sum_axes_input else [s1]

        keep += [
            helper.make_node("Reshape", [n.input[0], sname], [t], name=nm + "_rs"),
            _reduce("ReduceMax", max1, m1, nm + "_rmax1", None if max_axes_input else A1),
            _reduce("ReduceMax", max0, mm, nm + "_rmax2", None if max_axes_input else A0),
            helper.make_node("Sub", [t, mm], [t1], name=nm + "_sub_m"),
            helper.make_node("Exp", [t1], [e], name=nm + "_exp"),
            _reduce("ReduceSum", sum1, s1, nm + "_rsum1", None if sum_axes_input else A1),
            _reduce("ReduceSum", sum0, ss, nm + "_rsum2", None if sum_axes_input else A0),
            helper.make_node("Log", [ss], [ls], name=nm + "_log"),
            helper.make_node("Sub", [t, ls], [y2], name=nm + "_sub_ls"),
            helper.make_node("Reshape", [y2, oname], list(n.output), name=nm + "_rs_out"),
        ]
        records.append({"node": nm, "input_shape": list(src), "reshape": flat, "split": [R, C]})

    if not records:
        return []
    del g.node[:]
    g.node.extend(keep)
    g.initializer.extend(new_inits)
    return records
