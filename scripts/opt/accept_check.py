#!/usr/bin/env python3
"""Acceptance comparison: delivered substitutions vs the field-built zero graph.

Two processes, because the RX 460 has 2 GB of VRAM and both graphs together do
not fit with headroom:

    python3 scripts/opt/accept_check.py ref    # zero substitutions -> /tmp/accept_ref.npz
    python3 scripts/opt/accept_check.py test   # delivered config    -> /tmp/accept_test.npz
    python3 scripts/opt/accept_check.py compare [ref.npz test.npz]

rel_L2 is per output, on the GPU outputs themselves. The timed-host-copy rule
does not apply here: this is an offline correctness gate, not a benchmark.
"""
import os
import sys

import numpy as np

IMAGES = [
    "truFor-src/test_docker/images/tampered2.png",
    "truFor-src/test_docker/images/tampered1.png",
    "truFor-src/test_docker/images/pristine1.jpg",
]
_OUT_NAMES = ("pred_map", "confidence", "integrity")


def _root():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def run(tag):
    if tag == "ref":
        os.environ["TRUFOR_HANDWRITTEN"] = "0"
        os.environ["TRUFOR_DECODER"] = "0"
    import run_trufor
    import tvm
    from tvm import relax

    exe, dev, _ = run_trufor.build_vm(verbose=(tag != "ref"))
    vm = relax.VirtualMachine(exe.jit(), dev)
    store = {}
    for path in IMAGES:
        x = run_trufor.load(os.path.join(_root(), path))
        tin = tvm.runtime.tensor(x, dev)
        o = vm["main"](tin)
        dev.sync()
        for i in range(len(o)):
            store["%s__%d" % (os.path.basename(path), i)] = o[i].numpy()
    out = "/tmp/accept_%s.npz" % tag
    np.savez_compressed(out, **store)
    print("wrote %s (%d arrays)" % (out, len(store)))


def compare(refp="ref", testp="test"):
    a = np.load(refp if refp.endswith(".npz") else "/tmp/accept_%s.npz" % refp)
    b = np.load(testp if testp.endswith(".npz") else "/tmp/accept_%s.npz" % testp)
    worst = 0.0
    for key in sorted(a.files):
        ra, rb = a[key].astype(np.float64), b[key].astype(np.float64)
        rel = np.linalg.norm((rb - ra).ravel()) / max(np.linalg.norm(ra.ravel()), 1e-12)
        extra = ""
        if key.endswith("__0"):
            ca = ra - ra.mean()
            cb = rb - rb.mean()
            corr = float((ca * cb).sum() / np.sqrt((ca * ca).sum() * (cb * cb).sum()))
            extra = "  map_corr=%.7f" % corr
        worst = max(worst, rel)
        print("%-40s rel_L2=%.2e%s" % (key, rel, extra))
    print("worst rel_L2 = %.2e  (%s)" % (worst, "PASS" if worst < 1e-4 else "FAIL"))


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_root(), "scripts", "opt"))
    if len(sys.argv) < 2 or sys.argv[1] not in ("ref", "test", "compare"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "compare":
        compare(*sys.argv[2:4])
    else:
        run(sys.argv[1])
