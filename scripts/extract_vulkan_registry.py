#!/usr/bin/env python3
"""Extract the operator list from a pytorch/executorch backends/vulkan/op_registry.py
so it can be used as the registry for classify_export.py.

The op_registry.py lists supported kernels as references of the form
    exir_ops.edge.<namespace>.<op>
which is the authoritative source the VulkanPartitioner itself consults.
The published CSV in docs/ is generated from it, but the CSV is only refreshed
periodically and was observed to disagree with the code (main's CSV omits
aten::mul while main's op_registry.py still lists it under a different section).

Usage:
  curl -O https://raw.githubusercontent.com/pytorch/executorch/<tag>/backends/vulkan/op_registry.py
  python3 scripts/extract_vulkan_registry.py --ref v0.6.0 --out reports/operator-inventory/vulkan-registry-v0.6.0.csv
"""

import argparse
import re
import sys
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="git ref / tag, e.g. v0.6.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=None, help="override the raw URL")
    args = ap.parse_args()

    url = args.url or (
        "https://raw.githubusercontent.com/pytorch/executorch/%s/backends/vulkan/op_registry.py" % args.ref
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        src = r.read().decode("utf-8")
    print("fetched %s (%d bytes)" % (url, len(src)))

    ops = sorted(set("%s::%s" % (ns, op) for ns, op in re.findall(r"exir_ops\.edge\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)", src)))

    # blocklist registered in the same file, via BlockList/DenyList style structures
    blocked = sorted(set("%s::%s" % (ns, op) for ns, op in re.findall(r"BlockedOps[^\n]*?([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)", src)))

    with open(args.out, "w") as f:
        f.write("Namespace,Operator,Notes\n")
        for op in ops:
            ns, _, name = op.partition("::")
            f.write("%s,%s,extracted from op_registry.py@%s\n" % (ns, name, args.ref))
    print("wrote %s with %d operators" % (args.out, len(ops)))
    if blocked:
        print("explicitly blocked in registry: %s" % ", ".join(blocked))
    for probe in ("aten::mul", "aten::logsumexp", "aten::log_sigmoid", "aten::sum",
                  "aten::convolution", "aten::_softmax", "aten::native_layer_norm",
                  "aten::_native_batch_norm_legit_no_training", "aten::upsample_bilinear2d",
                  "aten::tile", "aten::expand_copy", "aten::cat", "aten::permute"):
        print("  %-52s %s" % (probe, "YES" if probe in ops else "no"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
