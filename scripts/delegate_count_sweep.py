#!/usr/bin/env python3
"""Isolate what makes the TruFor .pte crash the Vulkan runner.

Small models with 1 delegate run fine (verified). TruFor's graph is split into
88 delegates because aten.log has no Vulkan implementation and therefore cuts
the partition. This builds models with a controllable number of delegates by
placing unsupported ops between supported blocks, then reports the delegate
count and the resulting .pte size.

Run the produced models through vulkan_executor_runner to find where it breaks.
"""

import os
import subprocess
import sys

import torch
import torch.nn as nn


class Segmented(nn.Module):
    """n_blocks conv blocks separated by aten.log, which Vulkan cannot execute.

    Each separator forces the partitioner to cut, so delegates == n_blocks.
    """

    def __init__(self, n_blocks):
        super().__init__()
        self.n = n_blocks
        self.convs = nn.ModuleList([nn.Conv2d(3 if i == 0 else 8, 8, 3, padding=1) for i in range(n_blocks)])

    def forward(self, x):
        for i, c in enumerate(self.convs):
            x = torch.relu(c(x))
            if i < self.n - 1:
                x = torch.log(torch.abs(x) + 1.0)
        return x


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "reports/runner-control"
    runner = sys.argv[2] if len(sys.argv) > 2 else "vendor/executorch/build-vk/backends/vulkan/vulkan_executor_runner"
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    from executorch.backends.transforms.convert_dtype_pass import I64toI32
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export

    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"

    print("%-6s %-10s %-9s %-9s %s" % ("blocks", "delegates", "pte_MB", "runner", "note"))
    for n in (1, 2, 4, 8, 16, 32, 64, 88):
        mod = Segmented(n)
        sample = (torch.rand(1, 3, 128, 128),)
        try:
            prog = export(mod, sample, dynamic_shapes=None, strict=True)
            cfg = EdgeCompileConfig(_skip_dim_order=False)
            ep = to_edge_transform_and_lower(
                prog, compile_config=cfg,
                transform_passes=[I64toI32(cfg._skip_dim_order)],
                partitioner=[VulkanPartitioner({})],
            )
            et = ep.to_executorch()
            path = os.path.join(outdir, "seg-%03d.pte" % n)
            with open(path, "wb") as f:
                f.write(et.buffer)
            nd = len(et.executorch_program.execution_plan[0].delegates)
        except Exception as e:  # noqa: BLE001
            print("%-6d export/partition FAILED: %s" % (n, str(e)[:120]))
            continue

        r = subprocess.run([runner, "-model_path=%s" % os.path.abspath(path), "-num_executions=1"],
                           env=env, capture_output=True, text=True, timeout=600)
        note = ""
        if r.returncode != 0:
            note = (r.stderr.strip().splitlines() or [""])[-1][:80]
        print("%-6d %-10d %-9.2f %-9s %s" % (n, nd, os.path.getsize(path) / 1e6,
                                               "OK" if r.returncode == 0 else "rc=%d" % r.returncode, note))


if __name__ == "__main__":
    main()
