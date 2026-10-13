#!/usr/bin/env python3
"""Minimal control experiment for the Vulkan runner.

Exports and partitions a 3-layer MLP -- operators the Vulkan backend definitely
supports (linear, relu) -- so the resulting .pte can be fed to
vulkan_executor_runner. If this tiny model also crashes the runner, the problem
is the runner/Vulkan environment; if it runs, the problem is specific to
TruFor's heavily fragmented 88-delegate graph.

This is deliberately independent of the TruFor sources.
"""

import os
import sys

import torch
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Wide(nn.Module):
    """Adds the elementwise ops that TruFor's detection head needs."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def build(mod, size, outdir, tag):
    from executorch.backends.transforms.convert_dtype_pass import I64toI32
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export

    sample = (torch.rand(1, 3, size, size),)
    prog = export(mod, sample, dynamic_shapes=None, strict=True)
    cfg = EdgeCompileConfig(_skip_dim_order=False)
    ep = to_edge_transform_and_lower(
        prog,
        compile_config=cfg,
        transform_passes=[I64toI32(cfg._skip_dim_order)],
        partitioner=[VulkanPartitioner({})],
    )
    et = ep.to_executorch()
    path = os.path.join(outdir, "tiny-%s-%d.pte" % (tag, size))
    with open(path, "wb") as f:
        f.write(et.buffer)
    n_del = len(et.executorch_program.execution_plan[0].delegates)
    print("%-6s %dx%d -> %s (%.1f MB), delegates=%d" % (tag, size, size, path, os.path.getsize(path) / 1e6, n_del))
    return path


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "reports/runner-control"
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    for tag, mod, size in (("tiny", Tiny(), 64), ("wide", Wide(), 128), ("wide512", Wide(), 512)):
        try:
            build(mod, size, outdir, tag)
        except Exception as e:  # noqa: BLE001
            print("%-8s FAILED: %s: %s" % (tag, type(e).__name__, str(e)[:200]))


if __name__ == "__main__":
    main()
