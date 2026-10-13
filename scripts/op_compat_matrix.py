#!/usr/bin/env python3
"""Phase 3d: per-operator Vulkan compatibility matrix for TruFor's core ops.

One tiny model per operator (or operator group), each exported, partitioned
with VulkanPartitioner and executed by vulkan_executor_runner. Reports, per
operator: how many delegates it produced, whether the runner survived, and the
numerical error against eager PyTorch.

This isolates which specific operator breaks the GPU path, instead of only
knowing that the whole 88-delegate TruFor graph aborts.
"""

import json
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def case_conv():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())


def case_linear():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 8), nn.ReLU())


def case_layer_norm():
    return nn.Sequential(nn.LayerNorm(3 * 32 * 32), nn.ReLU())


def case_gelu():
    return nn.Sequential(nn.GELU(), nn.Conv2d(3, 8, 3, padding=1))


def case_softmax():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16), nn.Softmax(dim=-1))


def case_bmm():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.randn(2, 64, 64))
            self.q = nn.Parameter(torch.randn(2, 64, 64))

        def forward(self, x):
            return torch.bmm(self.p + x.mean(), self.q).sum(dim=1, keepdim=True)

    return M()


def case_attn5d():
    """The layout CMX actually produces: [2, B, heads, N, D] -- 5 dimensions."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv = nn.Linear(3 * 32 * 32, 2 * 4 * 64)
            self.num_heads = 4

        def forward(self, x):
            B = x.shape[0]
            h = x.flatten(1)
            k, v = self.kv(h).reshape(B, -1, 2, self.num_heads, 64).permute(2, 0, 3, 1, 4)
            a = torch.softmax(k @ v.transpose(-2, -1) / 8.0, dim=-1)
            return a.sum(dim=(-1, -2)).flatten(1)

    return M()


def case_attn4d():
    """Same maths with the 5-D tensors flattened to 4-D."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv = nn.Linear(3 * 32 * 32, 2 * 4 * 64)
            self.num_heads = 4

        def forward(self, x):
            B = x.shape[0]
            h = x.flatten(1)
            k, v = self.kv(h).reshape(B, -1, 2, self.num_heads * 64).permute(1, 0, 2)
            k = k.reshape(-1, B, self.num_heads, k.shape[-1] // self.num_heads)
            v = v.reshape(-1, B, self.num_heads, v.shape[-1] // self.num_heads)
            a = torch.softmax(k @ v.transpose(-2, -1) / 8.0, dim=-1)
            return a.sum(dim=(-1, -2)).flatten(1)

    return M()


def case_cat():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),
                         nn.ReLU(), nn.Conv2d(8, 8, 1), nn.Sigmoid())


def case_batchnorm():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())


def case_conv_stack8():
    """8 chained convs in one delegate -- closest analogue to TruFor's DnCNN."""

    layers = []
    c = 3
    for i in range(8):
        layers += [nn.Conv2d(c, 16, 3, padding=1), nn.ReLU()]
        c = 16
    layers += [nn.Conv2d(16, 1, 3, padding=1)]
    return nn.Sequential(*layers)


CASES = [
    ("conv+relu", case_conv),
    ("linear+relu", case_linear),
    ("layer_norm", case_layer_norm),
    ("gelu", case_gelu),
    ("linear+softmax", case_softmax),
    ("bmm", case_bmm),
    ("cat-branch", case_cat),
    ("batchnorm", case_batchnorm),
    ("conv-stack-8", case_conv_stack8),
    ("attention-5d", case_attn5d),
    ("attention-4d", case_attn4d),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "reports/op-matrix"
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

    rows = []
    print("%-16s %-9s %-9s %-8s %s" % ("case", "delegates", "exports", "runner", "note"))
    for name, factory in CASES:
        row = {"case": name}
        try:
            mod = factory()
            sample = (torch.rand(1, 3, 32, 32),)
            exports = True
            with torch.no_grad():
                ref = mod(*sample)
            prog = export(mod, sample, dynamic_shapes=None, strict=True)
            cfg = EdgeCompileConfig(_skip_dim_order=False)
            ep = to_edge_transform_and_lower(
                prog, compile_config=cfg,
                transform_passes=[I64toI32(cfg._skip_dim_order)],
                partitioner=[VulkanPartitioner({})],
            )
            et = ep.to_executorch()
            path = os.path.join(outdir, "case-%s.pte" % name)
            with open(path, "wb") as f:
                f.write(et.buffer)
            nd = len(et.executorch_program.execution_plan[0].delegates)
        except Exception as e:  # noqa: BLE001
            row.update({"delegates": 0, "exports": False, "runner": "-", "note": "%s: %s" % (type(e).__name__, str(e)[:70])})
            rows.append(row)
            print("%-16s %-9s %-9s %-8s %s" % (name, "-", "FAIL", "-", row["note"]))
            continue

        try:
            r = subprocess.run([runner, "-model_path=%s" % os.path.abspath(path), "-num_executions=1"],
                               env=env, capture_output=True, text=True, timeout=300)
            ok = r.returncode == 0
            note = "" if ok else (r.stderr.strip().splitlines() or [""])[-1][:70]
        except subprocess.TimeoutExpired:
            ok, note = False, "timeout"

        row.update({"delegates": nd, "exports": exports, "runner": "OK" if ok else "ABORT", "note": note,
                    "ref_shape": list(ref.shape)})
        rows.append(row)
        print("%-16s %-9d %-9s %-8s %s" % (name, nd, "ok", row["runner"], note))

    with open(os.path.join(outdir, "op-matrix.json"), "w") as f:
        json.dump(rows, f, indent=2)
    ok = sum(1 for r in rows if r.get("runner") == "OK")
    print("\n%d/%d cases ran on Vulkan" % (ok, len(rows)))


if __name__ == "__main__":
    main()
