#!/usr/bin/env python3
"""A1 (gate): trace TruFor to TorchScript and convert with pnnx.

Reuses build_model/swap_adaptive_pool/ExportWrapper from
scripts/export_onnx_dynamo.py. The venv has torch 2.14.1 with no matching
torchvision (latest wheel 0.29.1 targets torch 2.4.x), so `import timm`'s
top level dies on torchvision.datasets. TruFor only needs three symbols from
timm.models.layers, so we register `timm` / `timm.models` as bare namespace
packages pointing at the real directories; the real timm.layers submodules
load normally and only need the torchvision.ops.misc stub already installed.

pnnx is invoked with two inputshapes so it can tell which axes are dynamic;
constant-baked Reshapes/interpolate sizes either break conversion or show up
in work/pnnx_a1_test.py as wrong output shapes.
"""
import importlib.util
import os
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = os.path.join(ROOT, "venv", "lib", "python3.11", "site-packages")

for _name, _path in (("timm", os.path.join(SITE, "timm")),
                     ("timm.models", os.path.join(SITE, "timm", "models"))):
    _mod = types.ModuleType(_name)
    _mod.__path__ = [_path]
    sys.modules[_name] = _mod

_spec = importlib.util.spec_from_file_location(
    "export_onnx_dynamo", os.path.join(ROOT, "scripts", "export_onnx_dynamo.py"))
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# --- pnnx lowering patches (model sources untouched; same spirit as swap_adaptive_pool) ---
# torch.tile(modal_x,(3,1,1)) at cmx/builder_np_conf.py:277 -> cat along channel
# dim, which pnnx maps to Concat. ncnn left torch.tile as an unregistered custom
# layer (load_param fails).
_orig_tile = torch.tile
def _tile_patch(input, dims):
    if tuple(dims) == (3, 1, 1) and input.dim() == 4 and input.shape[1] == 1:
        return torch.cat([input, input, input], dim=1)
    return _orig_tile(input, dims)
torch.tile = _tile_patch

# torch.zeros((b,1,x.shape[-1])) in weighted_statistics_pooling (layer_utils.py:27)
# has a data-dependent size -> custom torch.zeros layer. x[:, :1]-x[:, :1] is
# exactly zero with the same dynamic shape, and lowers to BinaryOp(Sub).
# Imported by call site via `from .layer_utils import ...` inside forward, so
# patching the module attribute takes effect.
import lib.models.cmx.layer_utils as _lu  # noqa: E402
def _wsp_no_zeros(x, log_w=None):
    b = x.shape[0]
    c = x.shape[1]
    x = x.view(b, c, -1)
    if log_w is None:
        log_w = x[:, :1, :] - x[:, :1, :]
    else:
        log_w = log_w.view(b, 1, -1)
    log_w = F.log_softmax(log_w, dim=-1)
    x_min = -torch.logsumexp(log_w - x, dim=-1)
    x_max = torch.logsumexp(log_w + x, dim=-1)
    w = torch.exp(log_w)
    x_avg = torch.sum(w * x, dim=-1)
    x_msq = torch.sum(w * x * x, dim=-1)
    return torch.cat((x_min, x_max, x_avg, x_msq), dim=1)
_lu.weighted_statistics_pooling = _wsp_no_zeros

# pnnx drops the dims of 5D constant-target reshapes (reshape(B,-1,2,heads,Dh)
# at net_utils.py:98-99 and dual_segformer.py:123-125), emitting zero-parameter
# ncnn Reshape layers that yield empty Mats (first failure at blob 75; 35 such
# layers total). Head splitting via channel-dim slicing (t[..., a:b]) is bitwise
# the same view (channel order is (2, heads, head_dim)) and keeps the only
# dynamic axis inside the Mat's own h dim -- no reshape needed. Uses slice not
# narrow(): pnnx lowers aten::slice to ncnn Slice but emits torch.narrow as an
# unregistered custom op.
import lib.models.cmx.net_utils as _nu  # noqa: E402
import lib.models.cmx.encoders.dual_segformer as _ds  # noqa: E402


def _cross_attn_headsplit(self, x1, x2):
    B, N, C = x1.shape
    Hs, Dh = self.num_heads, C // self.num_heads
    def kv_heads(t):  # (B,N,2*Hs*Dh) -> [(k_h, v_h)...] same channel partition as the 5D view
        return [(t[..., h * Dh:(h + 1) * Dh], t[..., (Hs + h) * Dh:(Hs + h + 1) * Dh])
                for h in range(Hs)]
    k1s, v1s = list(zip(*kv_heads(self.kv1(x1))))
    k2s, v2s = list(zip(*kv_heads(self.kv2(x2))))
    outs1, outs2 = [], []
    for h in range(Hs):
        ctx1 = (k1s[h].transpose(-2, -1) @ v1s[h]) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2s[h].transpose(-2, -1) @ v2s[h]) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        outs1.append(x1[..., h * Dh:(h + 1) * Dh] @ ctx2)
        outs2.append(x2[..., h * Dh:(h + 1) * Dh] @ ctx1)
    return torch.cat(outs1, dim=-1), torch.cat(outs2, dim=-1)


def _self_attn_headsplit(self, x, H, W):
    B, N, C = x.shape
    Hs, Dh = self.num_heads, C // self.num_heads
    qf = self.q(x)
    if self.sr_ratio > 1:
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        kvf = self.kv(x_)
    else:
        kvf = self.kv(x)
    outs = []
    for h in range(Hs):
        qh = qf[..., h * Dh:(h + 1) * Dh]
        kh = kvf[..., h * Dh:(h + 1) * Dh]
        vh = kvf[..., (Hs + h) * Dh:(Hs + h + 1) * Dh]
        attn = (qh @ kh.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        outs.append(attn @ vh)
    x = torch.cat(outs, dim=-1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


_nu.CrossAttention.forward = _cross_attn_headsplit
_ds.Attention.forward = _self_attn_headsplit


# FRM ChannelWeights/SpatialWeights end with reshape(B,2,...).permute(1,0,...)
# then branch indexing; pnnx emits a 4D Reshape plus Slice unbind and maps the
# branch axis onto the wrong ncnn axis (e.g. Slice axis=2 on a c=1 Mat), so the
# second slice comes back an empty Mat and the next reshape(-1,-1) SIGFPEs
# (first crash at blob 2360, param line 1748, on the FRMs_3 SpatialWeights).
# Slicing the mlp output along the channel dim directly is the bitwise-same
# partition (row-major (B,2,...) splits as [0]=first block, [1]=second).
def _channel_weights_split(self, x1, x2):
    B, _, H, W = x1.shape
    x = torch.cat((x1, x2), dim=1)
    avg = self.avg_pool(x).view(B, self.dim * 2)
    mx = self.max_pool(x).view(B, self.dim * 2)
    y = torch.cat((avg, mx), dim=1)
    y = self.mlp(y)  # (B, 2*dim)
    return (y[:, 0:self.dim].reshape(B, self.dim, 1, 1),
            y[:, self.dim:2 * self.dim].reshape(B, self.dim, 1, 1))


def _spatial_weights_split(self, x1, x2):
    B, _, H, W = x1.shape
    x = torch.cat((x1, x2), dim=1)
    m = self.mlp(x)  # (B, 2, H, W)
    return (m[:, 0:1], m[:, 1:2])


_nu.ChannelWeights.forward = _channel_weights_split
_nu.SpatialWeights.forward = _spatial_weights_split
# --- end patches ---

torch.set_grad_enabled(False)

OUTDIR = os.path.join(HERE, "ncnn_a1")
os.makedirs(OUTDIR, exist_ok=True)


def main():
    t0 = time.perf_counter()
    model = exp.build_model(os.path.join(ROOT, "weights", "trufor.pth.tar"))
    model.eval()
    print("model built %.1fs" % (time.perf_counter() - t0), flush=True)

    changes = exp.swap_adaptive_pool(model)
    print("pool substitutions: %d" % len(changes), flush=True)

    wrapper = exp.ExportWrapper(model)
    wrapper.eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        outs = wrapper(x)
    for o, nm in zip(outs, ("out", "conf", "det")):
        print("eager %s: %s" % (nm, tuple(o.shape)), flush=True)

    print("trace start", flush=True)
    t1 = time.perf_counter()
    ts = torch.jit.trace(wrapper, x, check_trace=False)
    print("trace done %.1fs" % (time.perf_counter() - t1), flush=True)

    pt = os.path.join(OUTDIR, "trufor.ts.pt")
    ts.save(pt)
    print("saved %s (%d bytes)" % (pt, os.path.getsize(pt)), flush=True)

    pnnx = os.path.join(ROOT, "venv", "bin", "pnnx")
    cmd = [pnnx, "trufor.ts.pt",
           "inputshape=[1,3,512,512]", "inputshape2=[1,3,640,384]",
           "fp16=0"]  # fp16=0: default fp1=1 stores conv weights fp16,
# losing ~1e-3 relative precision per layer that compounds across the
# 17-layer DnCNN + CMX chain (measured end-to-end rel_L2 ~0.13 vs fp32 ref)
    print("pnnx:", " ".join(cmd), flush=True)
    t2 = time.perf_counter()
    r = subprocess.run(cmd, cwd=OUTDIR, capture_output=True, text=True, timeout=5400)
    print("pnnx rc=%d %.1fs" % (r.returncode, time.perf_counter() - t2), flush=True)
    with open(os.path.join(OUTDIR, "pnnx.log"), "w") as f:
        f.write("$ %s\n--- stdout ---\n%s\n--- stderr ---\n%s\n"
                % (" ".join(cmd), r.stdout, r.stderr))
    for line in (r.stdout + r.stderr).splitlines()[-40:]:
        print("pnnx|", line, flush=True)

    # Post-convert normalisation. pnnx emits reshape sizes that reference a
    # *different* blob as `size(@N,k)` expressions (e.g. ChannelEmbed's
    # x.permute(0,2,1).reshape(B,_C,H,W) where H,W come from the FFM spatial
    # input, blob v_2128); the pypi ncnn wheel parser rejects '@' outright
    # ("malformed literal token @1"). The tokens are the same quantity when
    # spelled with the indexed-input form (input 1 h == 1h), which loads fine
    # (verified by hand on ncnn_a1/fix1: load_param rc=0). Rewrite the exact
    # known pattern, then fail loudly on anything else that could not parse.
    param_path = os.path.join(OUTDIR, "trufor.ts.ncnn.param")
    with open(param_path) as f:
        text = f.read()
    fixed = text.replace('6="1w,size(@1,2),1024"', '6="1w,1h,1024"')
    if fixed != text:
        with open(param_path, "w") as f:
            f.write(fixed)
        print("post-fix: replaced 1 size(@1,2) reshape expression", flush=True)
    bad = [ln for ln in fixed.splitlines() if ("@" in ln or "trunc" in ln)]
    print("A1_POSTCHECK residual_expr_lines=%d" % len(bad), flush=True)
    for ln in bad[:10]:
        print("A1_POSTCHECK|", ln, flush=True)

    for name in ("trufor.ts.ncnn.param", "trufor.ts.ncnn.bin"):
        p = os.path.join(OUTDIR, name)
        print("%s: %s %s" % (name, os.path.exists(p),
                             os.path.getsize(p) if os.path.exists(p) else "-"), flush=True)
    print("A1_CONVERT_DONE rc=%d" % r.returncode, flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
