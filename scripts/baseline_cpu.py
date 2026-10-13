#!/usr/bin/env python3
"""Phase 1: TruFor CPU reference baseline.

Runs the unmodified official TruFor model on CPU, records per-stage latency and
saves every output tensor (map / conf / detection score / Noiseprint++) as .npz
plus a JSON manifest so later CPU-vs-GPU comparison is reproducible.

No third-party TruFor source file is modified; everything is done by importing
the official modules and wrapping them with hooks.
"""

import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np
import torch
from torch.nn import functional as F

# --- torch>=2.6 compatibility: official checkpoint pickles numpy scalars. ---
_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

TRUFOR_ROOT = os.environ.get(
    "TRUFOR_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "truFor-src")
)
TRAIN_DIR = os.path.join(TRUFOR_ROOT, "TruFor_train_test")
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from lib.config import config, update_config  # noqa: E402
from lib.utils import get_model  # noqa: E402
from dataset.dataset_test import TestDataset  # noqa: E402

STAGES = ["npp", "backbone", "loc_head", "conf_head", "post"]


def build_model(model_file, experiment="trufor_ph3"):
    class _Args:
        pass

    # lib/config/default.py merges the experiment yaml via a *relative* path.
    prev_cwd = os.getcwd()
    os.chdir(TRAIN_DIR)
    try:
        a = _Args()
        a.experiment = experiment
        a.opts = []
        update_config(config, a)
        model = get_model(config)
    finally:
        os.chdir(prev_cwd)
    checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model, {
        "epoch": checkpoint.get("epoch"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def attach_timers(model, timings):
    """Register forward hooks that accumulate wall time per sub-module."""

    def hook(name):
        def fn(_mod, _inp, _out):
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - _t0[name])

        return fn

    # _t0 is set by a pre-hook, so ordering is correct even for nested modules.
    handles = []
    targets = {
        "npp": model.dncnn,
        "backbone": model.backbone,
        "loc_head": model.decode_head,
        "conf_head": model.decode_head_conf,
    }
    for name, mod in targets.items():
        if mod is None:
            continue

        def pre(_m, _i, _name=name):
            _t0[_name] = time.perf_counter()

        def post(_m, _i, _o, _name=name):
            timings[_name] = timings.get(_name, 0.0) + (time.perf_counter() - _t0[_name])

        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    return handles


_t0 = {}


def list_images(path):
    if "*" in path:
        out = glob(path, recursive=True)
    elif os.path.isfile(path):
        out = [path]
    elif os.path.isdir(path):
        out = glob(os.path.join(path, "**/*"), recursive=True)
    else:
        raise ValueError("input is neither a file nor a folder: %s" % path)
    return sorted([f for f in out if not os.path.isdir(f)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", default="trufor_ph3")
    ap.add_argument("--repeats", type=int, default=1, help="timed repeats per image; run 1 of each is reported")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads; 0 = leave default")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    npz_dir = os.path.join(args.outdir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    model, meta = build_model(args.model, args.exp)
    model.eval()

    images = list_images(args.input)
    if not images:
        raise SystemExit("no images found under %s" % args.input)

    timings = {}
    handles = attach_timers(model, timings)

    manifest = {
        "trufor_root": TRUFOR_ROOT,
        "model_file": args.model,
        "experiment": args.exp,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "torch_threads": torch.get_num_threads(),
        "checkpoint_epoch": meta["epoch"],
        "missing_keys": meta["missing_keys"],
        "unexpected_keys": meta["unexpected_keys"],
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "images": [],
    }

    dataset = TestDataset(list_img=images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    t_total0 = time.perf_counter()
    for idx, (rgb, paths) in enumerate(loader):
        p = paths[0]
        rel = os.path.relpath(p, args.input) if os.path.isdir(args.input) else os.path.basename(p)
        stem = rel.rsplit(".", 1)[0]

        per_repeat = []
        for rep in range(args.repeats):
            timings.clear()
            t0 = time.perf_counter()
            out, conf, det, _ = model(rgb, save_np=False)
            t_total = time.perf_counter() - t0
            per_repeat.append({"total_s": t_total, **{k: timings.get(k, 0.0) for k in STAGES}})

        # ---- final tensors (computed once, from the last forward) ----
        conf_map = torch.squeeze(conf, 0)
        conf_map = torch.sigmoid(conf_map)[0].cpu().numpy()
        loc_map = torch.squeeze(out, 0)
        loc_map = F.softmax(loc_map, dim=0)[1].cpu().numpy()
        det_score = torch.sigmoid(det).item() if det is not None else None

        # Noiseprint++ output, recomputed explicitly (model returns it only with save_np)
        with torch.no_grad():
            modal_x = model.dncnn(rgb)
        npp = torch.squeeze(modal_x, 0)[0].cpu().numpy()

        npz_path = os.path.join(npz_dir, stem.replace(os.sep, "__") + ".npz")
        np.savez_compressed(
            npz_path,
            map=loc_map,
            conf=conf_map,
            npp=npp,
            rgb=rgb[0].cpu().numpy(),
            det=np.float64(det_score if det_score is not None else np.nan),
            imgsize=np.array(rgb.shape[2:]),
        )

        best = per_repeat[0]
        manifest["images"].append(
            {
                "path": p,
                "shape_hw": list(rgb.shape[2:]),
                "npz": os.path.relpath(npz_path, args.outdir),
                "integrity_score": det_score,
                "map_mean": float(loc_map.mean()),
                "map_max": float(loc_map.max()),
                "conf_mean": float(conf_map.mean()),
                "npp_shape": list(npp.shape),
                "timings": best,
                "all_timings": per_repeat,
            }
        )
        print(
            "[%d/%d] %-28s shape=%-12s det=%s  total=%.2fs  %s"
            % (
                idx + 1,
                len(loader),
                stem,
                "%dx%d" % (rgb.shape[2], rgb.shape[3]),
                ("%.6f" % det_score) if det_score is not None else "n/a",
                best["total_s"],
                "  ".join("%s=%.2f" % (k, best[k]) for k in STAGES),
            ),
            flush=True,
        )

    for h in handles:
        h.remove()

    manifest["wall_clock_total_s"] = time.perf_counter() - t_total0
    with open(os.path.join(args.outdir, "baseline.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nwrote %s" % os.path.join(args.outdir, "baseline.json"))


if __name__ == "__main__":
    main()
