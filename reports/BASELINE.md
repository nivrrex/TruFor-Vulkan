# BASELINE.md — TruFor CPU reference (Phase 1)

## Environment (verified on this machine)

| item | value |
|---|---|
| OS | Devuan 5 (Debian 12 base), no systemd |
| CPU | 4 cores |
| RAM | 15 GB |
| GPU | `01:00.0 AMD Baffin [Radeon RX 460/560D / Pro 450/455/460/555/555X/560/560X] (rev cf)` |
| GPU node | `/dev/dri/renderD128` |
| Mesa | 22.3.6-1+deb12u2 |
| Vulkan loader | `libvulkan.so.1` 1.3.239.0-1 |
| Vulkan ICDs present | `radeon_icd` (RADV), `lvp_icd` (lavapipe = **software**), `intel_icd`, `intel_hasvk_icd` |
| CUDA | none (`torch.version.cuda is None`) |
| ROCm | none |
| `vulkaninfo` | **not installed** (vulkan-tools missing) |
| Python | 3.11.2 |
| PyTorch | 2.7.1+**cpu** |
| torchvision | 0.22.1+cpu |
| timm | 1.0.28 |
| onnxruntime | 1.24.4 (CPU EP only — no Vulkan EP, see `reports/PROBE-REPORT.md`) |

## Source of truth

- TruFor upstream: `https://github.com/grip-unina/TruFor`
- Commit: `ae54475df6f41a491d7615100feb19263dec13f7` (cloned as `truFor-src/`)
- Final weights: `https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip` → `weights/trufor.pth.tar` (260 MB zip)
- Pretrained Noiseprint++ + SegFormer-B2: shipped inside the repo under `TruFor_train_test/pretrained_models/`

**No file in `truFor-src/` was modified.** `git -C truFor-src status --porcelain` is empty except for
`__pycache__`. All adaptation lives in `scripts/`.

## Compatibility patches applied (in our scripts, not upstream)

1. `torch.load(..., weights_only=False)` — the official checkpoint is a torch-1.11-era pickle
   containing `numpy.core.multiarray.scalar`; torch ≥ 2.6 defaults to `weights_only=True` and
   refuses it.
2. `os.chdir(TruFor_train_test)` around `update_config` — `lib/config/default.py` merges
   `lib/config/<experiment>.yaml` through a *relative* path.
3. Installed `matplotlib` (imported by `lib/utils.py`) and `torch-dct`.

Full `mmcv-full 1.5.3` / `mmsegmentation 0.25.0` / `mmcls 0.23.1` from `trufor_conda.yaml`
were **not** installed: inference on TruFor does not touch them, and those 2022-era pins do
not build against Python 3.11 + torch 2.7. Verified: `get_model()` and a full forward pass run
without them.

## Model

- `lib.models.cmx.builder_np_conf.EncoderDecoder`
- 68,697,421 parameters
- `load_state_dict(strict=True)` → **0 missing keys, 0 unexpected keys** (verified twice: the
  baseline runner asserts this via `strict=False` + reporting, and the inventory runner uses
  `strict=True` outright)

## Reproduce

```bash
cd <repo root>
python3 scripts/baseline_cpu.py \
  --input truFor-src/test_docker/images \
  --outdir artifacts/baseline_cpu \
  --model weights/trufor.pth.tar
```

Wall clock: 4 m 01 s for the 4 bundled test images (single run, no warm-up split yet).

## CPU baseline results (4 torch threads)

| image | size (HxW) | integrity score | localization map mean | confidence mean | total (s) | NP++ (s) | backbone (s) | loc_head (s) | conf_head (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pristine1.jpg | 256x384 | 0.189690 | 0.0216 | 0.9736 | 1.20 | 0.56 | 0.34 | 0.15 | 0.14 |
| pristine2.jpg | 1773x2048 | 0.188444 | 0.1935 | 0.8962 | 100.42 | 24.40 | 62.63 | 6.51 | 6.73 |
| tampered1.png | 1536x2048 | 0.997614 | 0.1161 | 0.9773 | 80.12 | 21.60 | 46.59 | 5.84 | 5.99 |
| tampered2.png | 256x256 | 0.998346 | 0.7135 | 0.9456 | 0.70 | 0.33 | 0.20 | 0.09 | 0.08 |

Sanity check on the integrity score: pristine images → 0.189 / 0.188, tampered → 0.998 / 0.998.
This matches the intended semantics of the TruFor integrity score, which confirms the weights
loaded correctly and the graph is wired as upstream intends.

Stage split is captured with forward pre/post hooks on `dncnn`, `backbone`, `decode_head`,
`decode_head_conf`; `post` is everything after the last hook (interpolate + detection head) and
came out at ~0.00 s.

## Reference artifacts

`artifacts/baseline_cpu/baseline.json` — manifest with per-image shapes, scores, per-stage timings.
`artifacts/baseline_cpu/npz/*.npz` — per image: `map` (localization), `conf` (confidence),
`npp` (Noiseprint++ raw), `rgb` (input), `det` (integrity score), `imgsize`.

These are the sole reference for every later CPU-vs-GPU comparison (Phase 6). Re-running the
script overwrites them; git-track `baseline.json` and a checksum of the npz files if you want
regression protection.

## Acceptance

- [x] inference runs repeatably on CPU
- [x] checkpoint loads with zero missing/unexpected keys
- [x] scores are semantically correct (pristine ≪ tampered)
- [x] per-stage timings recorded
- [x] all four output tensors persisted
- [x] upstream sources untouched

## Known limitations of this baseline

- Single run per image, no warm-up/steady-state split — timing numbers include lazy allocation
  and first-touch page faults. Phase 5 must use `--repeats >= 5` before quoting any speedup.
- 4 threads on a 4-core box; `torch.set_num_threads` was left at the default.
- `post` stage timing is coarse (it is "everything after conf_head") and not separately resolved.
