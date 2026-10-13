# PACKAGE-NOTE.md — what this package is

**A document-review snapshot. It contains the project's written findings and the raw measurement
data behind them, and deliberately nothing else.**

## What is here

| | |
|---|---|
| `README.md` | entry point: one-line answer, the driver-bug narrative, report index, honest landing summary |
| `CHANGELOG.md` | how the project arrived at its conclusions, including three retractions and a `Not answered` section |
| `DEPLOY.md` | the working configuration, build steps, known-bad environments |
| `reports/**/*.md` | 14 reports, one per phase, each with its own evidence |
| `reports/**/*.json`, `*.csv` | the raw measurements every number in the reports comes from |

Nothing here needs to be executed to be read. Every quantitative claim in the reports can be checked
against the JSON next to it.

## What is not here, and why

Per the packaging request this is a documents-only snapshot. Excluded:

| excluded | where it lives |
|---|---|
| `scripts/`, `scripts/bench/` | source project — 46 scripts, each runnable standalone |
| `truFor-src/` | source project — upstream TruFor at commit `ae54475`, never modified |
| `vendor/tvm-src/` | source project — Apache TVM 0.27.0, byte-identical to upstream |
| `vendor/mesa-ab/` | source project — Mesa 25.0.7 unpacked, loadable via `VK_DRIVER_FILES` |
| `weights/`, the 280 MB production ONNX | source project — needed to re-run, not to evaluate |
| `upstream/*.cpp`, `*.py` | source project — material intended for the TVM project |

**Consequence for reference checking:** the reports quote paths in those trees — `scripts/gate04…`,
`src/backend/vulkan/runtime/vulkan_device.cc`, `lib/models/cmx/net_utils.py`, `include/spirv-tools/
libspirv.h`. Those are *inline citations locating a behaviour*, not links expected to resolve here.
Each affected report states this at the top. The full development package
(`TruFor_Vulkan_dev_*.tar.gz`) contains every one of them.

## Where to start

1. `README.md` — the conclusion and the index.
2. `CHANGELOG.md`, the **`Not answered`** section — the honest list of what is still unexplained,
   which is the part most worth attacking.
3. `reports/capability/CAPABILITY-REPORT.md` — the measurement study that establishes the current
   performance position (Gate 0–7).
4. `reports/FINAL-REPORT.md` — the root cause and the retractions.

## If you want to verify rather than read

The measurement protocol is fixed and stated in the capability report: warm-up 3, 15 repeats,
min/p10/median/p90/max, VRAM sampled in the background. The three claims most worth an independent
attempt are:

- **the 0.35% overhead decomposition** (Gate 2) — it is what rules out every cheap fix;
- **the 0-of-23 operator sweep** (Gate 4) — it is what separates "hardware is too weak" from
  "the software stack leaves performance unused";
- **the rejected `thread_warp_size` experiment** — the most plausible single-line explanation, which
  measurement killed, and which was reverted rather than kept.

Known gaps in the harness are recorded in the capability report so they are not re-derived: Conv and
ReduceSum cases fail inside TVM's ONNX frontend, so the operator sweep covers elementwise, matmul
and softmax only; VRAM must be sampled through `radeontop` because
`/sys/class/drm/card0/mem_info_vram_*` does not exist on this kernel; and Gate 6 is only partially
answered because batching was blocked upstream of it.
