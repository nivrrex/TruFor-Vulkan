# DEPLOY.md — running TruFor on the RX 560 (Baffin) over Vulkan

Verified configuration. Every number here was measured on the reference machine; nothing is
inherited from a tutorial or assumed from documentation.

## Deployment matrix

| component | version | note |
|---|---|---|
| GPU | AMD Radeon RX 560 (Baffin), enumerates as `AMD Radeon RX 460 Graphics (RADV POLARIS11)` | `deviceType = DISCRETE_GPU`, PCI `01:00.0` |
| driver | **Mesa / RADV ≥ 25.0.7** | **hard requirement.** Mesa 22.3.6 produces silently wrong results — see "Known-bad" |
| Vulkan API | 1.4.305 | loader 1.3.239 is fine; the driver version is what matters |
| inference runtime | Apache TVM 0.27.0, built from source with `USE_VULKAN=ON` | the PyPI wheel has `USE_VULKAN=OFF` and cannot produce a Vulkan executable |
| model | `reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx` | the **original** dynamo export; the Slice+Squeeze rewrite is *not* part of the production path |
| input | 512×512, fp32, batch 1 | static shape only |

## What you get

| | |
|---|---|
| correctness | `map corr = 1.0000000`, `IoU = Dice = 1.0`, integrity score error 2.4e-06 … 2.7e-05 |
| latency (512×512) | ONNX Runtime CPU ≈ 2.15–2.20 s, TVM Vulkan ≈ 2.59–2.62 s |
| **speedup** | **0.82 – 0.84× — the GPU path is ~18% slower. No performance benefit.** |

A *correct* GPU execution, not a useful one. These are the numbers any optimisation has to beat;
the CPU path is a tuned oneDNN/MLAS implementation on 4 cores.

## What this package contains

| path | why |
|---|---|
| `DEPLOY.md` | this file |
| `README.md` | project overview and report index |
| `reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx` | the production model, weights embedded |
| `reports/mesa-ab/root/` | Mesa 25.0.7 unpacked, ready to load (no system install is touched) |
| `scripts/final_benchmark.py` | the entry point — correctness + latency |
| `scripts/tvm_cpu_gate.py` | TVM-LLVM correctness gate, used to separate driver problems from model problems |
| `scripts/tvm_vulkan_run.py` | end-to-end Vulkan run with per-image reporting |
| `scripts/onnx_gather_to_slice.py` | driver-bisection canary only; **not** part of the production path |
| `truFor-src/test_docker/images/` | the four reference test images |

**Not included, and not needed to run:** the TruFor PyTorch sources, the `.pth.tar` checkpoint, the
Vulkan SDK archive, and every TVM build product. Steps 0–2 below reproduce the third party pieces.
Re-exporting the ONNX from PyTorch (or changing the input resolution) needs the sources; see the
development package or the upstream repository for that.

---

## Step 0 — Vulkan SDK (for `glslc` and SPIRV-Tools)

Only needed to *build* TVM. 288 MB archive, unpacks to ~1.6 GB.

```bash
mkdir -p vendor && cd vendor
curl -L -o vulkan_sdk.tar.xz \
  https://sdk.lunarg.com/sdk/download/1.4.341.1/linux/vulkan_sdk.tar.xz
echo "3bf0f762afb6c79bc6a9d9fb5998745ccff928800a29619b501ed9de7fd9789b  vulkan_sdk.tar.xz" | sha256sum -c -
tar xf vulkan_sdk.tar.xz
cd ..
export VULKAN_SDK=$PWD/vendor/1.4.341.1/x86_64
```

1.4.341.1 is the version TVM's own Vulkan docs require. Any newer SDK works as long as it ships
`glslc` and `libSPIRV-Tools`.

## Step 1 — Mesa 25.0.7 (side by side, no system change)

Already unpacked in this package. To fetch it again:

```bash
echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm-backports main" \
  > /etc/apt/sources.list.d/bw-backports.list
apt-get update
mkdir -p vendor/mesa-ab && cd vendor/mesa-ab
apt-get download -t bookworm-backports mesa-vulkan-drivers=25.0.7-2~bpo12+1
for f in *.deb; do dpkg-deb -x "$f" root; done
cd ../..
```

## Step 2 — TVM with the Vulkan backend

```bash
git clone --recursive --branch v0.27.0 https://github.com/apache/tvm.git vendor/tvm-src
# llvm-15 or newer required; this was built against llvm-18 from apt.llvm.org
cd vendor/tvm-src && mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSE_VULKAN=$VULKAN_SDK -DUSE_KHRONOS_SPIRV=$VULKAN_SDK \
  -DUSE_LLVM=/usr/bin/llvm-config-18 -DUSE_GTEST=OFF
ninja -j$(nproc)
cd ../../..
```

Two build pitfalls that will stop you:

- `USE_VULKAN` must be the SDK **path**. With `ON`, the build finds the system loader but not
  SPIRV-Tools, and CMake then aborts with `Vulkan_SPIRV_TOOLS_LIBRARY-NOTFOUND`.
- `-DUSE_GTEST=OFF` is required on Debian 12: GTest 1.12.1 does not define the legacy
  `GTest::GTest` target the build looks up.

## Step 3 — Python environment

```bash
python3 -m venv --system-site-packages vendor/tvm-venv
vendor/tvm-venv/bin/pip install apache-tvm==0.27.0 onnx onnxscript onnxruntime
```

The wheel provides the binding layer and the ONNX frontend; the Vulkan runtime comes from the build
in step 2.

## Step 4 — Environment

```bash
export PYTHONPATH=$PWD/vendor/tvm-src/python
export TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build
export LD_LIBRARY_PATH=$PWD/vendor/mesa-ab/root/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=$PWD/vendor/mesa-ab/root/usr/share/vulkan/icd.d/radeon_icd.json
```

**Do not** set `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json` — that is the *old*
Mesa 22.3.6 ICD and would defeat the whole point.

## Step 5 — Verify before trusting any result

```bash
vulkaninfo --summary | grep -E 'driverVersion|apiVersion|deviceName|deviceType'
vendor/tvm-venv/bin/python -c "import tvm; d=tvm.vulkan(); print(d.device_name, d.driver_version, d.api_version)"
```

Expect `driverVersion 25.0.7`, `apiVersion 1.4.305`, and
`AMD Radeon RX 460 Graphics (RADV POLARIS11)`. `llvmpipe` also enumerates, always with
`deviceType = CPU`; assert the device type rather than trusting the device index.

Note: `tvm.support.libinfo()["USE_VULKAN"]` reports `OFF` even on a working Vulkan build. Test
`tvm.vulkan().exist` and the loaded libraries instead.

## Step 6 — Run

```bash
vendor/tvm-venv/bin/python scripts/final_benchmark.py \
  --onnx reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx \
  --images truFor-src/test_docker/images/tampered2.png \
  --outdir reports/bench --size 512 --repeats 5 --only ort-cpu,tvm-vulkan
```

Every run must print `map_corr = 1.0000000`. Anything less means the driver is wrong for this
workload — go back to step 1. A full four-image run is in `reports/FINAL-REPORT.md` §4.

## Step 7 — the hand-written kernels (what this project actually changes)

### There is no module_pass, and that is deliberate

The hand-written kernels are substituted on the **Python side**, in
`scripts/opt/swap_conv.py`, immediately before `relax.build`. There are five
families: a scalar-output maximum, a 3x3 and a 1x1 convolution, a depthwise 3x3,
and an NCHW/NHWC layout change with an optional GELU epilogue. Matching is by
**buffer signature**, never by function name, because the fusion pass assigns the
names. So:

* **you do NOT need to rebuild TVM to use them.** An unmodified TVM 0.27.0
  at commit `4d0ac168c7` is enough.
* there is no C++ module pass to build, and no `ModulePass` to register. The
  only upstream change in `patches/` is the per-kernel profiler in
  `src/runtime/vm/vm.cc`, which is diagnostic and not required to run.

Apply the patch only if you want per-kernel timings:

```sh
cd vendor/tvm-src
git apply ../../patches/tvm-vulkan-perf-and-profiling.patch   # 2 files, +73/-9
# then rebuild, see Step 2
```

Verified with `git apply --check` against a clean tree at `4d0ac168c7`.

### Run it

```sh
export PYTHONPATH=$PWD/vendor/tvm-src/python:$PWD/scripts/opt
export TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build
export LD_LIBRARY_PATH=$PWD/vendor/mesa-ab/root/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=$PWD/vendor/mesa-ab/root/usr/share/vulkan/icd.d/radeon_icd.json

vendor/tvm-venv/bin/python scripts/opt/run_trufor.py IMAGE
vendor/tvm-venv/bin/python scripts/opt/run_trufor.py --bench --reps 15 IMAGE
```

Measured here, 512x512, **one session**, ABAB interleaved, median of 15 after 3
warmups, 5 real images:

| path | median | vs ONNX Runtime CPU |
|---|---:|---:|
| `TRUFOR_HANDWRITTEN=0`, stock DLight schedules | 2478.1 ms | 0.86x |
| default, 26 substitutions | 1105.0 ms | **1.919x** |

Throughput, `scripts/opt/tile_throughput.py`, 8 tiles: CPU only 0.443 tile/s, GPU
only 0.892 tile/s, CPU and GPU workers over a shared queue **1.080 tile/s, 2.44x
over pure CPU**, per-tile outputs identical between lanes on 8 of 8.

For batch throughput rather than single-image latency, see
`scripts/opt/tile_throughput.py`: one CPU worker and one GPU worker over a shared
queue of tiles gives 1.003 tile/s against 0.448 for CPU alone and 0.683 for GPU
alone, i.e. 2.24x over pure CPU, with per-tile outputs identical between lanes.

### Fallback when a signature does not match

`swap_conv` **verifies every candidate against the PrimFunc it replaces, full
tensor, before accepting it**, and skips anything that does not reproduce. A
model whose fusion differs from the one these numbers were taken on therefore
degrades to the stock DLight schedules, not to a wrong answer. On a model where
nothing matches you get `substituted 0 PrimFuncs` on stderr and a correct, just
slower, run.

The depthwise kernel needs this most: half of the fused depthwise convolutions
end in a bare residual add and half in a ReLU, and the name does not say which.
The code tries both epilogues and keeps the one that matches, because guessing
wrong gives rel_L2 0.71 against 2.2e-08 when right.

### Environment switches

| variable | default | effect |
|---|---|---|
| `TRUFOR_HANDWRITTEN` | `1` | `0` disables all substitutions, stock schedules |
| `TRUFOR_VERIFY` | `1` | `0` skips full-tensor verification. **Unsafe**: the depthwise ReLU cannot be inferred without it, and with it off roughly half the depthwise kernels get the wrong epilogue |
| `TVM_VM_SKIP` | unset | diagnostic only: comma-separated PrimFunc names to skip at runtime, for measuring what a function costs. A skip is refused if any tensor argument is an integer or bool dtype, since skipping an index producer is how this build's out-of-bounds warning gets triggered. **Never set it in production**: a skipped call leaves its outputs stale, so the results are meaningless |
| `TVM_VULKAN_TARGET` | `{"kind":"vulkan","from_device":0}` | target override |

### What is verified, and how

Correctness is measured against the **un-substituted** TVM graph, not against a
reference implementation, with 3 real images and a per-buffer rel_L2 below 1e-4:

Against the un-substituted TVM graph, whole graph, all three outputs:

| comparison | worst rel_L2 |
|---|---:|
| scalar max substitution | 0.00e+00 |
| NCHW/NHWC + GELU layout | 0.00e+00 |
| depthwise 3x3 | 2.36e-08 |
| **whole graph** | **0.00e+00** |

Against ONNX Runtime CPU, which additionally carries ORT's own differences, over
5 images, map correlation is 1.0000000 on every one; the largest relative figure
is 1.96e-04 on the scalar detector output, which is a squashed sigmoid and
corresponds to a score difference of 2.5e-05 in absolute terms.

`det_err` is **not** used as an acceptance criterion. It sat at 2.46e-05 under
every kernel change in this project, including changes that broke the output
completely, so it carries no information.

### Known limitations to read before trusting the speedup

The transpose kernel runs at 8.7 GB/s against the 20 GB/s the hardware could
reach, because the load stays strided by the channel count. A shared-memory tile
is the fix and **the flat form supports it**: a probe with and without an explicit
`T.tvm_storage_sync` both give correct results, since ThreadSync inserts the
barrier. An earlier note in this project claimed the flat form returns zeros;
that was a bug in the probe, not a property of the form, and the claim is
withdrawn. The tiled transpose has not been written.

The decoder multi-scale fusion is the largest item still in the graph, 180.3 ms by
`TVM_VM_SKIP` measurement. Its replacement kernels are written and verified in
isolation but are **not** in the pipeline: the Relax-level rewrite that would
insert them matches zero call sites. The graph is correct and 1.919x without it.

## Known-bad legacy environment

| | |
|---|---|
| **Mesa / RADV 22.3.6** (the Devuan daedalus default) | The model runs to completion and returns **silently wrong** results: `map corr = -0.2874`, IoU = Dice = 0, integrity score 0.63 instead of 0.99. No crash, no error, no validation-layer message. Some pipeline variants additionally abort with `corrupted double-linked list`. |

Do not work around it. The ONNX-level `Gather(axis=0)` → `Slice`+`Squeeze` rewrite is **verified
equivalent and unnecessary** on Mesa 25.0.7 — both models produce byte-identical errors — so it is
deliberately kept out of the production path. Use `scripts/onnx_gather_to_slice.py` only as a canary
when bisecting a driver problem.

## Input resolution

The **Vulkan / TVM delivery path is fixed at 512×512** — its hand-written kernels match on static
512×512 buffer signatures and cannot consume a dynamic graph.

A separate **dynamic-H/W ONNX exists for the ONNX Runtime CPU path only**:
`scripts/export_onnx_dynamo_dynamic.py` re-exports with `rgb` shaped `1×3×height×width` so any
input size works and the output map comes back at exactly the input size (the model's own
`F.interpolate(size=orisize[2:])` is carried through the export). `scripts/verify_dynamic_ort.py`
checks it against the PyTorch reference across sizes including non-multiples of 32. It is a
regenerable build artifact and is git-ignored, not committed. It does **not** speed up or replace the
Vulkan path.

Re-exporting from PyTorch needs the TruFor sources under `truFor-src/` (git-ignored; fetch separately).
The workload is activation-bound, so the GPU/CPU crossover has not been located — see
`reports/FINAL-REPORT.md` §10.

## If the numbers regress

1. `driverVersion` is not 25.0.7 → step 1.
2. `map_corr < 1.0` → the driver is wrong for this workload, not the model. The model and the CPU
   reference are known-good; `scripts/tvm_cpu_gate.py` must still report `corr = 0.9999999999` on the
   LLVM backend.
3. Latency far from 2.6 s → check that the timed region includes the host copy, otherwise you are
   measuring dispatch rather than execution.
