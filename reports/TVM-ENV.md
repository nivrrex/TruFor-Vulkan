# TVM-ENV.md — Phase 6A: Apache TVM environment

## What was checked

| item | result |
|---|---|
| TVM installed in system Python | no (`ModuleNotFoundError: tvm`) |
| PyPI package name | `apache-tvm`, latest `0.27.0` |
| Linux x86_64 wheel on PyPI | **yes** — `apache_tvm-0.27.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` (100.5 MB) |
| wheel `USE_VULKAN` | **OFF** |
| wheel `USE_LLVM` | ON |
| wheel `USE_CUDA` | ON |
| wheel `USE_OPENCL` / `USE_ROCM` / `USE_METAL` | OFF / OFF / OFF |

So the wheel is usable for import + LLVM CPU work but **cannot** produce a Vulkan
executable; a source build is required, exactly like the ExecuTorch situation.

## Isolation

Installed into a dedicated venv so the system Python (and therefore the TruFor
CPU baseline environment) is untouched:

```bash
apt-get install python3-venv                      # venv module was missing
python3 -m venv --system-site-packages vendor/tvm-venv
vendor/tvm-venv/bin/pip install apache-tvm==0.27.0
```

`--system-site-packages` lets the venv reuse `onnx`, `PIL`, `numpy` from the
system without reinstalling them; `apache-tvm` itself and its own dependency
`apache-tvm-ffi` live only in the venv.

## Source build prerequisites (verified, not assumed)

| requirement | status |
|---|---|
| TVM source | cloned `v0.27.0` = `4d0ac168c78c1e20bf05a6ec5926760bb1c52b05` |
| LLVM | `cmake/config.cmake:125` → **"Requires LLVM version >= 15"**; `FindLLVM.cmake` has a `>= 200` (LLVM 20) branch, so 18 is in range |
| system LLVM | only `llvm-14-dev` present → not enough |
| LLVM 18 | installed from `apt.llvm.org` (`llvm-18-dev 18.1.8`) → `llvm-config-18` |
| Vulkan headers | `/usr/include/vulkan/vulkan.h` present; `libvulkan.so` from `libvulkan-dev 1.3.239` |
| Vulkan SDK | `vendor/1.4.341.1/x86_64` (glslc `shaderc v2026.1`) — already downloaded for the ExecuTorch phase |
| **SPIRV-Tools** | required by `cmake/utils/FindVulkan.cmake`; the SDK already ships it: `lib/libSPIRV-Tools.a`, `lib/libSPIRV-Tools-shared.so`, `include/spirv-tools/libspirv.h`, `include/spirv/unified1/{spirv.hpp,GLSL.std.450.h}` |

Nothing had to be built separately for SPIRV-Tools.

## Two configuration gotchas, both resolved

1. **`USE_GTEST`**: with `AUTO` (the default) `CMakeLists.txt:419` aborts on this
   machine — Debian's GTest 1.12.1 defines `GTest::gtest` but not the legacy
   `GTest::GTest` target the file looks up. Fixed with `-DUSE_GTEST=OFF`; TVM's
   C++ unit tests are not needed here.

2. **`USE_VULKAN` must be the SDK *path*, not a boolean.** `FindVulkan.cmake` does
   `if(IS_DIRECTORY ${__use_vulkan})` first, then falls back to
   `find_package(Vulkan)`. Passing `ON` makes it find the *system* loader
   (`/usr/lib/x86_64-linux-gnu/libvulkan.so`) but leaves
   `Vulkan_SPIRV_TOOLS_LIBRARY-NOTFOUND`, which CMake then treats as a hard error
   because the value is used in `target_link_libraries`. Passing the SDK
   directory fixes both, and also gives absolute include paths (a relative one
   triggers `CMake Error: Found relative path while evaluating include
   directories of "tvm_runtime_vulkan_objs"`).

## Working configure line

```bash
SDK=$PWD/vendor/1.4.341.1/x86_64
cd vendor/tvm-src/build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSE_VULKAN=$SDK \
  -DUSE_KHRONOS_SPIRV=$SDK \
  -DUSE_LLVM=/usr/bin/llvm-config-18 \
  -DUSE_GTEST=OFF
# -> Build with Vulkan support
# -> Vulkan_SPIRV_TOOLS_LIBRARY=<SDK>/lib/libSPIRV-Tools.a
```

## Reproduce the environment from scratch

```bash
apt-get install -y python3-venv g++ cmake ninja-build
curl -sSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /etc/apt/trusted.gpg.d/llvm.asc
echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-18 main" \
  > /etc/apt/sources.list.d/llvm18.list
apt-get update && apt-get install -y llvm-18-dev
git clone --recursive --branch v0.27.0 https://github.com/apache/tvm.git vendor/tvm-src
python3 -m venv --system-site-packages vendor/tvm-venv   # optional: for the wheel path
```

## Hardware (unchanged, re-verified)

```
01:00.0 AMD Baffin [Radeon RX 460/560D / Pro 450/455/460/555/555X/560/560X] (rev cf)
vulkaninfo (radeon ICD): AMD Radeon RX 460 Graphics (RADV POLARIS11)
  deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, apiVersion 1.3.230
  driverID = DRIVER_ID_MESA_RADV, Mesa 22.3.6
llvmpipe enumerates separately as deviceType = PHYSICAL_DEVICE_TYPE_CPU
```

The second ICD is software (llvmpipe) and is excluded by device type; any Vulkan
work must pin `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`
and assert `deviceType == DISCRETE_GPU` rather than trusting device index 0.
