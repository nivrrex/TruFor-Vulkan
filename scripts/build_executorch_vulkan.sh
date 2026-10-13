#!/bin/bash
# Build ExecuTorch v0.6.0 with the Vulkan delegate linked into portable_lib.
# Re-runnable: every prerequisite is derived, nothing is hard-coded to a user path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ET_SRC="${ET_SRC:-$ROOT/vendor/executorch}"
SDK="${VULKAN_SDK:-$ROOT/vendor/1.4.341.1/x86_64}"
BUCK2="${BUCK2:-$ROOT/vendor/buck2-bin/buck2-edae27cfca00053d9c5f7c7be81b6b0d7d07573a50be374ce53a9d8692afa5fc}"
LOG="${1:-$ROOT/reports/et-build.log}"

export PATH="$SDK/bin:$PATH"
# ExecuTorch 0.6.0 vendors gflags with cmake_minimum_required < 3.5, which CMake 4 rejects.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export BUCK2
# setup.py's get_python_lib() returns Debian's /usr/lib/python3/dist-packages, where
# torch is not installed; point find_package(Torch) at the real one.
export CMAKE_PREFIX_PATH="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')"
# PYBIND=ON is what makes portable_lib (and therefore the delegate) get built.
export CMAKE_ARGS="-DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_PYBIND=ON"

command -v glslc >/dev/null || { echo "glslc not on PATH" >&2; exit 1; }
test -x "$BUCK2" || { echo "buck2 not executable: $BUCK2" >&2; exit 1; }

cd "$ET_SRC"
pip3 install . --no-build-isolation --no-deps -v >"$LOG" 2>&1
echo "PIP_EXIT=$?"
