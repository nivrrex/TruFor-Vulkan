#!/bin/bash
# Build the pure-C++ workspace reproducer against the ASan-instrumented TVM
# runtime. No Python, no tvm_ffi, so ASan's interceptors are never mixed with a
# prebuilt binding.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/vendor/tvm-src"
ASAN_BUILD="${ASAN_BUILD:-$SRC/build-asan}"
[[ -d "$ASAN_BUILD/lib" ]] || ASAN_BUILD="$SRC/build"      # fall back to the normal build
OUT="${1:-$ROOT/upstream/workspace_repro}"

INC=(-I "$SRC/include" -I "$SRC/3rdparty/tvm-ffi/include" -I "$SRC/3rdparty/tvm-ffi/3rdparty/dlpack/include")
LIBDIR="$ASAN_BUILD/lib"
echo "include: ${INC[*]}"
echo "libdir : $LIBDIR"
g++ -std=c++17 -fsanitize=address -fno-omit-frame-pointer -g -O0 \
    "${INC[@]}" \
    "$ROOT/upstream/workspace_repro.cpp" \
    -L "$LIBDIR" -ltvm_runtime -ltvm_ffi -ltvm_runtime_vulkan \
    -Wl,-rpath,"$LIBDIR" \
    -o "$OUT"
echo "built $OUT"
