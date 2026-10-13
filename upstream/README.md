# Upstream material

| file | status |
|---|---|
| `workspace_repro.cpp` + `build_cpp_repro.sh` | **usable.** Real TVM issue: cross-thread workspace alloc/free is unsupported by the Vulkan device API. Reproduces identically on Mesa 22.3.6 and 25.0.7, is not a driver artefact, and is a good robustness bug to report. Pure C++, ASan build, no Python. |
| `tvm-vulkan-take-select-repro.py` | **driver-sensitive.** On Mesa 22.3.6 it shows 9/15 wrong; on Mesa 25.0.7 it is 15/15 correct. Not a TVM bug. Useful only as a driver-A/B canary. |
| `ISSUE-take-select.md` | **DO NOT SUBMIT to apache/tvm.** Written while the root cause was still unknown; the behaviour it describes is caused by Mesa/RADV 22.3.6, not by TVM. Kept only as a record of the investigation. |

Verified with: TVM 0.27.0 (built from source, `USE_VULKAN=ON`), AMD RADV POLARIS11,
Mesa 22.3.6 and 25.0.7, LLVM 18.1.8, Vulkan SDK 1.4.341.1.
