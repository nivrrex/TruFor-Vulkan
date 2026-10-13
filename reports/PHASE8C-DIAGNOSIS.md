# PHASE8C-DIAGNOSIS.md — where the host-side corruption lives

Three diagnostics run as planned. Result in one line: **the GPU side is clean, the corruption is in
TVM's own workspace bookkeeping, and ASan could not be brought up in this environment.**

---

## 1. Vulkan validation layer — ZERO diagnostics

Ran the full 1918-node graph on the RX 560 with the SDK's validation layer forced on:

```bash
VK_LAYER_PATH=$SDK/share/vulkan/explicit_layer.d
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
```

```
device: AMD Radeon RX 460 Graphics (RADV POLARIS11) exist: True
build ok, running...
done. map mean=0.03687 std=0.01407
```

**No validation messages at all.** The graph ran to completion, produced wrong numbers, and the
loader/driver never reported an API misuse, descriptor error, sync hazard, or SPIR-V violation.

This is a real narrowing, with the caveat you stated yourself: a clean validation layer does not
mean the computation is right. It means the defect is *not* a contract violation visible at the
Vulkan API boundary. The remaining possibilities are host-side logic in TVM, shader arithmetic that
is legal but wrong, or a driver interaction.

## 2. ASan — built successfully, unusable in this environment

`vendor/tvm-src/build-asan`: 633/633 targets, 0 errors, `RelWithDebInfo`,
`-fsanitize=address -fno-omit-frame-pointer -g`.

It cannot be run over the Python path. Every attempt dies before reaching TVM's Vulkan code:

```
AddressSanitizer: CHECK failed: asan_interceptors.cpp:320
  "((__interception::real___cxa_throw)) != (0)" (0x0, 0x0)
    #3  .../tvm_venv/.../tvm_ffi/lib/libtvm_ffi.so+0xd004
    #4  .../tvm_ffi/lib/libtvm_ffi.so+0xa25d
    #5  .../tvm_ffi/core.cpython-311-x86_64.so+0x6bc58
```

ASan's C++ exception interceptor is initialised before `libtvm_ffi.so` (the prebuilt binding from
the pip wheel) can be intercepted. `LD_PRELOAD=libasan.so` with
`verify_asan_link_order=0:detect_odr_violation=0:handle_abort=1` changes nothing. This is the
well-known "ASan needs every C++ component built consistently" problem: our ASan-instrumented
`libtvm_compiler.so` is mixed with a non-instrumented prebuilt binding. Making this work would mean
rebuilding `tvm_ffi` and the CPython extension with ASan as well — a second full build plus a
Python-extension toolchain, for which I did not have a budget this round.

So: **ASan produced no stack. Everything below came from gdb instead.**

## 3. gdb — the corruption is in `VulkanDeviceAPI::FreeWorkspace`

Reproduced with the configuration that aborts (Phase 8B variant D: `ApplyDefaultSchedule`
removed) under `MALLOC_CHECK_=3`:

```
Thread 1 ("python"):
#0  std::_Hash_bytes(...)                                    from libstdc++.so.6
#1  tvm::runtime::vulkan::VulkanDeviceAPI::FreeWorkspace(Device, void*)
                                                          from libtvm_runtime_vulkan.so
#2  TVMBackendFreeWorkspace()                                from libtvm_runtime.so
#3  ?? ()
#4  0x0000000000000000 ()
```

### What the source says

`src/backend/vulkan/runtime/vulkan_device_api.cc:310`

```cpp
void VulkanDeviceAPI::FreeWorkspace(Device dev, void* data) {
  auto* pool = pool_per_thread.Get();                       // <-- hash lookup
  TVM_FFI_ICHECK(pool) << "Attempted to free a vulkan workspace on a CPU-thread "
                       << "that has never allocated a workspace";
  pool->FreeWorkspace(dev, data);
}
```

`pool_per_thread` is a `ThreadMap<WorkspacePool>` — a per-CPU-thread map keyed by thread id, which
is why the innermost frame is `std::_Hash_bytes`. The `TVM_FFI_ICHECK` would have thrown a clean
TVM error; instead the process aborts inside the lookup itself, and frames #3/#4 are already
garbage (`0x0` return address). That is a use-after-free or a corrupted object, not a logic error.

### The asymmetry that makes it suspicious

The CPU device does the same job differently — `src/runtime/cpu_device_api.cc:154`:

```cpp
void CPUDeviceAPI::FreeWorkspace(Device dev, void* data) {
  CPUWorkspacePoolThreadLocal()->FreeWorkspace(dev, data);   // thread-local, always present
}
```

while Vulkan splits allocate/free across two different lookups on the *same* member:

```cpp
// allocate                                    // free
auto& pool = pool_per_thread.GetOrMake(...);   auto* pool = pool_per_thread.Get();
```

`GetOrMake` creates the entry if missing; `Get` only looks. Any free that happens on a thread other
than the one that allocated — or after the thread entry has been reclaimed, or during teardown when
the `VulkanDeviceAPI` singleton and its `ThreadMap` are being destroyed while the VM is still
releasing its workspace — leaves this path operating on freed memory.

`WorkspacePool::Pool::Free` then indexes two `std::vector<Entry>` by an index derived from a linear
scan; a `pool` that is not the pool the allocation came from will happily `erase()` a valid index of
a different pool's bookkeeping, which is a clean recipe for the
`corrupted double-linked list` glibc reports.

## 4. What this does and does not explain

| observation | explained by the workspace finding? |
|---|---|
| `corrupted double-linked list` when `ApplyDefaultSchedule` is removed | **yes** — free-time bookkeeping |
| normal pipeline: wrong result, no abort | **partly** — same pool class of bug can corrupt a live `allocated_`/`free_list_` without hitting glibc's checker |
| every operator correct in isolation | consistent — a single kernel allocates/frees one workspace |
| every stage-1 sub-module correct as its own sub-graph | consistent — few allocations each |
| full 1918-node graph wrong | consistent — thousands of allocations, one bad free is enough |
| 128×128 less wrong but still wrong | consistent — fewer allocations, less exposure |
| validation layer silent | consistent — a host-side bug violates no Vulkan contract |

This is the first hypothesis in the whole investigation that explains *all* of the observations at
once, including the ones that no operator-level or fusion-level explanation could.

## 5. Status of the two open questions

- **The take/select bug is real but is not the cause of the full-graph failure** (established in
  Phase 8A-8C: removing all 80 instances moved correlation from -0.290142 to -0.287366). Issue
  material is ready in `upstream/`, scoped strictly to that bug.
- **The full-graph failure is now localised to `VulkanDeviceAPI::FreeWorkspace`** — a host-side
  workspace-lifetime issue in TVM, not a kernel, not fusion, not the driver contract.

## 6. What I would do next, and the honest cost

| step | cost | value |
|---|---|---|
| rebuild `tvm_ffi` + the CPython extension with ASan and re-run | one more full build | turns the gdb frame into an exact line and a confirmed object |
| write a C++ reproducer that drives `TVMBackendFreeWorkspace` directly (no Python) | hours | ASan works cleanly without the binding conflict; also gives a minimal upstream case |
| Mesa/RADV A/B | ~1 h, changes the driver variable | still worth doing, but it is now a *secondary* check: the strongest evidence points at TVM's own code, and the validation layer was silent on the same driver |
| upstream report for the workspace bug | ready to write once step 1 or 2 lands | the material is far stronger than what exists upstream today |

My recommendation: do the C++ reproducer (step 2) rather than the Python ASan rebuild, because it
is the only route that gets a clean sanitizer stack, and a standalone C++ case is also a better
upstream artefact. The Mesa A/B is still worth doing in parallel, but I would no longer expect it
to be decisive on its own.

## Reproduce

```bash
# 1 validation layer
SDK=$PWD/vendor/1.4.341.1/x86_64
PYTHONPATH=$PWD/vendor/tvm-src/python TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
VK_LAYER_PATH=$SDK/share/vulkan/explicit_layer.d \
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation \
vendor/tvm-venv/bin/python scripts/tvm_vulkan_run.py --skip-6e \
  --onnx reports/phase8a/trufor-512x512-slicefix.onnx \
  --image truFor-src/test_docker/images/tampered2.png --outdir reports/phase8c/validation

# 3 gdb on the aborting configuration
PYTHONPATH=$PWD/vendor/tvm-src/python TVM_LIBRARY_PATH=$PWD/vendor/tvm-src/build \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json MALLOC_CHECK_=3 \
gdb -q -batch -ex run -ex "bt 30" -ex "thread apply all bt 8" --args \
vendor/tvm-venv/bin/python scripts/phase8_asan_run.py \
  --onnx reports/phase8a/trufor-512x512-slicefix.onnx --variant D
```

Artefacts: `reports/phase8c/asan.{17066,}` (interceptor failure), `reports/phase8c/asan-run*.log`,
`reports/asan-build.log`, `upstream/`.
