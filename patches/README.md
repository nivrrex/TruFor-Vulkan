# Patches applied to the vendored Apache TVM 0.27.0

`vendor/tvm-src` is a git clone but is excluded from this repository (17 GB, not
redistributable). These patches record every change made to it, so the working
configuration is reproducible:

```bash
cd vendor/tvm-src && git apply ../../patches/tvm-vulkan-perf-and-profiling.patch
# then rebuild:
cd build && ninja -j$(nproc)
```

`vendor/tvm-src` is otherwise byte-identical to upstream `v0.27.0`
(`4d0ac168c78c1e20bf05a6ec5926760bb1c52b05`).

Patches apply to that exact commit; they will not apply to another TVM version
without adjustment.

## 1. `src/runtime/vm/vm.cc` — per-instruction profiling

Opt-in via `TVM_VM_PROFILE=1`. Around every VM instruction, when the instrument
callback is not installed, the patch syncs the device (resolved from the first
tensor argument through `DeviceAPI::StreamSync`) on both sides and prints
`TVMPROF <func> <ms>` to stderr. Without the sync this would only measure
dispatch (~3 ms for a ~2.5 s graph), because Vulkan submission is asynchronous.

Cost when enabled: ~0.18 ms per instruction. It also serialises the queue, so
the sum of per-instruction times came out at 192% of the wall time for the full
graph (6224 instructions, 243 of them PrimFuncs). The ranking is usable; the
absolute values are inflated.

## 2. `python/tvm/s_tir/dlight/gpu/transpose.py` — crash fix

`is_transpose` compared index lists that are `tirx` expressions, so

```python
list(lhs) != list(rhs) and set(lhs) == set(rhs)
```

returned an `Expr` rather than a bool and raised
`ValueError: Cannot use and / or / not operator to Expr`, aborting the entire
`ApplyDefaultSchedule` pass on any real graph. The patch reduces the indices to
Python ints first and treats any unmodellable block as "not a transpose".

With the fix the rule runs and is a large win **on standalone transposes**:

| transpose | default | +Transpose rule | change |
|---|---:|---:|---:|
| (1,16384,512) perm(0,2,1) | 65.74 ms | 13.50 ms | **-79.5%** |
| (1,1024,1280) perm(0,2,1) | 2.19 ms | 1.52 ms | -30.4% |
| (1,4096,512) perm(0,2,1) | 3.90 ms | 2.58 ms | -34.0% |
| (1,64,128,128) NCHW→NHWC | 1.74 ms | 1.93 ms | +10.6% |

But it is **inert on the full graph**, which got 1.3% slower (2452.5 → 2485.4 ms,
243 kernels either way). `ApplyDefaultSchedule` runs *after* `FuseOps`/`FuseTIR`
in the default `gpu_generic` pipeline, so transposes are already absorbed into
fused kernels (`fused_transpose64_reshape62_transpose65_reshape` 364 ms,
`fused_transpose7_reshape8` 278 ms) and no standalone transpose block remains for
the rule to match. Using it on this graph would require moving the schedule pass
ahead of `FuseOps`, which is a separate experiment, not a flag.

## 3. Profile output format

The `TVM_VM_PROFILE` line was widened to
`TVMPROF <func_idx> <func_name> <ms> <input_tensor_shape>`.

`func_idx` is needed because several distinct PrimFuncs can carry the same name
after TIR dedup, and the name alone cannot be mapped back to a shape. The shape is
read from the first tensor argument's `DLTensor` rather than inferred from the
function name — the profile of this graph contains a `max4` whose name suggests a
4-D channel reduction but whose real signature is `(1,1,262144) -> (1,1,1)`, and
guessing from names produced a 5x error in the first analysis of it.
