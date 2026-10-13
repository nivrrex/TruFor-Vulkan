/*
 * Minimal C++ reproducer for the suspected TVM Vulkan workspace lifetime bug.
 *
 * No Python, no PyTorch, no ONNX, no model. This links only libtvm_runtime.so
 * (the ASan build) and drives the workspace C API directly, which is the exact
 * call path gdb showed aborting:
 *
 *     TVMBackendFreeWorkspace
 *       -> DeviceAPI::FreeWorkspace
 *         -> tvm::runtime::vulkan::VulkanDeviceAPI::FreeWorkspace
 *           -> pool_per_thread.Get()          <-- ThreadMap hash lookup
 *
 * The asymmetry under test (vulkan_device_api.cc vs cpu_device_api.cc):
 *
 *     alloc : pool_per_thread.GetOrMake(kDLVulkan, this)   // creates if missing
 *     free  : pool_per_thread.Get()  + TVM_FFI_ICHECK     // looks up only
 *
 * while the CPU device uses one thread-local accessor for both.
 *
 * Build (see build_cpp_repro.sh):
 *   g++ -fsanitize=address -g ... repro.cpp -ltvm_runtime
 */

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>

#include <atomic>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

// In TVM 0.27 the DLPack enums (kDLVulkan, kDLUInt, ...) live at global scope,
// and tvm::Device is the alias for DLDevice; tvm::runtime::DLDevice no longer
// exists.
using tvm::Device;

namespace {

constexpr int kDev = 0;
std::atomic<int> g_failures{0};

void check(bool ok, const std::string& what) {
  if (ok) {
    printf("    [ok]   %s\n", what.c_str());
  } else {
    printf("    [FAIL] %s\n", what.c_str());
    g_failures.fetch_add(1);
  }
  fflush(stdout);
}

void* alloc(size_t nbytes) { return TVMBackendAllocWorkspace(kDLVulkan, kDev, nbytes, kDLUInt, 8); }

// TVM raises a tvm::ffi::Error (ICHECK failure) rather than returning non-zero
// for several workspace misuse patterns. Capture it so one case does not abort
// the whole run, and report it explicitly.
int free_ws(void* p) {
  try {
    return TVMBackendFreeWorkspace(kDLVulkan, kDev, p);
  } catch (const std::exception& e) {
    printf("      (tvm error: %s)\n", e.what());
    return -2;
  }
}

// ---------------------------------------------------------------- A1
void test_a1_single_thread() {
  printf("A1  single-thread alloc/free\n");
  void* w = alloc(4096);
  check(w != nullptr, "Alloc returns non-null");
  check(free_ws(w) == 0, "Free returns 0");

  void* a = alloc(1024);
  void* b = alloc(65536);
  check(a && b, "two live allocations");
  check(free_ws(b) == 0 && free_ws(a) == 0, "free in reverse order");

  // LIFO, the pattern the pool's quick path expects
  std::vector<void*> v;
  for (int i = 0; i < 64; ++i) v.push_back(alloc(1024 * (1 + (i % 8))));
  int rc = 0;
  for (size_t i = v.size(); i-- > 0;) rc |= free_ws(v[i]);
  check(rc == 0, "64 nested alloc/free, LIFO");
  printf("\n");
  fflush(stdout);
}

// ---------------------------------------------------------------- A2
void test_a2_cross_thread_free() {
  printf("A2  cross-thread free (the hypothesis under test)\n");
  for (int trial = 0; trial < 64; ++trial) {
    void* w = alloc(8192);
    if (w == nullptr) {
      check(false, "alloc before cross-thread free");
      return;
    }
    int rc = 1;
    std::thread t([&] { rc = free_ws(w); });
    t.join();
    if (rc != 0) {
      check(false, "cross-thread free failed on trial " + std::to_string(trial) +
                        " (rc=" + std::to_string(rc) + ")");
      return;
    }
  }
  check(true, "64 cross-thread alloc(thread A)/free(thread B) pairs");
  printf("\n");
  fflush(stdout);
}

// ---------------------------------------------------------------- A3
// Many threads each allocating, then a different thread freeing, in a
// deliberately unhelpful order: not the allocating thread, and not LIFO.
void test_a3_stress() {
  printf("A3  multi-thread stress, cross-thread and out-of-order free\n");
  constexpr int kThreads = 8;
  constexpr int kPer = 64;
  std::vector<std::vector<void*>> owned(kThreads);

  auto worker = [&](int id) {
    for (int i = 0; i < kPer; ++i) {
      // mixed sizes so the pool has to grow, shrink and re-search free lists
      size_t n = 4096 * (1 + ((id * 7 + i * 13) % 17));
      owned[id].push_back(alloc(n));
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < kThreads; ++i) ts.emplace_back(worker, i);
  for (auto& t : ts) t.join();
  ts.clear();

  size_t total = 0;
  for (auto& v : owned) total += v.size();
  check(total == (size_t)kThreads * kPer, "all allocations succeeded (" + std::to_string(total) + ")");

  // free from a *different* thread than the allocator, interleaved order
  int rc = 0;
  ts.emplace_back([&] {
    for (int i = 0; i < kThreads; ++i) {
      for (int j = 0; j < kPer; ++j) {
        int owner = (i + j) % kThreads;
        rc |= free_ws(owned[owner][j]);
      }
    }
  });
  for (auto& t : ts) t.join();

  check(rc == 0, "cross-thread, interleaved frees all returned 0");
  printf("\n");
  fflush(stdout);
}

// ---------------------------------------------------------------- A4
// Free on a thread that never allocated at all.
void test_a4_free_without_alloc() {
  printf("A4  free on a thread that never allocated\n");
  void* w = alloc(4096);
  int rc = 1;
  std::thread t([&] { rc = free_ws(w); });
  t.join();
  check(rc == 0, "free on a fresh thread (Get() must find an entry or throw)");
  printf("\n");
  fflush(stdout);
}

// ---------------------------------------------------------------- A5
// The dangerous variant of A2: the freeing thread has already allocated a
// workspace of its own, so VulkanDeviceAPI's
//   TVM_FFI_ICHECK(pool) << "Attempted to free a vulkan workspace on a
//       CPU-thread that has never allocated a workspace";
// PASSES, and the foreign pointer is then handed to the *wrong* pool's
// WorkspacePool::Free(), which indexes that pool's own std::vector<Entry>
// bookkeeping. This is the path that can corrupt the heap instead of
// producing a clean error.
void test_a5_cross_thread_with_own_pool() {
  printf("A5  cross-thread free where the freeing thread HAS its own pool\n");
  for (int trial = 0; trial < 8; ++trial) {
    void* foreign = alloc(8192);           // allocated on the main thread
    void* mine = nullptr;
    int rc = 1;
    std::thread t([&] {
      mine = alloc(4096);                   // give this thread its own pool
      rc = free_ws(foreign);                // then free the foreign pointer
    });
    t.join();
    if (rc != 0) {
      check(false, "foreign free returned " + std::to_string(rc) + " on trial " +
                        std::to_string(trial));
      if (mine) free_ws(mine);
      return;
    }
    if (mine) free_ws(mine);
  }
  check(true, "8 foreign frees into a non-empty pool all returned 0");
  printf("\n");
  fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::string which = argc > 1 ? argv[1] : "all";
  printf("TVM Vulkan workspace reproducer (C++, ASan build, no Python)\n");
  printf("  device_type=%d device_id=%d\n\n", (int)kDLVulkan, kDev);

  if (which == "all" || which == "A1") test_a1_single_thread();
  if (which == "all" || which == "A2") test_a2_cross_thread_free();
  if (which == "all" || which == "A3") test_a3_stress();
  if (which == "all" || which == "A4") test_a4_free_without_alloc();
  if (which == "all" || which == "A5") test_a5_cross_thread_with_own_pool();

  printf("failures: %d\n", g_failures.load());
  return g_failures.load() == 0 ? 0 : 1;
}
