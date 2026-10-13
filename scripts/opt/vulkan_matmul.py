"""B2: a Matmul schedule config for the Vulkan target.

`dlight/gpu/matmul.py:get_configs` only special-cases cuda, rocm and one
opencl profile. Vulkan falls into the `else` branch and gets the dataclass
defaults:

    block_size_x=8, block_size_y=8, micro_size_x=4, micro_size_y=4,
    micro_size_k=8, vector_size=1, use_shared=True

micro_size_x * micro_size_y is the number of accumulators per thread, so the
default schedule gives 16. The direct-conv sweep in this project found the
optimum at 64 (1086 GFLOPS) and that 32 sits between (acc>=128 spills to
scratch and collapses). Applying the same reasoning to Matmul is therefore the
obvious move, and this module exposes it as a rule that can be swapped into the
pipeline in place of dl.gpu.Matmul().

VULKAN_MICRO is the scan point: (mx, my) sets accumulators per thread as
mx * my, and micro_size_k is kept at 8 to match the default until proven
otherwise.
"""

from tvm.s_tir.dlight.gpu.matmul import Matmul
from tvm.target import Target


def make_rule(mx, my, mk=8, bx=8, by=8, vx=1, vy=1, vector_size=1, unroll=256,
              use_shared=True, storage_align=False, inner_x=False, storage_align_pad=0):
    class _VulkanMatmul(Matmul):
        def get_configs(self, target):
            return Matmul.Config(
                block_size_x=bx, block_size_y=by, vthread_x=vx, vthread_y=vy,
                micro_size_x=mx, micro_size_y=my, micro_size_k=mk,
                vector_size=vector_size, unroll=unroll, use_shared=use_shared,
                storage_align=storage_align, inner_x=inner_x,
            )

    return _VulkanMatmul()
