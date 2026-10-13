# _dev 自包含环境。所有路径基于本文件所在目录推导，禁止硬编码绝对路径。
# 用法: . ./env.sh
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export TVM_LIBRARY_PATH="$_here/vendor/tvm-src/build/lib"
export PYTHONPATH="$_here/vendor/tvm-src/python:$_here/scripts/opt"
export LD_LIBRARY_PATH="$_here/vendor/mesa-ab/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_DRIVER_FILES="$_here/vendor/mesa-ab/root/usr/share/vulkan/icd.d/radeon_icd.json"
export TVM_VULKAN_TARGET='{"kind":"vulkan","from_device":0}'
export _TRUFOR_HOME="$_here"
unset _here
