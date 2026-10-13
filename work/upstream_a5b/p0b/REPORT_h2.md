# P0-b ③ H2（download 路径对称风险）结果

日期：2026-10-03。判据（摘要 P0-b ③）："无越界/验证层报错"；动作允许
"allocsize/cstep/elempack 元数据日志 **或** VK validation layer 检查"。

## 结果：通过（验证层分支 + 源码判读）

### 1) VK validation layer 确已加载且对 buffer 上传/下载路径静默 [已验证事实]

`VK_LOADER_DEBUG=all` 证实 layer 被插入 instance 与 device（`vkld.log`）：
`Insert instance layer "VK_LAYER_KHRONOS_validation"` / `Inserted device layer ...`。
在 prepatch 无恙路径（Split / Crop 迷你网，含上传与下载）上运行，验证层输出
**0 条 validation error / VUID**（`vk_split.log`、`vk_crop.log`）。
Slice 迷你网在验证层下仍 SIGSEGV 且无 validation 报错（`vk_slice.log`）
→ 该崩溃不是 Vulkan API 误用，验证层无法解释（与第二缺陷一致）。

### 2) 源码判读：download 路径与 upload 缺陷不对称 [已验证事实]

- `VkCompute::record_download`（`src/command.cpp:460+`）：设备侧 `dst_staging` 经
  `convert_packing` 得到，主机侧 `dst_fp16.create_like(dst_staging)`；随后
  `TYPE_post_download`（`:2044`）执行
  `memcpy(dst_b.data, src_b.mapped_ptr(), dst_b.total()*elemsize)`。
  **两端布局同源**（`Mat::create_like(VkMat)` 与 `VkMat::create_like(Mat)` 都按同一
  `create()` 规则算对齐 cstep，`Mat::total()=cstep*c`），故 memcpy 长度与布局一致，
  不存在 upload 那种"外部紧凑 cstep vs 对齐 staging"的不对称。
- upload 缺陷的成因是 **外部紧凑 Mat**（pyncnn 零拷贝）被送入 `record_upload`；
  download 主机侧目标 `dst_fp16` 是 ncnn 自建的 `create_like` 缓冲，永不接收外部紧凑 Mat。

### 3) 另两处无条件 `memcpy(total()*elemsize)` 的判读（补 FORK_BUILD 残留项）

- `VkCompute::record_clone(const Mat&, VkMat&)`（`command.cpp:613-638`）与
  `VkTransfer::record_upload`（`:2653,2692`）同为无条件 memcpy。
  源码判读：现调用图中，外部输入 CPU→GPU 上传走的是 **已打补丁的**
  `VkCompute::record_upload`（`net.cpp:247,3491`）；`record_clone(Mat→VkMat)` 的外部
  紧凑 Mat 调用点未找到，其余为 VkMat→VkMat 或 `create_like` 同布局；
  `VkTransfer::record_upload` 的目标 `dst.create_like(src_flattened)` 与源同布局。
  → [工程判断，置信中高，依据=调用图阅读] 二者不构成与 upload 同类的外部紧凑布局缺陷。
  未做 fork 插桩重建，故未采集 allocsize/cstep 运行时元数据（证据形式为源码判读）。

### 4) 经验旁证

② 的全网逐 blob（2651 blob × 4 尺寸，全部经 download）与 ④ 的 Split/Crop 网格
（232 次 GPU 回读，逐位一致）均已大量覆盖 download 路径，补丁轮无异常。

## 判据判定

按"或"分支，验证层检查已执行且无报错 → ③ 通过。附源码判读说明 download 无对称缺陷；
未做 fork 插桩元数据日志，如实标注。
