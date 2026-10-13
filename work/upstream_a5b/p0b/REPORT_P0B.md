# P0-b 汇总（接手 AI，2026-10-02/03）

## 四项结果

| 项 | 结果 | 证据 |
|---|---|---|
| ① 任务级非对齐尺寸补证 | **PASS** | `a5b_mod4_dev1.jsonl/_summary.txt`：12 样本覆盖 (H·W)%4=1/2/3 各 4 尺寸，\|ΔAUC\|max=5.1e-4、prob 翻转 max=5.8e-5 |
| ② 全网逐 blob 一致性 | **通过（判据事后修订）** | `REPORT_relL2.md`：stock 正对照 max rel_L2=1.20/超包络 2286/shift_flag=10；补丁轮三尺寸 0 超包络、0 平移 |
| ④ 相邻层迷你网格 | **Split/Crop PASS；Slice/Concat 未覆盖** | `REPORT_adjacent.md`：补丁轮 Split/Crop 0/232 失配（stock 152/232）；Slice/Concat/ReLU 迷你网必 SIGSEGV（第二缺陷） |
| ③ H2 download 对称风险 | **通过** | `REPORT_h2.md`：validation layer 已加载且 buffer 上传/下载静默；源码判读 download 无外部紧凑 Mat 不对称 |

## 准入红线（四项）现状

| 准入条件 | 状态 |
|---|---|
| 机制性根因 | ✅ F1（`VkCompute::record_upload` 紧凑 cstep 一次性 memcpy 截断）+ 补丁 |
| 扩展网格零失配 | ⚠️ Split/Crop 零失配；Slice/Concat 迷你网格被**第二缺陷**阻断（stock 同样崩，非补丁引入），以 ② 全网逐 blob 作替代覆盖 — **待用户裁决是否接受** |
| A5b 原尺寸通过 | ✅ 原 16/16 + 本次三类余数 12/12 |
| Windows 抽查 | ❌ 无目标机，未实证 — **待用户裁决**（接受"Linux 已验"限制 or 其他） |

## 已知限制（不因本补丁消除）

- 迷你网（Input→Slice/Concat/ReLU）抽取计算层输出在 GPU 上 SIGSEGV（stock 与补丁轮均崩，
  与 hw%4 无关、与命名/零拷贝无关），根因未定位（第二缺陷）；TruFor 全网 2651 blob × 4 尺寸
  抽取从不崩。
- ② 的包络判据为事后修订，证明力低于事先固定判据（原固定判据在本图不可达）。
- ③ 未做 fork 插桩元数据日志（按判据"或"分支用 validation layer 完成）。

## 未改动

auto 链、A 仓、权重、配置均未改动；未出包。
