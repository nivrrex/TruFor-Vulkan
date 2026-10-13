# ncnn fork 修复轮构建与分发链（+trufor1）

日期：2026-10-02。全部动作本地完成，零外部发布。

## 缺陷与补丁

- 缺陷：`VkCompute::record_upload`（`src/command.cpp:414`）一次性 memcpy 假设
  host Mat 与 staging VkMat 平面步长一致；pyncnn 零拷贝包 numpy 时 cstep 为紧凑
  值（`python/src/main.cpp:371-372`），`hw*elemsize` 非 16B 对齐且 c≥2 时
  GPU 输入平面平移损坏（A5b b22 样本 rel_L2 0.23、ΔAUC 0.18 的根因）。
- 补丁：单文件 `src/command.cpp`，步长不等时逐平面拷贝。
  - 提交：`ncnn_upstream/ncnn` 本地树 `eab8352`（基于 master `9f9d4ec`，未 push）。
  - 导出：`0001-fix-vulkan-record_upload-for-compact-layout-external.patch`（本目录）。
  - 子模块（本次构建所用）：`3rdparty/glslang @ 780b6a2`、`python/pybind11 @ d03662f`；
    产物 `.so` md5 = `b0e53111dfe2a1cbbc6a8d56bae49947`（见下方验证表）。
- pypi 官方轮（1.0.20260526）与未打补丁的 master 构建均带此缺陷，
  **不得作为 ncnn-Vulkan 后端的生产分发物**；本目录 wheel 为修复版。

## 构建命令（Linux x86_64 / CPython 3.11）

```bash
# 1) 打补丁
cd <ncnn 源码树（master 9f9d4ec）>
git apply <本目录>/0001-*.patch          # 或 cherry-pick eab8352

# 2) 构建扩展（已有 cmake 配置时增量）
cd python && pip install .              # 产生 build/temp.*/、build/lib.*/（内含打补丁 .so）
#   cmake 侧参数随 python/CMakeLists.txt 默认（NCNN_VULKAN=ON 视机器）

# 3) 出 wheel（版本打本地标识，构建后还原 setup.py）
cp ../build/lib.linux-x86_64-cpython-311/ncnn/ncnn.cpython-311-*.so ncnn/
sed -i 's/1.0.20261002"/1.0.20261002+trufor1"/' setup.py
python setup.py bdist_wheel
# 产物 dist/ncnn-1.0.20261002+trufor1-cp311-cp311-linux_x86_64.whl（约 10.5MB）
```

注意：`ncnn.version()`/`__version__` 运行时打印仍是上游串 `1.0.20261002`
（编译期宏，不含 local 段）；区分补丁与否以 pip 元数据
（`pip show ncnn` → `1.0.20261002+trufor1`）或行为判据为准。

## 仓库内随附 wheel（clone 即用）

本仓库已在 `work/upstream_a5b/wheels/` 随附构建产物
`ncnn-1.0.20261002+trufor1-cp311-cp311-linux_x86_64.whl`（约 10.1MB，wheel md5
`9e803b698951762eee691d59797dd20f`，内含 `.so` md5 `b0e53111…`），
适用于 **CPython 3.11 / Linux x86_64**：

```bash
git clone <本仓库> && cd <本仓库>
python -m pip install --no-deps work/upstream_a5b/wheels/ncnn-1.0.20261002+trufor1-*.whl
```

其他 Python 版本/平台仍需按上节自建；详见 `wheels/README.md`。

## 验证（均实测，2026-10-02）

| 项 | 结果 |
|---|---|
| wheel 内 .so md5 = 全部实验所用构建产物 | `b0e53111dfe2a1cbbc6a8d56bae49947` |
| `micro/repro_split_pack4.py`（wheel 独立 venv，P4） | 全网格零损坏（77 个原损坏组合全净） |
| A5b 16 样本原尺寸任务级 vs ORT-CPU | P4 与 RX460 各 16/16 PASS（`../ablation_results/a5b_patched_native{,_dev0}.jsonl`） |
| 同设备跨 fresh 进程逐位 | PASS（P4/RX460 抽测均逐位一致） |
| 跨设备 | 非逐位一致（prob rel_L2≈3e-5~3e-4，conf max_abs≤9.5e-3），与既有分设备 golden 口径一致 |
| hw%4==0 尺寸 | 补丁不触发新分支（步长相等走原路径），既有基线/golden 全部继续有效 |

## 残留问题（不阻塞本分发链）

- 迷你 net（Input+Split+Conv / Input+ReLU）extract 实算层输出 SIGSEGV，补丁后
  仍在；TruFor 全网同形状不崩。判定为独立第二缺陷，未定位，见
  `micro/ISSUE_draft.md` "Separate issue"。
- image 上传路径（`command.cpp:614`）与 flatten 路径（2653/2692）疑似同类假设，
  未验证（本验证只走 buffer 路径）。
- Windows 构建/分发实证挂起到目标机阶段；wheel 平台标签 cp311/linux_x86_64，
  其余 Python 版本/平台需按上节命令重出。
- License：ncnn 本体 BSD-3 未改（补丁不动 LICENSE），上游权重非营利限制另行处理。

## 建议的上游动作

- `micro/ISSUE_draft.md` 已按定案机制重写（含建议补丁代码），是否提交由用户决定。
- fork 若长期自用：保留单 commit 便于 rebase master；一旦上游合入等效修复即弃用本 fork。

- 2026-10-13 复核：上游 master 对 `src/command.cpp` 的最近改动为 `201145bd`
  （fix crash reading past end of memory-mapped model file, #7021），与本补丁目标路径无关
  → **0001 仍然需要**。
