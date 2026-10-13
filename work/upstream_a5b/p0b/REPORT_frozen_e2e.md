# 补丁轮下冻结包 e2e（CPU-only / GPU 两态）+ wheel 分发落地

日期：2026-10-03。样本 `ctrl512_r0.01.png`（512×512）；`--heatmap-only`；weights 以
symlink 指向 `下游应用/src/weights`。PyInstaller 6.22.3。

## CPU-only 态（构建环境 `ncnn_pypi_env`：ort-CPU 1.29 + 补丁轮 ncnn + PyInstaller）

- 包体 324MB；`_internal/ncnn/ncnn.cpython-311-*.so` md5 = `b0e53111…`（补丁轮已入包）。
- `--trufor-probe`：vulkan ok、upload ok（device_index 1）。
- explicit cpu：heatmap md5 `58d5a76e982b4ce12068f689940d001e`
  = **系统 python3 参考逐位一致**（也与 S8 记录一致）。
- explicit ncnn：`backend_used=ncnn`、exit 0，heatmap md5 `76938cbb087c6401d071e3dc114d47c7`。
- auto：`backend_used=ncnn`（补丁轮入包 + 探针 PASS → 命中 ncnn，非降级）。

## GPU 态（构建环境 `ortgpu_env`：onnxruntime-gpu 1.23.2 + 补丁轮 ncnn + PyInstaller）

- 包体 4.0GB；`_internal/ncnn/...so` md5 = `b0e53111…`（补丁轮已入包）。
- `--trufor-probe`：cuda ok（providers 含 CUDAExecutionProvider）、vulkan ok、upload ok。
- explicit cuda：heatmap md5 `d24b680c17c4ccc20925ad5347484f6b`
  = **ortgpu_env 参考逐位一致**（也与 S8 记录一致）。
- auto：`backend_used=cuda`。

## wheel 随包分发（落地）

新增 `下游应用/packaging/bundle_ncnn_wheel.sh`：把补丁轮放入 `<distdir>/ncnn_wheels/`，
**分发前强制核对 md5 = `9e803b698951762eee691d59797dd20f`**，不符即拒绝。
负向测试：喂 stock 轮（md5 `9772dc43…`）→ 退出码 1、拒绝复制。冻结态已内置 ncnn，
本步骤只服务非冻结（pip）部署。

## 结论

补丁轮下冻结包两态 e2e 全部通过；冻结包内 ncnn 确为补丁产物（.so md5 核验）；
wheel 分发路径可复现且拒收官方轮。仍未出正式部署包（dual-package 待定）。
