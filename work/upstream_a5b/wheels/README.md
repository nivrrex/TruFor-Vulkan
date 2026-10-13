# 补丁轮 wheel（随仓库提供）

本目录随仓库提供**已构建的 ncnn 补丁轮**，便于 clone 后直接安装，无需自建 ncnn：

```bash
python -m pip install --no-deps ncnn-1.0.20261002+trufor1-cp311-cp311-linux_x86_64.whl
```

- 适用平台：**CPython 3.11 / Linux x86_64**
- wheel md5：`9e803b698951762eee691d59797dd20f`
- 内含 `.so` md5：`b0e53111dfe2a1cbbc6a8d56bae49947`
- 许可：ncnn BSD-3（补丁不改许可）
- 其他 Python 版本 / 平台：按 `../FORK_BUILD.md` 自建
- **判别是否补丁轮不要看 `ncnn.__version__`**（恒为上游串 `1.0.20261002`）；以 pip 元数据
  `+trufor1`、`.so` md5 或 `probe_ncnn_upload` 行为判据为准。
- 官方 pypi 轮带 Vulkan 上传缺陷，**不得用于 Vulkan 后端**。
