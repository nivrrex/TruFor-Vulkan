# work/STATE.md

**最后更新**：中断后先读这个文件。规则见 `work/TODO.md` 与 `AGENTS.md`。

## 交付状态

**环境已迁移到 _dev（自包含：vendor/tvm-src+mesa-ab、venv、env.sh 均在 _dev 下，
不依赖原工作区），机器也从 RX560 4G 换成 RX460 2G + P4 8G。以下数字为 _dev/RX460 实测。**

**最新交付（GEMM 接入后）**：整图 **853.3 ms**（3 图各 15 次 median：853.5 / 853.3 /
853.4，IQR 3.4-5.4），**60 处替换**（30 kernel + 新增 30 个 GEMM）。
GEMM 前基线 906-907 ms，GEMM 省 **~53 ms**（SKIP 上限 86.7 ms 的 61%，落在旧估 35-50 上沿）。
正确性（验收 = 对现场构建的零替换图）：worst rel_L2 **1.29e-06**、map_corr 1.0000000，
与 GEMM 接入前逐位一致（每个 GEMM 候选在 swap 内 rel_L2=0.00e+00）。

**_dev/RX460 同 session 合法比值（2026-10-01，重测）**：
- stock GPU（TRUFOR_HANDWRITTEN=0 DECODER=0，2 次）median **2455.7 / 2463.4**，IQR 11-17
  （与旧 RX560 的 2472.7 几乎一致，GPU 侧跨环境稳定）。
- **同 session 同构建路径 GPU 比 = 2463.4 / 853.3 = 2.887×**（这是本机主口径）。
- ORT CPU 本机 **Ryzen 5 2600（6C/12T）**：unpinned 2864.8/2997.5，pinned 0-3 反而
  3047.2/3087.6，median ~2950、IQR 高达 40-139。**这比旧机器的 2129.4 慢 30%+，
  原因是 CPU 换代不同**（旧 2129 不得再引用）。CPU 侧比 ~3.4×，但本机 CPU 测量噪声大，
  只作参考、不作主比值。

**P4 8G vs RX460 2G 同 pipeline A/B（2026-10-01）**：
- P4 stock（VK_DRIVER_FILES=nvidia_icd，`from_device:0`→Tesla P4，Vulkan 1.3）
  3 图各 15 次 median **1454.1 / 1465.2 / 1456.9**，IQR 2.4-3.3（比 RX460 stock 快 **1.6×**，
  P4 算力确实更强）。
- **P4 delivered（60 替换）median 4947.9 / 4935.9 / 4935.6，比 P4 stock 慢 3.4×，
  比 RX460 delivered 慢 5.8×**。手写 kernel 在 NVIDIA Vulkan 上不但没收益、还严重回退。
- **撤回**：曾把"P4 上 11 个候选被跳过"归因于驱动语义不同。实测两侧完全一致的 11 条
  `跳过 channel max mean × 4` 和 `跳过 GELU layout × 7`，两臂都是 rel_L2 5e+01~5e+02。
  swap_conv 按 (1,C,H,W)→(1,C,1,1) 形状过匹配 MEAN kernel，用 channel_MAX 候选去替换
  当然失败；GELU layout 同理（over-match 到不做 erf 的纯 transpose）。这不是 kernel 有
  bug，是签名不够严；`_verify_new` 拒得对。
- 顺带修了 `scalar_reduce.py` 里 `sh[t + st]` 的越界读（非 reader 线程 t+st ≥ NT，
  SPIR-V OpSelect 两边都求值；Mesa 静默容忍，NVIDIA 未必），改成 `sh[T.min(t+st, NT-1)]`。
  RX460 交付重跑 median 865.7/867.1/865.7（原 853，+13 ms 属噪声，替换集完全一致，
  channel-max 验证仍然 0.00e+00）；P4 独立单测 5 个 shape 全 0.00e+00。
- **结论：不切换生产默认设备。** P4 需要针对 NVIDIA Vulkan 语义（warp_size=1、
  max_threads_per_block=256、shared-mem 48 KB）重做 kernel 调优才可能吃到收益；工作量显著。
  RX460 上 853 ms 保持。

（旧环境 RX560 4G 存档）整图 **889.4 ms**（原 stock **2472.7 ms**），**同 session 同构建路径比 2.780×**，越过 ≤900 ms 理想线。
跨 ORT CPU 独立进程（median 2129.4 ms）为 **2.394×**。
（旧的 1.37× 是跨 session 错值，已撤回；同 session 旧值是 1.20×。）
正确性（**验收口径 = 对现场构建的零替换图**）：3 张真实图片 rel_L2 = 3.07e-07 / 5.66e-07 / 1.52e-07，map_corr 1.0000000。
**注意**：此前报告的"整图 rel_L2 = 0.00e+00"从未被测过，两个 0 分别来自
26-替换 vs 25-替换（只差 max4）与 tiled-vs-scalar（纯置换），二者确实逐位一致但都不是"对未替换图"。
**多 tile 吞吐**（`scripts/opt/tile_throughput.py`，不改生产路径）：
纯 CPU 0.448 tile/s，纯 GPU 0.683 tile/s，**CPU+GPU 流水线 1.003 tile/s = 2.24× 于纯 CPU**，
逐 tile 输出 0/8 不一致。ORT 用 4 线程优于 3 线程；GPU 等待用 sleep 优于 spin（4 核机器）。
**det_err 已废弃**：它在所有 kernel 改动下恒为 2.46e-05，包括输出完全错的情况。

## M2/B 嵌入决策（2026-10-02，外部评审后采纳）

- **决策：暂缓** 下游应用完整 in-process 嵌入 TVM/Vulkan 路径（选项1）；**不做选项2 PoC**——单独解 C4（冷启动缓存）不改变决策，C2/C3/C5 仍无低成本解。否决方案：原 M2"内嵌 TVM+手写 kernel"不实施。依据：评审材料 `(本地路径)/M2_B_评审材料.md` 全文 + 外部评审结论（附于该材料末尾）。
- 后续动作（全部不动 TVM 资产、不嵌入）：①resize-to-512 取证语义消融（C2 门控实验，ORT-CPU 本机可做）；②ORT CUDA EP 在 P4 实测或证伪；③DirectML EP 桌面调研（本机无 Windows，只能书面结论）；④ORT-CPU int8/fp16 量化探针。
- 分级修正（评审问题6采纳）：Mesa 22.3.6 静默错值 = 源仓库记录、本轮未复现；"换卡可能 ≤1×"是外推，属工程判断非事实。
- **口径警示**：对外引用只用 _dev/RX460 同 session 主口径（2463.4→853.3=2.887×；clamp 修复后重跑 866 属噪声），不得混引 2.780×/2.394×/2.297×/1.498× 四个旧口径。
- License（上游权重仅限非营利）优先级高于一切性能决策，待法务确认后才谈分发。
- 信息缺口（决策阈值判断所需，待用户补充）：实际日处理量、目标机器 GPU/OS 分布。

## M2 后续调研结果（2026-10-02，评审选项1采纳后的四项跟进）

**① resize 消融（C2 门控）— [事实]**（`work/ablation_results/runs.jsonl`，16 样本 × 4 路径，dynamic ONNX + ORT-CPU，splice 合成篡改，基线=原尺寸 native）：
- 大图（1190×1684~1264×1926）native 有效时，0.1%/1%/10% 面积篡改的 recall@FPR1% 均值 = 0.25/0.25/0.45；**resize 到长边 768 或 512 后全部归零**（AUC 均值 0.74–0.90 → 0.54–0.74）。连 cap768 都足以毁掉检出。
- 上采样类（380×540→512）退化较轻（1% 区 rec 0.947→0.614）但非无害（10% 区 0.073→0.003）；512² 对照组四路径逐位一致（harness 自检通过）。
- **结论：C2 锁死**——fixed-512 不只是"跑不了任意尺寸"，其输出在真实扫描大图上语义失效（检测信号随分辨率消失）。动态 ONNX 原尺寸推理是正确性要求，非便利。
- 附带发现（涉及 下游应用，仅指出未改动）：生产 CPU 路径 auto max-size 在小内存机器上可把输入缩到 600–1000，按本消融即落入"静默退化到无效"区间；建议 下游应用侧把 auto 缩放事件显式标注进报告。

**② ORT CUDA EP @ P4 — [事实]**（隔离 venv：onnxruntime-gpu **1.23.2** + **cuDNN 9.5.1** + CUDA12 代库，驱动 535）：
- CUDA EP 原生跑 dynamic ONNX（无 resize）：512² **224ms**（同环境 CPU ~2700ms，≈12×）；1264×1926 大图 native **2.8–3.8s**（CPU native 33s，≈9×），8GB 显存无 OOM；指标与 CPU native 一致（b5_r0.01 AUC 0.9995/rec 0.989/iou 0.500 vs CPU 1.000/0.989/0.501）。
- 版本矩阵脆：ORT≥1.29 要求 CUDA13（驱动 535 不可用，且 CUDA13 已弃 Pascal）；ORT 1.23.2 配 cuDNN 9.27 在 Conv 报 CUDNN_FE failure 11，降 9.5.1 修复。
- [工程判断，置信度高] 评审期待的"第四路径"实锤：零自研、任意尺寸、保精度；但仅限 NVIDIA 目标机，Windows 分发需捆 CUDA 运行时（增量数百 MB，远小于 TVM 方案 ~1GB），NVIDIA 弃 Pascal 趋势使该路径有支持期限。**是否作为 下游应用可选 EP 待用户决策。**

**③ DirectML EP 桌面调研 — [子代理报告，附来源，未本机实证]**：DML 进入维护模式；Resize-cubic/动态 sizes/Unfold 回退 CPU；动态形状重编译+GPU↔CPU 读回有 5× 减速实证（ORT #14550）；exe 增量 ~10–25MB。结论：值得低成本 PoC（固定分桶+查回退率），不值得作为长期主加速路径。

**④ ORT-CPU 量化探针 — [事实]**（`work/quant_probe.py` → `work/ablation_results/quant_probe.log`，dynamic ONNX + ORT-CPU 1.x fp32 基线）：
- 图结构（1979 节点）op 分布：Transpose 324 / Add 320 / Reshape 306 / MatMul 272 / Mul 141 / LayerNormalization 114 / Conv 111 / Gather 80 / Concat 53 / Div 41 / Softmax 40 / Relu 39。**MatMul+Gemm 282 > Conv 111**，即 Transformer 分支的 GEMM 主导算力，卷积不是热点。
- 线程消融（同机同模型，512²）：fp32 默认线程 3096ms → intra=4 2802ms → intra=6 2451ms。**[待验证假设] 本机此前登记的 CPU median ~2950 大概率是默认线程数（=逻辑核 12）下的值，把 intra 钉到 6 就已有 ~17% 收益，且未触碰任何数值路径。** 样本量 1、未做重复，须正式复测后才能写成结论。
- `quantize_dynamic(QUInt8, per_channel, ReduceRange=False)` **直接失败**：`ShapeInferenceError: Inferred shape and existing shape differ in dimension 0: (256) vs (128)`。根因判断：272 个 MatMul 的权重被 Transpose 就地改写布局，ORT 量化预处理按标准 [K,N] 假设重排 initializer，与实际形状冲突。**[工程判断，置信度中] 这条路径不是"多调一个参数"能救的，需要 pre-processing（折叠 Transpose 进 MatMul）才可能跑通，属独立工作量。**
- 未做：静态 QDQ / INT8 权重+fp16 激活的混合方案、fp16 CPU 推理。理由：CPU 路径的真实问题由上一条线程消融指向（配置而非量化），先复测配置。

**⑤ ncnn 路径阶段0/A1 — [阶段A闸门 A1–A5 全绿（A4 带边界：全分辨率≲2.4MP，瓶颈为主机 16GB 内存；Windows 复测转目标机）；用户已批准进入阶段 B（2026-10-02），后端链 CUDA EP → ncnn → CPU，DML 退出默认链]**：
- 环境事实：`venv/bin/pnnx` + pypi `ncnn 1.0.20260526` 可用；ncnn python 绑定的 `ncnn.Mat(np.float32)` 已验证接受 (C,H,W) 布局，`Option` 有 `use_fp16_packed/storage/arithmetic`、`use_bf16`、`use_vulkan_compute`。
- **环境陷阱（新发现，影响所有 torch 侧复现）**：venv 内 torch 为 **2.14.1+cpu**，而 PyPI 上 torchvision 最高只有 **0.29.1**，无匹配构建；装官方 torchvision 会报 `operator torchvision::nms does not exist`。TruFor 只需要 `timm.models.layers` 的 3 个符号（`DropPath`/`to_2tuple`/`trunc_normal_`），但 `import timm` 顶层链强制要 `torchvision.datasets`+`torchvision.transforms`。**[决策] 放弃伪造整个 torchvision 的 stub 路线（依赖面太宽、易静默错值），改为把 `timm`/`timm.models` 注册为裸命名空间包指向真实目录**，让 `timm.layers.*` 子模块正常加载（见 `work/pnnx_a1_convert.py`）。附带：`lib/utils.py:26` 硬依赖 matplotlib，venv 内已补装。
- 另外 `lib/` 不是直接位于项目根，实际路径为 `truFor-src/TruFor_train_test/lib/`；`weights/trufor.pth.tar`（281MB）为 checkpoint。
- A1 闸门脚本：`work/pnnx_a1_convert.py`（trace→pnnx，双 inputshape `[1,3,512,512]` + `[1,3,640,384]`）、`work/pnnx_a1_test.py`（fp16 全关 CPU 跑 256²、384×512、**1000×1500** 三尺寸，判据=主输出空间尺寸==输入）。止损条件按讨论：若只能固定尺寸或 reshape 被 trace 固化 → 停，ncnn 不可作通用路径。
- A1 缺陷-补丁链（已验证事实，2026-10-02，均为 harness 侧等价补丁、未动 TruFor 源文件；每条 eager 位等价核对通过）：
  1. `torch.tile` 不降级（`torch.tile not exists`）→ cmx 277 处 tile(3,1,1) 改 cat 通道。
  2. wsp 的 `torch.zeros` 常量 → 用 `x[:, :1, :]-x[:, :1, :]` 生成零。
  3. 5D 含 -1 reshape（kv `reshape(B,-1,2,Hs,Dh)`）被 pnnx 发成**零参数 Reshape**，静默产空 Mat（35 处，首个崩点 blob 75）→ attention 头切片（slice 语法；`narrow` 会发未注册 `torch.narrow` 层）。
  4. **`reshape(B,2,·).permute(1,0,·)+index`（FRM ChannelWeights/SpatialWeights，net_utils:28/45）→ pnnx 发 4D Reshape + `Slice unbind`（axis 映射错位）**：第二个分片为空 Mat（blob 2360 dims=0），下游 `reshape(-1,-1)` 除零 → **SIGFPE 硬崩**（CPU 64² 逐 blob 二分定位，param 行 1748 `unbind_7`）。补丁：mlp 输出直接通道切片返回二元组，FRM 消费端 `weights[0]/[1]` 索引语义不变。
  5. 跨张量尺寸引用（FFM ChannelEmbed `x.permute(0,2,1).reshape(B,_C,H,W)`，H,W 来自另一空间 blob v_2128）→ pnnx 发 `6="1w,size(@1,2),1024"`，pypi ncnn 解析器 `malformed literal token @1` 拒收（全 param 仅 1 行含 `@`；`1w` 类带索引 token 有 77 行先例且可加载）。手工替换 `size(@1,2)`→`1h` 后 load_param rc=0（`work/ncnn_a1/fix1`）；等价性（slot 序 (w,h,c) 下 `1h`=H）待 A2/A3 非方形数值判定，若错位则备选 `1w`。该替换已固化为 convert 脚本后处理步（含 A1_POSTCHECK 残留扫描）。
- CPU 侧性能事实：修复后 CPU 512² 单 extract >580s 未完成（12 核，fp32，fp16 全关）——若 A2 仍无法在可接受时间跑完，CPU 兜底链价值存疑，须单独评估。Vulkan 设备枚举正常：0=RX460(RADV 22.3.6)、1=P4、2=llvmpipe。
- A3 Vulkan 数值脚本：`work/pnnx_a3_vulkan_test.py`（--device/--fp16/判据同 A2，marker A3_GATE）。

- **A3 fp16 根因与修复（已验证事实，2026-10-02）**：pnnx 默认 `fp16=1` 把 conv/linear
  权重以 fp16 存入 .bin（全域 ~13% 误差主因）。`fp16=0` 重转（复用 ts.pt，param md5 不变
  214f22c3…，bin 137.6→274.8MB `work/ncnn_a1/`，旧 fp16 bin 归档 prev1/）。修复后
  backbone 边界与各段残差降到 ≤4.4e-4 段；显存代价 A4 须知（RX460 2G 压力↑）。
- **A3 残余分歧定位（含对旧误报的撤回，2026-10-02）**：旧结论"Gemm(blob2534) 已偏
  3.376e-2 → interp → fuse"**撤回**——系参考点错位伪报。以 `work/gemm_probe.py`
  隔离进程逐点重测（192²，fp32 bin，P4）：det/conf 两分支 cat/fuse/pred rel 全 ≤3.4e-3；
  2534 vs torch-eager rel=2.27e-3；2534 vs **float64 手工 GEMM** rel=2.27e-3 而 torch
  vs float64 =2.3e-7，ncnn-in blob2533 与 torch seq 也差 2.32e-3。→ 残差自 backbone
  末段（2529=post-BN c4，vs torch rel 2.32e-3）即存在，GEMM/decode 为 fp32 串行点积+
  对消（输入 std 0.87/输出 std 0.15）累积特征，非图错误。**旧 boundary/diff probe 的
  blob 级结论均不可靠**（同进程批量 extract 恰是不稳定协议，见下条）。
- **"ncnn Vulkan 跨进程非确定 → winograd"归因撤回；真根因 = python 绑定零拷贝 UAF（事实，2026-10-02 二次修正）**：
  反翻转/二分实验链（W1/W2/E1/E5/E6 全 3/3 好，P6 局部持有好，S inline 临时坏，X2 `del inp`
  后单抽 out0 即坏，X3 `del inp` 保 Mat 引用 SIGSEGV）证明唯一因子是**输入 numpy 数组是否存活至
  extract 完成**：`ncnn.Mat(numpy)` 为零拷贝包装（python/src/main.cpp:359-381 直接取 `info.ptr`，
  README 明示 "no memory copy"，未声明生命周期义务），而 Vulkan 上传延迟到首次 extract →
  `ex.input("in0", ncnn.Mat(临时数组))` 时数组已释放（442KB 大块走 mmap，free 即 munmap）→
  上传读已释放内存 = 跨进程随机/坏吸引子/同进程重跑一致/大数组更易中招等全部既往现象。
  winograd **无罪**（t1 协议下 win=0 仍坏 6/6；数组存活下 win=1 默认即逐位确定且=好吸引子）；
  上轮"winograd 归因"系测试脚本协议混杂（`work/pnnx_dump_outputs.py` 用 inline 临时，其余
  a3_vulkan_test/err_profile/gemm_probe 均为局部变量持有=天然好协议）所致，一并**撤回**。
  应用层修复已实施：`work/pnnx_dump_outputs.py` 改为数组引用存活至全部 extract 完成。
  修复后 A3 复跑（winograd 开、fp32 bin）：P4 3/3、RX460 3/3 fresh 进程逐位一致；
  192² vs ORT：out rel_L2 5.1e-4 PASS、det max_abs 3.2e-4 PASS、conf max_abs 1.44e-2 FAIL（同旧值）。
  上游路线状态：master(9f9d4ec) 同缺陷（同一绑定代码）；PR #6954（CPU 侧 Mat 清零）与本问题无关，
  实验作废；真正可提的 upstream 修改点在绑定层（文档警告/生命周期保持），待用户决定是否提 PR。
- 数值判据现状（UAF 修复后，winograd 默认开，确定性已恢复）：out rel_L2 PASS（P4 512² 5.9e-4）、
  det max_abs PASS（1.2e-3）、conf max_abs：P4 1.19e-2(512²)/1.36e-2(384×512)/1.44e-2(192²)、
  RX460 1.19e-2(512²)/1.35e-2(384×512)/1.44e-2(192²)，双设备确定性一致。残差源头
  =backbone 末段 2529 的 2.32e-3 rel（性质未查明，疑 fp32 内核累加序差异）。
- **A5-lite 任务级一致性 PASS（事实，2026-10-02，评审新口径）**：`work/pnnx_a5_tasklevel.py`
  （16 篡改样本 stretch512，ORT-CPU dynamic ONNX 基线 vs ncnn-P4，指标复用
  resize_ablation.compute_metrics）。结果：ΔAUC max 1.3e-3、Δrecall@FPR1% =0、
  Δrecall@0.5 max -8e-4、prob/conf 阈值翻转占比 max 0.007%（≪0.1% 判据）、
  Δdet max 2.5e-3；conf max_abs 1.4–2.6e-2 降为纯记录项，不再作门限。
  → A3 conf 门限口径按此改定，backbone 残差不再单独追。附带性能数据：512² 单样本
  ncnn-P4 mean 1078ms vs ORT-CPU mean 2908ms（A4 参考）。
  环境注：解释器 `(本地路径)/venv_master`（本会话补装 onnxruntime 1.30
  + opencv；系统 python3 无 ncnn，不能单解释器跑双后端对比）。
- **A4 性能/显存/主机内存实测完成（事实，2026-10-02，PASS 带边界）**：`work/pnnx_a4_perf.py`
  （三输出全计时、VramPoller 50ms 采样 AMD sysfs+nvidia-smi、conf_md5 指纹、fresh
  extractor/rep、fp32 bin、winograd 默认开、fp16 三项关；每 size 独立进程）。steady min：
  512² P4 1.04s / RX460 1.97s；1024² P4 8.46s / RX460 18.3s；1926×1264(2.4MP) P4 35.4s /
  RX460 88.4s。峰值：RX460 1024² card0 **1859MiB（2G 卡仅剩 ~190MiB）**、2.4MP 1556MiB
  （非单调，随长宽比分配模式）；P4 2.4MP nv 6714MiB、RSS 5521MB。**4MP（2048²）双设备
  均 exit=137 主机 OOM killer 杀（dmesg `__vm_enough_memory`；3072² mmap 失败后 ncnn.so
  内 NULL segfault）→ 全分辨率路径瓶颈是 16GB 主机内存，非显存；ncnn-Vulkan 全分辨率
  上限 ≈2.4–3MP**。计时随像素超线性（9.4×像素→~34×时间）。
  确定性顺带闭环：P4 与 RX460 各自 2.4MP fresh 进程 conf_md5 逐位一致
  （698ec932f420 / 43f83a024fc9）；跨设备指纹不同（conf_sum 相对差 ~7e-7）属设备算术
  差异，符合预期。**推翻条件（RX460 2MP+ OOM）未触发**。生产工作点 stretch512 双设备
  无压力。脚本注：曾于 Net 存活时调 `destroy_gpu_instance` 致 teardown SIGSEGV(139)，
  已改为进程退出自然释放；首轮 512²/2.4MP 计时数据在崩溃前打印，仍有效。
  日志：`work/ablation_results/a4_P4.log` / `a4_RX460.log` / `a4_followup.log`。
- 待复测项（2026-10-02 评审提出）：①2MP+ 大图与 Windows 上重跑跨进程逐位一致（UAF 触发
  与分配大小相关）——**Linux 侧 2.4MP 双设备已闭环（见上条 A4），Windows 项转目标机阶段**；②CUDA EP 跨进程一致性——**已闭环（见下条）**；③A2 CPU
  在数组存活协议下重跑（旧 SIGSEGV/>580s 疑同 UAF 表象，结论未知）——**已闭环（见下条）**；④旧疑点复核——**已闭环（见末条）**。
- **A2 CPU 重跑 PASS + 旧 ">580s" 撤回（事实，2026-10-02）**：`work/pnnx_a2_test.py`
  （npz 键错位 bug 修复 `d[oname]`→`d[rname]`；CASES 可 argv 覆盖）对 ORT-CPU fp32 参考：
  192²/384×512/512² 三尺寸全 PASS——out rel_L2 ≤3.65e-5、conf max_abs ≤6.4e-4、
  det ≤1.0e-5。**512² 单 extract 实测 ~7s（12 核，fp32，fp16 全关），旧登记
  ">580s 未完成"撤回**（成因未查明，疑当时坏协议/资源竞争异常态，工程判断置信中）；
  CPU 兜底链价值恢复（7s vs ORT-CPU 2.9s，慢 2.4×但可用）。ncnn-CPU conf 残差仅
  ~4e-4 → 坐实 Vulkan conf 1.4e-2 差异属设备内核算术序，非图错误。
- **CUDA EP 跨进程一致性：out/conf 成立，det 不成立（事实，2026-10-02 S3 复测推翻本条旧登记）**：
  `work/ort_cuda_crossproc.py`（/root/ortgpu_env ORT 1.23.2，需
  `LD_LIBRARY_PATH`=venv 内 `nvidia/*/lib` 各目录，否则 libcurand.so.10 加载失败并
  **静默回退 CPU**——首轮"双一致"即 CPU 假象，第二组才是 CUDA 实证）。512² fresh
  进程×2 CUDA EP 三输出 md5 逐位一致（out 5a4772bb4016、conf 909891b928c9、
  det 94b5d5099968）；CPU EP ×2 亦一致；CUDA vs ORT-CPU 参考求和相对差 ~1e-5。
  **更正（S3，同脚本 9 个新样本 + 原 2 样本共 11）：out/conf 仍逐位一致；det 恰有
  两种离散模式——94b5d5099968（sum −1.58750772e+00）×8、18757f258295
  （sum −1.58750761e+00）×3，Δlogit≈1.1e-7（Δscore≈3e-8）。疑 cuBLAS 算法启发式
  随进程内存状态变化【待验证假设】。报告口径改"out/conf 逐位可复现，det 双模式
  容差"。A 侧 `下游应用/src/tests/cuda_numerics_check.py` 已按此设门。**
  ④旧疑点随全套件证据关闭：decode 区各点好协议（gemm_probe 逐点）与 A2/A3/A5/A4
  全部成功抽取中无崩溃、无多模式离散，"conf 互差 0.92"未再现。
  **阶段 A 闸门 A1–A5 至此全绿（Windows 项除外，转目标机阶段）。**

## 阶段 B 开工（2026-10-02，用户批准）

- 用户决策：①下游应用先 `git init` 基线（bf496d5）再改；②默认后端切 **auto**（链
  cuda→ncnn→cpu，report 落 backend_used/reason/fallback）；③目标机=**Linux 为主
  （NVIDIA+AMD 混合）**（2026-10-02 答复），捆库策略可按 Linux 定稿，Windows 实证
  仍挂起。实施方案（S0–S8）：
  `/root/.qoder-cn/plans/dawn-ledge-asp.md`。
- 硬约束入案：C2 一律原尺寸+auto 缩图显式告警；ncnn 封装层结构性自持输入副本（UAF）
  +X2 故障注入+fresh 进程确定性测试；CUDA providers[0] 断言防静默回退；ncnn 进 auto
  前置 **A5b**（16 样本原尺寸 ncnn-Vulkan vs ORT-CPU native 任务级，判据同 A5）。
- S0：ncnn 资产已拷 `(本地项目目录)/trufor_ncnn.{param,bin}`；
  golden=推理级 md5 三图（ctrl512/b22/b5=512²/967×714/1926×1264），校验脚本
  `src/tests/check_golden.py`（R0 红线，每步必跑）。
- A 侧进度（事实，2026-10-02）：S1 纯搬移重构（dce87b4）、S2 探测模块
  `trufor_backends.py`+CLI `--trufor-probe`（ba24515）均 GOLDEN=PASS；探测三态与
  本仓实测一致（RX460 2.0GiB→1.0MP / P4 7.5GiB→2.4MP / llvmpipe 排除；CUDA
  providers[0] 断言在缺库态正确拒判）。**新事实：运行时改 `os.environ` 的
  LD_LIBRARY_PATH 对本进程后续 dlopen 无效（ld.so 启动时缓存），进程内 CUDA
  依赖修复只能 ctypes.RTLD_GLOBAL 多轮预加载**——S8 打包侧预加载方案的依据。
- A 侧进度（事实，2026-10-02 续）：S3 CUDA 接入（3c8e34b，假回退断言 12 用例、
  512²/2.4MP 数值门 cuda_numerics_check.py=PASS，det 双模式已按撤回后口径入判据）、
  S4 ncnn 封装+3 unittest（cd0b825，pypi 轮与 master 构建指纹逐位一致→pypi 可分发；
  双设备 192² golden 落 `src/tests/golden/ncnn_vulkan_192.json`）、
  S5 auto 路由+report 字段（4bdd4f4）：**auto 链=cuda→cpu，ncnn 停留 explicit**
  （当时预期"A5b 过则转正"，A5b 已判 FAIL，终态见下节）；backend_info(requested/backend_used/gpu_index/fallback_reason) 为
  infer_trufor 返回 dict 附加键，overlay 报告字段名定为 `backend_used`/
  `backend_reason`（heatmap-only 与 pipeline 两路都透传）；实测系统 py auto→cpu
  带原因、ortgpu auto→cuda、ncnn explicit 缺省编号解析=P4、4MP 守卫抛错含 C2
  文案。CLI 白名单仍 {"cpu"}（扩 auto/cuda/ncnn+默认值切 auto 在 S7）。每步
  R0 均 GOLDEN=PASS。
- A 侧进度（事实，2026-10-02 续2）：S7 CLI/config/告警落地——CLI 白名单
  cpu/auto/cuda[:N]/ncnn[:N]（非法值 CLI 阶段拒绝）、DEFAULT_TRUFOR_DEVICE 与
  config.json trufor.device 切 **auto**（默认链终态 cuda→cpu）、新增可空 config
  键 `trufor.ncnn_max_megapixels`（null=按显存分档，数值只收紧不放宽，
  min(分档,值) 且 CLI 校验 0<x≤2.4）、C2 缩图告警：mem-auto 实际缩图且输入
  长边<1000px 时 `trufor_input.forensic_warning` 落 report+stderr（真实构造
  用例：reserve=20 逼出 1926→600px，警示出现）。infer_trufor 新增可选参数
  ncnn_max_megapixels（v0.5.3），缓存键扩为 4 元组。33 单测+R0 全 PASS。
  S8 落地（A仓 bec82b8）：requirements-cpu/gpu/ncnn三清单钉版（本机实测
  组合）、onedir spec/build.sh（nvidia收集开关）、ensure_nvidia_libs frozen分支、
  开发态CUDA wrapper、部署摘要〇.2节；CPU包261MB与GPU包4.0GB两态冻结e2e
  heatmap md5与各自python参考逐位一致，probe/路由/降级原因在冻结态全部正确。
  阶段B S0–S8 全部完成；挂起项不变（Windows实证、捆库策略、License法务、
  上游ncnn PR、cuBLAS det双模式）。

## A5b（ncnn 进 auto 前置门，2026-10-02）：**FAIL，ncnn 长期停留 explicit**

- 实验：`work/pnnx_a5b_native.py` 三相（ort/ncnn/metrics，**分相原因：两后端同
  进程 b5=2.43MP 被 OOM-killer 击杀 exit137，ORT-CPU 同进程连续多样本 RSS 累积
  到 11.6GB 亦被杀——单进程峰值 6.0GB 可存活**；另 fresh 进程逐样本续跑用
  `--only/--skip-done`）。16 样本原尺寸，ncnn-Vulkan(P4) vs ORT-CPU。
- 汇总（`work/ablation_results/a5b_summary.txt`）：|ΔAUC|max=0.182（门 2e-3）、
  prob 翻转占比@thr99 max=1.03%（门 0.1%）。**仅 b22（714×967 竖版）系 3 样本崩，
  其余 13 样本 prob max_abs≤5e-3、flip≤1e-4，含 b5 2.43MP**。
- 根因定位链（均实测）：①内容无关维度相关——b5/b3 内容缩到 714×967 同样崩，
  b22 缩到 1000×1721 正常；②三方对照 **torch 原生≈ORT-CPU（rel_L2 6e-4），
  ncnn-CPU≈ORT（7.7e-4），ncnn-Vulkan 崩（0.83）**；③RX460 与 P4 结果逐位同错
  （非驱动）；④关 winograd/sgemm 无效；⑤逐 blob 比对：崩点在最末 Interp 之
  **前**——首个 Interp 输入 latent(512,23,31) GPU 已偏 CPU 0.62，即 Swin encoder
  链中某算子在 latent 奇数尺寸(23×31)上 Vulkan 实现数值偏离，Resize 约定无罪。
  同进程重复逐位确定（A4 确定性结论不被推翻——它只保证同形状自洽，未跨形状/
  跨后端比对过）。
- 边界画像（GPU 崩的尺寸需同时满足）：输入 W=967 且 H∈{714,715}、W=714 且
  H=967 崩；967×712/716、968×714、1000×967、967×1000 正常。非"奇数"单一规则，
  疑与 23×31 latent 的窗口/对齐分支相关【待验证假设：上游 ncnn 定位前不再细化】。
- **决策：按方案预案，A5b 不过 → ncnn 不进 auto；下游应用 auto=cuda→cpu 为终态。
  ncnn 保留 explicit 仅供用户自担风险使用（默认守卫 1.0/2.4MP 表不变），文档
  写明"特定尺寸存在已被复现的任务级偏离"。上游 ncnn 缺陷登记升级为持有最小
  复现（param+bin+b22 单图+尺寸对），非阻塞。**
- **【2026-10-03 接手 AI 就地作废】本节两处已被下游推翻：(a) “边界画像（23×31
  latent 窗口/对齐分支）”真因是 **hw%4 上传步长**（见本文件 :268-272 与下游 fork
  `eab8352`）；(b) “A5b 不过 → ncnn 不进 auto、auto=cuda→cpu 为终态”已被下游改为
  **cuda→ncnn→cpu**（A 仓 v0.5.4，A5b 补丁轮 16/16 + mod4 12/12 双设备达标）。
  原文保留仅作历史，勿再引用为现行结论。**

### A5b 根因改判（2026-10-02 二次修正，推翻上节⑤"Swin encoder 某算子"归因；三次修正见下）

- **真根因（2026-10-02 三次修正，机制级定案）= `VkCompute::record_upload` 上传
  memcpy 步长不匹配**：pyncnn 把 numpy 包成零拷贝 Mat 时刻意将 cstep 设为紧凑
  平面步长 H·W（`python/src/main.cpp:371-372`），而 staging VkMat 由
  `create_like` 重算为 16B 对齐 cstep=align(H·W·4,16)/4；`command.cpp:414` 一次性
  `memcpy(staging, src.data, total()*elemsize)`（`total()=cstep*c`，紧凑值）把平面
  q 写到 GPU 偏移 q·hw、消费侧按 q·cstep 读 → **c≥2 且 hw%4≠0 时平面平移
  (cstep−hw)、末平面尾部 c·(cstep−hw) 字节从未写入**。与全部既有刻画逐字节自洽
  （`[1..9,13..18,0,0,0]`＝平移 3＋staging 新鲜零；in0 抽取干净＝net 缓存 CPU 侧
  Mat 不经下载，非上传无恙；`use_packing_layout=False` 不免疫＝截断与 packing 决策
  无关；weights 不受影响＝ModelBin 自建对齐 Mat 两侧步长相等；"上传往返对照=干净"
  系 in0 缓存假象，作废）。fork 补丁 `ncnn_upstream@eab8352`（单文件
  `src/command.cpp`，步长不等时逐平面拷贝）：micro 全网格零损坏、967×714
  max_abs=0、b22_r0.01 任务级 ΔAUC=4.2e-4/翻转 5.1e-5 过 A5b 判据（损坏态
  ΔAUC=0.182/翻转 2.8%）。patch 导出
  `work/upstream_a5b/0001-fix-vulkan-record_upload-for-compact-layout-external.patch`。
- ~~真根因 = Vulkan 路径 pack 转换尺寸缺陷~~（规则画像正确、机制归因错误：
  损坏发生在上传而非 pack 转换/回读；Split 纯透传本就不触数据）。
  规则本身仍成立：blob 满足 **c≥2 且 (H·W)%4≠0** 时 GPU 端数据回读损坏；
  最小复现仅需 `Input+Split` 3 行 param（已验证事实，work/upstream_a5b/micro/）。
- 复现规则经 c=1..8 × 17 尺寸网格零失配；c=1 全净、(H·W)%4==0 全净；
  P4/RX460/pypi 轮与 master(9f9d4ec) 构建逐位一致；CPU 全净；跨进程确定。
- 有序值刻画（c=2，3×3，输入 1..18）：Split 输出 = `[1..9, 13..18, 0,0,0]`，
  第二平面平移 4−(9%4)=3 个元素、尾部读进 pack4 padding 零。
- 统计筛查教训：mean/std 聚合对平面错位型损坏严重低估（噪声全链 std_rel≤5e-3
  而 out0 逐元素 rel_L2=0.27），定位 onset 必须逐元素 rel_L2。
- 与 A5b 全部既往实测自洽：边界画像（967×712/716、968×714、1000×967 好 vs
  967×714、714×967 崩）恰是 hw%4 边界；内容无关成立（噪声也崩）；
  winograd/sgemm 无关成立（非算术 kernel）。原 rel_L2 表（photo blob193=0.954/
  out0=0.229；noise out0=0.268）不改，作废的只是算子归因。
- 连带发现：损坏尺寸下同一 extractor 先 extract Split 输出再 extract "in0"
  → SIGSEGV（exit 139）；正序与反序单取均正常。**（2026-10-02 补丁后重测：
  Input+Split+Conv 迷你 net extract conv 输出仍 SIGSEGV，与上传损坏不同源，
  属第二缺陷；TruFor 全网从未崩，不阻塞修复门，未定位、挂起。）**
- 产出（均本地，零外部动作）：`work/upstream_a5b/micro/repro_split_pack4.py`
  （自包含复现，双设备验证过）、`split_roundtrip.param`、
  `upload_roundtrip.param`（上传往返对照=干净）、`ISSUE_draft.md`（英文草稿，
  用户自行决定是否/何时提交）。"最小复现需 param+bin+b22"的旧登记作废：
  实为 3 行 param + 随机张量，无需 275MB bin。
- **规避路径实测（2026-10-02，回应外部审阅建议）**：`use_packing_layout=False`
  在 P4 上对同一迷你 net（c∈{2,3,4}×尺寸{3×3,967×714,3×5,4×4}）仍 9/12 损坏，
  与默认开 packing 的 9/12 **同样本同模式**——关 packing **不能**规避本缺陷
  （已验证事实）。故 ncnn 准入维持 explicit 不变：可行规避只剩改 param（Split→
  Slice/Crop，需证无损）或 padding 输入（引入 C2 同类边界风险，否决）。绕开
  失败记录进 ISSUE_draft 供上游参考。**（2026-10-02 三次修正后作废：补丁即
  根治，无需规避。）**

### A5b 补丁后重验（2026-10-02，F3/F4；推翻上条"维持 explicit 不变"的前提升级为待发版）

- 补丁（`ncnn_upstream@eab8352`，单文件 command.cpp）后 ncnn-Vulkan 原尺寸 16 样本
  任务级重跑：**P4 与 RX460 均 16/16 PASS**（判据 ΔAUC≤2e-3、prob/conf 翻转<0.1%，
  实测最大 ΔAUC=5.6e-4[b28]、最大翻转=3.0e-4[b28]；conf_maxabs 全列 0.014~0.043，
  仅记录项）。结果文件 `work/ablation_results/a5b_patched_native{,_dev0}.jsonl`；
  损坏态基线保真存档 `a5b_dumps/ncnn_buggy_20261002`。
- 确定性口径澄清（与 golden 文件 `ncnn_vulkan_192.json` 分设备基线一致，非新缺陷）：
  **同设备跨 fresh 进程逐位确定**（P4/RX460 复测均 True）；**跨设备非逐位一致**，
  补丁后残余差异为算术级（prob rel_L2≈3e-5~3e-4，conf max_abs≤9.5e-3）；损坏态
  "双设备逐位同错"系 memcpy 搬运型缺陷掩盖算术差异，修复后暴露属预期。
- 显存：补丁不改分配尺寸；A4 数值沿用。RX460 2GiB 实测补充：同进程连跑
  3×2.4MP 后第 4 样本 `vkQueueSubmit -4`（amdgpu CS OOM）——生产守卫 <3GiB→1.0MP
  档位本就拒绝该组合，实验侧改每样本 fresh 进程绕过（顺带满足确定性口径）。
- **决策更新：fork 计划 S6 准入门（机制性解释+全部回归）已满足。ncnn 回 auto
  （cuda→ncnn→cpu）+ A 侧换装自建 `+trufor1` 轮，属 A 仓改动，待 F5 发版链闭环
  后实施；pypi 官方轮带缺陷，auto 接入前禁止放量。**hw%4==0 尺寸不受补丁影响
  （步长相等走原分支），既有 golden/基线全部继续有效。

### A 仓 S6 落地完成（2026-10-02，A commits 935a4cb/5c9587f/d8fe152）

- auto 链改 **cuda→ncnn→cpu**（trufor_inprocess v0.5.4 / overlay v0.10.2）。
  ncnn 入选三关门控：probe_vulkan 物理设备 → `probe_ncnn_upload()` 行为判据 →
  本图 MP≤min(显存分档, `trufor.ncnn_max_megapixels`)；任一不过落 CPU，原因拼接
  写 `report.backend_reason`；ncnn 入选不记降级；**explicit 不门控**（契约不变）。
- `probe_ncnn_upload()`（A 仓 trufor_backends）= 3 行 split 迷你网 `load_param_mem`
  + c=2 3×3 有序值 extract("1") 逐位判据；行为识别 stock 轮（版本串无 local 段
  无法区补丁，必须行为判据）。双环境实测：stock 1.0.20260526 → FAIL
  （读回 [1..9,13..18,0,0,0]）；+trufor1 轮/补丁 build → PASS。迷你网第二缺陷
  （extractor 混用 SIGSEGV）经单 extractor 单目标规避。
- e2e（本机 P4）：714×967 → ncnn 6.4s；3MP → cpu 带完整原因。回归：路由+cuda
  40 用例、ncnn 3 用例（补丁 build 下跑）、R0 golden 全 PASS。
- 环境换装：`/root/ncnn_pypi_env` 已 force-reinstall 为 `1.0.20261002+trufor1`
  （--no-deps），并补装 onnxruntime 1.29.0+pillow 成自包含测试环境——43 用例
  （含 ncnn golden 逐位）在该环境全绿；补丁轮运行时 `ncnn.__version__` 实测
  仍打印 `1.0.20261002`（无 local 段），再次佐证版本串不可作判据。stock 轮存档
  `work/upstream_a5b/wheels/ncnn-1.0.20260526-*.whl`
  作 known-bad A/B 参照。`venv_master` 内 .so 仍为**补丁前**旧物（md5
  58316cd4…），实验须显式 PYTHONPATH 指 `ncnn/build/lib.linux-x86_64-cpython-311`
  （b0e53111…）或直接用 ncnn_pypi_env。
- 未做（明示）：Windows/冻结包在 +trufor1 轮下的 e2e（S8 两态此前用 stock 轮
  跑的仅是 explicit 资产链路）；放量分发仍等目标机清单。


## 动态尺寸 ONNX（ORT-CPU 回退路径，2026-10-01）

**新增自包含脚本**：`scripts/export_onnx_dynamo_dynamic.py`（复用 export_onnx_dynamo 的
build_model/swap_adaptive_pool/ExportWrapper，非方形样本 1×3×480×640 +
`dynamic_axes` 使 H/W 成为符号轴，batch=1 固定）、`scripts/verify_dynamic_ort.py`。

**产物**：`reports/onnx-dynamo/trufor_dynamic_dynamo.onnx`，279 MB 内联（导出后把
torch 2.13 dynamo 写出的 `.onnx.data` sidecar 重新内联为单文件），1979 节点。

**已验证 PASS（事实）**：ORT-dynamic vs 本项目 truFor-src torch，尺寸
256²/512²/640×480/1200×900/**518×731**（含非 32 整除），rel_L2 3–6e-05、corr 1.0，
**map==input 契约在每个尺寸都成立**（out/conf 空间维==输入 H,W；内部
`F.interpolate(size=orisize[2:])` 随符号 H/W 一起导出）。det 折叠空间维、尺寸无关，留静态。
交叉核对：512² 处 dynamic vs 生产 fixed-512 ONNX rel_L2 ~3e-07 —— 同一模型，只是放开 H/W。

**归一化结论（事实）**：ONNX 图本身与归一化无关，÷255/÷256 是调用方约定。重写后的
`work/torch_vs_ort_numerics.py` 以 truFor-src /256 为参考，对同一图：ORT/256 vs torch/256
rel_L2 1.2e-03 < ORT/255 vs torch/256 的 2.25e-03 —— 要复现上游 GRIP-UNINA 数值就喂 /256。

**限制（事实，勿混淆）**：本 dynamic ONNX 只服务 ORT-CPU。TVM/Vulkan 手写 kernel 按静态
512×512 buffer 签名匹配，**无法**消费动态图；Vulkan 交付路径仍是 512-only。两路径的语义差
（非 512 输入 ORT 能原生跑、Vulkan 不能）需在嵌入下游消费方时体现。

**导出环境**：需 torch2.13 + timm + yacs + matplotlib + onnx/onnxscript/ort（本机项目 venv 的
torch2.14 与可用 torchvision 冲突，改用带 `--system-site-packages` 的临时 venv）。dynamo 下
`dynamic_axes` 会告警建议 `dynamic_shapes`，实测仍能正确产出符号轴。

**去外部依赖（本轮）**：`work/cpu_torch_vs_ort.py`、`work/torch_vs_ort_numerics.py` 原先
import 外部项目的 torch 封装（其 vendored 模型），已改为用本项目 `truFor-src`（权重同为
`weights/trufor.pth.tar`）。仓库内已无任何外部项目引用 / 外部绝对路径，可独立上传 GitHub；
后续按需再集成到下游消费方。

## 环境（只读记录，不改系统设置）

- CPU：4 物理核，governor = powersave。**已实测：切到 performance 反而慢 1.3%，
  IQR 未收窄 —— governor 不是 CPU 漂移主因，该说法已撤回**，原因仍未定位。
- GPU：RX 560 Baffin，`power_dpm_force_performance_level = auto`（部署默认，保持）。
  实测切 high **快 6.4% 且 IQR 从 30.8 收窄到 13.5 ms**，是首个能解释 GPU 波动的因素。
- 允许临时切换 governor/DPM 做测量，**必须 trap 恢复**（已验证有效）。
- 同 session 实测：CPU default 2170.8 / GPU auto 1448.8 → **默认对默认 1.498×**；
  钉 DPM high 可到 1.601×，但那不是默认配置，不得作为结果引用。
- TVM 0.27.0 @ 4d0ac168c7，Mesa 25.0.7，ORT 1.30.0
- 峰值一律按 **1903 GFLOPS = 896 SP × 2 × 1.062 GHz**（DPM 最高档；此前误用 1.176 GHz 得 2107）
- 设备为 Baffin，驱动报 RX 460 (POLARIS11)，**896 SP / 56 CU**（此前误记 448 SH）（896 lane × 2 × 1.176 GHz）

## 已知不可比（勿直接引用）

- 跨 session 的 CPU/GPU 比值一律无效。历史上出现过 2317 / 2438.3 / 2298 三个 CPU
  median，GPU 2464 / 2476.8 / 2516 / 2535。
- 曾经三次跨 session 比较出错：CPU 基线漂移、B2 matmul 配置假收益、`only` 参数漏约束。

## 已有 kernel（均按 buffer 签名替换，不改 TVM）

| kernel | 形状 | 状态 |
|---|---|---|
| `scripts/opt/direct_conv.py` | 3x3 64->64 @512x512，调用 15 次 | 16.45 ms / 1175 GFLOPS |
| `direct_conv.build_1x1` | 1x1 2048->512 @128x128，调用 2 次 | 46.0 ms / 748 GFLOPS |
| `scripts/opt/nhwc2nchw.py` | (1,H*W,C)->(1,C,H,W)，11 处 | tiled 1.19 ms / **28.30 GB/s**（3.25×）。整图收益 **48.6 ms（4.4%）** |
| `scripts/opt/depthwise.py` | depthwise 3x3 stride-1，8 处 | 2.36 ms / 28.47 GB/s（8.9×） |
| `nhwc2nchw(act="gelu")` | NCHW(1,C,H,W)→NHWC+GELU，4 处 | 5.55 ms / 6.05 GB/s（15.7×） |
| `scripts/opt/gemm.py::build_smem` | 手写 shared-mem GEMM，BM/BN/BK=64/64/4，relu 与 residual 双 epilogue | shape1 523 / shape2 668 / shape3 410 GF vs DLight 265/305/177（2.0-2.4×），**已接入 swap_conv，新增 30 处替换** |

## 已知未修问题

- `nhwc2nchw` 8.71 GB/s，目标 20+。**DLight 的 transpose TIR 里根本没有 shared
  memory / barrier / 向量化**（全文 16 行，grep 全空），所以改那条规则没用。
  剩下唯一路径是 s_tir block 形式 + 可用的 barrier；flat 形式的
  `alloc_shared`+`tvm_storage_sync` 返回全 0。
- **`&` 连接的 bool 表达式在 TIRX flat 形式下必然编译失败**（SPIR-V 固定指令
  index 报 -14），即使该变量未被使用。这是已知症状的确定原因，避开 `&`。
- depthwise 的 epilogue 必须逐个验证：`_relu` 结尾的要 ReLU，纯 add 的不要。
  猜错 rel_L2 = 0.71，对则 2.2e-08。`swap_conv._pick_epilogue` 自动二分。
- 消融（常量 0.5 填充）**7 次尝试全部 native 崩溃**，已 BLOCKED，别再走。
- **【部分撤回】上一条 T0 消融 BLOCKED**：`scripts/opt/ablate.py::make_noop` 生成的
  Python 源码有两处硬伤——(1) `def` 顶格被缩进 4 空格，是 Python IndentationError；
  (2) 函数体写成 `y[...] = T.float32(0)`，`[...]` 是 Python Ellipsis 字面量而不是
  TIR 索引，即使修好缩进也过不了 TIR 解析。所以"7 次崩溃"是 generator 自身的问题，
  不是 TVM 消融不可行。手写一个 1-block-per-row、256-thread grid 的合法 no-op
  （见 `work/ablate_poc.py`），走同样的 `tvm.compile(..., vulkan) → built(...)`
  路径，输出 buffer unique 值 = {0.5}、全元素写入到位。**整图逐类别 ablation
  仍未跑**，通用 make_noop 需要按 rank 展 div/mod 索引，工作量另计。
  注：STATE/TODO 里 SKIP 数字（max4 330.1、transpose 546 等）**都来自 `TVM_VM_SKIP`
  instrumented profiling 而不是 ablate.py**，撤回这条 BLOCKED 不改变 SKIP 表。
- **我自己的测试代码里的形状/广播错误会读成被测对象或后端的问题**（已发生 4 次）。
  写任何 rel_L2 比较前先确认两端形状能正确广播。
- **【已撤回自己的撤回】max4 曾是 330.1 ms（22.7%），现已修掉。** 替换 kernel 见
  `scripts/opt/scalar_reduce.py`（512 线程 + 9 步树形归约），实测省 **342.3 ms**
  （预测 327，偏差 +5%），三张图三个输出 **rel_L2 全部 0.00e+00**（逐位一致）。
  根因：Vulkan 归约按**输出** extent 取 grid，标量输出 → grid=1、256 线程屏蔽
  255 个、单线程串行扫 262144 个元素（82.5 ms/次）。
- **【又撤回一次】"flat 形式不给 shared memory" 是我的探针 bug，不是形式限制。**
  探针没写输出 buffer，且树形索引 `((t//st)*st*2 + (t%st)*2)` 在第一步就跳过了
  整个上半区。改正后：有无手动 `tvm_storage_sync` **都正确**（ThreadSync 会自动插）。
  flat 形式支持 shared memory，N4 不必回避它。
- 历史说明： 两种独立方法一致：
  插桩 327.3、SKIP 330.1（1% 一致），4 次调用 = 82.5 ms/次，与最初诊断的
  84.2 ms 独立复现吻合。**我此前用 B3 类别表（softmax_log 0.2 ms）撤回它是错的**——
  那张表无效：同一次运行显示 41 个 conv 只替换了 2 个（生成的 TIR 对另外 39 个
  静默解析失败），四分之三为空的替换表撑不起 0.2 ms 的上界。
  机制一直成立：max4 输出标量、输入 262144，Vulkan 归约按**输出**extent 取 grid，
  于是 grid=1、256 线程屏蔽 255 个、单线程串行扫 262144 个元素。
  **修复原型早已存在**：两级归约 82.98 → 0.59 ms（140×，bit-exact），当年因施做
  在 ONNX 层（改不到 legalize 之后的调度）而放弃。**现在是第一优先。**
- **`resize_concat` 向量化**：VEC=4/32×8 使孤立 24.58 → 5.45 ms（4.51×，逐位一致），
  整图 +38.4 ms。**关键：选 NTX×VEC 精确整除 128 宽输出行，循环体内不需要边界判断**——
  上一次失败就是在 `elif` 分支里嵌了内层 `if x < 128`。
- **`scalar_reduce` 树形归约曾有两个潜伏 bug**（越界读 `sh[t+NT]`、缺屏障且所有
  线程都写 out），在标量形状上两次全张量验证都通过并已交付；写逐通道版本才暴露。
  已修（步长从 NT/2 起、每步加 `tvm_storage_sync`、仅 t==0 写），四个 NT 各 12 次
  重复全 0。**一次验证通过不能证明没有越界/缺屏障，同族形状是静默 bug 的藏身处。**
- **⚠ 替换后必须断言内容而非数量**：N4 的整图收益曾报 1.8 ms，实际 48.6 ms，
  因为 swap_conv 被改成无条件传 `tiled=True`，A/B 两臂构建了同一个 kernel
  （导出 .so 的 md5 相同）。校验用 md5 或直接统计 `alloc_shared` 出现数。
- **`TVM_VM_SKIP` 是进程级**：库加载时解析为 static const，同进程内改
  `os.environ` 无效，必须分进程。
- **TIRX bool 故障 = 内联规则，不是 codegen 缺陷**（已实测）：条件**内联**可编译，
  条件**绑定到变量**则 `T.int8 vs T.bool` 失败。不需要改 C++。
- **decoder 融合：已解决并接入（撤回此前"BLOCKED/无法新增 PrimFunc"的旧结论）**。
  `decoder_split.register_kernels` 用 `relax.BlockBuilder.add_func(fn, name)` 新增
  5 个**带类型** GlobalVar——这正是此前那条"Python 无法创建带类型 GlobalVar"被卡住
  时的解法，此前 AI 没找到就下了 BLOCKED 结论。配合 `typed_call_tir`
  （同时补 `ty_args` 与 `ret_ty`，NormalizeCallTIR 查的是 ty_args）通过
  EmitNormalized/NormalizeCallTIR 检查。`run_trufor.py` 里已按
  PRE→decoder_split.rewrite→POST 的正确顺序接线，每轮打印 `rewrote 2 call sites`，
  验收逐位一致。kernel 计时 stage1 19.30 + stage2 23.50 = 42.8 ms/次，图内 100.2，
  两次省约 114.8 ms（与 T2 记录一致）。
- **T7 收尾已完成**：生产入口 `scripts/opt/run_trufor.py`、DEPLOY.md Step 7
  （说明"没有 module_pass、也不需要重建 TVM"）、`reports/VERIFIED-RESULTS.md`
  （仅已验证结论 + 预测vs实测 + 8 条撤回清单）、patch 已重新生成并
  `git apply --check` 通过（2 文件 +73/-9）。**评审包未重打**（需 dual-package skill，
  且用户选择 both 后 T6 排在最后）。
- **【撤回】"TVM 流水线无法表达 align_corners=false resize"（LOG.md:1651）**：
  最小复现（`onnx.Resize` mode=linear, coordinate_transformation_mode=pytorch_half_pixel
  + 常量 scales，opset 18）经 `relax.frontend.onnx.from_onnx` → `relax.build`
  → Vulkan VM，两处 shape 都跑得通、对 PyTorch `F.interpolate(align_corners=False)`
  逐位对齐：`(1,3,8,8)→(1,3,16,16)` rel_L2 6.14e-08，TruFor decoder 实际
  shape `(1,512,128,128)→(1,512,512,512)` rel 4.71e-08；再叠一层 Transpose
  NHWC→NCHW 后接 Resize 也仍然 4.71e-08。旧结论把"融合后的单 kernel 不高效"
  混同于"流水线不能表达"。ONNX 层拆分这条路是通的。

## 下一轮起点（已量清楚）

**当前起点 = ONNX 侧（T10 之后）**：本仓库已自包含，产物 `trufor_dynamic_dynamo.onnx`
就绪，可独立上传 GitHub；GEMM/decoder 线（下方旧起点）已收尾。**集成到下游消费方的后续动作
不在本仓库做**，留到用户要求时按需执行，届时需体现两路径语义差：ORT-CPU 走动态 ONNX
（任意尺寸、喂 /256 复现上游），Vulkan 走静态 512×512（非 512 输入需 resize 或回落 ORT）。
本仓库对下游消费方不做 import、不写其绝对路径。

**matmul 簇是唯一成片的未优化块**：6 个函数，SKIP 合计 **86.7 ms**（当前 926.8 ms 的
9.4%）。DLight 的效率是 **177-305 GFLOPS，仅峰值 1903 的 9-16%**，而我手写的 3×3
卷积是 1175 GFLOPS（62%）。DLight 已用 8×8 micro tile（acc=64，正是最优值），
所以问题在访存与并行度：M/64×N/64 只有 80 个 block，铺不满 56 个 CU。
**解法是 shared-memory A/B tile + split-K**；本轮 tiled transpose 已证明
flat TIRX 下 shared memory + `tvm_storage_sync` 可用（28.30 GB/s）。
**但已实测：只做寄存器分块（不上 shared memory）最好 217 GFLOPS，比 DLight 的
265 还慢 22%。** 因为 DLight 的 Matmul 规则本身就 `use_shared=True` 并对 A/B 做
`cache_read`——结构上已经做了我以为缺的那一步。所以 35-50 ms 的估计过于乐观：
要超过 DLight 需要一个**完整的** shared-memory GEMM（双 tile 载入并向量化、
无 bank conflict、搬运与计算重叠、可能 split-K），收益不确定。

## 关键提醒

替换处**必须断言替换数**，不符即失败。本项目已发生三次无声空操作。
消融必须写常量 0.5（不是空操作），否则 denormal/NaN 会污染墙钟。

## 2026-10-03 接手 AI：P0-b 准入补证（① ② ③ ④）结论

- **① 任务级非对齐尺寸补证：PASS**。夹具 `work/a5b_mod4_make_samples.py`（纯裁剪、
  无重采样）覆盖 (H·W)%4=1/2/3 各 4 个尺寸；补丁轮 P4 12 样本
  |ΔAUC|max=5.1e-4、prob 翻转 max=5.8e-5（门 2e-3/1e-3）→ `A5B_GATE=PASS`。
- **② 全网逐 blob：通过（判据事后修订）**。原 TSV mean/std 判据不可达（对照自身
  827/2605 超阈）；改用逐元素 rel_L2 仪器 `work/pnnx_blob_relL2.py`。stock 正对照
  max rel_L2=1.20、超包络 2286/2605、shift_flag=10（含 blob1/2）；补丁轮
  714×967/713×965/713×967 全部 ≤ 包络、shift_flag=0。包络为事后修订，证明力降级，
  已登记。
- **④ 相邻层网格：Split/Crop PASS，Slice/Concat 未覆盖**。补丁轮 Split/Crop
  0/232 失配（stock 152/232 = 灵敏度成立）；迷你网抽取 Slice/Concat/ReLU 的 GPU 输出
  **必 SIGSEGV(139)**（stock 同样），属已知第二缺陷（`:280-283`），非本补丁引入 →
  Slice/Concat 网格无法执行，替代覆盖为 ② 的全网逐 blob。
- **③ H2（download 对称风险）：通过**。VK validation layer 确已加载、对 buffer
  上传/下载路径静默（0 VUID）；源码判读 download 主机侧 `dst_fp16=create_like(dst_staging)`
  同布局，无外部紧凑 Mat，不构成与 upload 对称的缺陷。未做 fork 插桩元数据日志。
- **仍缺**：Windows 抽查（无目标机）；Slice/Concat 迷你网格（被第二缺陷阻断）。
  两项的用户裁决见对话；auto 链未改动，未出包。

### 2026-10-13 perf_opt 阶段0（落地在 下游应用，本仓留档）

- **改动**：`src/封装层` 增 `use_winograd_convolution=False`（仅此一项）；
  `scripts/param_add_featmask.py` 生成 `trufor_ncnn_featmask16.param`（softmax_361/362 追加 `31=16`）。
- **实测**（RX560 4GiB/Mesa 22.3.6）：1000×1721 25.47→23.15 s；1264×1926 54.11→40.56 s（−25.0%）；
  GTT 4835→2131 MiB；out/conf rel_L2 ≤2.44e-05；跨 fresh 逐位一致；**仍慢于 ORT-CPU 28.67 s**。
- **闸门**：G1/G2/G3/G5/G6 **通过**（G1 覆盖 192²/384×512；1000×1721 因 ncnn-CPU 参照 OOM 未覆盖）；
  **G4 未执行**（延后到最终配置）；**G7 P4 待用户实测**。
- **阶段1 spike**：无需 rebase；0001 仍需；fp16 关闭 ⇒ SDPA 走两遍 cross 路径。
