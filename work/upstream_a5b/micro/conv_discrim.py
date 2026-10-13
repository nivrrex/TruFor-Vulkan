import os, sys, numpy as np, ncnn
HERE = os.path.dirname(os.path.abspath(__file__))
C, H, W = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (2, 7, 7)
ncnn.create_gpu_instance()
n = ncnn.Net()
n.opt.use_vulkan_compute = True
n.opt.use_fp16_packed = n.opt.use_fp16_storage = n.opt.use_fp16_arithmetic = False
assert n.load_param(os.path.join(HERE, "conv_discrim.param")) == 0
assert n.load_model(os.path.join(HERE, "conv_discrim.bin")) == 0
n.set_vulkan_device(1)
x = np.ascontiguousarray(np.arange(C*H*W, dtype="float32").reshape(C,H,W)+1.0)
ex = n.create_extractor()
ex.input("in0", ncnn.Mat(x))
ret, m = ex.extract("3")
print("ret", ret)
got = np.asarray(m, dtype="float32").ravel()
print("conv out   :", got[:15], "...")
p0 = x[0].ravel()
compact = p0 + x[1].ravel()
shift = (4 - (H*W) % 4) % 4
p1_aligned = np.concatenate([x[1].ravel()[shift:], np.zeros(shift)])
print("compact-read expect:", compact[:15], "...")
print("aligned-read expect:", (p0 + p1_aligned)[:15], "...")
print("MATCH compact:", np.array_equal(got, compact))
print("MATCH aligned:", np.array_equal(got, p0 + p1_aligned))
