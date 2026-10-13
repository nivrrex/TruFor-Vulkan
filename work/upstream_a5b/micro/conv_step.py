import os, sys, numpy as np, ncnn
HERE = os.path.dirname(os.path.abspath(__file__))
C, H, W = (int(v) for v in sys.argv[1:4])
name = sys.argv[4]
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
ret, m = ex.extract(name)
print("extract", name, "ret", ret)
a = np.asarray(m, dtype="float32").ravel()
print(a[:12], "...")
