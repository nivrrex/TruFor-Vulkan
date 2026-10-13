import os, sys, numpy as np, ncnn
HERE = os.path.dirname(os.path.abspath(__file__))
param = sys.argv[1]
names = sys.argv[2].split(",")
ncnn.create_gpu_instance()
n = ncnn.Net()
n.opt.use_vulkan_compute = True
n.opt.use_fp16_packed = n.opt.use_fp16_storage = n.opt.use_fp16_arithmetic = False
assert n.load_param(os.path.join(HERE, param)) == 0
n.set_vulkan_device(1)
x = np.ascontiguousarray(np.arange(2*3*3, dtype="float32").reshape(2,3,3)+1.0)
for name in names:
    ex = n.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    ret, m = ex.extract(name)
    if ret != 0:
        print(name, "extract FAILED ret=%d" % ret)
        continue
    print(name, np.asarray(m, dtype="float32").ravel().tolist())
print("CHAIN_DONE")
