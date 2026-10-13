import sys, numpy as np, ncnn
kind=sys.argv[1]
c,h,w=2,3,3
if kind=="split":
    p="7767517\n2 3\nInput  in0 0 1 in0 0=3 1=3 2=2\nSplit sp 1 2 in0 1 2\n"; blob="1"
elif kind=="crop":
    p="7767517\n2 2\nInput  in0 0 1 in0 0=3 1=3 2=2\nCrop cr 1 1 in0 1 0=0 1=0 2=0 3=3 4=3 5=2\n"; blob="1"
else:
    p="7767517\n2 3\nInput  in0 0 1 in0 0=3 1=3 2=2\nSlice sl 1 2 in0 1 2 -23300=2,1,-233 1=0\n"; blob="1"
n=ncnn.Net(); n.opt.use_vulkan_compute=True
for a in ("use_fp16_packed","use_fp16_storage","use_fp16_arithmetic"): setattr(n.opt,a,False)
assert n.load_param_mem(p)==0
ncnn.create_gpu_instance(); n.set_vulkan_device(1)
x=np.ascontiguousarray(np.arange(18,dtype='float32').reshape(2,3,3)+1)
e=n.create_extractor(); e.input("in0", ncnn.Mat(x)); r,m=e.extract(blob)
print("OK ret",r, "equal", np.array_equal(np.array(m),x[:1]) if r==0 and not m.empty() else None)
