import numpy as np, ncnn, sys
dev=int(sys.argv[1]) if len(sys.argv)>1 else 1
ncnn.create_gpu_instance()
n=ncnn.Net(); n.opt.use_vulkan_compute=True; n.set_vulkan_device(dev)
n.opt.use_fp16_packed=False; n.opt.use_fp16_storage=False; n.opt.use_fp16_arithmetic=False
assert n.load_param("work/upstream_a5b/micro/split_roundtrip.param")==0
sizes=[(1,1),(2,1),(1,2),(1,3),(3,1),(2,2),(2,3),(3,2),(3,3),(4,4),(2,6),(6,2),(3,5),(5,3),(4,1),(967,714),(967,712)]
for c in range(1,9):
    row=[]
    for (H,W) in sizes:
        x=np.ascontiguousarray(np.random.RandomState(0).rand(c,H,W).astype("float32"))
        ex=n.create_extractor(); ex.input("in0",ncnn.Mat(x))
        ret,m=ex.extract("1"); assert ret==0
        y=np.array(m); d=float(np.abs(y.astype(np.float64)-x.astype(np.float64)).max())
        row.append((H,W,(H*W)%4,d>0))
        del ex
    corr=[(h,w,r) for h,w,r,bad in row if bad]
    print("c=%d corrupt(HW,hw%%4):"%c, corr)
