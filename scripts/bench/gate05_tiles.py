"""Gate 5: real large images, tiled at 512x512.

The single-image result (Gate 1/2/4) is a latency statement about one 512x512
tile. The question a user actually has is: what happens on a 2K or 4K forensic
image, which is many tiles of real work.

The model is not touched and nothing is re-exported: the production ONNX stays
512x512 batch 1, and a large image is cut into tiles that are fed through the
same graph one at a time. That keeps the model fixed and measures exactly the
deployment pattern that would be used.

Also measured, because it is the same code path and changes the conclusion
completely: tiles processed as a batch instead of sequentially. That needs a
batch-capable export, so it is attempted and reported as available or not
rather than assumed.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ONNX = os.path.join(REPO, "reports/onnx-dynamo/trufor_512x512_dynamo_inline.onnx")
DEFAULT_IMAGE = os.path.join(REPO, "truFor-src/test_docker/images/pristine2.jpg")  # 2048x1773


def tile_image(img, tile):
    """Non-overlapping tiles, edge tiles cropped rather than padded."""
    w, h = img.size
    ys, xs = list(range(0, h, tile)), list(range(0, w, tile))
    out = []
    for y in ys:
        for x in xs:
            out.append((x, y, min(x + tile, w), min(y + tile, h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--outdir", default=os.path.join(REPO, "reports/capability"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--virtual", default="",
                    help="also synthesise a WxH canvas of this size, e.g. 3840x2160")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    import tvm
    import tvm_ffi
    from PIL import Image
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    print("=" * 78)
    print("GATE 5  large-image tiling throughput (model fixed at %dx%d)" % (args.tile, args.tile))
    print("=" * 78)

    dev = tvm.vulkan()
    tgt = tvm.target.Target({"kind": "vulkan", "from_device": 0})
    m = onnx.load(args.onnx, load_external_data=False)
    ir = from_onnx(m, shape_dict={"rgb": [1, 3, args.tile, args.tile]}, dtype_dict="float32")
    exe = relax.build(ir, target=tgt)
    vm = relax.VirtualMachine(exe.jit(), dev)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = args.cpu_threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def gpu_one(arr):
        tin = tvm.runtime.tensor(arr, dev)
        out = vm["main"](tin)
        if isinstance(out, tvm_ffi.Array):
            return [out[i].numpy() for i in range(len(out))]
        return [out.numpy()]

    def cpu_one(arr):
        return sess.run(None, {iname: arr})

    def crop(arr, w, h):
        return arr[:, :, :h, :w]

    canvases = []
    img = Image.open(args.image).convert("RGB")
    canvases.append((os.path.basename(args.image), img.size, img))
    if args.virtual:
        w, h = (int(v) for v in args.virtual.lower().split("x"))
        canvases.append(("synthetic %dx%d" % (w, h), (w, h),
                         Image.new("RGB", (w, h), (127, 127, 127))))

    results = []
    for label, (W, H), im in canvases:
        boxes = tile_image(im, args.tile)
        tiles = []
        for (x0, y0, x1, y1) in boxes:
            patch = im.crop((x0, y0, x1, y1))
            if patch.size != (args.tile, args.tile):
                patch = patch.resize((args.tile, args.tile), Image.BILINEAR)
            tiles.append((np.asarray(patch, np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32))

        print("\n--- %s  %dx%d  ->  %d tiles of %dx%d" % (label, W, H, len(tiles), args.tile, args.tile))

        # warm both engines on one tile
        for _ in range(args.warmup):
            cpu_one(tiles[0]); gpu_one(tiles[0])

        # sequential
        t0 = time.perf_counter()
        for t in tiles:
            cpu_one(t)
        cpu_seq = (time.perf_counter() - t0) / len(tiles)

        t0 = time.perf_counter()
        for t in tiles:
            gpu_one(t)
        gpu_seq = (time.perf_counter() - t0) / len(tiles)

        # CPU with a thread pool (the 4-core box can overlap ORT sessions)
        for w in (2, 4):
            with ThreadPoolExecutor(max_workers=w) as ex:
                t0 = time.perf_counter()
                list(ex.map(cpu_one, tiles))
                cpu_par = (time.perf_counter() - t0) / len(tiles)
            print("    CPU %d threads    : %7.1f ms/tile" % (w, cpu_par * 1e3))
            results.append({"canvas": label, "size": [W, H], "tiles": len(tiles),
                            "engine": "cpu-%dt" % w, "ms_per_tile": cpu_par * 1e3})

        print("    CPU sequential   : %7.1f ms/tile" % (cpu_seq * 1e3))
        print("    GPU sequential   : %7.1f ms/tile" % (gpu_seq * 1e3))
        for eng, v in (("cpu-seq", cpu_seq), ("cpu-4t", cpu_par), ("gpu-seq", gpu_seq)):
            results.append({"canvas": label, "size": [W, H], "tiles": len(tiles),
                            "engine": eng, "ms_per_tile": v * 1e3,
                            "mpix_per_s": (W * H / 1e6) / v})

        best_cpu = min(cpu_seq, cpu_par)
        sp = best_cpu / gpu_seq
        print("    best CPU / GPU   : %.3fx  ->  %s"
              % (sp, "GPU faster" if sp > 1 else "CPU still faster"))
        results.append({"canvas": label, "gpu_speedup_vs_best_cpu": sp,
                        "verdict": "GPU faster" if sp > 1 else "CPU faster"})

    C.write_json(os.path.join(args.outdir, "gate5_tiles.json"), results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
