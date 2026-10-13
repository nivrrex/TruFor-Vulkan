#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-b ① 生成 (H·W)%4 三类余数覆盖的 A5b 尺寸样本（纯裁剪，无重采样）。

背景：既有 16 样本中只有 b22=714×967 的 (H·W)%4==2 真正走 ncnn 补丁新分支，
任务级证据实际只覆盖 1 个尺寸。本脚本用**裁剪**（不 resize，保持原生像素与
原生尺度篡改）构造 rem1/rem2/rem3 三类余数尺寸，供 pnnx_a5b_native.py 跑
任务级一致性。

裁剪点选在左上角 origin (0,0)，尺寸保证原篡改矩形完全落在裁剪内：
  b22 (H=967,W=714)，篡改范围 x∈[114,701], y∈[114,709]
    rem2 = 967×714 原生（直接用既有样本文件）
    rem1 = 965×713  (965*713=688045, %4=1)
    rem3 = 967×713  (967*713=689471, %4=3)
  b3 (H=1721,W=1000)，篡改范围 x∈[205,965], y∈[67,538]
    rem1 = 1719×999 (1719*999=1717281, %4=1)
    rem2 = 1721×998 (1721*998=1717558, %4=2)
    rem3 = 1721×999 (1721*999=1719279, %4=3)

输出：
  work/ablation_inputs/samples/<name>{,_mask}.png   （gitignore，随脚本再生）
  work/upstream_a5b/p0b/manifest_mod4.json          （入库）

所有断言失败即非零退出。
"""
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HERE, "ablation_inputs")
SAMPLES = os.path.join(INPUT_DIR, "samples")
OUT_MANIFEST = os.path.join(HERE, "upstream_a5b", "p0b", "manifest_mod4.json")

# (目标名, 源样本名, 裁剪 H, 裁剪 W)
CROPS = [
    ("b22_r0.001_rem1", "b22_r0.001", 965, 713),
    ("b22_r0.001_rem3", "b22_r0.001", 967, 713),
    ("b22_r0.01_rem1", "b22_r0.01", 965, 713),
    ("b22_r0.01_rem3", "b22_r0.01", 967, 713),
    ("b22_r0.1_rem1", "b22_r0.1", 965, 713),
    ("b22_r0.1_rem3", "b22_r0.1", 967, 713),
    ("b3_r0.01_rem1", "b3_r0.01", 1719, 999),
    ("b3_r0.01_rem2", "b3_r0.01", 1721, 998),
    ("b3_r0.01_rem3", "b3_r0.01", 1721, 999),
]
# rem2 原生复用既有样本文件（不新建）
NATIVE_REM2 = [("b22_r0.001", "b22_r0.001"), ("b22_r0.01", "b22_r0.01"),
               ("b22_r0.1", "b22_r0.1")]


def crop_one(src_name, out_name, h, w):
    img_p = os.path.join(SAMPLES, "%s.png" % src_name)
    msk_p = os.path.join(SAMPLES, "%s_mask.png" % src_name)
    img = cv2.imread(img_p, cv2.IMREAD_COLOR)
    msk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "缺图: %s" % img_p
    assert msk is not None, "缺mask: %s" % msk_p
    ih, iw = img.shape[:2]
    mh, mw = msk.shape[:2]
    assert (ih, iw) == (mh, mw), "图文尺寸不一致 %s" % src_name
    assert h <= ih and w <= iw, "裁剪 H=%d W=%d 超出源 H=%d W=%d" % (h, w, ih, iw)
    ci, cm = img[:h, :w], msk[:h, :w]
    cv2.imwrite(os.path.join(SAMPLES, "%s.png" % out_name), ci)
    cv2.imwrite(os.path.join(SAMPLES, "%s_mask.png" % out_name), cm)
    return (h, w), int(cm.sum()) // 255


def main():
    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
    manifest = []
    for out_name, src_name, h, w in CROPS:
        (ch, cw), npx = crop_one(src_name, out_name, h, w)
        rem = (ch * cw) % 4
        assert rem in (1, 2, 3), "%s (%dx%d) 余数=%d，不在1/2/3" % (out_name, ch, cw, rem)
        assert npx > 0, "%s 裁剪后无篡改像素" % out_name
        ratio = float(src_name.split("_r")[1])
        manifest.append({"name": out_name, "base": src_name, "ratio": ratio,
                         "size": "%dx%d" % (cw, ch), "hw_mod4": rem,
                         "src": src_name})
        print("CROP %-20s %dx%d hw%%4=%d mask_px=%d" % (out_name, cw, ch, rem, npx))

    for out_name, src_name in NATIVE_REM2:
        msk = cv2.imread(os.path.join(SAMPLES, "%s_mask.png" % src_name), cv2.IMREAD_GRAYSCALE)
        assert msk is not None, "缺mask: %s" % src_name
        h, w = msk.shape[:2]
        rem = (h * w) % 4
        assert rem == 2, "%s %dx%d 余数=%d 非2" % (src_name, w, h, rem)
        ratio = float(src_name.split("_r")[1])
        manifest.append({"name": out_name, "base": src_name, "ratio": ratio,
                         "size": "%dx%d" % (w, h), "hw_mod4": rem,
                         "src": src_name})
        print("NATIVE %-19s %dx%d hw%%4=%d" % (out_name, w, h, rem))

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    from collections import Counter
    c = Counter(m["hw_mod4"] for m in manifest)
    print("MANIFEST %d 项 -> %s ; 余数分布 %s" % (len(manifest), OUT_MANIFEST, dict(sorted(c.items()))))
    assert set(c) == {1, 2, 3}, "余数覆盖不完整: %s" % dict(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
