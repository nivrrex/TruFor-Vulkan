#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""param 后处理：给指定的层追加通用层参数 `31=<featmask>`（ncnn 的逐层 opt 覆盖）。

背景（源码依据，ncnn 9f9d4ec `src/net.cpp:115-132`）：
    layer->featmask = pd.get(31, 0)               # net.cpp:1762
    opt1.use_fp16_arithmetic = ... && !(featmask & (1<<0))
    opt1.use_fp16_packed     = ... && !(featmask & (1<<1))
    opt1.use_fp16_storage    = ... && !(featmask & (1<<1))   # bit1 同时关 packed 与 storage
    opt1.use_bf16_*          = ... && !(featmask & (1<<2))
    opt1.use_int8_*          = ... && !(featmask & (1<<3))
    opt1.use_vulkan_compute  = ... && !(featmask & (1<<4))   # bit4 = 该层回退 CPU
    opt1.use_tensor_storage  = ... && !(featmask & (1<<4))
    opt1.use_sgemm_convolution   = ... && !(featmask & (1<<5))
    opt1.use_winograd_convolution = ... && !(featmask & (1<<6))
故 `31=16`（bit4）= 让该层不走 Vulkan、回退 CPU；`31=2`（bit1）= 关该层 fp16 packed/storage。

匹配方式（两种同时计算、互为交叉校验，不一致即报错退出，不硬凑）：
  A) 层名匹配：--layers 显式给出（默认 softmax_361,softmax_362）
  B) 特征匹配：Softmax 层的输入 blob 沿产出层向上追溯（默认上限 8 跳），若链上存在
     `0=-1` 的 Reshape 则命中（用户给出的特征："Softmax 的输入来自 Reshape 0=-1"）。

输出：写出新 param；**原文件绝不被修改**。默认还会打印逐层 diff 与断言结果。

用法:
    python3 scripts/param_add_featmask.py --param <in.param> --out <out.param> \
        --mask 16 [--layers softmax_361,softmax_362] [--expect 2] [--depth 8]
"""
import argparse
import os
import sys

APEX = 8  # 特征追溯的最大跳数默认值


def parse_param(path):
    """返回 (magic_line, [layer dict], [原始文本行])。layer 保留原始 token 便于原样输出。"""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().splitlines()
    if not raw:
        raise ValueError("空 param 文件")
    magic = raw[0].strip()
    layers, lines = [], []
    for idx in range(1, len(raw)):
        text = raw[idx]
        lines.append(text)
        toks = text.split()
        if len(toks) < 5:
            continue
        try:
            nb, nt = int(toks[2]), int(toks[3])
        except ValueError:
            continue
        if 4 + nb + nt > len(toks):
            continue
        bottoms = toks[4:4 + nb]
        tops = toks[4 + nb:4 + nb + nt]
        params = toks[4 + nb + nt:]
        layers.append({"i": idx - 1, "line_no": idx + 1, "type": toks[0],
                       "name": toks[1], "bottoms": bottoms, "tops": tops,
                       "params": params, "raw": text})
    return magic, layers, lines


def owner_map(layers):
    """blob 数字 id → 产出该 blob 的层（取该层最后一个数字 top 作为主输出）。"""
    owners = {}
    for L in layers:
        nums = [t for t in L["tops"] if t.isdigit()]
        for b in nums:
            owners[b] = L
    return owners


def has_minus1_reshape_on_input_chain(L, owners, depth):
    """从 L 的输入 blob 向上追溯，链上是否出现 `0=-1` 的 Reshape。返回 (命中, 链描述)。"""
    cur, seen, chain = L, set(), []
    for _ in range(depth):
        nxt = None
        for b in cur["bottoms"]:
            if not b.isdigit():
                continue
            o = owners.get(b)
            if o is None:
                continue
            chain.append("%s(%s)" % (o["name"], o["type"]))
            if o["type"] == "Reshape" and any(p.startswith("0=") and p.split("=", 1)[1].split(",")[0] == "-1"
                                              for p in o["params"]):
                return True, " -> ".join(chain)
            if o["name"] in seen:
                continue
            seen.add(o["name"])
            nxt = o
            break
        if nxt is None:
            break
        cur = nxt
    return False, " -> ".join(chain)


def main():
    ap = argparse.ArgumentParser(description="给指定层追加 featmask（31=<mask>），不改原文件")
    ap.add_argument("--param", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", type=int, required=True, help="featmask 值，如 16(bit4=回退CPU)/2(bit1=关fp16)")
    ap.add_argument("--layers", default="softmax_361,softmax_362")
    ap.add_argument("--expect", type=int, default=2, help="期望命中的层数（断言）")
    ap.add_argument("--depth", type=int, default=APEX, help="特征追溯最大跳数")
    ap.add_argument("--no-feature-check", action="store_true",
                    help="目标不是 Softmax 层时跳过“输入链上存在 0=-1 Reshape”的特征匹配，"
                         "仅按 --layers 指定的层名匹配（批量给非 Softmax 层加 featmask 时用）")
    args = ap.parse_args()

    if os.path.abspath(args.param) == os.path.abspath(args.out):
        print("错误：--out 不得与 --param 相同（本脚本不修改原文件）", file=sys.stderr)
        return 2
    if args.mask <= 0:
        print("错误：--mask 必须为正整数且有实际意义的位", file=sys.stderr)
        return 2

    magic, layers, lines = parse_param(args.param)
    owners = owner_map(layers)
    by_name = {L["name"]: L for L in layers}

    want = [s.strip() for s in args.layers.split(",") if s.strip()]
    print("== 匹配方式 A：层名 ==")
    hit_a = []
    for nm in want:
        L = by_name.get(nm)
        if L is None:
            print("  未找到层 %s" % nm)
            continue
        ok = (L["type"] == "Softmax") or args.no_feature_check
        print("  %-14s type=%s 当前参数=%s %s" % (nm, L["type"], " ".join(L["params"]) or "(无)",
                                                "" if ok else "← 类型不是 Softmax，跳过"))
        if ok:
            hit_a.append(L)

    print("== 匹配方式 B：特征（输入链上存在 0=-1 的 Reshape） ==")
    hit_b = []
    if args.no_feature_check:
        print("  已跳过（--no-feature-check；目标不是 Softmax 层）")
        hit_b = list(hit_a)
    else:
        for L in layers:
            if L["type"] != "Softmax":
                continue
            ok, chain = has_minus1_reshape_on_input_chain(L, owners, args.depth)
            if ok:
                hit_b.append(L)
                print("  命中 %-14s 链: %s" % (L["name"], chain))
        for L in layers:
            if L["type"] == "Softmax" and L not in hit_b:
                _, chain = has_minus1_reshape_on_input_chain(L, owners, args.depth)
                print("  未命中 %-12s 链: %s" % (L["name"], chain or "(无上游)"))

    names_a = sorted(L["name"] for L in hit_a)
    names_b = sorted(L["name"] for L in hit_b)
    print("方式A 命中 %d 层: %s" % (len(hit_a), names_a))
    print("方式B 命中 %d 层: %s" % (len(hit_b), names_b))
    if names_a != names_b:
        print("错误：两种匹配结果不一致，拒绝硬凑，终止（不做任何写入）", file=sys.stderr)
        return 3
    if len(hit_a) != args.expect:
        print("错误：命中 %d 层，与 --expect %d 不符，终止" % (len(hit_a), args.expect),
              file=sys.stderr)
        return 4
    print("断言通过：两种匹配一致且恰好 %d 层" % len(hit_a))

    tag = "31=%d" % args.mask
    changed = {L["i"]: L for L in hit_a}
    out_lines = []
    print("== 写入 diff ==")
    for i, text in enumerate(lines):
        if i in changed:
            L = changed[i]
            new = text + " " + tag
            out_lines.append(new)
            print("  - %s\n  + %s" % (text, new))
        else:
            out_lines.append(text)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(magic + "\n")
        fh.write("\n".join(out_lines) + "\n")
    print("已写出：%s（原文件未改动）" % args.out)
    print("层数校验：原 %d 行层 / 新 %d 行层" % (len(lines), len(out_lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
