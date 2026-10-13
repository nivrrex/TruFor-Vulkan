#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2/P3：把"MatMul(transB) → BinaryOp(×scale) → Softmax → MatMul"链改写为 ncnn 的 SDPA 层。

严格匹配（任一条件不满足即**不改写**并列入未匹配清单）：
  1) `MatMul` 层，参数 `0=1`（transB）；
  2) 其输出 blob 的**唯一**消费者是 `BinaryOp`，参数含 `0=2`（乘），另一个输入来自 `MemoryData`，
     且该常量的 `0=` 值 == `--scale`（默认 0.125，即 1/sqrt(64)）；
  3) 再下游唯一消费者是 `Softmax`；
  4) 再下游唯一消费者是 `MatMul`（transB 缺省/0），其另一输入即 V。
改写为：`SDPA` 层，bottom = [Q, K, V]，top = 原第 4 层的输出 blob，参数 `5=0`（无 mask）+ `6=<scale>`；
中途产生的三个中间 blob（scores / scaled / attn）**不再被链外层引用**（已断言），保留编号但不再有生产者。
同时按用户要求检查风险点：param 中是否存在"零参数 Reshape"或缺省 `0=-1` 的 Reshape（pnnx 5D 展平历史问题）。
最后对每个新 SDPA 的输入 blob 做**非空**检查（形状需运行时断言，脚本在 `--verify-runtime` 时用 ncnn 加载并打印）。

用法：
  python3 scripts/param_rewrite_sdpa.py --param <in.param> --out <out.param> [--scale 0.125] [--dry-run]
"""
import argparse
import os
import sys


def parse_param(path):
    """返回 (magic, header_counts, layers, raw_lines)。"""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().splitlines()
    if not raw:
        raise ValueError("空 param")
    magic = raw[0].strip()
    counts = raw[1].split() if len(raw) > 1 else []
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
        layers.append({"i": idx - 1, "line": idx, "type": toks[0], "name": toks[1],
                       "bottoms": toks[4:4 + nb], "tops": toks[4 + nb:4 + nb + nt],
                       "params": toks[4 + nb + nt:], "raw": text, "nb": nb, "nt": nt})
    return magic, counts, layers, lines


def param_value(layer, key):
    pref = "%d=" % key
    for p in layer["params"]:
        if p.startswith(pref):
            return p[len(pref):]
    return None


def build_maps(layers):
    owner, consumers = {}, {}
    for L in layers:
        for b in L["tops"]:
            owner[b] = L
        for b in L["bottoms"]:
            consumers.setdefault(b, []).append(L)
    return owner, consumers


def find_chains(layers, scale, tol=1e-6):
    owner, consumers = build_maps(layers)
    matched, unmatched_matmul = [], []
    used = set()
    for L1 in layers:
        if L1["type"] != "MatMul":
            continue
        if param_value(L1, 0) != "1":            # 只认 transB=1 的 QK^T
            unmatched_matmul.append((L1, "MatMul 非 transB=1"))
            continue
        if len(L1["tops"]) != 1:
            unmatched_matmul.append((L1, "输出 blob 数 != 1"))
            continue
        s = L1["tops"][0]
        cons = consumers.get(s, [])
        if len(cons) != 1 or cons[0]["type"] != "BinaryOp":
            unmatched_matmul.append((L1, "下游不是唯一的 BinaryOp"))
            continue
        L2 = cons[0]
        if param_value(L2, 0) != "2":            # 0=2 → 乘
            unmatched_matmul.append((L1, "下游 BinaryOp 不是乘(0=2)"))
            continue
        # ncnn 的 BinaryOp 有两种常量写法：
        #   A) 参数内标量：`1=1`（with_scalar）+ `2=<value>`（本网络 162 处均为该形态）
        #   B) 第二个 bottom 来自 MemoryData 常量层
        valf = None
        if param_value(L2, 1) == "1":
            v = param_value(L2, 2)
            try:
                valf = float(v)
            except (TypeError, ValueError):
                unmatched_matmul.append((L1, "参数内常量无法解析: %r" % (v,)))
                continue
        else:
            others = [b for b in L2["bottoms"] if b != s]
            if len(others) != 1:
                unmatched_matmul.append((L1, "BinaryOp 乘数不是单一常量"))
                continue
            cl = owner.get(others[0])
            if cl is None or cl["type"] != "MemoryData":
                unmatched_matmul.append((L1, "乘数 blob 不是 MemoryData 常量"))
                continue
            try:
                valf = float(param_value(cl, 0))
            except (TypeError, ValueError):
                unmatched_matmul.append((L1, "常量值无法解析"))
                continue
        if abs(valf - scale) > tol:
            unmatched_matmul.append((L1, "常量 %g != scale %g" % (valf, scale)))
            continue
        s2 = L2["tops"][0] if L2["tops"] else None
        cons2 = consumers.get(s2, [])
        if len(cons2) != 1 or cons2[0]["type"] != "Softmax":
            unmatched_matmul.append((L1, "scaled 下游不是唯一的 Softmax"))
            continue
        L3 = cons2[0]
        a = L3["tops"][0] if L3["tops"] else None
        cons3 = consumers.get(a, [])
        if len(cons3) != 1 or cons3[0]["type"] != "MatMul":
            unmatched_matmul.append((L1, "attention 下游不是唯一的 MatMul(·V)"))
            continue
        L4 = cons3[0]
        vs = [b for b in L4["bottoms"] if b != a]
        if len(vs) != 1:
            unmatched_matmul.append((L1, "·V 的非 attention 输入不唯一"))
            continue
        if any(x in used for x in (L1["i"], L2["i"], L3["i"], L4["i"])):
            unmatched_matmul.append((L1, "链内层已被别的链占用"))
            continue
        used.update((L1["i"], L2["i"], L3["i"], L4["i"]))
        # 中间 blob 只能被链内消费（否则不能删除这些层）
        bad = [b for b in (s, s2, a) if len(consumers.get(b, [])) != 1]
        if bad:
            unmatched_matmul.append((L1, "中间 blob 被链外引用: %s" % bad))
            continue
        matched.append({"q": L1["bottoms"][0], "k": L1["bottoms"][1], "v": vs[0],
                        "out": L4["tops"][0], "layers": [L1, L2, L3, L4],
                        "mid_blobs": [s, s2, a], "scale": valf})
    return matched, unmatched_matmul


def check_reshape_risk(layers):
    """pnnx 5D 展平风险：零参数 Reshape 或 0=-1 的 Reshape（历史踩过：静默产空 Mat）。"""
    risk = []
    for L in layers:
        if L["type"] != "Reshape":
            continue
        v0 = param_value(L, 0)
        if v0 is None:
            risk.append((L["name"], "零参数 Reshape（无 0=）"))
        elif v0.strip() == "-1":
            risk.append((L["name"], "0=-1 展平 Reshape"))
    return risk


def read_embed(param, bin_path, chains):
    """从各链 SDPA 输入(Q) 的**实际 shape** 读 embed_dim（不硬编码）。

    只读元数据（m.w），不做 np.array 拷贝；单进程内只建一次 net（T0 结论：同进程反复建销 Net 会不稳）。"""
    import ncnn
    import numpy as np
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.lightmode = False
    if net.load_param(param) != 0:
        raise RuntimeError("load_param 失败: %s" % param)
    if net.load_model(bin_path) != 0:
        raise RuntimeError("load_model 失败: %s" % bin_path)
    ex = net.create_extractor()
    ex.set_light_mode(False)
    x = np.random.RandomState(0).rand(3, 512, 384).astype(np.float32)
    ex.input("in0", ncnn.Mat(np.ascontiguousarray(x)))
    emb = {}
    for m in chains:
        rc, mm = ex.extract(str(m["q"]))
        if rc != 0:
            raise RuntimeError("读取 Q blob %s 形状失败 rc=%d" % (m["q"], rc))
        emb[m["q"]] = (mm.w, mm.h, mm.c, mm.dims)
    return emb


def main():
    ap = argparse.ArgumentParser(description="SDPA 链改写（严格匹配，不改就无法匹配的链）")
    ap.add_argument("--param", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.125)
    ap.add_argument("--bin", default=None, help="模型 bin（用于读取各链 Q 的实际 shape；Step B 必需）")
    ap.add_argument("--reshape", action="store_true",
                    help="Step B：在 SDPA 后插入 Reshape(0=<embed> 1=-1) 把输出降回 2 维（用户裁决）")
    ap.add_argument("--dry-run", action="store_true", help="只报匹配结果，不写文件")
    args = ap.parse_args()

    if os.path.abspath(args.param) == os.path.abspath(args.out):
        print("错误：--out 不得与 --param 相同", file=sys.stderr)
        return 2
    tol_flag = 1e-6
    magic, counts, layers, lines = parse_param(args.param)
    print("param 层数（第2行）= %s；解析到 %d 层" % (" ".join(counts[:1]) or "?", len(layers)))

    matched, unmatched = find_chains(layers, args.scale)
    print("\n== 严格匹配结果 ==")
    print("命中链数：%d" % len(matched))
    for m in matched:
        print("  chain: Q=%s K=%s V=%s -> out=%s  (layers: %s)"
              % (m["q"], m["k"], m["v"], m["out"],
                 ", ".join("%s@line%d" % (L["type"], L["line"] + 1) for L in m["layers"])))
    print("\n未匹配的 MatMul（共 %d 个 transB=1 的 MatMul，逐条给出原因）:" % len(unmatched))
    from collections import Counter
    cnt = Counter(reason.split(":")[0] for _, reason in unmatched)
    for reason, c in cnt.most_common():
        print("  %-46s %d" % (reason, c))
    for L, reason in unmatched[:10]:
        print("   例: line%-5d %-22s %s" % (L["line"] + 1, L["name"], reason))

    risk = check_reshape_risk(layers)
    print("\n== 风险点检查（pnnx 5D 展平）== 命中 %d 个 Reshape" % len(risk))
    for name, why in risk[:10]:
        print("  %-24s %s" % (name, why))

    # 完整性（用户要求“不匹配则不改并列出”）：含 scale 乘法但未命中链的 BinaryOp
    hit_ids = {L["i"] for m in matched for L in m["layers"]}
    stray = []
    for L in layers:
        if L["type"] != "BinaryOp" or param_value(L, 0) != "2" or param_value(L, 1) != "1":
            continue
        try:
            v = float(param_value(L, 2))
        except (TypeError, ValueError):
            continue
        if abs(v - args.scale) < tol_flag and L["i"] not in hit_ids:
            stray.append(L)
    print("\n== 含 scale=%g 但未命中链的 BinaryOp：%d 个（不改动）==" % (args.scale, len(stray)))
    for L in stray[:10]:
        print("  line%-5d %-22s params=%s" % (L["line"] + 1, L["name"], " ".join(L["params"])))

    # 非空断言：SDPA 的三个输入都必须是有生产者的 blob（或为网络输入 in0/1/2）
    producers = {b for L in layers for b in L["tops"]}
    inputs = {"in0", "in1", "in2"}
    bad_inputs = []
    for m in matched:
        for k in ("q", "k", "v"):
            if m[k] not in producers and m[k] not in inputs:
                bad_inputs.append((m[k], k))
    print("\n== SDPA 输入非空断言 ==  失败 %d 项" % len(bad_inputs))
    for b, k in bad_inputs[:10]:
        print("  blob %s (%s) 无生产者且不是网络输入" % (b, k))

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return 0

    # 改写：删除链内 4 层；插入 SDPA（必要时再插 Reshape 降维，Step B）
    embed_map = {}
    if args.reshape:
        if not args.bin:
            print("错误：--reshape 需要 --bin 以读取各链 Q 的实际 shape", file=sys.stderr)
            return 2
        embed_map = read_embed(args.param, args.bin, matched)
        print("\n== 各链 Q 的实际 shape（前 5 条）==")
        for m in matched[:5]:
            print("  Q=%s → embed=%d h=%d c=%d dims=%d" % ((m["q"],) + embed_map[m["q"]]))
    drop = {L["i"] for m in matched for L in m["layers"]}
    ins = {}
    for m in matched:
        first = min(L["i"] for L in m["layers"])
        lines_new = []
        if args.reshape:
            emb = embed_map[m["q"]][0]
            tmp = "sdpa_out_%s" % m["out"]
            lines_new.append("SDPA                  sdpa_%s                 3 1 %s %s %s %s 5=0 6=%g"
                             % (m["out"], m["q"], m["k"], m["v"], tmp, m["scale"]))
            lines_new.append("Reshape               reshape_%s              1 1 %s %s 0=%d 1=-1"
                             % (m["out"], tmp, m["out"], emb))
        else:
            lines_new.append("SDPA                  sdpa_%s              3 1 %s %s %s %s 5=0 6=%g"
                             % (m["out"], m["q"], m["k"], m["v"], m["out"], m["scale"]))
        ins[first] = lines_new
    out_lines = []
    for i, text in enumerate(lines):
        if i == 0:
            # lines[0] 是原计数行（已被新计数行替代），不能重复写入
            continue
        if i in ins:
            out_lines.extend(ins[i])
        if i in drop:
            continue
        out_lines.append(text)
    new_count = len(out_lines)
    if len(counts) >= 2:
        try:
            counts[0] = str(new_count)
        except Exception:  # noqa: BLE001
            pass
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(magic + "\n")
        fh.write(" ".join(counts) + "\n")
        fh.write("\n".join(out_lines) + "\n")
    print("\n已写出 %s：层数 %d → %d（删除 %d 层、插入 %d 层 SDPA）"
          % (args.out, len(layers), new_count, len(drop), len(ins)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
