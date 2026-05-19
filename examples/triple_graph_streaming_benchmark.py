#!/usr/bin/env python3
"""
§6.2.8 顺序 3 扩展：三路 Graph 流式基准 - 三路均使用 CUDA Graph 快路径。

与 dual_graph_streaming_benchmark.py 的区别：
- 本脚本三路均使用 Graph 快路径（parity_mode=False）
- 用于测试三路并发时的资源争抢和调度公平性
- 调度顺序：路 B → 路 A → 路 C（每轮各取一个 chunk）

架构：
- 路 A：CUDA Graph 快路径（parity_mode=False）
- 路 B：CUDA Graph 快路径（parity_mode=False）
- 路 C：CUDA Graph 快路径（parity_mode=False）
- 调度：块级轮询，每轮从三路各拉一个 PCM chunk（如有）
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_qwen3_tts.model import FasterQwen3TTS


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _run_streaming_generator(model, text, ref_audio, ref_text, chunk_size, max_new_tokens):
    """返回 generate_voice_clone_streaming 生成器（Graph 快路径，parity_mode=False）"""
    return model.generate_voice_clone_streaming(
        text=text,
        language="Auto",
        ref_audio=ref_audio,
        ref_text=ref_text,
        xvec_only=False,
        non_streaming_mode=False,
        append_silence=True,
        chunk_size=chunk_size,
        max_new_tokens=max_new_tokens,
        parity_mode=False,  # 三路都使用 Graph 快路径
    )


def main() -> None:
    p = argparse.ArgumentParser(description="三路 Graph 流式基准：三路均使用 CUDA Graph 快路径")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref-audio", type=str, required=True)
    p.add_argument("--ref-text", type=str, default="")
    p.add_argument("--text-a", type=str, required=True, help="路 A 的文本")
    p.add_argument("--text-b", type=str, required=True, help="路 B 的文本")
    p.add_argument("--text-c", type=str, required=True, help="路 C 的文本")
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    # 加载模型（CUDA Graph 启用）
    print("=== 加载模型 ===")
    model = FasterQwen3TTS.from_pretrained(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    print(f"talker_graph={model.talker_graph is not None}, predictor_graph={model.predictor_graph is not None}")

    # --- 预热（Graph 路径需要 capture） ---
    print("\n=== 预热（CUDA Graph capture）===")
    _warmup_gen = _run_streaming_generator(
        model, "预热", args.ref_audio, args.ref_text,
        args.chunk_size, 48,
    )
    with torch.inference_mode():
        for _ in _warmup_gen:
            pass
    print("CUDA Graph capture 完成")

    # --- 三路首包预热：分别触发首次计算 ---
    print("\n=== 三路首包预热 ===")
    with torch.inference_mode():
        for label, text in [("A", "预热A"), ("B", "预热B"), ("C", "预热C")]:
            _warmup_gen = _run_streaming_generator(
                model, text, args.ref_audio, args.ref_text,
                args.chunk_size, 48,
            )
            try:
                next(_warmup_gen)
                print(f"路 {label} 首包预热完成")
            except StopIteration:
                pass
    print("全部预热完成")

    # --- 准备三路流式生成器（都是 Graph 快路径） ---
    gen_a = _run_streaming_generator(
        model, args.text_a, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens,
    )
    gen_b = _run_streaming_generator(
        model, args.text_b, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens,
    )
    gen_c = _run_streaming_generator(
        model, args.text_c, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens,
    )

    # --- 混合调度：块级轮询，边产边解边发 ---
    wall_pcm: DefaultDict[str, List[float]] = defaultdict(list)
    total_gen_ms: DefaultDict[str, float] = defaultdict(float)
    audio_s: DefaultDict[str, float] = defaultdict(float)
    first_pcm_wall: DefaultDict[str, Optional[float]] = defaultdict(lambda: None)
    finished: DefaultDict[str, bool] = defaultdict(bool)

    t0 = time.perf_counter()

    with torch.inference_mode():
        while True:
            got_a = False
            got_b = False
            got_c = False

            # --- 路 A：尝试拉一个 PCM chunk ---
            if not finished["A"]:
                try:
                    audio_chunk, sr, timing = next(gen_a)
                    now = time.perf_counter()
                    wall_pcm["A"].append((now - t0) * 1000.0)
                    if first_pcm_wall["A"] is None:
                        first_pcm_wall["A"] = (now - t0) * 1000.0
                    total_gen_ms["A"] += float(timing.get("prefill_ms", 0.0)) + float(timing.get("decode_ms", 0.0))
                    audio_s["A"] += len(audio_chunk) / float(sr or 24000)
                    got_a = True
                except StopIteration:
                    finished["A"] = True
            
            # --- 路 B：尝试拉一个 PCM chunk ---
            if not finished["B"]:
                try:
                    audio_chunk, sr, timing = next(gen_b)
                    now = time.perf_counter()
                    wall_pcm["B"].append((now - t0) * 1000.0)
                    if first_pcm_wall["B"] is None:
                        first_pcm_wall["B"] = (now - t0) * 1000.0
                    total_gen_ms["B"] += float(timing.get("prefill_ms", 0.0)) + float(timing.get("decode_ms", 0.0))
                    audio_s["B"] += len(audio_chunk) / float(sr or 24000)
                    got_b = True
                except StopIteration:
                    finished["B"] = True

            # --- 路 C：尝试拉一个 PCM chunk ---
            if not finished["C"]:
                try:
                    audio_chunk, sr, timing = next(gen_c)
                    now = time.perf_counter()
                    wall_pcm["C"].append((now - t0) * 1000.0)
                    if first_pcm_wall["C"] is None:
                        first_pcm_wall["C"] = (now - t0) * 1000.0
                    total_gen_ms["C"] += float(timing.get("prefill_ms", 0.0)) + float(timing.get("decode_ms", 0.0))
                    audio_s["C"] += len(audio_chunk) / float(sr or 24000)
                    got_c = True
                except StopIteration:
                    finished["C"] = True

            if not got_a and not got_b and not got_c:
                break  # 三路都结束

    t_total = time.perf_counter()

    # --- 统计输出 ---
    print("\n=== 三路 Graph 模式统计（三路均使用 CUDA Graph）===")
    for label_key, label_name in [("A", "路 A (Graph)"), ("B", "路 B (Graph)"), ("C", "路 C (Graph)")]:
        times = wall_pcm[label_key]
        if not times:
            print(f"  {label_name}: no PCM chunks")
            continue

        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        ttfa_pcm = first_pcm_wall[label_key] or times[0]
        rtf = (audio_s[label_key] / (total_gen_ms[label_key] / 1000.0)) if total_gen_ms[label_key] > 0 else 0.0

        print(
            f"  {label_name}: "
            f"ttfa_wall_ms(audio)={ttfa_pcm:.1f} "
            f"inter_chunk_max_ms={max(gaps) if gaps else 0:.1f} "
            f"inter_chunk_p95_ms={_p95(gaps):.1f} "
            f"n_chunks={len(times)} "
            f"audio_s={audio_s[label_key]:.2f} "
            f"total_gen_ms={total_gen_ms[label_key]:.2f} "
            f"rtf(model_timing)={rtf:.3f}"
        )

    print(f"\n  total_wall_ms={(t_total - t0) * 1000:.1f}")

    # --- 三路对比 ---
    all_keys = ["A", "B", "C"]
    if all(wall_pcm[k] for k in all_keys):
        print("\n=== 三路 Graph 对比 ===")

        # 提取各路的 gaps
        gaps_map = {}
        for key in all_keys:
            times = wall_pcm[key]
            gaps_map[key] = [times[i] - times[i - 1] for i in range(1, len(times))]

        # TTFA 分析（按调度顺序 B -> A -> C）
        ttfa_order = [("A", "路A"), ("B", "路B"), ("C", "路C")]
        print("\n  TTFA（按调度顺序）:")
        for i, (key, name) in enumerate(ttfa_order):
            ttfa = first_pcm_wall[key]
            print(f"    {name}: {ttfa:.1f}ms (第{i+1}个调度)")

        # TTFA 差距（最大 - 最小）
        ttfas = [first_pcm_wall[k] for k in all_keys]
        ttfa_diff = max(ttfas) - min(ttfas)
        print(f"  TTFA 最大差距    = {ttfa_diff:.1f}ms")

        # inter_chunk 分析
        print("\n  inter_chunk_p95:")
        for key, name in [("A", "路A"), ("B", "路B"), ("C", "路C")]:
            print(f"    {name}: {_p95(gaps_map[key]):.1f}ms")
        ic_values = [_p95(gaps_map[k]) for k in all_keys]
        ic_diff = max(ic_values) - min(ic_values)
        print(f"  inter_chunk_p95 差距 = {ic_diff:.1f}ms")

        # RTF 分析
        print("\n  RTF:")
        rtfs = {}
        for key, name in [("A", "路A"), ("B", "路B"), ("C", "路C")]:
            rtf = audio_s[key] / (total_gen_ms[key] / 1000.0) if total_gen_ms[key] > 0 else 0
            rtfs[key] = rtf
            print(f"    {name}: {rtf:.3f}")
        rtf_diff = max(rtfs.values()) - min(rtfs.values())
        print(f"  RTF 差距         = {rtf_diff:.3f}")

        # 与单路 Graph 预期对比（基于之前测试结果 ~180ms inter_chunk）
        if gaps_map["A"]:
            slowdown = _p95(gaps_map["A"]) / 180.0
            print(f"\n  相对单路 slowdown = {slowdown:.2f}x （inter_chunk 相对于单路 Graph 的倍数）")
            print(f"  理论 3路 slowdown ≈ 3.00x")


if __name__ == "__main__":
    main()