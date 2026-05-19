#!/usr/bin/env python3
"""
§6.2.8 顺序 3 扩展：双路 Graph 流式基准 - 两路均使用 CUDA Graph 快路径。

与 hybrid_graph_parity_streaming_benchmark.py 的区别：
- 本脚本两路均使用 Graph 快路径（parity_mode=False）
- 用于对比测试：两路都是"快"的时候，是否会有资源争抢或调度问题
- 与混合策略（1 Graph + 1 parity）形成对照

架构：
- 路 A：CUDA Graph 快路径（parity_mode=False）
- 路 B：CUDA Graph 快路径（parity_mode=False）
- 调度：块级轮询，每轮从两路各拉一个 PCM chunk（如有）
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
        parity_mode=False,  # 两路都使用 Graph 快路径
    )


def main() -> None:
    p = argparse.ArgumentParser(description="双路 Graph 流式基准：两路均使用 CUDA Graph 快路径")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref-audio", type=str, required=True)
    p.add_argument("--ref-text", type=str, default="")
    p.add_argument("--text-a", type=str, required=True, help="路 A 的文本")
    p.add_argument("--text-b", type=str, required=True, help="路 B 的文本")
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    # 加载模型（CUDA Graph 启用）
    print("=== 加载模型 ===")
    model = FasterQwen3TTS.from_pretrained(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    print(f"talker_graph={model.talker_graph is not None}, predictor_graph={model.predictor_graph is not None}")

    # --- 预热（Graph 路径需要 capture + 两路首包预热） ---
    print("\n=== 预热（CUDA Graph capture）===")
    _warmup_gen = _run_streaming_generator(
        model, "预热", args.ref_audio, args.ref_text,
        args.chunk_size, 48,
    )
    with torch.inference_mode():
        for _ in _warmup_gen:
            pass
    print("CUDA Graph capture 完成")

    # --- 两路首包预热：分别触发首次计算，消除正式测试时的 TTFA 差距 ---
    print("\n=== 两路首包预热 ===")
    with torch.inference_mode():
        # 路 A 首包预热
        _warmup_gen_a = _run_streaming_generator(
            model, "预热A", args.ref_audio, args.ref_text,
            args.chunk_size, 48,
        )
        try:
            next(_warmup_gen_a)  # 只取第一个 chunk，触发预填充
            print("路 A 首包预热完成")
        except StopIteration:
            pass

        # 路 B 首包预热
        _warmup_gen_b = _run_streaming_generator(
            model, "预热B", args.ref_audio, args.ref_text,
            args.chunk_size, 48,
        )
        try:
            next(_warmup_gen_b)  # 只取第一个 chunk，触发预填充
            print("路 B 首包预热完成")
        except StopIteration:
            pass
    print("全部预热完成")

    # --- 准备两路流式生成器（都是 Graph 快路径） ---
    # 路 A：Graph 快路径
    gen_a = _run_streaming_generator(
        model, args.text_a, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens,
    )
    # 路 B：Graph 快路径
    gen_b = _run_streaming_generator(
        model, args.text_b, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens,
    )

    # --- 混合调度：块级轮询，边产边解边发 ---
    # 记录每路 PCM 块的墙钟时间戳
    wall_pcm: DefaultDict[str, List[float]] = defaultdict(list)
    # 记录每路的累计指标
    total_gen_ms: DefaultDict[str, float] = defaultdict(float)
    audio_s: DefaultDict[str, float] = defaultdict(float)
    first_pcm_wall: DefaultDict[str, Optional[float]] = defaultdict(lambda: None)
    finished: DefaultDict[str, bool] = defaultdict(bool)

    t0 = time.perf_counter()

    with torch.inference_mode():
        while True:
            got_a = False
            got_b = False

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

            # --- 路 A：尝试拉一个 PCM chunk ---
            if not finished["A"]:
                try:
                    audio_chunk, sr, timing = next(gen_a)
                    now = time.perf_counter()
                    # 每产出一个chunk，记录一次墙钟时间
                    wall_pcm["A"].append((now - t0) * 1000.0)
                    if first_pcm_wall["A"] is None:
                        first_pcm_wall["A"] = (now - t0) * 1000.0
                    total_gen_ms["A"] += float(timing.get("prefill_ms", 0.0)) + float(timing.get("decode_ms", 0.0))
                    audio_s["A"] += len(audio_chunk) / float(sr or 24000)
                    got_a = True
                except StopIteration:
                    finished["A"] = True

            if not got_a and not got_b:
                break  # 两路都结束

    t_total = time.perf_counter()

    # --- 统计输出 ---
    print("\n=== 双路 Graph 模式统计（两路均使用 CUDA Graph）===")
    for label_key, label_name in [("A", "路 A (Graph)"), ("B", "路 B (Graph)")]:
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

    # --- 两路对比 ---
    if wall_pcm["A"] and wall_pcm["B"]:
        print("\n=== 两路 Graph 对比 ===")
        a_times = wall_pcm["A"]
        b_times = wall_pcm["B"]
        a_gaps = [a_times[i] - a_times[i - 1] for i in range(1, len(a_times))]
        b_gaps = [b_times[i] - b_times[i - 1] for i in range(1, len(b_times))]

        # TTFA 差距
        if first_pcm_wall["B"] and first_pcm_wall["A"]:
            ttfa_diff = abs(first_pcm_wall["B"] - first_pcm_wall["A"])
            print(f"  TTFA 差距        = {ttfa_diff:.1f}ms （两路首包时间差）")

        # inter_chunk 差距
        if b_gaps and a_gaps:
            ic_diff = abs(_p95(b_gaps) - _p95(a_gaps))
            print(f"  inter_chunk_p95 差距 = {ic_diff:.1f}ms （两路块间间隔差）")

        # RTF 差距
        rtf_a = audio_s["A"] / (total_gen_ms["A"] / 1000.0) if total_gen_ms["A"] > 0 else 0
        rtf_b = audio_s["B"] / (total_gen_ms["B"] / 1000.0) if total_gen_ms["B"] > 0 else 0
        rtf_diff = abs(rtf_a - rtf_b)
        print(f"  RTF 差距         = {rtf_diff:.3f} （两路计算速度差）")

        # 与单路 Graph 预期对比（基于之前测试结果 ~180ms inter_chunk）
        if a_gaps:
            slowdown = _p95(a_gaps) / 180.0  # 假设单路 Graph inter_chunk ~180ms
            print(f"  相对单路 slowdown = {slowdown:.2f}x （inter_chunk 相对于单路 Graph 的倍数）")


if __name__ == "__main__":
    main()