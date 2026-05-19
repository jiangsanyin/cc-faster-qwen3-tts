#!/usr/bin/env python3
"""
§6.2.8 顺序 3 原型：混合策略流式基准 - 真实 HTTP 流式模式（边产 codec 边解码边发送）。

与 hybrid_graph_parity_benchmark.py 的区别：
- 本脚本模拟真实 HTTP 流式服务：每产出一个 codec chunk 立即解码为 PCM 并"发送"
- 不再是"先收集全部 codec 再批量解码"，而是 codec 产出和解码交错进行
- 更准确反映真实服务的 TTFA / inter_chunk / 端到端延迟

架构：
- 路 A：CUDA Graph 快路径（generate_voice_clone_streaming, parity_mode=False）
- 路 B：parity 动态路径（generate_voice_clone_streaming, parity_mode=True）
- 调度：块级轮询，每轮从两路各拉一个 PCM chunk（如有），记录 wall 时间戳
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


def _run_streaming_generator(model, text, ref_audio, ref_text, chunk_size, max_new_tokens, parity_mode: bool):
    """返回 generate_voice_clone_streaming 生成器，产出 (audio_chunk, sr, timing)"""
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
        parity_mode=parity_mode,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="混合策略流式基准：真实 HTTP 流式模式")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref-audio", type=str, required=True)
    p.add_argument("--ref-text", type=str, default="")
    p.add_argument("--text-a", type=str, required=True, help="走 Graph 快路径的文本")
    p.add_argument("--text-b", type=str, required=True, help="走 parity 动态路径的文本")
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
        args.chunk_size, 48, parity_mode=False,
    )
    with torch.inference_mode():
        for _ in _warmup_gen:
            pass
    print("预热完成")

    # --- 准备两路流式生成器 ---
    # 路 A：Graph 快路径
    gen_a = _run_streaming_generator(
        model, args.text_a, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens, parity_mode=False,
    )
    # 路 B：parity 动态路径
    gen_b = _run_streaming_generator(
        model, args.text_b, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens, parity_mode=True,
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

            if not got_a and not got_b:
                break  # 两路都结束

    t_total = time.perf_counter()

    # --- 统计输出 ---
    print("\n=== 真实流式模式统计（边产边解边发）===")
    for label_key, label_name in [("A", "A (Graph)"), ("B", "B (parity)")]:
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
            f"rtf(model_timing)={rtf:.3f}"
        )

    print(f"\n  total_wall_ms={(t_total - t0) * 1000:.1f}")

    # --- 加速比（如果两路都有数据） ---
    if wall_pcm["A"] and wall_pcm["B"]:
        print("\n=== 对比（Graph / Parity）===")
        a_times = wall_pcm["A"]
        b_times = wall_pcm["B"]
        a_gaps = [a_times[i] - a_times[i - 1] for i in range(1, len(a_times))]
        b_gaps = [b_times[i] - b_times[i - 1] for i in range(1, len(b_times))]

        # TTFA 加速比（越小越好）
        if first_pcm_wall["B"] and first_pcm_wall["A"]:
            ttfa_speedup = first_pcm_wall["B"] / first_pcm_wall["A"]
            print(f"  TTFA 加速比      = {ttfa_speedup:.2f}x （Graph 首包比 Parity 快 {ttfa_speedup:.2f} 倍）")

        # inter_chunk 加速比（越小越好）
        if b_gaps and a_gaps:
            ic_speedup = _p95(b_gaps) / _p95(a_gaps)
            print(f"  inter_chunk_p95  = {ic_speedup:.2f}x （Graph 块间间隔比 Parity 紧凑 {ic_speedup:.2f} 倍）")

        # RTF 加速比（越大越好）
        rtf_a = audio_s["A"] / (total_gen_ms["A"] / 1000.0) if total_gen_ms["A"] > 0 else 0
        rtf_b = audio_s["B"] / (total_gen_ms["B"] / 1000.0) if total_gen_ms["B"] > 0 else 0
        if rtf_b > 0:
            rtf_speedup = rtf_a / rtf_b
            print(f"  RTF(model_timing)= {rtf_speedup:.2f}x （Graph 计算速度比 Parity 快 {rtf_speedup:.2f} 倍）")


if __name__ == "__main__":
    main()