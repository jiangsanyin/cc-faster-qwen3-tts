#!/usr/bin/env python3
"""
§6.2.8 顺序 3 原型：**CUDA Graph 快路径 vs Parity 动态路径** 单路基线对比。

目的：量化 Graph 路径相对于 Parity 路径的 **TTFA / inter_chunk / RTF** 加速比，
为后续「混合策略」（单路 Graph + 并发 parity 回退）提供基线数据。

**关键设计**：使用 **同一个模型实例** 分别跑 Graph 和 Parity 路径，
避免二次加载模型带来的状态差异。Graph 路径需要 TalkerGraph（默认加载），
Parity 路径在同一实例上通过 ``parity_mode=True`` 走动态 KV。

在 ``faster-qwen3-tts`` 目录下示例：

    python examples/graph_vs_parity_benchmark.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --ref-audio /path/to/ref.wav \\
        --ref-text "参考文本" \\
        --text "合成文本"

**RTF 口径**：``audio_s / (prefill_ms + decode_ms) / 1000``，**越大越好**（> 1.0 即实时）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_qwen3_tts.model import FasterQwen3TTS


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _run_streaming(model: FasterQwen3TTS, text, ref_audio, ref_text, chunk_size, max_new_tokens,
                   parity_mode: bool, label: str) -> dict:
    """跑一次流式合成，返回指标字典。"""
    wall_times: List[float] = []
    total_gen_ms = 0.0
    audio_s = 0.0
    sr0 = 24000
    hit_max_tokens = False

    t0 = time.perf_counter()
    gen = model.generate_voice_clone_streaming(
        text=text,
        language="Auto",
        ref_audio=ref_audio,
        ref_text=ref_text,
        xvec_only=False,
        non_streaming_mode=False,
        append_silence=True,
        chunk_size=chunk_size,
        # max_new_tokens: 流式合成的安全上限，控制 decode 阶段最多生成多少个 codec token
        # 防止无限生成（即使模型异常不触发 EOS，也会在达到上限后强制停止）
        max_new_tokens=max_new_tokens,
        parity_mode=parity_mode,
    )
    with torch.inference_mode():
        # gen 是生成器，每次迭代返回一个音频块及其元数据
        # audio_chunk: np.ndarray - PCM 音频块（波形数据），shape 为 (n_samples,)
        # sr: int - 采样率（sample rate），单位 Hz，通常为 24000（24kHz）
        # timing: dict - 计时信息，包含 prefill_ms/decode_ms/chunk_index/is_final 等字段
        for audio_chunk, sr, timing in gen:
            now = time.perf_counter()
            wall_times.append((now - t0) * 1000.0)
            total_gen_ms += float(timing.get("prefill_ms", 0.0)) + float(timing.get("decode_ms", 0.0))
            if sr:
                sr0 = sr
            audio_s += len(audio_chunk) / float(sr0)
            # 检测是否命中 max_new_tokens 上限（is_final=False 且累计步数已达上限）
            total_steps = timing.get("total_steps_so_far", 0)
            is_final = timing.get("is_final", False)

    t_total = time.perf_counter()

    # 估算总 token 数
    total_tokens_est = len(wall_times) * chunk_size

    # 判断是否异常：产出音频时长远超正常范围（12字文本 ≈ 2~10 秒音频）
    # EOS = End Of Sequence（序列结束标记），模型生成完语音后会输出此标记表示结束
    # 如果参考音频与文本不匹配或模型状态异常，模型可能永远不输出 EOS，一直生成下去
    if audio_s > 30.0:
        print(f"  ** 异常：{label} 产出 {audio_s:.2f}s 音频（远超正常范围），可能未触发 EOS **")

    # 达到 max_new_tokens 上限仍未 EOS，说明模型异常未正常结束
    if total_tokens_est >= max_new_tokens:
        print(f"  ** 异常：{label} 累计约 {total_tokens_est} tokens（达 max_new_tokens 上限），可能未触发 EOS **")
        hit_max_tokens = True

    gaps = [wall_times[i] - wall_times[i - 1] for i in range(1, len(wall_times))]
    rtf = (audio_s / (total_gen_ms / 1000.0)) if total_gen_ms > 0 else 0.0
    wall_rtf = (audio_s / ((t_total - t0))) if (t_total - t0) > 0 else 0.0

    return {
        "mode": "parity" if parity_mode else "graph",
        "ttfa_wall_ms": wall_times[0] if wall_times else 0.0,
        "inter_chunk_max_ms": max(gaps) if gaps else 0.0,
        "inter_chunk_p95_ms": _p95(gaps),
        "n_chunks": len(wall_times),
        "audio_s": audio_s,
        "total_gen_ms": total_gen_ms,
        "rtf_model_timing": rtf,
        "rtf_wall_clock": wall_rtf,
        "total_wall_ms": (t_total - t0) * 1000.0,
        "total_tokens_est": total_tokens_est,
        "hit_max_tokens": hit_max_tokens,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="CUDA Graph vs Parity 基线对比基准")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref-audio", type=str, required=True)
    p.add_argument("--ref-text", type=str, default="")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    # --- 加载一个模型实例（CUDA Graph 启用），两条路径共用 ---
    print("=== 加载模型（Graph + Parity 共用）===")
    model = FasterQwen3TTS.from_pretrained(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    print(f"talker_graph={model.talker_graph is not None}, predictor_graph={model.predictor_graph is not None}")

    # --- 0. 预热线程（Graph 路径必须先完成 capture，否则第一次调用会包含 capture 时间） ---
    print("\n=== 预热（CUDA Graph capture）===")
    # 用 dummy 预热：跑一条短文本完成 Graph capture，但不计时
    _ = _run_streaming(
        model, "预热", args.ref_audio, args.ref_text,
        args.chunk_size, 48, parity_mode=False, label="Warmup",
    )
    print("预热完成，Graph 已捕获")

    # --- 1. Graph 快路径（正式计时） ---
    print("\n=== 跑 Graph 快路径（正式计时） ===")
    graph_result = _run_streaming(
        model, args.text, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens, parity_mode=False, label="Graph",
    )

    # 清理 GPU 缓存，避免 Graph 的 KV Cache 残留影响 Parity 路径
    torch.cuda.empty_cache()

    # --- 2. Parity 动态路径（同一模型实例，切换 parity_mode） ---
    print("\n=== 跑 Parity 动态路径 ===")
    parity_result = _run_streaming(
        model, args.text, args.ref_audio, args.ref_text,
        args.chunk_size, args.max_new_tokens, parity_mode=True, label="Parity",
    )

    # --- 3. 对比输出 ---
    print("\n=== Graph 快路径 ===")
    _print_result(graph_result)

    print("\n=== Parity 动态路径 ===")
    _print_result(parity_result)

    # --- 4. 有效性检查 ---
    graph_audio = graph_result["audio_s"]
    parity_audio = parity_result["audio_s"]
    if graph_result["hit_max_tokens"] or parity_result["hit_max_tokens"]:
        print("\n** 加速比计算无效：至少一条路径未正常触发 EOS，音频时长差异过大 **")
        print(f"   Graph audio_s={graph_audio:.2f}s, Parity audio_s={parity_audio:.2f}s")
        print(f"   加速比仅供参考，不建议用于产品决策。")
    else:
        print("\n=== 加速比（Graph / Parity）===")
        _print_speedup(graph_result, parity_result)


def _print_result(r: dict) -> None:
    print(f"  ttfa_wall_ms       = {r['ttfa_wall_ms']:.1f}")
    print(f"  inter_chunk_max_ms = {r['inter_chunk_max_ms']:.1f}")
    print(f"  inter_chunk_p95_ms = {r['inter_chunk_p95_ms']:.1f}")
    print(f"  n_chunks           = {r['n_chunks']}")
    print(f"  audio_s            = {r['audio_s']:.2f}")
    print(f"  total_tokens_est   = {r['total_tokens_est']}")
    print(f"  rtf(model_timing)  = {r['rtf_model_timing']:.3f}  （越大越好，>1.0 即实时）")
    print(f"  rtf(wall_clock)    = {r['rtf_wall_clock']:.3f}  （越大越好，>1.0 即实时）")
    print(f"  total_wall_ms      = {r['total_wall_ms']:.1f}")
    if r["hit_max_tokens"]:
        print(f"  ** 异常：未触发 EOS，打到 max_new_tokens 上限 **")


def _print_speedup(graph: dict, parity: dict) -> None:
    # TTFA: 越小越好 → 用 parity/graph 比来表示加速
    if parity["ttfa_wall_ms"] > 0:
        ttfa_speedup = parity["ttfa_wall_ms"] / graph["ttfa_wall_ms"]
        print(f"  TTFA 加速比      = {ttfa_speedup:.2f}x （Graph 首包比 Parity 快 {ttfa_speedup:.2f} 倍）")
    else:
        print(f"  TTFA 加速比      = N/A")

    # inter_chunk: 越小越好 → 用 parity/graph 比来表示加速
    if parity["inter_chunk_p95_ms"] > 0 and graph["inter_chunk_p95_ms"] > 0:
        ic_speedup = parity["inter_chunk_p95_ms"] / graph["inter_chunk_p95_ms"]
        print(f"  inter_chunk_p95  = {ic_speedup:.2f}x （Graph 块间间隔比 Parity 紧凑 {ic_speedup:.2f} 倍）")
    else:
        print(f"  inter_chunk_p95  = N/A")

    # RTF: 越大越好 → 用 graph/parity 比来表示加速
    if parity["rtf_model_timing"] > 0:
        rtf_speedup = graph["rtf_model_timing"] / parity["rtf_model_timing"]
        print(f"  RTF(model_timing)= {rtf_speedup:.2f}x （Graph 计算速度比 Parity 快 {rtf_speedup:.2f} 倍）")
    else:
        print(f"  RTF(model_timing)= N/A")

    if parity["rtf_wall_clock"] > 0:
        rtf_wall_speedup = graph["rtf_wall_clock"] / parity["rtf_wall_clock"]
        print(f"  RTF(wall_clock)  = {rtf_wall_speedup:.2f}x （Graph 端到端速度比 Parity 快 {rtf_wall_speedup:.2f} 倍）")
    else:
        print(f"  RTF(wall_clock)  = N/A")

    # total_wall_ms: 越小越好 → 用 parity/graph 比来表示加速
    if graph["total_wall_ms"] > 0:
        total_speedup = parity["total_wall_ms"] / graph["total_wall_ms"]
        print(f"  总耗时加速比     = {total_speedup:.2f}x （Graph 总耗时比 Parity 短 {total_speedup:.2f} 倍）")


if __name__ == "__main__":
    main()