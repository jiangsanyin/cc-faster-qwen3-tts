#!/usr/bin/env python3
"""
§6.2.8 顺序 3 原型：混合策略基准 - 一路 CUDA Graph 快路径 + 一路 parity 动态路径并行执行。

验证「单路 Graph + 并发 parity 回退」混合策略的可行性：同一 GPU 上，
路 A 用 Graph（快但独占缓冲区）、路 B 用 parity（慢但灵活穿插），
通过轮询调度交替推进两路 decode，观察 TTFA / inter_chunk / RTF。

RTF 口径：audio_s / (prefill_ms + decode_ms) / 1000，越大越好（> 1.0 即实时）。
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_qwen3_tts.model import FasterQwen3TTS
from faster_qwen3_tts.parity_stream_session import ParityStreamSession
from faster_qwen3_tts.streaming import fast_generate_streaming


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _build_stream_kwargs(talker, tie, tam, tth, tpe, config,
                         chunk_size, max_new_tokens) -> dict:
    """
    构建流式合成的关键字参数字典。
    
    参数说明：
    - max_new_tokens: 安全上限，控制 decode 阶段最多生成多少个 codec token
      防止无限生成（即使模型异常不触发 EOS，也会在达到上限后强制停止）
    - chunk_size: 每个音频块包含的 codec token 数，影响流式 chunk 的粒度
    """
    return dict(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=config,
        # 流式合成的安全上限，防止模型异常时无限生成
        max_new_tokens=max_new_tokens,
        min_new_tokens=2,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        do_sample=True,
        repetition_penalty=1.05,
        chunk_size=chunk_size,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="混合策略基准：Graph + parity 并行")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref-audio", type=str, required=True)
    p.add_argument("--ref-text", type=str, default="")
    p.add_argument("--text-a", type=str, required=True, help="走 Graph 快路径的文本")
    p.add_argument("--text-b", type=str, required=True, help="走 parity 动态路径的文本")
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    # 加载模型（CUDA Graph 启用，路 A 需要）
    model = FasterQwen3TTS.from_pretrained(
        args.model, device=args.device, dtype=torch.bfloat16, # 默认启用CUDA Graph
    )

    # --- 预热：先跑一条短文本完成 Graph capture，避免第一次正式计时包含 capture 开销 ---
    print("=== 预热（CUDA Graph capture）===")
    _warmup_m, _warmup_talker, _warmup_cfg, _warmup_tie, _warmup_tam, _warmup_tth, _warmup_tpe, _ = model._prepare_generation(
        text="预热", language="Auto",
        ref_audio=args.ref_audio, ref_text=args.ref_text,
        xvec_only=False, non_streaming_mode=False,
        append_silence=True, voice_clone_prompt=None, instruct=None,
    )
    _warmup_kw = _build_stream_kwargs(
        _warmup_talker, _warmup_tie, _warmup_tam, _warmup_tth, _warmup_tpe, _warmup_cfg,
        args.chunk_size, 48,
    )
    _warmup_kw["predictor_graph"] = model.predictor_graph
    _warmup_kw["talker_graph"] = model.talker_graph
    # 简单跑完预热
    _warmup_gen = fast_generate_streaming(**_warmup_kw)
    with torch.inference_mode():
        for _ in _warmup_gen:
            pass
    print("预热完成，Graph 已捕获")

    # --- 路 A（Graph）：准备 codec 流 ---
    m_a, talker_a, config_a, tie_a, tam_a, tth_a, tpe_a, ref_codes_a = model._prepare_generation(
        text=args.text_a, language="Auto",
        ref_audio=args.ref_audio, ref_text=args.ref_text,
        xvec_only=False, non_streaming_mode=False,
        append_silence=True, voice_clone_prompt=None, instruct=None,
    )
    stream_kwargs_a = _build_stream_kwargs(
        talker_a, tie_a, tam_a, tth_a, tpe_a, config_a,
        args.chunk_size, args.max_new_tokens,
    )
    stream_kwargs_a["predictor_graph"] = model.predictor_graph
    stream_kwargs_a["talker_graph"] = model.talker_graph

    # --- 路 B（parity）：准备 session ---
    m_b, talker_b, config_b, tie_b, tam_b, tth_b, tpe_b, ref_codes_b = model._prepare_generation(
        text=args.text_b, language="Auto",
        ref_audio=args.ref_audio, ref_text=args.ref_text,
        xvec_only=False, non_streaming_mode=False,
        append_silence=True, voice_clone_prompt=None, instruct=None,
    )
    stream_kwargs_b = _build_stream_kwargs(
        talker_b, tie_b, tam_b, tth_b, tpe_b, config_b,
        args.chunk_size, args.max_new_tokens,
    )
    parity_session_b = ParityStreamSession(**stream_kwargs_b)

    # --- 混合调度：每轮先拉路 A 的一个 chunk，再拉路 B 的一个 chunk ---
    gen_a = fast_generate_streaming(**stream_kwargs_a)

    per_codec: DefaultDict[str, List[Tuple[torch.Tensor, dict]]] = defaultdict(list)
    wall_codec: DefaultDict[str, List[float]] = defaultdict(list)

    t0 = time.perf_counter()

    with torch.inference_mode():
        while True:
            stepped_a = False
            stepped_b = False

            # --- 路 A：尝试拉一个 chunk ---
            # gen_a 是生成器，next() 返回 (codec_chunk, timing_dict)
            # codec_chunk: torch.Tensor - codec token 块，shape 为 [chunk_steps, 16]
            # timing: dict - 计时信息，包含 prefill_ms/decode_ms/chunk_index/is_final 等
            try:
                chunk_a, timing_a = next(gen_a)
                wall_codec["A"].append((time.perf_counter() - t0) * 1000.0)
                per_codec["A"].append((chunk_a, timing_a))
                stepped_a = True
            except StopIteration:
                pass

            # --- 路 B：连续 step() 直到产出一个 chunk 或 finished ---
            # finished: bool - ParityStreamSession 属性，表示会话是否已彻底结束
            # step() 返回 None 表示未凑满 chunk，返回 (codec_chunk, timing) 表示产出一块
            # codec_chunk: torch.Tensor - codec token 块，shape 为 [chunk_steps, 16]
            # timing: dict - 计时信息，包含 prefill_ms/decode_ms/chunk_index/is_final 等
            if not parity_session_b.finished:
                while not parity_session_b.finished:
                    item_b = parity_session_b.step()
                    if item_b is not None:
                        chunk_b, timing_b = item_b
                        wall_codec["B"].append((time.perf_counter() - t0) * 1000.0)
                        per_codec["B"].append((chunk_b, timing_b))
                        stepped_b = True
                        break  # 一轮只产出一个 chunk，与路 A 对齐

            if not stepped_a and not stepped_b:
                break  # 两路都结束

    t_codec_done = time.perf_counter()

    # --- Codec 阶段统计 ---
    print("=== Codec 阶段（混合调度，墙钟）===")
    for label in ["A (Graph)", "B (parity)"]:
        key = label.split()[0]
        times = wall_codec[key]
        if not times:
            print(f"  {label}: no chunks")
            continue
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        ttfa_wall = times[0]
        print(
            f"  {label}: ttfa_wall_ms(codec)={ttfa_wall:.1f} "
            f"inter_chunk_max_ms={max(gaps) if gaps else 0:.1f} "
            f"inter_chunk_p95_ms={_p95(gaps):.1f} n_chunks={len(times)}"
        )

    # --- 波形阶段 ---
    speech_tokenizer = m_a.speech_tokenizer

    print("=== 波形阶段（每路单独 decode）===")
    for label, ref in [("A", ref_codes_a), ("B", ref_codes_b)]:
        pairs = per_codec[label]
        if not pairs:
            continue

        def pair_iter():
            yield from pairs

        total_gen_ms = 0.0
        audio_s = 0.0
        sr0 = 24000
        first_audio_wall = None
        mode_label = "A (Graph)" if label == "A" else "B (parity)"
        # 记录解码阶段开始时间，用于准确测量首块音频的解码耗时
        t_dec = time.perf_counter()
        for audio, sr, timing in model._iter_voice_clone_audio_from_codec_stream(
            speech_tokenizer, ref, args.chunk_size, pair_iter()
        ):
            if first_audio_wall is None:
                # 使用 t_dec 而非 t0，排除 codec 收集和统计打印的时间
                first_audio_wall = (time.perf_counter() - t_dec) * 1000.0
            # print(f"{mode_label} sr: {sr}")
            sr0 = sr or sr0
            total_gen_ms += float(timing.get("prefill_ms", 0.0)) + float(timing.get("decode_ms", 0.0))
            audio_s += len(audio) / float(sr0)
        t_done = time.perf_counter()
        wall_total_ms = (t_done - t_dec) * 1000.0
        rtf = (audio_s / (total_gen_ms / 1000.0)) if total_gen_ms > 0 else 0.0

        print(
            f"  {mode_label}: ttfa_wall_ms(audio)={first_audio_wall:.1f} "
            f"audio_s={audio_s:.2f} rtf(model_timing)={rtf:.3f} "
            f"decode_wall_ms={wall_total_ms:.1f} "
            # f"(codec_phase_wall_ms={(t_codec_done - t0) * 1000:.1f})"
        )

    print(f"  total_wall_ms={(time.perf_counter() - t0) * 1000:.1f}")


if __name__ == "__main__":
    main()