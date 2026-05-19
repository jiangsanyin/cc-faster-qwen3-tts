#!/usr/bin/env python3
"""
§6.2.8 顺序 2 原型：两条 parity session 轮流 ``step()``，统计近似 **TTFA / inter_chunk / RTF**。

在 ``faster-qwen3-tts`` 目录下示例：

    python examples/parity_dual_session_benchmark.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --ref-audio /path/to/ref.wav \\
        --ref-text "参考文本" \\
        --text-a "第一段合成文本" \\
        --text-b "第二段合成文本"

**TTFA（codec）**：自进程 ``t0`` 到该 session **首块 codec** 的墙钟 ms。  
**inter_chunk_*（codec）**：同 session 相邻两次 **codec 块产出** 的墙钟间隔（与服务端队列入队间隔同阶）。  
**RTF**：在 **整段 codec 已收集后**，对每路单独跑 ``_iter_voice_clone_audio_from_codec_stream``，
用 timing 中 ``prefill_ms/decode_ms`` 累计与 **audio_s** 估算（与 METRICS 定义同向）。
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
from faster_qwen3_tts.parity_dual_round_robin import iter_round_robin_codec_chunks
from faster_qwen3_tts.parity_stream_session import ParityStreamSession


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _build_stream_kwargs(
    talker, tie, tam, tth, tpe, config, chunk_size: int, max_new_tokens: int
) -> dict:
    return dict(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=config,
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
    p = argparse.ArgumentParser(description="Parity 双 session 轮流 step 基准")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref-audio", type=str, required=True)
    p.add_argument("--ref-text", type=str, default="")
    p.add_argument("--text-a", type=str, required=True)
    p.add_argument("--text-b", type=str, required=True)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    dtype = torch.bfloat16
    model = FasterQwen3TTS.from_pretrained(
        args.model, 
        device=args.device, 
        dtype=dtype,
        disable_cuda_graph=True  # OOM FIX: Disable static cache allocation in dual-session benchmark
    )

    def make_session(text: str) -> Tuple[ParityStreamSession, torch.Tensor, object]:
        m, talker, config, tie, tam, tth, tpe, ref_codes = model._prepare_generation(
            text=text,
            language="Auto",
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=False,
            non_streaming_mode=False,
            append_silence=True,
            voice_clone_prompt=None,
            instruct=None,
            skip_warmup=True,  # 避免 OOM: 双 session benchmark 用 parity（非 Graph）路径，无需且不该捕捉 CUDA Graph
        )
        kw = _build_stream_kwargs(
            talker, tie, tam, tth, tpe, config, args.chunk_size, args.max_new_tokens
        )
        return ParityStreamSession(**kw), ref_codes, m.speech_tokenizer

    s0, ref0, tok = make_session(args.text_a)
    s1, ref1, _ = make_session(args.text_b)

    per_codec: DefaultDict[int, List[Tuple[torch.Tensor, dict]]] = defaultdict(list)
    wall_codec: DefaultDict[int, List[float]] = defaultdict(list)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for sid, chunk, timing in iter_round_robin_codec_chunks([s0, s1]):
            wall_codec[sid].append((time.perf_counter() - t0) * 1000.0)
            per_codec[sid].append((chunk, timing))
    t_codec_done = time.perf_counter()

    print("=== Codec 阶段（轮流 step，墙钟）===")
    for sid, label in [(0, "A"), (1, "B")]:
        times = wall_codec[sid]
        if not times:
            print(f"session {label}: no chunks")
            continue
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        ttfa_wall = times[0]
        print(
            f"session {label}: ttfa_wall_ms(codec)={ttfa_wall:.1f} "
            f"inter_chunk_max_ms={max(gaps) if gaps else 0:.1f} "
            f"inter_chunk_p95_ms={_p95(gaps):.1f} n_chunks={len(times)}"
        )

    print("=== 波形阶段（每路单独 decode，与线上一致）===")
    for sid, label, ref in [(0, "A", ref0), (1, "B", ref1)]:
        pairs = per_codec[sid]
        if not pairs:
            continue

        def pair_iter():
            yield from pairs

        t_dec = time.perf_counter()
        total_gen_ms = 0.0
        audio_s = 0.0
        sr0 = 24000
        first_audio_wall = None
        for audio, sr, timing in model._iter_voice_clone_audio_from_codec_stream(
            tok, ref, args.chunk_size, pair_iter()
        ):
            if first_audio_wall is None:
                first_audio_wall = (time.perf_counter() - t0) * 1000.0
            sr0 = sr or sr0
            total_gen_ms += float(timing.get("prefill_ms", 0.0)) + float(timing.get("decode_ms", 0.0))
            audio_s += len(audio) / float(sr0)
        t_done = time.perf_counter()
        wall_total_ms = (t_done - t_dec) * 1000.0
        rtf = (audio_s / (total_gen_ms / 1000.0)) if total_gen_ms > 0 else 0.0
        print(
            f"session {label}: ttfa_wall_ms(audio)={first_audio_wall:.1f} "
            f"audio_s={audio_s:.2f} rtf(model_timing)={rtf:.3f} "
            f"decode_wall_ms={wall_total_ms:.1f} "
            f"(codec_phase_wall_ms={(t_codec_done - t0) * 1000:.1f})"
        )

    print(f"total_wall_ms={ (time.perf_counter() - t0) * 1000:.1f}")


if __name__ == "__main__":
    main()
