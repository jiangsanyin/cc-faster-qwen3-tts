#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Custom model and port:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio voice.wav --ref-text "transcript" \\
        --port 8000

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import time
import struct
import sys
import threading
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
tts_models = []
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
# 统一限制“在飞”请求数量，避免并发过高导致OOM/抖动
_request_sem: Optional[asyncio.Semaphore] = None
# 轮询与请求ID都需要线程安全更新
_rr_lock = threading.Lock()
_rr_idx = 0
_req_seq = 0
_metrics_enabled = True


def _next_req_id() -> int:
    """生成递增请求ID，便于日志追踪单请求指标。"""
    global _req_seq
    with _rr_lock:
        _req_seq += 1
        return _req_seq


def _pick_model():
    """Round-robin pick from loaded model replicas."""
    global _rr_idx
    if not tts_models:
        # 兼容 replicas=1 的场景
        return tts_model
    with _rr_lock:
        # 多副本下做轮询分发，尽量均摊请求压力
        m = tts_models[_rr_idx % len(tts_models)]
        _rr_idx += 1
    return m


def _gpu_stats() -> dict:
    """Best-effort GPU stats; requires pynvml for utilization."""
    stats = {}
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        stats["gpu_util"] = util.gpu
        stats["mem_util"] = util.memory
        stats["vram_used_gb"] = round(mem.used / (1024**3), 2)
        stats["vram_total_gb"] = round(mem.total / (1024**3), 2)
    except Exception:
        # 没有 pynvml 时至少记录显存占用
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            used = torch.cuda.memory_allocated(idx)
            stats["vram_used_gb"] = round(used / (1024**3), 2)
            stats["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
    return stats


def _log_metrics(req_id: int, mode: str, metrics: dict) -> None:
    if not _metrics_enabled:
        return
    gpu = _gpu_stats()
    merged = {**metrics, **gpu}
    parts = [f"req_id={req_id}", f"mode={mode}"] + [f"{k}={v}" for k, v in merged.items()]
    logger.info("METRICS %s", " ".join(parts))

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    # 与 OpenAI /v1/audio/speech 风格保持兼容
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    # 流式时总长度未知，使用占位长度，客户端可边收边播
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        # 兼容客户端传了未知 voice 的情况：回退到默认 voice
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


def _prime_voice_caches(
    models: list,
    voices_dict: dict,
    *,
    prime_text: str,
    do_stream_first_chunk: bool,
    stream_max_new_tokens: int,
) -> None:
    """预热线程：把参考音频编码进 _voice_prompt_cache，避免首个真实请求的 TTFA 被拖慢。

    说明：启动时的 _warmup 只 capture 通用 CUDA Graph；首次语音克隆请求仍会走
    「读参考 wav → 提 speaker / codec 特征 → 拼进上下文」等路径，这些结果会缓存在
    FasterQwen3TTS._voice_prompt_cache 里。本函数在对外服务前主动执行一遍准备逻辑。
    """
    for replica_idx, m in enumerate(models):
        for vname, cfg in voices_dict.items():
            ref = cfg.get("ref_audio")
            if not ref:
                logger.warning("跳过预热 voice=%r：缺少 ref_audio", vname)
                continue
            if not os.path.isfile(ref):
                logger.warning("跳过预热 voice=%r：ref_audio 不存在 %s", vname, ref)
                continue
            lang = cfg.get("language", "Auto")
            ref_text = cfg.get("ref_text", "")
            chunk_sz = int(cfg.get("chunk_size", 12))
            try:
                logger.info(
                    "Priming voice clone cache: voice=%r replica=%d/%d",
                    vname,
                    replica_idx + 1,
                    len(models),
                )
                # 与 generate_voice_clone_streaming 默认一致：ICL、与线上一致的模式开关
                m._prepare_generation(
                    text=prime_text,
                    ref_audio=ref,
                    ref_text=ref_text,
                    language=lang,
                    xvec_only=False,
                    non_streaming_mode=False,
                )
                if do_stream_first_chunk:
                    # 可选：再跑一小段流式首包，覆盖纯 _prepare 未触发的 decode 懒路径
                    gen = m.generate_voice_clone_streaming(
                        text=prime_text,
                        language=lang,
                        ref_audio=ref,
                        ref_text=ref_text,
                        chunk_size=chunk_sz,
                        non_streaming_mode=False,
                        max_new_tokens=stream_max_new_tokens,
                    )
                    next(gen, None)
            except Exception as exc:
                logger.warning("预热 voice=%r 失败（将仍可在首请求时懒加载）: %s", vname, exc)


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(voice_cfg: dict, text: str, req_id: int) -> AsyncGenerator[bytes, None]:
    """
    Run generate_voice_clone_streaming in a background thread and yield
    raw PCM bytes for each chunk as they arrive.
    """
    # 线程 -> 协程 的桥接队列：后台线程产出 chunk，主协程消费并返回 HTTP 流
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        # 在后台线程里跑同步生成器；主协程只负责向客户端推流
        t0 = time.perf_counter()
        ttfa_ms = None
        total_gen_ms = 0.0
        total_audio_s = 0.0
        try:
            # 每个请求绑定一个副本执行，避免所有请求都挤到同一副本
            model = _pick_model()
            for chunk, _sr, _timing in model.generate_voice_clone_streaming(
                text=text,
                language=voice_cfg.get("language", "Auto"),
                ref_audio=voice_cfg["ref_audio"],
                ref_text=voice_cfg.get("ref_text", ""),
                chunk_size=voice_cfg.get("chunk_size", 12),
                non_streaming_mode=False,
            ):
                prefill = float(_timing.get("prefill_ms", 0.0))
                decode = float(_timing.get("decode_ms", 0.0))
                # 这里的TTFA/RTF按模型侧 timing 聚合，便于评估推理性能
                total_gen_ms += prefill + decode
                if ttfa_ms is None and total_gen_ms > 0:
                    ttfa_ms = total_gen_ms
                chunk_len = len(chunk) if isinstance(chunk, np.ndarray) else len(np.array(chunk))
                if _sr:
                    total_audio_s += chunk_len / float(_sr)
                # 仅投递音频 chunk 本体；时延/吞吐统计在本线程内累计
                q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            total_ms = round((time.perf_counter() - t0) * 1000, 1)
            rtf = round(total_audio_s / (total_gen_ms / 1000), 3) if total_gen_ms > 0 else 0.0
            # 请求结束后统一打点，便于后续离线聚合分析
            _log_metrics(
                req_id,
                "stream",
                {
                    "ttfa_ms": round(ttfa_ms or 0.0, 1),
                    "rtf": rtf,
                    "audio_s": round(total_audio_s, 3),
                    "total_ms": total_ms,
                },
            )
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    # 入参基础校验，尽早失败，减少无效 GPU 占用
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # 每个请求都带 req_id，方便在日志平台按请求串联排障
    req_id = _next_req_id()
    sem = _request_sem or asyncio.Semaphore(1)
    # 并发闸门：超过并发上限的请求会在这里排队
    await sem.acquire()
    sem_released = False
    try:
        # --- MP3: generate all audio, then encode (non-streaming) ---
        if fmt == "mp3":
            loop = asyncio.get_event_loop()
            t0 = time.perf_counter()

            def _generate():
                model = _pick_model()
                # 非流式（mp3）路径：先全量生成，再编码返回
                return model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                )

            audio_arrays, sr = await loop.run_in_executor(None, _generate)
            audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
            total_ms = (time.perf_counter() - t0) * 1000
            audio_s = len(audio) / sr if sr else 0.0
            rtf = round(audio_s / (total_ms / 1000), 3) if total_ms > 0 else 0.0
            _log_metrics(
                req_id,
                "non_stream_mp3",
                {
                    "ttfa_ms": round(total_ms, 1),
                    "rtf": rtf,
                    "audio_s": round(audio_s, 3),
                    "total_ms": round(total_ms, 1),
                },
            )
            # 非流式响应在返回前即可释放并发令牌
            sem.release()
            sem_released = True
            return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

        # --- WAV / PCM: stream chunks as they are generated ---
        async def audio_stream():
            try:
                if fmt == "wav":
                    yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
                async for raw_chunk in _stream_chunks(voice_cfg, req.input, req_id):
                    yield raw_chunk
            finally:
                # 流式场景必须在“流真正结束”时释放令牌，避免提前放量
                if not sem_released:
                    sem.release()

        return StreamingResponse(audio_stream(), media_type=content_type)
    except Exception:
        # 兜底释放，避免异常路径把并发令牌泄漏掉
        if not sem_released:
            sem.release()
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("QWEN_TTS_CONCURRENCY", "2")),
        help="Max in-flight inference requests (default: 2)",
    )
    p.add_argument(
        "--replicas",
        type=int,
        default=int(os.environ.get("QWEN_TTS_REPLICAS", "1")),
        help="Model replicas loaded in one process (default: 1)",
    )
    p.add_argument(
        "--no-metrics-log",
        action="store_true",
        help="Disable per-request metrics logs",
    )
    p.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip startup CUDA graph warmup (first user request will pay capture cost)",
    )
    p.add_argument(
        "--warmup-prefill-len",
        type=int,
        default=100,
        help="Talker graph warmup prefill length (default: 100, same as demo/server.py)",
    )
    p.add_argument(
        "--skip-prime-voices",
        action="store_true",
        help="跳过启动时对各 voice 的参考音频缓存预热（首个真实请求 TTFA 可能偏高）",
    )
    p.add_argument(
        "--prime-text",
        type=str,
        default=".",
        help="预热时用的极短合成文本（默认 '.'）",
    )
    p.add_argument(
        "--prime-stream-first-chunk",
        action="store_true",
        help="预热时额外拉取一次流式首块（更贴近首请求，但启动更慢）",
    )
    p.add_argument(
        "--prime-stream-max-new-tokens",
        type=int,
        default=48,
        help="与 --prime-stream-first-chunk 配合，限制预热解码步数（默认 48）",
    )
    return p.parse_args()


def main():
    global tts_model, tts_models, voices, default_voice, SAMPLE_RATE, _request_sem, _metrics_enabled

    args = _parse_args()

    # 构建 voice 配置源：优先 --voices，多 voice；否则退化为单 voice
    if args.voices:
        with open(args.voices) as f:
            voices = json.load(f)
        default_voice = next(iter(voices))
        logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        print(
            "ERROR: provide --ref-audio <file> or --voices <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    from faster_qwen3_tts import FasterQwen3TTS

    # 参数合法性检查
    if args.replicas < 1:
        raise ValueError("--replicas must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    _metrics_enabled = not args.no_metrics_log
    # 启动时初始化并发控制器
    _request_sem = asyncio.Semaphore(args.concurrency)

    logger.info(
        "Loading %d replica(s) of model %s on %s …",
        args.replicas,
        args.model,
        args.device,
    )
    tts_models = []
    for i in range(args.replicas):
        logger.info("Loading replica %d/%d", i + 1, args.replicas)
        # 每个副本都是完整模型实例；副本数增加会明显提高显存占用
        m = FasterQwen3TTS.from_pretrained(
            args.model,
            device=args.device,
            dtype=torch.bfloat16,
        )
        # 启动阶段完成 CUDA Graph capture，避免首个真实请求的 TTFA 被拖慢
        if not args.skip_warmup:
            logger.info(
                "Warming up replica %d/%d (prefill_len=%d)…",
                i + 1,
                args.replicas,
                args.warmup_prefill_len,
            )
            m._warmup(prefill_len=args.warmup_prefill_len)
        tts_models.append(m)
    # 保留第一个副本作为健康检查与兜底引用
    tts_model = tts_models[0]
    SAMPLE_RATE = tts_model.sample_rate
    if not args.skip_prime_voices:
        _prime_voice_caches(
            tts_models,
            voices,
            prime_text=args.prime_text,
            do_stream_first_chunk=args.prime_stream_first_chunk,
            stream_max_new_tokens=args.prime_stream_max_new_tokens,
        )
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info(
        "Concurrency enabled: max_in_flight=%d replicas=%d",
        args.concurrency,
        args.replicas,
    )
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    # 单进程内由 asyncio + 线程桥接处理流式输出
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
