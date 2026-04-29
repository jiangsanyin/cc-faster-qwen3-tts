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

    # Multiple named voices + optional inference log / stream_chunk_size via config.json:
    python examples/openai_server.py \
        --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
        --voices ./voices.json  \
        --stream-chunk-size 10 \
        --language Chinese --port 8000

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
import shlex
import struct
import sys
import threading
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
INFERENCE_LOGGER_NAME = "faster_qwen3_tts.inference"

_metrics_enabled = True
_metrics_log_mode = "DEV"
_stream_chunk_size = 12
_req_seq = 0
_req_seq_lock = threading.Lock()

_DEV_METRIC_KEYS = (
    "ttfa_wall_ms",
    "ttfa_cuda_graphs_ms",
    "inter_chunk_max_ms",
    "inter_chunk_p95_ms",
    "rtf",
    "audio_s",
    "total_gen_ms",
    "total_ms",
    "n_chunks",
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference

# ---------------------------------------------------------------------------
# Inference METRICS logging (aligned with examples/openai_server_v5.py semantics)
# ---------------------------------------------------------------------------


def _next_req_id() -> int:
    global _req_seq
    with _req_seq_lock:
        _req_seq += 1
        return _req_seq


def _percentile_95(values: list) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _normalize_metrics_log_mode(value: Optional[str]) -> str:
    """Same `inference_logging.level` as v5: DEBUG -> full METRICS; anything else -> DEV subset."""
    mode = str(value or "INFO").strip().upper()
    if mode == "DEBUG":
        return "DEBUG"
    return "DEV"


def _log_metrics(req_id: int, mode: str, metrics: dict) -> None:
    if not _metrics_enabled:
        return
    if _metrics_log_mode == "DEV":
        merged = {k: metrics[k] for k in _DEV_METRIC_KEYS if k in metrics}
    else:
        merged = {**metrics}
    inf = logging.getLogger(INFERENCE_LOGGER_NAME)
    parts = [f"{k}={v}" for k, v in merged.items()]
    if _metrics_log_mode == "DEV":
        inf.info("METRICS mode=%s %s", mode, " ".join(parts))
    else:
        inf.info("METRICS req_id=%d mode=%s %s", req_id, mode, " ".join(parts))


def _resolve_cfg_path(config_path: str, p: str) -> str:
    """将配置中的相对路径按配置文件所在目录解析为绝对路径。"""
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(config_path)), p))


def _inference_log_path_with_timestamp(resolved_path: str) -> str:
    """在已解析的日志路径主文件名与扩展名之间插入时间缀 _YYYY-M-D_HHMMSS。"""
    d, base = os.path.dirname(resolved_path), os.path.basename(resolved_path)
    stem, ext = os.path.splitext(base)
    now = datetime.now()
    ts = f"_{now.year}-{now.month}-{now.day}_{now.strftime('%H%M%S')}"
    return os.path.join(d, f"{stem}{ts}{ext}")


def _setup_inference_logging(inference_cfg: dict, config_path: str) -> Optional[str]:
    """按 config.json 的 inference_logging 段配置推理摘要 logger（与 config_v5 / openai_server_v5 语义一致）。

    读取的配置项：
    - console: 是否向控制台（stderr）挂载 StreamHandler
    - file: 相对 config_path 或绝对路径；空字符串表示不写文件
    - add_timestamp: True 时在文件名主干与扩展名之间插入 _YYYY-M-D_HHMMSS
    - level: Python logging 级别名（DEBUG/INFO/WARNING/...）；同时用于 METRICS 详略（仅 DEBUG 输出完整字段）
    """
    console = bool(inference_cfg.get("console", True))
    file_rel = str(inference_cfg.get("file", "") or "").strip()
    add_timestamp = bool(inference_cfg.get("add_timestamp", True))
    level_name = str(inference_cfg.get("level", "INFO")).strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    log = logging.getLogger(INFERENCE_LOGGER_NAME)
    log.handlers.clear()
    log.setLevel(level)
    log.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        log.addHandler(sh)

    out_path: Optional[str] = None
    if file_rel:
        abs_file = _resolve_cfg_path(config_path, file_rel)
        if add_timestamp:
            abs_file = _inference_log_path_with_timestamp(abs_file)
        parent = os.path.dirname(abs_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fh = logging.FileHandler(abs_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        log.addHandler(fh)
        out_path = abs_file
        logger.info("Inference log file: %s", abs_file)
    return out_path


def _log_startup_summary(config_path: Optional[str], inference_log_file: Optional[str]) -> None:
    inf = logging.getLogger(INFERENCE_LOGGER_NAME)
    log = inf if (inf.handlers or inf.propagate) else logger
    try:
        cmdline = shlex.join(sys.argv)
    except (AttributeError, ValueError, TypeError):
        cmdline = " ".join(sys.argv)
    log.info("STARTUP cmdline=%s", cmdline)
    if config_path:
        log.info("STARTUP --config=%s", os.path.abspath(config_path))
    log.info("STARTUP stream_chunk_size(effective)=%d", _stream_chunk_size)
    log.info("STARTUP metrics_log_mode=%s (from inference_logging.level)", _metrics_log_mode)
    if inference_log_file:
        log.info("STARTUP inference_log_file=%s", inference_log_file)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
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


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(
    req_id: int,
    voice_cfg: dict,
    text: str,
    t_start: float,
    t_ttfa0: float,
) -> AsyncGenerator[bytes, None]:
    """
    Run generate_voice_clone_streaming in a background thread and yield
    raw PCM bytes; log METRICS once at end (same fields as openai_server_v5).
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        ttfa_wall_ms: Optional[float] = None
        ttfa_ms: Optional[float] = None
        ttfa_cuda_graphs_ms: Optional[float] = None
        first_prefill_ms: Optional[float] = None
        first_decode_ms: Optional[float] = None
        prefill_len_v: Optional[int] = None
        attention_mask_shape = ""
        trailing_text_len_v: Optional[int] = None
        icl_v: Optional[int] = None
        parity_mode_v: Optional[int] = None
        chunk_gaps_ms: List[float] = []
        total_gen_ms = 0.0
        total_audio_s = 0.0
        n_chunks = 0
        t_prev_put: Optional[float] = None
        metrics_logged = False
        err: Optional[BaseException] = None

        def finalize_log() -> None:
            nonlocal metrics_logged
            if metrics_logged:
                return
            metrics_logged = True
            ic_max = max(chunk_gaps_ms) if chunk_gaps_ms else 0.0
            ic_p95 = _percentile_95(chunk_gaps_ms)
            _ttfa_ms = ttfa_ms or 0.0
            first_overhead_ms = (
                (ttfa_cuda_graphs_ms or 0.0) - _ttfa_ms
                if ttfa_cuda_graphs_ms is not None and ttfa_ms is not None
                else 0.0
            )
            _log_metrics(
                req_id,
                "stream",
                {
                    "ttfa_wall_ms": round(ttfa_wall_ms or 0.0, 1),
                    "ttfa_ms": round(_ttfa_ms, 1),
                    "ttfa_cuda_graphs_ms": round(ttfa_cuda_graphs_ms or 0.0, 1),
                    "first_prefill_ms": round(first_prefill_ms or 0.0, 1),
                    "first_decode_ms": round(first_decode_ms or 0.0, 1),
                    "first_overhead_ms": round(first_overhead_ms, 1),
                    "prefill_len": prefill_len_v or 0,
                    "attention_mask_shape": attention_mask_shape or "",
                    "trailing_text_len": trailing_text_len_v or 0,
                    "icl": icl_v or 0,
                    "parity_mode": parity_mode_v or 0,
                    "inter_chunk_max_ms": round(ic_max, 1),
                    "inter_chunk_p95_ms": round(ic_p95, 1),
                    "rtf": round(
                        total_audio_s / (total_gen_ms / 1000.0), 3,
                    ) if total_gen_ms > 0 else 0.0,
                    "audio_s": round(total_audio_s, 3),
                    "total_gen_ms": round(total_gen_ms, 1),
                    "total_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
                    "n_chunks": n_chunks,
                },
            )

        try:
            with _model_lock:
                parity_bool = tts_model.talker_graph is None
                parity_int = int(parity_bool)
                parity_mode_v = parity_int
                t_bench0 = time.perf_counter()
                gen = tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=int(voice_cfg.get("chunk_size", _stream_chunk_size)),
                    non_streaming_mode=False,
                    parity_mode=parity_bool,
                )
                for chunk, sr, timing in gen:
                    if ttfa_cuda_graphs_ms is None:
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        ttfa_cuda_graphs_ms = (time.perf_counter() - t_bench0) * 1000.0

                    chunk_prefill_ms = float(timing.get("prefill_ms", 0.0))
                    chunk_decode_ms = float(timing.get("decode_ms", 0.0))
                    total_gen_ms += chunk_prefill_ms + chunk_decode_ms
                    if sr:
                        chunk_len = len(chunk) if isinstance(chunk, np.ndarray) else len(np.array(chunk))
                        total_audio_s += chunk_len / float(sr)
                    n_chunks += 1

                    now = time.perf_counter()
                    if t_prev_put is None:
                        ttfa_wall_ms = (now - t_ttfa0) * 1000.0
                        if ttfa_ms is None and total_gen_ms > 0:
                            ttfa_ms = total_gen_ms
                            first_prefill_ms = chunk_prefill_ms
                            first_decode_ms = chunk_decode_ms
                            prefill_len_v = int(timing.get("prefill_len", 0) or 0)
                            attention_mask_shape = str(timing.get("attention_mask_shape", "") or "")
                            trailing_text_len_v = int(timing.get("trailing_text_len", 0) or 0)
                            icl_v = int(timing.get("icl", 0) or 0)
                            parity_mode_v = int(timing.get("parity_mode", parity_int) or 0)
                    else:
                        chunk_gaps_ms.append((now - t_prev_put) * 1000.0)
                    t_prev_put = now

                    q.put(_to_pcm16(chunk))
        except Exception as exc:
            err = exc
        finally:
            finalize_log()
            if err is not None:
                q.put(err)
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield item


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
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

    req_id = _next_req_id()
    t_req_start = time.perf_counter()
    t_ttfa0 = t_req_start

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_running_loop()

        def _generate():
            with _model_lock:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                t_bench0 = time.perf_counter()
                t0 = time.perf_counter()
                audio_arrays, sr = tts_model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                )
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                total_ms = (time.perf_counter() - t0) * 1000
                ttfa_cuda_graphs_ms = (time.perf_counter() - t_bench0) * 1000
                ttfa_wall_ms = (time.perf_counter() - t_ttfa0) * 1000
                return audio_arrays, sr, total_ms, ttfa_wall_ms, ttfa_cuda_graphs_ms

        audio_arrays, sr, total_ms, ttfa_wall_ms, ttfa_cuda_graphs_ms = await loop.run_in_executor(
            None, _generate,
        )
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        audio_len_s = len(audio) / sr if sr else 0.0
        parity_bool = tts_model.talker_graph is None

        mp3_metrics = {
            "ttfa_wall_ms": round(ttfa_wall_ms, 1),
            "ttfa_ms": round(total_ms, 1),
            "ttfa_cuda_graphs_ms": round(ttfa_cuda_graphs_ms, 1),
            "first_prefill_ms": 0.0,
            "first_decode_ms": 0.0,
            "first_overhead_ms": round(max(0.0, ttfa_cuda_graphs_ms - total_ms), 1),
            "prefill_len": 0,
            "attention_mask_shape": "",
            "trailing_text_len": 0,
            "icl": 0,
            "parity_mode": int(parity_bool),
            "inter_chunk_max_ms": 0.0,
            "inter_chunk_p95_ms": 0.0,
            "rtf": round(audio_len_s / (total_ms / 1000.0), 3) if total_ms > 0 else 0.0,
            "audio_s": round(audio_len_s, 3),
            "total_gen_ms": round(total_ms, 1),
            "total_ms": round(total_ms, 1),
            "n_chunks": 0,
        }
        _log_metrics(req_id, "non_stream_mp3", mp3_metrics)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(req_id, voice_cfg, req.input, t_req_start, t_ttfa0):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


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
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.json"),
        help="Optional JSON with inference_logging and stream_chunk_size (default: ../config.json); --stream-chunk-size overrides stream_chunk_size when set.",
    )
    p.add_argument(
        "--stream-chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Codec steps per streaming chunk; overrides config stream_chunk_size when set.",
    )
    p.add_argument(
        "--no-metrics-log",
        action="store_true",
        help="Disable METRICS lines (inference logger may still run for STARTUP if configured).",
    )
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE
    global _metrics_enabled, _metrics_log_mode, _stream_chunk_size

    args = _parse_args()

    cfg: Dict = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)

    _metrics_enabled = not args.no_metrics_log
    _metrics_log_mode = _normalize_metrics_log_mode((cfg.get("inference_logging") or {}).get("level", "INFO"))
    try:
        if args.stream_chunk_size is not None:
            _stream_chunk_size = max(1, int(args.stream_chunk_size))
        else:
            _stream_chunk_size = max(1, int(cfg.get("stream_chunk_size", 12)))
    except (TypeError, ValueError):
        logger.warning("stream_chunk_size invalid, using default 12")
        _stream_chunk_size = 12

    infer_log_path: Optional[str] = None
    if "inference_logging" in cfg and args.config and os.path.isfile(args.config):
        infer_log_path = _setup_inference_logging(cfg.get("inference_logging") or {}, args.config)

    cfg_abs = os.path.abspath(args.config) if (args.config and os.path.isfile(args.config)) else None
    _log_startup_summary(cfg_abs, infer_log_path)

    # Build voice registry
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

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = tts_model.sample_rate
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
