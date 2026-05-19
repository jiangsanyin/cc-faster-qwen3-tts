#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server (v3 热加载版) for faster-qwen3-tts.

与 v2 的区别：
  - 音色元数据来自 JSON 注册表 + 磁盘参考音频（config_v3.json），无 MySQL/Redis。
  - 管理 API 使用 voice_manager_api_v3.py；本服务通过注册表 mtime 热加载。
  - 可选按阈值清理各模型副本上的 _voice_prompt_cache 孤儿条目（见 config）。

Usage:
    python examples/openai_server_concurrent_v3.py \
        --config config_v3.json \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --host 0.0.0.0 --port 8000 --concurrency 4

API usage:
    curl -s http://localhost:8000/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d '{"model": "tts-1", "input": "Hello!", "voice": "<voice_id>", "response_format": "wav"}' \
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
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from voice_registry_v3 import (
    HotVoiceRegistryV3,
    VoiceRegistryV3,
    load_config_v3,
    maybe_cleanup_voice_prompt_cache,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
tts_models = []
hot_registry: Optional[HotVoiceRegistryV3] = None
_v3_cleanup_enable: bool = True
_v3_cleanup_threshold: int = 30

_v3_prime_token: Optional[str] = None
_v3_prime_lock: Optional[asyncio.Lock] = None  # 串行化处理内部预热，保护 GPU
SAMPLE_RATE = 24000  # updated once the model loads
# 统一限制“在飞”请求数量，避免并发过高导致OOM/抖动
_request_sem: Optional[asyncio.Semaphore] = None
# 轮询与请求ID都需要线程安全更新
_rr_lock = threading.Lock()
_rr_idx = 0
_req_seq = 0
_metrics_enabled = True


def _next_req_id() -> int:
    """
    生成递增请求ID，便于日志追踪单请求指标。
    
    Returns:
        int: 请求序列号。
    """
    global _req_seq
    with _rr_lock:
        _req_seq += 1
        return _req_seq


def _pick_model():
    """
    Round-robin 方式从加载的模型副本中选择一个。
    
    Returns:
        FasterQwen3TTS: 模型实例。
    """
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
    """
    尽力获取 GPU 统计信息（利用率、显存占用）。
    如果安装了 pynvml 则获取详细信息，否则仅记录显存占用。
    
    Returns:
        dict: GPU 状态字典。
    """
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
    """
    记录请求指标日志，包含性能数据和 GPU 状态。
    
    Args:
        req_id: 请求ID。
        mode: 模式（stream/non_stream_mp3等）。
        metrics: 性能指标字典。
    """
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
    """
    将 float32 numpy 数组转换为 16 位小端 PCM 字节。
    
    Args:
        pcm: float32 音频数据。
        
    Returns:
        bytes: PCM16 字节数据。
    """
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """
    构建 WAV 文件头。
    对于流式传输，data_len 使用 0xFFFFFFFF 表示未知大小。
    
    Args:
        sample_rate: 采样率。
        data_len: 音频数据长度。
        
    Returns:
        bytes: WAV 头字节。
    """
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
    """
    将 float32 numpy 数组转换为完整的 WAV 文件字节。
    
    Args:
        pcm: 音频数据。
        sample_rate: 采样率。
        
    Returns:
        bytes: WAV 文件内容。
    """
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """
    将 float32 numpy 数组转换为 MP3 字节（需要 pydub + ffmpeg）。
    
    Args:
        pcm: 音频数据。
        sample_rate: 采样率。
        
    Returns:
        bytes: MP3 文件内容。
        
    Raises:
        HTTPException: 如果未安装 pydub。
    """
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


def _resolve_cfg_path(config_path: str, p: str) -> str:
    """
    解析配置文件中的路径，支持相对路径。
    
    Args:
        config_path: 配置文件路径。
        p: 待解析的路径。
        
    Returns:
        str: 规范化的绝对路径。
    """
    if os.path.isabs(p):
        return os.path.normpath(p)
    base = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(base, p))


def _run_orphan_cleanup_all() -> None:
    """
    对每个模型副本按当前注册表快照做一次孤儿缓存清理（受配置阈值控制）。
    """
    if hot_registry is None or not tts_models:
        return
    reg = hot_registry.get_registry_copy()
    for m in tts_models:
        maybe_cleanup_voice_prompt_cache(
            m,
            reg,
            enable=_v3_cleanup_enable,
            threshold=_v3_cleanup_threshold,
        )


def _prime_single_voice_sync(models: list, cfg: dict, prime_text: str) -> None:
    """
    同步方式对所有副本预填指定音色的缓存。
    
    Args:
        models: 模型副本列表。
        cfg: 音色配置字典。
        prime_text: 预热文本。
    """
    vname = cfg.get("voice_id", "?")
    ref = cfg.get("ref_audio")
    if not ref or not os.path.isfile(ref):
        logger.warning("跳过预热 voice=%r：ref_audio 无效", vname)
        return
    lang = cfg.get("language", "Auto")
    ref_text = cfg.get("ref_text", "")
    for i, m in enumerate(models):
        try:
            logger.info("Priming voice cache: voice=%r replica=%d/%d", vname, i + 1, len(models))
            m._prepare_generation(
                text=prime_text,
                ref_audio=ref,
                ref_text=ref_text,
                language=lang,
                xvec_only=False,
                non_streaming_mode=False,
            )
        except Exception as exc:
            logger.warning("预热 voice=%r 失败: %s", vname, exc)


@app.post("/internal/v1/prime-voice")
async def internal_prime_voice(
    voice_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    内部接口：供管理 API 在新增/更新音色后触发 TTS 预热。
    
    Args:
        voice_id: 音色ID。
        x_internal_token: 内部鉴权 Token。
        
    Returns:
        dict: 成功状态。
    """
    if _v3_prime_token and x_internal_token != _v3_prime_token:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    if hot_registry is None:
        raise HTTPException(status_code=503, detail="Registry not ready")

    # 强制重载注册表以获取最新变更
    hot_registry.get_registry_copy()
    cfg = hot_registry.resolve_active_voice(voice_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not active")

    lock = _v3_prime_lock or asyncio.Lock()
    async with lock:
        loop = asyncio.get_event_loop()
        # 借用启动时的 prime_text
        await loop.run_in_executor(None, _prime_single_voice_sync, tts_models, cfg, ".")

    # 预热后顺便触发一次孤儿清理（因为注册表变了，可能有旧音色变孤儿）
    _run_orphan_cleanup_all()

    return {"success": True, "voice_id": voice_id}


def resolve_voice(voice_id: str) -> dict:
    """
    从热加载注册表解析 active 音色；未找到则 400。
    
    Args:
        voice_id: 音色ID。
        
    Returns:
        dict: 音色配置。
        
    Raises:
        HTTPException: 503 (未初始化) 或 400 (不存在/禁用)。
    """
    if hot_registry is None:
        raise HTTPException(status_code=503, detail="Voice registry not initialized")
    cfg = hot_registry.resolve_active_voice(voice_id)
    if cfg:
        return cfg
    raise HTTPException(
        status_code=400,
        detail=f"Voice '{voice_id}' 不存在或已禁用",
    )


def _prime_voice_caches(
    models: list,
    voice_list: list,
    *,
    prime_text: str,
    do_stream_first_chunk: bool,
    stream_max_new_tokens: int,
) -> None:
    """
    启动时预热：把参考音频编码进 _voice_prompt_cache，避免首个真实请求的 TTFA 被拖慢。

    Args:
        models: 模型副本列表。
        voice_list: active 音色配置列表。
        prime_text: 预热文本。
        do_stream_first_chunk: 是否拉取流式首块。
        stream_max_new_tokens: 预热解码步数上限。
    """
    if not voice_list:
        logger.info("无 active 音色，跳过 voice prime")
        return
    for replica_idx, m in enumerate(models):
        for cfg in voice_list:
            vname = cfg.get("voice_id", "?")
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
                m._prepare_generation(
                    text=prime_text,
                    ref_audio=ref,
                    ref_text=ref_text,
                    language=lang,
                    xvec_only=False,
                    non_streaming_mode=False,
                )
                if do_stream_first_chunk:
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
    在后台线程中运行 generate_voice_clone_streaming，并随着 chunk 到达产生原始 PCM 字节。
    
    Args:
        voice_cfg: 音色配置。
        text: 输入文本。
        req_id: 请求ID。
        
    Yields:
        bytes: PCM16 音频块字节。
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
    """
    健康检查接口。
    
    Returns:
        dict: 状态信息。
    """
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    """
    OpenAI 兼容的语音合成接口。
    
    Args:
        req: 合成请求。
        
    Returns:
        Response/StreamingResponse: 音频数据。
    """
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
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数。
    """
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server (v3 JSON 注册表热加载版) for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        default=os.environ.get(
            "QWEN_TTS_CONFIG",
            os.path.join(os.path.dirname(__file__), "..", "config_v3.json"),
        ),
        help="v3 配置文件路径（注册表/目录/孤儿清理等），默认 ../config_v3.json",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
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
    """
    主入口函数：加载配置、初始化模型副本、启动服务。
    """
    global tts_model, tts_models, hot_registry, SAMPLE_RATE, _request_sem, _metrics_enabled
    global _v3_cleanup_enable, _v3_cleanup_threshold, _v3_prime_token, _v3_prime_lock

    args = _parse_args()

    config = load_config_v3(args.config)
    reg_path = _resolve_cfg_path(args.config, config["voices_registry_path"])
    VoiceRegistryV3(reg_path).ensure_file_exists()
    hot_registry = HotVoiceRegistryV3(reg_path)
    hot_registry.get_registry_copy()

    _v3_cleanup_enable = bool(config.get("enable_orphan_cache_cleanup", True))
    _v3_cleanup_threshold = int(config.get("orphan_cache_cleanup_threshold", 30))
    _v3_prime_token = config.get("tts_internal_prime_token")

    active_voices = hot_registry.list_active_voice_cfgs()
    logger.info("从注册表 %s 加载到 %d 个 active 音色", reg_path, len(active_voices))

    from faster_qwen3_tts import FasterQwen3TTS

    # 参数合法性检查
    if args.replicas < 1:
        raise ValueError("--replicas must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    _metrics_enabled = not args.no_metrics_log
    # 启动时初始化并发控制器
    _request_sem = asyncio.Semaphore(args.concurrency)
    _v3_prime_lock = asyncio.Lock()

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
            active_voices,
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
