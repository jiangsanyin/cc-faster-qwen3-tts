#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server (v4 合并单进程版) for faster-qwen3-tts.

合并了音色管理 API 与 TTS 推理服务，监听单一端口。
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import time
import shlex
import struct
import sys
import threading
from datetime import datetime
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from voice_registry_v4 import (
    HotVoiceRegistryV4,
    VoiceRegistryV4,
    load_config_v4,
    maybe_cleanup_voice_prompt_cache,
)
from voice_manager_router_v4 import router as voice_router, init_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 与 faster_qwen3_tts.model._infer_log 同名；由 config_v4.json 的 inference_logging 配置输出目标
INFERENCE_LOGGER_NAME = "faster_qwen3_tts.inference"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API (v4)")

tts_model = None
tts_models = []
hot_registry: Optional[HotVoiceRegistryV4] = None
_v4_cleanup_enable: bool = True
_v4_cleanup_threshold: int = 30
_v4_prime_lock: Optional[asyncio.Lock] = None
_v4_prime_text: str = "."
_v4_prime_stream_first_chunk: bool = False
_v4_prime_stream_max_tokens: int = 48
# 流式合成 codec 步进块大小（与 README benchmark 中 chunk_size 含义一致）；由 config_v4.json stream_chunk_size 配置
_v4_stream_chunk_size: int = 8

SAMPLE_RATE = 24000
_request_sem: Optional[asyncio.Semaphore] = None
_rr_lock = threading.Lock()
_rr_idx = 0
_req_seq = 0
_metrics_enabled = True


def _next_req_id() -> int:
    """
    生成单调递增的请求 ID。

    用于在 METRICS 日志中区分每条推理请求。

    Returns:
        当前全局计数器自增后的整型 ID。
    """
    global _req_seq
    with _rr_lock:
        _req_seq += 1
        return _req_seq


def _pick_model():
    """
    轮询选取一个模型副本。

    在配置了多 replica 时在各副本间均衡分配单次推理负载。

    Returns:
        选中的 FasterQwen3TTS 实例；若尚未加载副本列表则回退为全局 tts_model。
    """
    global _rr_idx
    if not tts_models: return tts_model
    with _rr_lock:
        m = tts_models[_rr_idx % len(tts_models)]
        _rr_idx += 1
    return m


def _gpu_stats() -> dict:
    """
    采集当前 GPU 状态摘要。

    优先通过 pynvml 读取利用率与显存；不可用时回退为 torch 显存信息。

    Returns:
        包含 gpu_util、mem_util、vram_used_gb、vram_total_gb 等键的字典（视可用性部分键可能缺失）。
    """
    stats = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        stats.update({"gpu_util": util.gpu, "mem_util": util.memory, "vram_used_gb": round(mem.used / (1024**3), 2), "vram_total_gb": round(mem.total / (1024**3), 2)})
    except:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            stats.update({"vram_used_gb": round(torch.cuda.memory_allocated(idx) / (1024**3), 2), "vram_total_gb": round(props.total_memory / (1024**3), 2)})
    return stats


def _log_metrics(req_id: int, mode: str, metrics: dict) -> None:
    """
    输出合成指标日志。

    在开启指标开关时，将业务侧 metrics 与 _gpu_stats() 合并后写入 INFO。

    Args:
        req_id: 请求编号。
        mode: 指标场景标识（如 stream、non_stream_mp3）。
        metrics: 业务指标字典（ttfa_wall_ms、ttfa_ms、ttfa_cuda_graphs_ms、rtf、inter_chunk_* 等）。
    """
    if not _metrics_enabled: return
    merged = {**metrics, **_gpu_stats()}
    logging.getLogger(INFERENCE_LOGGER_NAME).info(
        "METRICS req_id=%d mode=%s %s", req_id, mode, " ".join(f"{k}={v}" for k, v in merged.items())
    )


class SpeechRequest(BaseModel):
    """
    OpenAI 兼容的语音合成请求体。

    承载待合成文本、音色 ID、输出格式等字段，对应 POST /v1/audio/speech 的 JSON 体。
    """

    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = 1.0


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """
    将浮点波形转换为 16-bit PCM 字节流。

    输入约为 [-1, 1] 的 float32，输出为小端 int16 原始字节。

    Args:
        pcm: 一维或可被视作波形的 ndarray。

    Returns:
        PCM16 二进制数据。
    """
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """
    构造 WAV 文件头（单声道、16-bit PCM）。

    流式场景下可将 data_len 设为 0xFFFFFFFF 表示长度未知。

    Args:
        sample_rate: 采样率（Hz）。
        data_len: data 段长度字节数，默认流式占位。

    Returns:
        符合 RIFF/WAVE 格式的头部字节。
    """
    n_channels, bits = 1, 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """
    将 PCM 波形编码为 MP3。

    依赖 pydub，在内存中整段导出，适用于非流式 mp3 响应。

    Args:
        pcm: 浮点波形数组。
        sample_rate: 采样率。

    Returns:
        MP3 文件二进制内容。
    """
    from pydub import AudioSegment
    segment = AudioSegment(_to_pcm16(pcm), frame_rate=sample_rate, sample_width=2, channels=1)
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


def _resolve_cfg_path(config_path: str, p: str) -> str:
    """
    解析配置文件中的路径项。

    绝对路径原样规范化；相对路径相对于配置文件所在目录解析。

    Args:
        config_path: 当前使用的配置文件路径。
        p: 配置项中的路径字符串。

    Returns:
        规范化后的绝对路径。
    """
    if os.path.isabs(p): return os.path.normpath(p)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(config_path)), p))


def _inference_log_path_with_timestamp(resolved_path: str) -> str:
    """
    在已解析的日志路径的主文件名与扩展名之间插入时间缀 ``_YYYY-M-D_HHMMSS``。

    例如 ``.../inference.log`` → ``.../inference_2026-4-11_140346.log``。
    """
    d, base = os.path.dirname(resolved_path), os.path.basename(resolved_path)
    stem, ext = os.path.splitext(base)
    now = datetime.now()
    ts = f"_{now.year}-{now.month}-{now.day}_{now.strftime('%H%M%S')}"
    return os.path.join(d, f"{stem}{ts}{ext}")


def _setup_inference_logging(inference_cfg: dict, config_path: str) -> Optional[str]:
    """
    按 config_v4 的 inference_logging 段配置推理摘要日志（RTF、METRICS 等）的输出目标。

    - console: 是否输出到 stderr
    - file: 日志文件路径模板；空字符串表示不写文件。相对路径相对于配置文件所在目录。
    - add_timestamp: 为 ``true``（默认）时，在 ``file`` 主文件名与扩展名之间插入 ``_YYYY-M-D_HHMMSS``；
      为 ``false`` 时直接使用 ``file`` 解析后的路径，便于固定写入 ``inference.log`` 等单文件。
    - level: 日志级别名，默认 INFO

    未在配置文件中包含 inference_logging 键时，不在此函数中改动 logger，行为与原先一致（随 root 打到控制台）。

    Returns:
        启用文件日志时返回实际写入的绝对路径，否则 ``None``。
    """
    console = bool(inference_cfg.get("console", True))
    file_rel = str(inference_cfg.get("file", "") or "").strip()
    add_timestamp = bool(inference_cfg.get("add_timestamp", True))
    level_name = str(inference_cfg.get("level", "INFO")).upper()
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


def _log_service_startup_summary(
    args: argparse.Namespace,
    *,
    resolved_prime_text: str,
    resolved_prime_stream_first_chunk: bool,
    resolved_prime_stream_max_tokens: int,
    resolved_stream_chunk_size: int,
    inference_log_file: Optional[str],
) -> None:
    """
    将完整启动命令行与关键运行参数写入推理摘要 logger（与 METRICS / RTF 同通道）。

    若 inference_logging 将 faster_qwen3_tts.inference 配成无 handler 且不向上传播，则回退到本模块 logger，避免启动信息被静默丢弃。
    """
    inf = logging.getLogger(INFERENCE_LOGGER_NAME)
    if not inf.handlers and not inf.propagate:
        log = logger
    else:
        log = inf

    try:
        cmdline = shlex.join(sys.argv)
    except (AttributeError, ValueError, TypeError):
        cmdline = " ".join(sys.argv)

    log.info("STARTUP cmdline=%s", cmdline)
    log.info("STARTUP --config=%s", os.path.abspath(args.config))
    log.info("STARTUP --model=%s", args.model)
    log.info("STARTUP --host=%s", args.host)
    log.info("STARTUP --port=%d", args.port)
    log.info("STARTUP --device=%s", args.device)
    log.info("STARTUP --concurrency=%d", args.concurrency)
    log.info("STARTUP --replicas=%d", args.replicas)
    log.info("STARTUP --prime-stream-first-chunk=%s (effective=%s)", args.prime_stream_first_chunk, resolved_prime_stream_first_chunk)
    log.info("STARTUP --prime-text(cli)=%r", args.prime_text)
    log.info("STARTUP --prime-text(effective)=%r", resolved_prime_text)
    log.info("STARTUP prime_stream_max_new_tokens=%s (effective=%d)", args.prime_stream_max_new_tokens, resolved_prime_stream_max_tokens)
    log.info("STARTUP stream_chunk_size(config effective)=%d", resolved_stream_chunk_size)
    log.info("STARTUP --no-metrics-log=%s", args.no_metrics_log)
    log.info("STARTUP --skip-warmup=%s", args.skip_warmup)
    log.info("STARTUP --warmup-prefill-len=%d", args.warmup_prefill_len)
    log.info("STARTUP --skip-prime-voices=%s", args.skip_prime_voices)
    if inference_log_file:
        log.info("STARTUP inference_log_file=%s", inference_log_file)


def _run_orphan_cleanup_all() -> None:
    """
    对所有模型副本执行语音克隆缓存孤儿清理。

    依据 hot_registry 当前快照与配置项 enable_orphan_cache_cleanup、threshold 调用 maybe_cleanup_voice_prompt_cache。
    """
    if hot_registry is None or not tts_models: return
    reg = hot_registry.get_registry_copy()
    for m in tts_models:
        maybe_cleanup_voice_prompt_cache(m, reg, enable=_v4_cleanup_enable, threshold=_v4_cleanup_threshold)


def _run_prime_for_voice(m, cfg: dict, prime_text: str, do_stream: bool, max_tokens: int) -> None:
    """
    对单个模型与单条音色执行预热。

    先调用 _prepare_generation 填充语音克隆相关缓存；若 do_stream 为 True，再拉取流式生成器首块以预热解码路径。

    Args:
        m: FasterQwen3TTS 模型实例。
        cfg: 音色配置（含 ref_audio、ref_text、language 等）。
        prime_text: 预热用的目标文本（通常与配置 prime_text 一致）。
        do_stream: 是否额外执行流式首块预热。
        max_tokens: 流式预热时 generate 的最大 new token 数。
    """
    ref = cfg.get("ref_audio")
    if not ref or not os.path.isfile(ref):
        return
    lang = cfg.get("language", "Auto")
    rt = cfg.get("ref_text", "")
    m._prepare_generation(text=prime_text, ref_audio=ref, ref_text=rt, language=lang, xvec_only=False, non_streaming_mode=False)
    if do_stream:
        gen = m.generate_voice_clone_streaming(
            text=prime_text, language=lang, ref_audio=ref, ref_text=rt,
            chunk_size=_v4_stream_chunk_size, non_streaming_mode=False, max_new_tokens=max_tokens,
        )
        next(gen, None)


def _prime_single_voice_sync(models: list, cfg: dict) -> None:
    """
    同步预热单个音色（全部副本）。

    使用全局 _v4_prime_text、_v4_prime_stream_first_chunk、_v4_prime_stream_max_tokens，供 run_in_executor 在线程中调用。

    Args:
        models: 全部 TTS 模型副本列表。
        cfg: 该音色的 active 配置字典。
    """
    vname, ref = cfg.get("voice_id", "?"), cfg.get("ref_audio")
    if not ref or not os.path.isfile(ref):
        return
    for i, m in enumerate(models):
        try:
            logger.info("Priming voice cache: voice=%r replica=%d/%d", vname, i + 1, len(models))
            _run_prime_for_voice(
                m, cfg, _v4_prime_text, _v4_prime_stream_first_chunk, _v4_prime_stream_max_tokens,
            )
        except Exception as exc:
            logger.warning("预热失败: %s", exc)


async def prime_voice_v4(voice_id: str) -> None:
    """
    进程内音色预热入口。

    在管理端成功写入注册表后由路由异步触发：解析 active 音色、在锁内于线程池执行同步预热，随后做一次孤儿缓存清理。

    Args:
        voice_id: 注册表中的音色 ID。
    """
    if hot_registry is None: return
    hot_registry.get_registry_copy()
    if not (cfg := hot_registry.resolve_active_voice(voice_id)): return
    # 预热会占 GPU：用全局 asyncio.Lock 串行化多次 prime_voice_v4，避免并发预热与推理抢资源。
    # _v4_prime_lock 在 main() 中赋值；若为 None 则退化为每次新建 Lock（仅兜底，一般不应依赖）。
    async with (_v4_prime_lock or asyncio.Lock()):
        await asyncio.get_event_loop().run_in_executor(None, _prime_single_voice_sync, tts_models, cfg)
    _run_orphan_cleanup_all()


def resolve_voice(voice_id: str) -> dict:
    """
    解析并返回可用于合成的 active 音色配置。

    Args:
        voice_id: 请求中指定的音色 ID。

    Returns:
        热注册表解析后的音色配置字典。

    Raises:
        HTTPException: 503 注册表未就绪；400 音色不存在或已禁用。
    """
    if hot_registry is None: raise HTTPException(status_code=503, detail="Registry not ready")
    if cfg := hot_registry.resolve_active_voice(voice_id): return cfg
    raise HTTPException(status_code=400, detail=f"Voice '{voice_id}' 不存在或已禁用")


def _prime_voice_caches(models: list, voice_list: list, prime_text: str, do_stream: bool, max_tokens: int) -> None:
    """
    启动阶段批量预热注册表中的音色。

    对每个模型副本与 voice_list 中每条 ref_audio 存在的条目调用 _run_prime_for_voice。

    Args:
        models: 全部 TTS 副本。
        voice_list: active 音色配置列表。
        prime_text: 预热用文本。
        do_stream: 是否流式首块预热。
        max_tokens: 流式预热 token 上限。
    """
    if not voice_list:
        return
    for i, m in enumerate(models):
        for cfg in voice_list:
            if not cfg.get("ref_audio") or not os.path.isfile(cfg["ref_audio"]):
                continue
            try:
                logger.info("Priming: voice=%r replica=%d/%d", cfg.get("voice_id"), i + 1, len(models))
                _run_prime_for_voice(m, cfg, prime_text, do_stream, max_tokens)
            except Exception as exc:
                logger.warning("预热失败: %s", exc)


def _percentile_95(values: list) -> float:
    """样本的 95 分位；空列表返回 0.0。"""
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


async def _stream_chunks(
    voice_cfg: dict,
    text: str,
    req_id: int,
    t_ttfa0: float,
) -> AsyncGenerator[bytes, None]:
    """
    异步迭代流式合成 PCM16 数据块。

    在后台线程中运行 generate_voice_clone_streaming ，通过队列与 run_in_executor 桥接到 asyncio，结束时写入 METRICS。

    Args:
        voice_cfg: 音色配置（ref_audio、ref_text、language）。
        text: 待合成文本。
        req_id: 用于日志关联的请求 ID。
        t_ttfa0: ``time.perf_counter()`` 锚点（请求在校验通过后、进入合成路径前），用于墙钟 TTFA。

    Yields:
        每个音频块的 PCM16 字节。
    """
    q, _DONE = queue.Queue(), object()

    def producer():
        """
        流式合成生产者（运行于守护线程）。

        将模型输出的 chunk 放入队列；异常时放入异常对象；finally 中打点并放入结束标记。
        """
        # 计时与累计量初值（finally 中 _log_metrics 的 stream 字段对应）：
        # t0 — producer 起点 perf_counter → METRICS total_ms（本线程内整段流式墙钟毫秒）。
        # ttfa_ms — 当 total_gen_ms 首次 >0 时冻结为当时累计值 → METRICS ttfa_ms（模型 _timing 累计 prefill+decode 的首段口径）。
        # total_gen_ms — 各 chunk 的 prefill_ms+decode_ms 之和 → 参与 METRICS rtf 分母（不单独落字段）。
        # total_audio_s — 各 chunk 换算的音频时长（秒）之和 → METRICS audio_s，且作 rtf 分子。
        t0, ttfa_ms, total_gen_ms, total_audio_s = time.perf_counter(), None, 0.0, 0.0
        
        # TTFA（墙钟）、Graphs 首段耗时与块间间隔统计初值：
        # ttfa_wall_ms — 自 t_ttfa0（async 侧进入合成路径前锚点）至首块 q.put 的墙钟 ms → METRICS ttfa_wall_ms。
        # t_prev_put — 上一次 q.put 前的 perf_counter；不落字段，用于与本次 now 计算相邻入队间隔。
        # chunk_gaps_ms — 各相邻两次入队间隔（ms）列表 → METRICS inter_chunk_max_ms / inter_chunk_p95_ms。
        ttfa_wall_ms, t_prev_put, chunk_gaps_ms = None, None, []
        
        # ttfa_cuda_graphs_ms — t_bench0（cuda sync 后、调用 generate 前）至首块迭代内再 sync 的 ms → METRICS ttfa_cuda_graphs_ms。
        ttfa_cuda_graphs_ms: Optional[float] = None
        
        try:
            # 对于多replicas（replica为1是多replicas的特殊情况，此时返回的总是同一个model）时，需要先pick_model，然后才能生成voice_clone_streaming
            model = _pick_model()
            # ttfa_cuda_graphs_ms 的测量流程 ： 与 README「CUDA Graphs TTFA」及 benchmarks/throughput.py 一致：cuda sync → t_bench0 →
            # generate_voice_clone_streaming(...) → 首块 next → cuda sync → ttfa_cuda_graphs_ms = (now - t_bench0)*1000
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            t_bench0 = time.perf_counter()
            # parity_mode 由模型是否加载了 CUDA Graph 决定：
            # 无 Graph（disable_cuda_graph=True）时走 parity（动态 KV Cache）路径，
            # 有 Graph 时走 Graph 快路径（静态 KV Cache + CUDA Graph replay）。
            parity_mode = (model.talker_graph is None)
            gen = model.generate_voice_clone_streaming(
                text=text,
                language=voice_cfg.get("language", "Auto"),
                ref_audio=voice_cfg["ref_audio"],
                ref_text=voice_cfg.get("ref_text", ""),
                chunk_size=_v4_stream_chunk_size,
                non_streaming_mode=False,
                parity_mode=parity_mode,
            )
            # generate_voice_clone_streaming 每步 yield (audio_chunk, sample_rate, timing_dict)：
            # chunk — 本步波形（numpy）；经 _to_pcm16 写入 HTTP 流。
            # _sr — 采样率 Hz，与 chunk 长度合计 audio_s / rtf。
            # _timing — 本步计时 dict（如 prefill_ms、decode_ms），累计 total_gen_ms / ttfa_ms。
            for chunk, _sr, _timing in gen:
                # 第一次迭代时，记录 ttfa_cuda_graphs_ms，之后不再记录
                if ttfa_cuda_graphs_ms is None:
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    ttfa_cuda_graphs_ms = (time.perf_counter() - t_bench0) * 1000.0
                total_gen_ms += float(_timing.get("prefill_ms", 0.0)) + float(_timing.get("decode_ms", 0.0))
                if ttfa_ms is None and total_gen_ms > 0: ttfa_ms = total_gen_ms
                if _sr: total_audio_s += (len(chunk) if isinstance(chunk, np.ndarray) else len(np.array(chunk))) / float(_sr)
                now = time.perf_counter()
                if t_prev_put is not None:
                    chunk_gaps_ms.append((now - t_prev_put) * 1000)
                else:
                    ttfa_wall_ms = (now - t_ttfa0) * 1000
                t_prev_put = now
                q.put(chunk)
        except Exception as exc: q.put(exc)
        finally:
            ic_max = max(chunk_gaps_ms) if chunk_gaps_ms else 0.0
            ic_p95 = _percentile_95(chunk_gaps_ms)
            _log_metrics(
                req_id,
                "stream",
                {
                    "ttfa_wall_ms": round(ttfa_wall_ms or 0.0, 1),
                    "ttfa_ms": round(ttfa_ms or 0.0, 1),
                    "ttfa_cuda_graphs_ms": round(ttfa_cuda_graphs_ms or 0.0, 1),
                    "inter_chunk_max_ms": round(ic_max, 1),
                    "inter_chunk_p95_ms": round(ic_p95, 1),
                    "rtf": round(total_audio_s / (total_gen_ms / 1000), 3) if total_gen_ms > 0 else 0.0,
                    "audio_s": round(total_audio_s, 3),
                    "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                },
            )
            q.put(_DONE)
    # 启动守护线程运行 producer，producer 在后台线程中运行 generate_voice_clone_streaming ，通过队列与 run_in_executor 桥接到 asyncio，结束时写入 METRICS。
    threading.Thread(target=producer, daemon=True).start()
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE: break
        if isinstance(item, Exception): raise item
        yield _to_pcm16(item)


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    """
    OpenAI 兼容的语音合成接口。

    支持 wav/pcm 流式响应与 mp3 非流式整段响应；并发由全局信号量限制，避免 GPU 过载。

    Args:
        req: 解析后的请求体（input、voice、response_format 等）。

    Returns:
        StreamingResponse（wav/pcm）或含 MP3 二进制内容的 Response。

    Raises:
        HTTPException: 模型未加载、input 为空、音色无效或格式不支持等。
    """
    if tts_model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip(): raise HTTPException(status_code=400, detail="input empty")
    voice_cfg, fmt = resolve_voice(req.voice), req.response_format.lower()
    _CT = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg"}
    if fmt not in _CT: raise HTTPException(status_code=400, detail="format not supported")
    req_id = _next_req_id()
    # 墙钟 TTFA 锚点：校验通过后、排队/合成开始前（含 --concurrency 等待）
    t_ttfa0 = time.perf_counter()
    await (_request_sem or asyncio.Semaphore(1)).acquire()
    sem_released = False
    try:
        if fmt == "mp3":
            loop, t0 = asyncio.get_event_loop(), time.perf_counter()
            # 线程池中跑非流式整段克隆，避免阻塞事件循环
            audio_arrays, sr = await loop.run_in_executor(None, lambda: _pick_model().generate_voice_clone(text=req.input, language=voice_cfg.get("language", "Auto"), ref_audio=voice_cfg["ref_audio"], ref_text=voice_cfg.get("ref_text", "")))
            audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
            total_ms = (time.perf_counter() - t0) * 1000
            ttfa_wall_ms = (time.perf_counter() - t_ttfa0) * 1000
            _log_metrics(
                req_id,
                "non_stream_mp3",
                {
                    "ttfa_wall_ms": round(ttfa_wall_ms, 1),
                    "ttfa_ms": round(total_ms, 1),
                    "rtf": round((len(audio) / sr if sr else 0.0) / (total_ms / 1000), 3) if total_ms > 0 else 0.0,
                    "audio_s": round(len(audio) / sr if sr else 0.0, 3),
                    "total_ms": round(total_ms, 1),
                },
            )
            _request_sem.release(); sem_released = True
            return Response(content=_to_mp3_bytes(audio, sr), media_type=_CT[fmt])

        async def audio_stream():
            """
            构造流式 HTTP 响应体。

            wav 格式时先输出 WAV 头，再经 _stream_chunks 输出 PCM；在 finally 中释放并发信号量。
            """
            try:
                # _wav_header(SAMPLE_RATE) 的返回值如下， 是WAV文件的头部，包含文件类型、格式、采样率、通道数、位数、数据长度等信息：
                # b'RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\xc0]\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00data\xff\xff\xff\xff'
                if fmt == "wav": yield _wav_header(SAMPLE_RATE)
                async for raw in _stream_chunks(voice_cfg, req.input, req_id, t_ttfa0): yield raw
            finally:
                if not sem_released: _request_sem.release()
        return StreamingResponse(audio_stream(), media_type=_CT[fmt])
    except Exception:
        if not sem_released: _request_sem.release()
        raise


def _parse_args():
    """
    解析本服务所需的命令行参数。

    包含配置文件路径、模型、监听地址、并发副本数、CUDA 预热与 prime 相关开关。

    Returns:
        argparse 解析结果 Namespace。
    """
    p = argparse.ArgumentParser(description="OpenAI-compatible TTS server (v4 合并单进程版)")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config_v4.json"))
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--no-metrics-log", action="store_true")
    p.add_argument("--disable-cuda-graph", action="store_true", help="禁用 CUDA Graph")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--warmup-prefill-len", type=int, default=100)
    p.add_argument("--skip-prime-voices", action="store_true")
    p.add_argument(
        "--prime-text",
        default=None,
        help="覆盖配置文件 prime_text；未指定则使用 config 中 prime_text，无配置键时为 .",
    )
    p.add_argument("--prime-stream-first-chunk", action="store_true", help="强制开启流式首块预热（叠加配置）")
    p.add_argument("--prime-stream-max-new-tokens", type=int, default=None, help="覆盖配置中的 prime_stream_max_new_tokens")
    return p.parse_args()


def main():
    """
    服务进程入口。

    加载 config_v4、初始化注册表与 HotVoiceRegistry、挂载 voice 管理路由、加载 TTS 模型与 CUDA 预热、可选批量 prime 活跃音色，最后启动 Uvicorn 阻塞运行。
    """
    global tts_model, tts_models, hot_registry, SAMPLE_RATE, _request_sem, _metrics_enabled, _v4_cleanup_enable, _v4_cleanup_threshold, _v4_prime_lock
    global _v4_prime_text, _v4_prime_stream_first_chunk, _v4_prime_stream_max_tokens, _v4_stream_chunk_size
    args = _parse_args()
    cfg = load_config_v4(args.config)
    reg_path = _resolve_cfg_path(args.config, cfg["voices_registry_path"])
    VoiceRegistryV4(reg_path).ensure_file_exists()
    hot_registry = HotVoiceRegistryV4(reg_path)
    hot_registry.get_registry_copy()
    _v4_cleanup_enable, _v4_cleanup_threshold = bool(cfg.get("enable_orphan_cache_cleanup", True)), int(cfg.get("orphan_cache_cleanup_threshold", 30))
    _v4_prime_text = args.prime_text if args.prime_text is not None else str(cfg.get("prime_text", ".") or ".")
    _v4_prime_stream_first_chunk = bool(cfg.get("prime_stream_first_chunk", False)) or bool(args.prime_stream_first_chunk)
    _v4_prime_stream_max_tokens = int(
        args.prime_stream_max_new_tokens
        if args.prime_stream_max_new_tokens is not None
        else cfg.get("prime_stream_max_new_tokens", 48)
    )
    try:
        _v4_stream_chunk_size = max(1, int(cfg.get("stream_chunk_size", 8)))
    except (TypeError, ValueError):
        logger.warning("config stream_chunk_size 无效，使用默认值 8")
        _v4_stream_chunk_size = 8
    infer_log_path: Optional[str] = None
    if "inference_logging" in cfg:
        infer_log_path = _setup_inference_logging(cfg.get("inference_logging") or {}, args.config)
    _log_service_startup_summary(
        args,
        resolved_prime_text=_v4_prime_text,
        resolved_prime_stream_first_chunk=_v4_prime_stream_first_chunk,
        resolved_prime_stream_max_tokens=_v4_prime_stream_max_tokens,
        resolved_stream_chunk_size=_v4_stream_chunk_size,
        inference_log_file=infer_log_path,
    )

    # 初始化管理路由并挂载
    init_router(
        registry=VoiceRegistryV4(reg_path),
        tone_dir=_resolve_cfg_path(args.config, cfg["tone_wav_file_dir"]),
        allowed_formats=cfg.get("allowed_audio_formats", ["wav", "mp3"]),
        max_file_mb=float(cfg.get("max_audio_file_size_mb", 20)),
        prime_fn=prime_voice_v4,
        model_loaded_fn=lambda: tts_model is not None,
    )
    app.include_router(voice_router)

    from faster_qwen3_tts import FasterQwen3TTS
    _metrics_enabled, _request_sem, _v4_prime_lock = not args.no_metrics_log, asyncio.Semaphore(args.concurrency), asyncio.Lock()
    tts_models = [FasterQwen3TTS.from_pretrained(args.model, device=args.device, dtype=torch.bfloat16, disable_cuda_graph=args.disable_cuda_graph) for _ in range(args.replicas)]
    for i, m in enumerate(tts_models):
        if not args.skip_warmup:
            logger.info("Warming up replica %d/%d...", i+1, args.replicas)
            m._warmup(prefill_len=args.warmup_prefill_len)
    tts_model = tts_models[0]; SAMPLE_RATE = tts_model.sample_rate
    if not args.skip_prime_voices:
        _prime_voice_caches(
            tts_models,
            hot_registry.list_active_voice_cfgs(),
            _v4_prime_text,
            _v4_prime_stream_first_chunk,
            _v4_prime_stream_max_tokens,
        )
    logger.info("Server ready. Listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
