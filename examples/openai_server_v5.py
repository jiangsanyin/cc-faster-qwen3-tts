#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-compatible TTS API server (v5 多路Graph轮询调度版) for faster-qwen3-tts.

基于 §6.4.1 路线B综合方案实现：
- 线程池管理正在执行的请求（大小由 --concurrency 控制）
- 线程池外的请求进入队列等待
- 全局轮询调度器统一分配GPU时间，每轮各路各产一个PCM chunk

与 v4 的关键区别：
- v4: 单路独占GPU，多路随机争抢
- v5: 多路轮询调度，追求公平性+流畅性平衡
"""
import argparse
import asyncio
import io
import logging
import os
import queue
import shlex
import struct
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from voice_registry_v5 import (
    HotVoiceRegistryV5,
    VoiceRegistryV5,
    load_config_v5,
    maybe_cleanup_voice_prompt_cache,
)
from voice_manager_router_v5 import router as voice_router, init_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
INFERENCE_LOGGER_NAME = "faster_qwen3_tts.inference"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API (v5 多路Graph轮询调度版)")

tts_model = None
tts_models: list = []
hot_registry: Optional[HotVoiceRegistryV5] = None
_v5_cleanup_enable: bool = True
_v5_cleanup_threshold: int = 30
_v5_prime_lock: Optional[asyncio.Lock] = None
_v5_prime_text: str = "."
_v5_prime_stream_first_chunk: bool = False
_v5_prime_stream_max_new_tokens: int = 48
_v5_stream_chunk_size: int = 8

SAMPLE_RATE = 24000
_rr_lock = threading.Lock()
_rr_idx = 0
_req_seq = 0
_req_seq_lock = threading.Lock()
_metrics_enabled = True
_metrics_log_mode = "DEV"
_DEV_METRIC_KEYS = (
    "ttfa_wall_ms",
    "ttfa_cuda_graphs_ms",
    "inter_chunk_max_ms",
    "inter_chunk_p95_ms",
    "inter_chunk_p99_ms",
    "rtf",
    "audio_s",
    "total_gen_ms",
    "total_ms",
    "n_chunks",
)

# §6.7.1：与 `_log_metrics` 输出的 `merged` 一致旁路到 HTTP（供试用页拉取）
_UI_METRICS_MAX = 512
_ui_metrics_lock = threading.Lock()
_ui_metrics_order: "OrderedDict[int, dict]" = OrderedDict()

_DEBUG_METRIC_EXTRA_KEYS = (
    "ttfa_ms",
    "first_prefill_ms",
    "first_decode_ms",
    "first_overhead_ms",
    "prefill_len",
    "attention_mask_shape",
    "trailing_text_len",
    "icl",
    "parity_mode",
)

# v5: 线程池与调度器全局状态
_v5_thread_pool_sem: Optional[threading.Semaphore] = None
_v5_request_queue: "queue.Queue" = queue.Queue()
_v5_active_workers: Dict[int, "TTSWorker"] = {}
_v5_scheduler_thread: Optional[threading.Thread] = None
_v5_queue_manager_thread: Optional[threading.Thread] = None
_v5_scheduler_lock: threading.Lock = threading.Lock()
_v5_scheduler_running: bool = False

# v5: 流式 vs 非流式 互斥 (对应 §6.4.1.3 第 5 条)
# - 流式调度器在每次 worker.step() 前后获取/释放「读锁」（允许多个流式 step 交替推进）；
# - 非流式 MP3 请求获取「写锁」后整段独占 GPU；
# - 写者优先：一旦有 MP3 在等写锁，调度器随后就不再获取新的读锁，等 MP3 放行后再继续；
# - 非流式之间另用 `asyncio.Lock` 做 FIFO 串行（asyncio.Lock 在 CPython 里是 FIFO 的）。
_v5_nonstream_fifo_lock: Optional[asyncio.Lock] = None  # 在 main() 里懒创建
# 「多路并行流」是否可走 CUDA Graph：仅当 replicas >= concurrency —— 否则会话交错共用同一 pair of graph → 杂音/超长 runaway。
_v5_stream_use_cuda_graph: bool = False


class _WriterPreferringRWLock:
    """轻量写者优先读写锁：流式=读者（可并发）、非流式=写者（独占 + 优先）。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    def has_waiting_writer(self) -> bool:
        with self._cond:
            return self._writers_waiting > 0 or self._writer


_v5_gpu_rwlock = _WriterPreferringRWLock()


def _next_req_id() -> int:
    """生成单调递增的请求 ID（线程安全）。"""
    global _req_seq
    with _req_seq_lock:
        _req_seq += 1
        return _req_seq


def _pick_model():
    """轮询选取一个模型副本（线程安全）。"""
    global _rr_idx
    if not tts_models:
        return tts_model
    with _rr_lock:
        m = tts_models[_rr_idx % len(tts_models)]
        _rr_idx = (_rr_idx + 1) % max(1, len(tts_models))
    return m


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """将浮点波形 [-1, 1] 转为有符号 16-bit PCM 字节。"""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int) -> bytes:
    """构造 16-bit mono WAV 文件头（data 段长度留空表示流式）。"""
    n_channels, bits = 1, 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 0xFFFFFFFF))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", 0xFFFFFFFF))
    return buf.getvalue()


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """将 PCM 波形编码为 MP3。"""
    from pydub import AudioSegment
    segment = AudioSegment(_to_pcm16(pcm), frame_rate=sample_rate, sample_width=2, channels=1)
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


def _percentile_95(values: list) -> float:
    """样本的 95 分位；空列表返回 0.0。"""
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _percentile_99(values: list) -> float:
    """样本的 99 分位；空列表返回 0.0。"""
    if not values:
        return 0.0
    return float(np.percentile(values, 99))


def _gpu_stats() -> dict:
    """采集当前 GPU 状态摘要（失败时返回空）。"""
    stats: Dict[str, float] = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        stats.update({
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "vram_used_gb": round(mem.used / (1024 ** 3), 2),
            "vram_total_gb": round(mem.total / (1024 ** 3), 2),
        })
    except Exception:
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                stats.update({
                    "vram_used_gb": round(torch.cuda.memory_allocated(idx) / (1024 ** 3), 2),
                    "vram_total_gb": round(props.total_memory / (1024 ** 3), 2),
                })
            except Exception:
                pass
    return stats


def _normalize_metrics_log_mode(value: Optional[str]) -> str:
    """Normalize metrics mode; DEBUG keeps all fields, DEV keeps production fields only."""
    mode = str(value or "DEBUG").strip().upper()
    if mode == "DEBUG":
        return "DEBUG"
    else:
        return "DEV"


def _ui_metrics_store(req_id: int, mode: str, merged: dict) -> None:
    """线程安全：保存与 METRICS 日志行一致的 merged（DEV/DEBUG 裁剪后）。"""
    entry = {
        "req_id": req_id,
        "mode": mode,
        "ts": time.time(),
        "metrics": dict(merged),
    }
    with _ui_metrics_lock:
        if req_id in _ui_metrics_order:
            _ui_metrics_order.pop(req_id, None)
        _ui_metrics_order[req_id] = entry
        while len(_ui_metrics_order) > _UI_METRICS_MAX:
            _ui_metrics_order.popitem(last=False)


def _log_metrics(req_id: int, mode: str, metrics: dict) -> None:
    """将推理指标写入推理摘要 logger。"""
    if not _metrics_enabled:
        return
    if _metrics_log_mode == "DEV":
        merged = {k: metrics[k] for k in _DEV_METRIC_KEYS if k in metrics}
    else:
        merged = {**metrics}
    _ui_metrics_store(req_id, mode, merged)
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
    """按 config_v5 的 inference_logging 段配置推理摘要日志输出目标。"""
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


# ---------------------------------------------------------------------------
# TTSWorker 类
# ---------------------------------------------------------------------------

@dataclass
class TTSWorker:
    """每个流式TTS请求对应的Worker。"""
    req_id: int
    voice_cfg: dict
    text: str

    generator: Optional[Iterator] = None
    response_queue: "queue.Queue" = field(default_factory=queue.Queue)

    t_start: float = field(default_factory=time.perf_counter)
    t_ttfa0: float = 0.0
    ttfa_wall_ms: Optional[float] = None
    ttfa_ms: Optional[float] = None  # 首块产出时冻结的累计推理耗时（对齐 v4）
    ttfa_cuda_graphs_ms: Optional[float] = None
    first_prefill_ms: Optional[float] = None
    first_decode_ms: Optional[float] = None
    prefill_len: Optional[int] = None
    attention_mask_shape: str = ""
    trailing_text_len: Optional[int] = None
    icl: Optional[int] = None
    parity_mode: Optional[int] = None
    chunk_gaps_ms: List[float] = field(default_factory=list)
    total_gen_ms: float = 0.0
    total_audio_s: float = 0.0
    n_chunks: int = 0
    t_prev_put: Optional[float] = None

    finished: bool = False
    error: Optional[BaseException] = None
    client_cancelled: bool = False
    t_bench0: float = 0.0
    _metrics_logged: bool = False

    def init_generator(self) -> None:
        """初始化底层生成器（在调度器线程中首次 step 前被调用）。"""
        if self.generator is not None or self.finished:
            return
        try:
            model = _pick_model()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            self.t_bench0 = time.perf_counter()
            graphs_ok = (
                _v5_stream_use_cuda_graph
                and getattr(model, "predictor_graph", None) is not None
                and getattr(model, "talker_graph", None) is not None
            )
            parity_mode = not graphs_ok

            self.generator = model.generate_voice_clone_streaming(
                text=self.text,
                language=self.voice_cfg.get("language", "Auto"),
                ref_audio=self.voice_cfg["ref_audio"],
                ref_text=self.voice_cfg.get("ref_text", ""),
                chunk_size=_v5_stream_chunk_size,
                non_streaming_mode=False,
                parity_mode=parity_mode,
            )
        except Exception as e:
            self.error = e
            self.finished = True
            logger.error("Worker %d init failed: %s", self.req_id, e)

    def step(self) -> None:
        """推进一步：从生成器拉取一个PCM chunk。"""
        if self.finished or self.error:
            return
        if self.generator is None:
            self.init_generator()
            if self.finished:
                return

        try:
            with torch.inference_mode():
                chunk, sr, timing = next(self.generator)

            # 首块：cuda sync 后记录 ttfa_cuda_graphs_ms
            if self.ttfa_cuda_graphs_ms is None:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                self.ttfa_cuda_graphs_ms = (time.perf_counter() - self.t_bench0) * 1000.0

            chunk_prefill_ms = float(timing.get("prefill_ms", 0.0))
            chunk_decode_ms = float(timing.get("decode_ms", 0.0))
            self.total_gen_ms += chunk_prefill_ms + chunk_decode_ms
            if sr:
                chunk_len = len(chunk) if isinstance(chunk, np.ndarray) else len(np.array(chunk))
                self.total_audio_s += chunk_len / float(sr)
            self.n_chunks += 1

            now = time.perf_counter()
            if self.t_prev_put is None:
                self.ttfa_wall_ms = (now - self.t_ttfa0) * 1000.0
                # 对齐 v4：ttfa_ms 冻结在首块首次出现 total_gen_ms>0 时的累计值
                if self.ttfa_ms is None and self.total_gen_ms > 0:
                    self.ttfa_ms = self.total_gen_ms
                    self.first_prefill_ms = chunk_prefill_ms
                    self.first_decode_ms = chunk_decode_ms
                    self.prefill_len = int(timing.get("prefill_len", 0) or 0)
                    self.attention_mask_shape = str(timing.get("attention_mask_shape", "") or "")
                    self.trailing_text_len = int(timing.get("trailing_text_len", 0) or 0)
                    self.icl = int(timing.get("icl", 0) or 0)
                    self.parity_mode = int(timing.get("parity_mode", 0) or 0)
            else:
                self.chunk_gaps_ms.append((now - self.t_prev_put) * 1000.0)
            self.t_prev_put = now

            if not self.client_cancelled:
                self.response_queue.put(_to_pcm16(chunk))

        except StopIteration:
            self.finished = True
            self._log_final_metrics()
        except Exception as e:
            self.error = e
            self.finished = True
            self._log_final_metrics()
            logger.error("Worker %d step error: %s", self.req_id, e)

    def _log_final_metrics(self) -> None:
        """记录最终指标（幂等）。"""
        if self._metrics_logged:
            return
        self._metrics_logged = True
        ic_max = max(self.chunk_gaps_ms) if self.chunk_gaps_ms else 0.0
        ic_p95 = _percentile_95(self.chunk_gaps_ms)
        ic_p99 = _percentile_99(self.chunk_gaps_ms)
        ttfa_ms = self.ttfa_ms or 0.0
        first_overhead_ms = (
            (self.ttfa_cuda_graphs_ms or 0.0) - ttfa_ms
            if self.ttfa_cuda_graphs_ms is not None and self.ttfa_ms is not None
            else 0.0
        )
        _log_metrics(
            self.req_id,
            "stream",
            {
                "ttfa_wall_ms": round(self.ttfa_wall_ms or 0.0, 1),
                "ttfa_ms": round(ttfa_ms, 1),
                "ttfa_cuda_graphs_ms": round(self.ttfa_cuda_graphs_ms or 0.0, 1),
                "first_prefill_ms": round(self.first_prefill_ms or 0.0, 1),
                "first_decode_ms": round(self.first_decode_ms or 0.0, 1),
                "first_overhead_ms": round(first_overhead_ms, 1),
                "prefill_len": self.prefill_len or 0,
                "attention_mask_shape": self.attention_mask_shape or "",
                "trailing_text_len": self.trailing_text_len or 0,
                "icl": self.icl or 0,
                "parity_mode": self.parity_mode or 0,
                "inter_chunk_max_ms": round(ic_max, 1),
                "inter_chunk_p95_ms": round(ic_p95, 1),
                "inter_chunk_p99_ms": round(ic_p99, 1),
                "rtf": round(self.total_audio_s / (self.total_gen_ms / 1000), 3) if self.total_gen_ms > 0 else 0.0,
                "audio_s": round(self.total_audio_s, 3),
                "total_gen_ms": round(self.total_gen_ms, 1),
                "total_ms": round((time.perf_counter() - self.t_start) * 1000, 1),
                "n_chunks": self.n_chunks,
                # "cancelled": int(self.client_cancelled),
            },
        )


# ---------------------------------------------------------------------------
# 全局轮询调度器
# ---------------------------------------------------------------------------

def _scheduler_loop():
    """轮询调度器主循环（运行于独立线程）。"""
    global _v5_scheduler_running
    logger.info("Scheduler thread started")

    while _v5_scheduler_running:
        with _v5_scheduler_lock:
            active_workers = list(_v5_active_workers.values())

        if not active_workers:
            time.sleep(0.01)
            continue

        finished_ids: List[int] = []
        for worker in active_workers:
            if worker.finished or worker.error:
                finished_ids.append(worker.req_id)
                continue
            # 每次 step() 前都获取一次读锁；若此时有非流式写者在等，
            # acquire_read() 会阻塞到写者完成后再继续，实现"写者优先 + 流式让步"。
            _v5_gpu_rwlock.acquire_read()
            try:
                worker.step()
                if worker.finished or worker.error:
                    finished_ids.append(worker.req_id)
            except Exception as e:
                logger.error("Scheduler step error for worker %d: %s", worker.req_id, e)
                worker.error = e
                worker.finished = True
                finished_ids.append(worker.req_id)
            finally:
                _v5_gpu_rwlock.release_read()

        if finished_ids:
            with _v5_scheduler_lock:
                for req_id in finished_ids:
                    w = _v5_active_workers.pop(req_id, None)
                    if w is not None:
                        w.response_queue.put(None)  # 结束标记
            for _ in finished_ids:
                if _v5_thread_pool_sem:
                    _v5_thread_pool_sem.release()

        time.sleep(0.001)

    logger.info("Scheduler thread stopped")


def _start_scheduler():
    """启动轮询调度器线程（幂等）。"""
    global _v5_scheduler_thread, _v5_scheduler_running
    if _v5_scheduler_thread is None or not _v5_scheduler_thread.is_alive():
        _v5_scheduler_running = True
        _v5_scheduler_thread = threading.Thread(target=_scheduler_loop, name="v5-scheduler", daemon=True)
        _v5_scheduler_thread.start()
        logger.info("Scheduler started")


def _stop_scheduler():
    """停止轮询调度器线程与队列管理器线程。"""
    global _v5_scheduler_running
    _v5_scheduler_running = False
    if _v5_scheduler_thread:
        _v5_scheduler_thread.join(timeout=2.0)
    # 给队列管理器一个结束信号
    try:
        _v5_request_queue.put_nowait(None)
    except Exception:
        pass
    if _v5_queue_manager_thread:
        _v5_queue_manager_thread.join(timeout=2.0)


def _submit_to_scheduler(worker: TTSWorker) -> bool:
    """将 worker 提交到调度器；线程池已满则返回 False。"""
    if _v5_thread_pool_sem is None:
        return False
    if not _v5_thread_pool_sem.acquire(blocking=False):
        return False
    with _v5_scheduler_lock:
        _v5_active_workers[worker.req_id] = worker
    _start_scheduler()
    return True


def _queue_manager_loop():
    """请求队列管理器：将等待队列中的请求逐一提交到调度器。"""
    logger.info("Queue manager thread started")
    while True:
        try:
            worker = _v5_request_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if worker is None:
            break
        try:
            while not _submit_to_scheduler(worker):
                time.sleep(0.05)
        except Exception as e:
            logger.error("Queue manager error: %s", e)
    logger.info("Queue manager thread stopped")


# ---------------------------------------------------------------------------
# 音色预热（对齐 v4 语义）
# ---------------------------------------------------------------------------

def _run_prime_for_voice(m, cfg: dict, prime_text: str, do_stream: bool, max_tokens: int) -> None:
    """对单个模型与单条音色执行预热。"""
    ref = cfg.get("ref_audio")
    if not ref or not os.path.isfile(ref):
        return
    lang = cfg.get("language", "Auto")
    rt = cfg.get("ref_text", "")
    m._prepare_generation(
        text=prime_text, ref_audio=ref, ref_text=rt, language=lang,
        xvec_only=False, non_streaming_mode=False,
    )
    if do_stream:
        gen = m.generate_voice_clone_streaming(
            text=prime_text, language=lang, ref_audio=ref, ref_text=rt,
            chunk_size=_v5_stream_chunk_size, non_streaming_mode=False,
            max_new_tokens=max_tokens,
        )
        next(gen, None)


def _prime_voice_caches(models: list, voice_list: list, prime_text: str, do_stream: bool, max_tokens: int) -> None:
    """启动阶段批量预热注册表中的音色。"""
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


def _prime_single_voice_sync(models: list, cfg: dict) -> None:
    """同步预热单个音色（全部副本）。"""
    vname, ref = cfg.get("voice_id", "?"), cfg.get("ref_audio")
    if not ref or not os.path.isfile(ref):
        return
    for i, m in enumerate(models):
        try:
            logger.info("Priming voice cache: voice=%r replica=%d/%d", vname, i + 1, len(models))
            _run_prime_for_voice(
                m, cfg, _v5_prime_text, _v5_prime_stream_first_chunk, _v5_prime_stream_max_new_tokens,
            )
        except Exception as exc:
            logger.warning("预热失败: %s", exc)


def _run_orphan_cleanup_all() -> None:
    """对所有模型副本执行语音克隆缓存孤儿清理。"""
    if hot_registry is None or not tts_models:
        return
    reg = hot_registry.get_registry_copy()
    for m in tts_models:
        maybe_cleanup_voice_prompt_cache(
            m, reg, enable=_v5_cleanup_enable, threshold=_v5_cleanup_threshold,
        )


async def prime_voice_v5(voice_id: str) -> None:
    """进程内音色预热入口（路由在写入注册表后异步触发）。"""
    if hot_registry is None:
        return
    hot_registry.get_registry_copy()
    cfg = hot_registry.resolve_active_voice(voice_id)
    if not cfg:
        return
    async with (_v5_prime_lock or asyncio.Lock()):
        await asyncio.get_running_loop().run_in_executor(
            None, _prime_single_voice_sync, tts_models, cfg,
        )
    _run_orphan_cleanup_all()


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech 请求体。"""
    model: str = "tts-1"
    input: str
    voice: str
    response_format: str = "wav"


async def _stream_chunks_v5(worker: TTSWorker, fmt: str) -> AsyncGenerator[bytes, None]:
    """v5 流式合成：从 worker 的响应队列拉取 PCM chunks。"""
    if fmt == "wav":
        yield _wav_header(SAMPLE_RATE)

    loop = asyncio.get_running_loop()
    while True:
        try:
            chunk = await loop.run_in_executor(None, worker.response_queue.get)
        except Exception as e:
            logger.error("Stream error for worker %d: %s", worker.req_id, e)
            worker.client_cancelled = True
            raise
        if chunk is None:
            break
        yield chunk


def _build_cors_middleware_kw(args: argparse.Namespace) -> Dict[str, Any]:
    """§6.7 试用壳跨端口 fetch：STARLETTE CORS 配置。

    优先级：
    - `TTS_V5_CORS_ORIGINS` 若为 `*` 或 `all`（不区分大小写）：允许任意 Origin（**allow_credentials=false**，浏览器限制）。
    - 否则：`localhost:7860` 默认两项 + `TTS_V5_CORS_ORIGINS`（逗号）+ `--cors-origins`（逗号），去重合并。
    - 可选：`TTS_V5_CORS_ORIGIN_REGEX`，与上面的列表**任一**命中即放行（便于 `http://10.x.x.x:*`）。
    """
    raw_env = (os.environ.get("TTS_V5_CORS_ORIGINS") or "").strip()
    rx = (os.environ.get("TTS_V5_CORS_ORIGIN_REGEX") or "").strip() or None
    cli_origins = ((getattr(args, "cors_origins", None) or "").strip())

    defaults = ["http://127.0.0.1:7860", "http://localhost:7860"]

    if raw_env.lower() in ("*", "all"):
        logger.warning(
            "CORS: TTS_V5_CORS_ORIGINS=%r → allow_origins=['*'], allow_credentials=false（仅建议内网联调）",
            raw_env,
        )
        return {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_origin_regex": None,
        }

    merged: List[str] = []
    seen = set()
    for bucket in (
        defaults,
        [x.strip() for x in raw_env.split(",") if x.strip()] if raw_env else [],
        [x.strip() for x in cli_origins.split(",") if x.strip()] if cli_origins else [],
    ):
        for o in bucket:
            if o not in seen:
                seen.add(o)
                merged.append(o)
    if not merged:
        merged = list(defaults)

    kw: Dict[str, Any] = {
        "allow_origins": merged,
        "allow_credentials": True,
    }
    if rx:
        kw["allow_origin_regex"] = rx
    return kw


@app.get("/v1/ui/last-metrics")
async def ui_last_metrics(req_id: int = Query(..., ge=1)):
    """按 `req_id` 取最近一次与 METRICS 日志一致的旁路条目（可能因 LRU 被逐出）。"""
    with _ui_metrics_lock:
        entry = _ui_metrics_order.get(req_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="metrics not found for req_id (expired or unknown)")
    return entry


@app.get("/v1/ui/metrics-tail")
async def ui_metrics_tail(limit: int = Query(20, ge=1, le=200)):
    """按插入顺序返回最近若干条旁路记录。"""
    with _ui_metrics_lock:
        items = list(_ui_metrics_order.values())[-limit:]
    return {"count": len(items), "items": items}


@app.get("/v1/ui/metrics-schema")
async def ui_metrics_schema():
    """前后端展示字段对齐：DEV 子集 vs DEBUG 额外列。"""
    return {
        "metrics_log_mode": _metrics_log_mode,
        "stream": {
            "dev_keys": list(_DEV_METRIC_KEYS),
            "debug_extra_keys": list(_DEBUG_METRIC_EXTRA_KEYS),
        },
        "non_stream_mp3": {
            "typical_keys": ["ttfa_wall_ms", "ttfa_ms", "rtf", "audio_s", "total_ms"],
        },
    }


@app.get("/v1/ui/effective-config")
async def ui_effective_config():
    """试用页只读：全局 `stream_chunk_size` 等。"""
    return {
        "stream_chunk_size": _v5_stream_chunk_size,
        "sample_rate": SAMPLE_RATE,
        "metrics_log_mode": _metrics_log_mode,
    }


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    """OpenAI 兼容的语音合成接口（v5 多路Graph轮询调度版）。"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input empty")
    if hot_registry is None:
        raise HTTPException(status_code=503, detail="Registry not ready")

    voice_cfg = hot_registry.resolve_active_voice(req.voice)
    # logger.info("voice_cfg: %s", voice_cfg)
    if not voice_cfg:
        raise HTTPException(status_code=400, detail=f"Voice '{req.voice}' 不存在或已禁用")

    fmt = req.response_format.lower()
    _CT = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg"}
    if fmt not in _CT:
        raise HTTPException(status_code=400, detail="format not supported")

    req_id = _next_req_id()

    # 非流式 mp3：按 §6.4.1.3 第 5 条
    #  ① 非流式之间 FIFO 串行（asyncio.Lock）；
    #  ② 与流式之间通过读写锁互斥（非流式 = 写者，独占 GPU）；
    #  ③ 不占用流式的 `_v5_thread_pool_sem` 槽位，不与轮询调度时分复用。
    if fmt == "mp3":
        loop = asyncio.get_running_loop()
        t_ttfa0 = time.perf_counter()  # 含 FIFO 等待 + 写锁等待
        fifo_lock = _v5_nonstream_fifo_lock or asyncio.Lock()
        async with fifo_lock:
            # 等待调度器把当前正在推进的流式 step 全部让出（写者优先）
            await loop.run_in_executor(None, _v5_gpu_rwlock.acquire_write)
            try:
                t0 = time.perf_counter()
                audio_arrays, sr = await loop.run_in_executor(
                    None,
                    lambda: _pick_model().generate_voice_clone(
                        text=req.input,
                        language=voice_cfg.get("language", "Auto"),
                        ref_audio=voice_cfg["ref_audio"],
                        ref_text=voice_cfg.get("ref_text", ""),
                    ),
                )
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
                return Response(
                    content=_to_mp3_bytes(audio, sr),
                    media_type=_CT[fmt],
                    headers={"X-Req-Id": str(req_id)},
                )
            finally:
                _v5_gpu_rwlock.release_write()

    # 流式 wav/pcm：创建 worker，提交到调度器或等待队列
    t_ttfa0 = time.perf_counter()
    worker = TTSWorker(
        req_id=req_id,
        voice_cfg=voice_cfg,
        text=req.input,
        t_ttfa0=t_ttfa0,
    )

    if not _submit_to_scheduler(worker):
        logger.info("Request %d queued (thread pool full)", req_id)
        _v5_request_queue.put(worker)

    async def audio_stream():
        try:
            async for raw in _stream_chunks_v5(worker, fmt):
                yield raw
        except (asyncio.CancelledError, GeneratorExit):
            worker.client_cancelled = True
            raise
        except Exception as e:
            logger.error("Stream error for request %d: %s", req_id, e)
            worker.client_cancelled = True
            raise

    return StreamingResponse(
        audio_stream(),
        media_type=_CT[fmt],
        headers={"X-Req-Id": str(req_id)},
    )


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

def _parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="OpenAI-compatible TTS server (v5 多路Graph轮询调度版)")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config_v5.json"))
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--concurrency", type=int, default=3, help="最大并发数（线程池大小），建议3（舒适区）或4（临界点）")
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--no-metrics-log", action="store_true")
    p.add_argument("--disable-cuda-graph", action="store_true", help="禁用 CUDA Graph（不推荐，会大幅降速）")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--warmup-prefill-len", type=int, default=100)
    p.add_argument("--skip-prime-voices", action="store_true")
    p.add_argument("--prime-text", default=None)
    p.add_argument("--prime-stream-first-chunk", action="store_true")
    p.add_argument("--prime-stream-max-new-tokens", type=int, default=None)
    p.add_argument("--stream-chunk-size", type=int, default=None,
                   help="流式chunk大小（codec帧数），未指定则读取 config 的 stream_chunk_size")
    p.add_argument(
        "--cors-origins",
        default="",
        help=(
            "§6.7 试用页浏览器 Origin（逗号分隔），与默认 localhost:7860 及 "
            "环境变量 TTS_V5_CORS_ORIGINS **合并**。例：http://10.0.0.1:10017"
        ),
    )
    return p.parse_args()


def _log_startup_summary(args, inference_log_file: Optional[str]) -> None:
    """将完整启动命令行与关键运行参数写入推理摘要 logger。"""
    inf = logging.getLogger(INFERENCE_LOGGER_NAME)
    log = inf if (inf.handlers or inf.propagate) else logger
    try:
        cmdline = shlex.join(sys.argv)
    except (AttributeError, ValueError, TypeError):
        cmdline = " ".join(sys.argv)
    log.info("STARTUP cmdline=%s", cmdline)
    log.info("STARTUP --config=%s", os.path.abspath(args.config))
    log.info("STARTUP --model=%s", args.model)
    log.info("STARTUP --host=%s --port=%d", args.host, args.port)
    log.info("STARTUP --device=%s", args.device)
    log.info("STARTUP --concurrency=%d --replicas=%d", args.concurrency, args.replicas)
    log.info("STARTUP stream_chunk_size(effective)=%d", _v5_stream_chunk_size)
    log.info("STARTUP prime_text=%r prime_stream_first_chunk=%s max_new_tokens=%d",
             _v5_prime_text, _v5_prime_stream_first_chunk, _v5_prime_stream_max_new_tokens)
    log.info("STARTUP metrics_log_mode=%s", _metrics_log_mode)
    log.info("STARTUP --disable-cuda-graph=%s --skip-warmup=%s --skip-prime-voices=%s",
             args.disable_cuda_graph, args.skip_warmup, args.skip_prime_voices)
    if inference_log_file:
        log.info("STARTUP inference_log_file=%s", inference_log_file)


def main():
    """服务进程入口（同步）。"""
    global tts_model, tts_models, hot_registry, SAMPLE_RATE
    global _v5_thread_pool_sem, _v5_stream_chunk_size, _metrics_enabled
    global _v5_cleanup_enable, _v5_cleanup_threshold, _v5_prime_lock
    global _v5_prime_text, _v5_prime_stream_first_chunk, _v5_prime_stream_max_new_tokens
    global _v5_queue_manager_thread, _v5_nonstream_fifo_lock
    global _metrics_log_mode
    global _v5_stream_use_cuda_graph

    args = _parse_args()
    cfg = load_config_v5(args.config)

    # 路径与通用配置
    reg_path = _resolve_cfg_path(args.config, cfg["voices_registry_path"])
    tone_dir = _resolve_cfg_path(args.config, cfg["tone_wav_file_dir"])
    VoiceRegistryV5(reg_path).ensure_file_exists()
    hot_registry = HotVoiceRegistryV5(reg_path)
    hot_registry.get_registry_copy()

    _v5_cleanup_enable = bool(cfg.get("enable_orphan_cache_cleanup", True))
    _v5_cleanup_threshold = int(cfg.get("orphan_cache_cleanup_threshold", 30))

    _v5_prime_text = args.prime_text if args.prime_text is not None else str(cfg.get("prime_text", ".") or ".")
    _v5_prime_stream_first_chunk = bool(cfg.get("prime_stream_first_chunk", False)) or bool(args.prime_stream_first_chunk)
    _v5_prime_stream_max_new_tokens = int(
        args.prime_stream_max_new_tokens
        if args.prime_stream_max_new_tokens is not None
        else cfg.get("prime_stream_max_new_tokens", 48)
    )

    try:
        if args.stream_chunk_size is not None:
            _v5_stream_chunk_size = max(1, int(args.stream_chunk_size))
        else:
            _v5_stream_chunk_size = max(1, int(cfg.get("stream_chunk_size", 8)))
    except (TypeError, ValueError):
        logger.warning("stream_chunk_size 无效，使用默认值 8")
        _v5_stream_chunk_size = 8

    _v5_thread_pool_sem = threading.Semaphore(max(1, int(args.concurrency)))
    _v5_prime_lock = asyncio.Lock()
    _v5_nonstream_fifo_lock = asyncio.Lock()

    # 推理日志
    infer_log_path: Optional[str] = None
    if "inference_logging" in cfg:
        infer_log_path = _setup_inference_logging(cfg.get("inference_logging") or {}, args.config)

    level_name = str((cfg.get("inference_logging") or {}).get("level", "INFO")).upper()
    _metrics_enabled = not args.no_metrics_log
    _metrics_log_mode = _normalize_metrics_log_mode(level_name)

    # 挂载音色管理路由
    init_router(
        registry=VoiceRegistryV5(reg_path),
        tone_dir=tone_dir,
        allowed_formats=cfg.get("allowed_audio_formats", ["wav", "mp3"]),
        max_file_mb=float(cfg.get("max_audio_file_size_mb", 20)),
        prime_fn=prime_voice_v5,
        model_loaded_fn=lambda: tts_model is not None,
    )
    app.include_router(voice_router)

    _cors_kw = _build_cors_middleware_kw(args)
    logger.info(
        "STARTUP CORS allow_origins=%s allow_credentials=%s origin_regex=%r",
        _cors_kw["allow_origins"],
        _cors_kw["allow_credentials"],
        _cors_kw.get("allow_origin_regex"),
    )
    _cors_mid = dict(
        allow_origins=_cors_kw["allow_origins"],
        allow_credentials=_cors_kw["allow_credentials"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Req-Id"],
    )
    if _cors_kw.get("allow_origin_regex"):
        _cors_mid["allow_origin_regex"] = _cors_kw["allow_origin_regex"]
    app.add_middleware(CORSMiddleware, **_cors_mid)

    _log_startup_summary(args, infer_log_path)

    # 加载模型（含副本）
    from faster_qwen3_tts import FasterQwen3TTS
    logger.info("Loading model: %s (replicas=%d)", args.model, args.replicas)
    tts_models = [
        FasterQwen3TTS.from_pretrained(
            args.model, device=args.device, dtype=torch.bfloat16,
            disable_cuda_graph=args.disable_cuda_graph,
        )
        for _ in range(max(1, int(args.replicas)))
    ]
    for i, m in enumerate(tts_models):
        if not args.skip_warmup:
            logger.info("Warming up replica %d/%d...", i + 1, len(tts_models))
            m._warmup(prefill_len=args.warmup_prefill_len)
    tts_model = tts_models[0]
    SAMPLE_RATE = tts_model.sample_rate

    _conc = max(1, int(args.concurrency))
    _nrep = len(tts_models)
    _graphs = (
        not bool(args.disable_cuda_graph)
        and getattr(tts_model, "predictor_graph", None) is not None
        and getattr(tts_model, "talker_graph", None) is not None
    )
    _v5_stream_use_cuda_graph = bool(_graphs and _nrep >= _conc)
    if _graphs and not _v5_stream_use_cuda_graph:
        logger.warning(
            "流式会话数可能超过 CUDA Graph 安全副本：replicas=%d < concurrency=%d。"
            "已自动改用 Parity（非 Graph）流式，避免出现异常时长/chunk、『卡碟』杂音；"
            "若要高性能 Graph 并行，请加 --replicas 且使其 ≥ --concurrency。",
            _nrep,
            _conc,
        )
    logger.info(
        "STARTUP stream_cuda_graph=%s parity_stream_fallback=%s replicas=%d concurrency=%d",
        _v5_stream_use_cuda_graph,
        bool(_graphs and not _v5_stream_use_cuda_graph),
        _nrep,
        _conc,
    )

    # 批量预热注册表音色
    if not args.skip_prime_voices:
        _prime_voice_caches(
            tts_models,
            hot_registry.list_active_voice_cfgs(),
            _v5_prime_text,
            _v5_prime_stream_first_chunk,
            _v5_prime_stream_max_new_tokens,
        )

    # 启动队列管理线程
    _v5_queue_manager_thread = threading.Thread(
        target=_queue_manager_loop, name="v5-queue-manager", daemon=True,
    )
    _v5_queue_manager_thread.start()
    _start_scheduler()

    logger.info("=" * 60)
    logger.info("Server v5 ready: host=%s port=%d concurrency=%d chunk_size=%d",
                args.host, args.port, args.concurrency, _v5_stream_chunk_size)
    logger.info("Model: %s (CUDA Graph enabled: %s)", args.model, not args.disable_cuda_graph)
    logger.info("=" * 60)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        _stop_scheduler()


if __name__ == "__main__":
    main()
