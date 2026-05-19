#!/usr/bin/env python3
"""
音色管理 API（v3）：JSON 注册表 + 磁盘参考音频，无 MySQL/Redis。

启动方式：
    python examples/voice_manager_api_v3.py --config ../config_v3.json --port 8001

接口与 v2 对齐：
    POST   /voices              新增音色（multipart 上传音频 + 元数据）
    GET    /voices              列出音色（status=active|disabled）
    GET    /voices/{voice_id}   查询单个 active 音色
    PUT    /voices/{voice_id}   更新（可选新音频；换音频时递增 version 并写入新路径）
    DELETE /voices/{voice_id}   软删 / 硬删
    POST   /voices/{voice_id}/restore
    GET    /health              注册表可读 + 音频目录可写
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from voice_registry_v3 import (
    VoiceRegistryV3,
    audio_path_for_version,
    load_config_v3,
    utc_now_iso,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS Voice Manager API (v3)")

_registry: Optional[VoiceRegistryV3] = None
_tone_dir: str = ""
_allowed_formats: List[str] = []
_max_file_mb: float = 50.0

# 内部预热配置
_tts_prime_url: Optional[str] = None
_tts_prime_token: Optional[str] = None


async def _notify_tts_prime(voice_id: str) -> None:
    """
    异步通知 TTS 服务预热指定音色。
    
    通过 HTTP POST 请求触发 TTS 侧将参考音频编码并存入内存缓存，
    以降低该音色首次真实合成请求的延迟。
    
    Args:
        voice_id: 需要预热的音色唯一标识。
    """
    if not _tts_prime_url:
        return
    try:
        headers = {}
        if _tts_prime_token:
            headers["x-internal-token"] = _tts_prime_token
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _tts_prime_url,
                params={"voice_id": voice_id},
                headers=headers,
            )
            if resp.status_code == 200:
                logger.info("成功触发 TTS 预热: voice_id=%s", voice_id)
            else:
                logger.warning("触发 TTS 预热失败: code=%d resp=%s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("通知 TTS 预热异常（不影响注册表）: %s", exc)


def _resolve_cfg_path(config_path: str, p: str) -> str:
    """
    解析配置文件中的路径。
    
    如果路径是相对路径，则相对于配置文件所在的目录进行解析。
    
    Args:
        config_path: 配置文件本身的绝对路径。
        p: 待解析的路径字符串。
        
    Returns:
        解析后的规范化路径字符串。
    """
    if os.path.isabs(p):
        return os.path.normpath(p)
    base = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(base, p))


def _validate_audio_file(file: UploadFile) -> str:
    """
    校验上传的音频文件格式。
    
    检查文件是否存在以及扩展名是否在允许的白名单内。
    
    Args:
        file: FastAPI 上传的文件对象。
        
    Returns:
        文件扩展名（小写，不含点）。
        
    Raises:
        HTTPException: 400 错误，如果未上传文件或格式不支持。
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="未上传音频文件")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in _allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式 '{ext}'，允许：{_allowed_formats}",
        )
    return ext


def _check_size_and_maybe_remove(path: str) -> None:
    """
    检查文件大小，如果超出限制则删除该文件。
    
    Args:
        path: 待检查文件的磁盘路径。
        
    Raises:
        HTTPException: 400 错误，如果文件大小超出配置的上限。
    """
    file_mb = os.path.getsize(path) / (1024 * 1024)
    if file_mb > _max_file_mb:
        os.remove(path)
        raise HTTPException(
            status_code=400,
            detail=f"文件大小 {file_mb:.1f}MB 超出上限 {_max_file_mb}MB",
        )


def _save_upload_to_path(file: UploadFile, dest: str) -> None:
    """
    将上传的文件保存到指定的目标路径，并执行大小校验。
    
    Args:
        file: FastAPI 上传的文件对象。
        dest: 目标存储路径。
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _check_size_and_maybe_remove(dest)


def _remove_voice_audio_files(voice_id: str) -> None:
    """
    清理指定音色在磁盘上的所有相关音频文件。
    
    会匹配该音色 ID 下的所有版本（如 {id}.wav, {id}_v2.wav 等）。
    用于硬删除场景。
    
    Args:
        voice_id: 音色唯一标识。
    """
    if not _tone_dir or not os.path.isdir(_tone_dir):
        return
    patterns = [
        os.path.join(_tone_dir, f"{voice_id}.*"),
        os.path.join(_tone_dir, f"{voice_id}_v*.*"),
    ]
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            ap = os.path.abspath(p)
            if ap in seen:
                continue
            seen.add(ap)
            try:
                if os.path.isfile(ap):
                    os.remove(ap)
                    logger.info("已删除音频文件: %s", ap)
            except OSError as exc:
                logger.warning("删除音频文件失败: %s %s", ap, exc)


@app.get("/health")
async def health():
    """
    健康检查接口。
    
    验证注册表文件是否可读，以及音频存储目录是否具有写权限。
    
    Returns:
        包含各组件状态的 JSON 对象。
    """
    registry_readable = False
    tone_dir_writable = False
    detail: Dict[str, Any] = {}
    try:
        assert _registry is not None
        _registry.read_all()
        registry_readable = True
    except Exception as exc:
        detail["registry_error"] = str(exc)
        logger.warning("注册表健康检查失败: %s", exc)
    try:
        os.makedirs(_tone_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="health_", dir=_tone_dir)
        os.close(fd)
        os.remove(tmp)
        tone_dir_writable = True
    except Exception as exc:
        detail["tone_dir_error"] = str(exc)
        logger.warning("音频目录写检测失败: %s", exc)
    ok = registry_readable and tone_dir_writable
    return {"status": "ok" if ok else "degraded", "registry_readable": registry_readable, "tone_dir_writable": tone_dir_writable, **detail}


@app.get("/voices")
async def list_voices(status: str = Query("active", description="筛选状态 active|disabled")):
    """
    列出音色列表。
    
    Args:
        status: 筛选音色状态，支持 'active'（默认）或 'disabled'。
        
    Returns:
        包含音色数量和音色详情列表的 JSON 对象。
    """
    assert _registry is not None
    if status not in ("active", "disabled"):
        raise HTTPException(status_code=400, detail="status 仅支持 active 或 disabled")
    voices = _registry.list_voices(status=status)
    return {"count": len(voices), "voices": voices}


@app.get("/voices/{voice_id}")
async def get_voice(voice_id: str):
    """
    查询单个 active 音色的详细配置。
    
    Args:
        voice_id: 音色唯一标识。
        
    Returns:
        音色配置对象。
        
    Raises:
        HTTPException: 404 错误，如果音色不存在或已被禁用。
    """
    assert _registry is not None
    cfg = _registry.get_voice_active(voice_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")
    return cfg


@app.post("/voices")
async def add_voice(
    audio_file: UploadFile = File(..., description="参考音频文件"),
    voice_id: Optional[str] = Form(None, description="自定义音色ID（留空自动生成）"),
    ref_text: str = Form(..., description="参考音频对应的文本"),
    language: str = Form("Auto", description="语言：Chinese / English / Auto"),
):
    """
    新增音色。
    
    上传参考音频并保存元数据到注册表。成功后会触发 TTS 侧的异步预热。
    
    Args:
        audio_file: 参考音频文件（multipart）。
        voice_id: 可选的自定义 ID，不传则生成 UUID。
        ref_text: 音频对应的文本内容。
        language: 语言标识。
        
    Returns:
        包含成功状态和新增音色详情的 JSON 响应。
    """
    assert _registry is not None
    ext = _validate_audio_file(audio_file)
    if not voice_id:
        voice_id = str(uuid.uuid4())

    if _registry.get_entry_raw(voice_id) is not None:
        raise HTTPException(status_code=409, detail=f"音色 '{voice_id}' 已存在")

    version = 1
    dest = audio_path_for_version(_tone_dir, voice_id, ext, version)
    try:
        _save_upload_to_path(audio_file, dest)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("保存音频失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"保存音频失败: {exc}")

    ref_audio_path = os.path.abspath(dest)

    def _fn(data: Dict[str, Any]) -> None:
        if voice_id in data:
            raise ValueError("exists")
        data[voice_id] = {
            "ref_audio": ref_audio_path,
            "ref_text": ref_text,
            "language": language,
            "status": "active",
            "version": version,
            "updated_at": utc_now_iso(),
        }

    try:
        _registry.mutate(_fn)
    except ValueError:
        if os.path.isfile(ref_audio_path):
            os.remove(ref_audio_path)
        raise HTTPException(status_code=409, detail=f"音色 '{voice_id}' 已存在")
    except Exception as exc:
        if os.path.isfile(ref_audio_path):
            os.remove(ref_audio_path)
        logger.error("写入注册表失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"写入注册表失败: {exc}")

    # 成功写入注册表后，异步触发 TTS 预热
    asyncio.create_task(_notify_tts_prime(voice_id))

    cfg = _registry.get_voice_active(voice_id)
    logger.info("新增音色: voice_id=%s path=%s", voice_id, ref_audio_path)
    return JSONResponse(
        status_code=201,
        content={"success": True, "voice_id": voice_id, "voice": cfg},
    )


@app.put("/voices/{voice_id}")
async def update_voice(
    voice_id: str,
    audio_file: Optional[UploadFile] = File(None, description="新的参考音频（可选）"),
    ref_text: Optional[str] = Form(None, description="新的参考文本（可选）"),
    language: Optional[str] = Form(None, description="新的语言（可选）"),
):
    """
    更新现有音色。
    
    支持更新音频文件、文本或语言。如果更新了音频或文本，版本号会递增。
    当更新音频时，会存入对应新版本号的新路径。
    仅当音频或文本变更时，才会触发 TTS 异步预热。
    
    Args:
        voice_id: 待更新的音色 ID。
        audio_file: 可选的新音频文件。
        ref_text: 可选的新文本。
        language: 可选的新语言。
        
    Returns:
        更新后的音色详情。
    """
    assert _registry is not None
    existing = _registry.get_voice_active(voice_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")

    raw = _registry.get_entry_raw(voice_id) or {}
    new_path: Optional[str] = None
    old_path_to_remove: Optional[str] = None
    update_kwargs: Dict[str, Any] = {}

    # 判断核心字段是否发生变更
    audio_changing = bool(audio_file and audio_file.filename)
    # 仅当传入了新文本且与旧文本不同时，视为文本变更
    text_changing = bool(ref_text is not None and ref_text != existing.get("ref_text"))

    # 如果音频或文本任一发生变更，则递增版本号
    if audio_changing or text_changing:
        new_version = int(raw.get("version", 1)) + 1
        update_kwargs["version"] = new_version

        if audio_changing:
            # 记录旧路径以便成功后清理
            old_path_to_remove = existing.get("ref_audio")

            # 处理新音频上传：使用新版本号生成新路径
            ext = _validate_audio_file(audio_file)
            dest = audio_path_for_version(_tone_dir, voice_id, ext, new_version)
            try:
                _save_upload_to_path(audio_file, dest)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"保存音频失败: {exc}")
            new_path = os.path.abspath(dest)
            update_kwargs["ref_audio"] = new_path
        
        if text_changing:
            update_kwargs["ref_text"] = ref_text

    if language is not None and language != existing.get("language"):
        update_kwargs["language"] = language

    if not update_kwargs:
        return {"success": True, "voice_id": voice_id, "voice": existing, "message": "无变更"}

    def _fn(data: Dict[str, Any]) -> None:
        e = data.get(voice_id)
        if not e or e.get("status", "active") == "disabled":
            raise KeyError("missing")
        if "ref_audio" in update_kwargs:
            e["ref_audio"] = update_kwargs["ref_audio"]
        if "version" in update_kwargs:
            e["version"] = int(update_kwargs["version"])
        if "ref_text" in update_kwargs:
            e["ref_text"] = update_kwargs["ref_text"]
        if "language" in update_kwargs:
            e["language"] = update_kwargs["language"]
        e["updated_at"] = utc_now_iso()

    try:
        _registry.mutate(_fn)
    except KeyError:
        if new_path and os.path.isfile(new_path):
            os.remove(new_path)
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")

    # 注册表更新成功后，清理旧音频文件
    if old_path_to_remove and new_path and os.path.isfile(old_path_to_remove):
        # 只有当新旧路径都存在且不一致时，才安全删除旧文件
        if os.path.abspath(old_path_to_remove) != os.path.abspath(new_path):
            try:
                os.remove(old_path_to_remove)
                logger.info("更新音色成功，已清理旧音频: %s", old_path_to_remove)
            except Exception as exc:
                logger.warning("清理旧音频失败: %s %s", old_path_to_remove, exc)

    # 只有更新了 audio_file 或 ref_text 时，才触发 TTS 预热
    if "version" in update_kwargs:
        asyncio.create_task(_notify_tts_prime(voice_id))

    cfg = _registry.get_voice_active(voice_id)
    logger.info("更新音色: voice_id=%s fields=%s", voice_id, list(update_kwargs.keys()))
    return {"success": True, "voice_id": voice_id, "voice": cfg}


@app.delete("/voices/{voice_id}")
async def delete_voice(
    voice_id: str,
    hard: bool = Query(False, description="是否硬删除（移除注册表项并尽量清理音频文件）"),
):
    """
    删除音色。
    
    默认执行软删除（标记状态为 'disabled'）。硬删除会移除注册表项并尝试删除磁盘文件。
    
    Args:
        voice_id: 音色唯一标识。
        hard: 是否执行物理硬删除。
        
    Returns:
        包含操作结果的 JSON 对象。
    """
    assert _registry is not None

    if not hard:
        existing = _registry.get_voice_active(voice_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")

        def _fn_soft(data: Dict[str, Any]) -> None:
            e = data.get(voice_id)
            if not e:
                raise KeyError("missing")
            if e.get("status", "active") == "disabled":
                raise ValueError("already_disabled")
            e["status"] = "disabled"
            e["updated_at"] = utc_now_iso()

        try:
            _registry.mutate(_fn_soft)
        except ValueError:
            return {"success": False, "voice_id": voice_id, "hard": False}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")

        logger.info("软删除音色: voice_id=%s", voice_id)
        return {"success": True, "voice_id": voice_id, "hard": False}

    snap = _registry.read_all().get(voice_id)
    if not snap:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在")

    def _fn_hard(data: Dict[str, Any]) -> None:
        if voice_id not in data:
            raise KeyError("missing")
        del data[voice_id]

    try:
        _registry.mutate(_fn_hard)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在")

    _remove_voice_audio_files(voice_id)
    logger.info("硬删除音色: voice_id=%s", voice_id)
    return {"success": True, "voice_id": voice_id, "hard": True}


@app.post("/voices/{voice_id}/restore")
async def restore_voice(voice_id: str):
    """
    恢复被软删除的音色。
    
    将音色状态从 'disabled' 改回 'active'。
    
    Args:
        voice_id: 音色唯一标识。
        
    Returns:
        恢复后的音色详情。
    """
    assert _registry is not None

    def _fn(data: Dict[str, Any]) -> None:
        e = data.get(voice_id)
        if not e:
            raise KeyError("missing")
        if e.get("status", "active") != "disabled":
            raise ValueError("not_disabled")
        e["status"] = "active"
        e["updated_at"] = utc_now_iso()

    try:
        _registry.mutate(_fn)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"音色 '{voice_id}' 不存在",
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"音色 '{voice_id}' 不存在、已是 active，或从未被软删除",
        )

    cfg = _registry.get_voice_active(voice_id)
    logger.info("恢复音色: voice_id=%s", voice_id)
    return {"success": True, "voice_id": voice_id, "voice": cfg}


def _parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="TTS Voice Manager API (v3, JSON registry)")
    p.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config_v3.json"),
        help="v3 配置文件路径（默认: ../config_v3.json）",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    return p.parse_args()


def main():
    """主入口函数，负责初始化配置、注册表并启动 FastAPI 服务。"""
    global _registry, _tone_dir, _allowed_formats, _max_file_mb
    global _tts_prime_url, _tts_prime_token
    args = _parse_args()

    cfg = load_config_v3(args.config)
    reg_path = _resolve_cfg_path(args.config, cfg["voices_registry_path"])
    tone = _resolve_cfg_path(args.config, cfg["tone_wav_file_dir"])
    _tone_dir = tone
    _allowed_formats = [x.lower().lstrip(".") for x in cfg.get("allowed_audio_formats", ["wav", "mp3"])]
    _max_file_mb = float(cfg.get("max_audio_file_size_mb", 50))

    _tts_prime_url = cfg.get("tts_internal_prime_url")
    _tts_prime_token = cfg.get("tts_internal_prime_token")

    _registry = VoiceRegistryV3(reg_path)
    _registry.ensure_file_exists()
    os.makedirs(_tone_dir, exist_ok=True)

    logger.info("voices_registry_path=%s", reg_path)
    logger.info("tone_wav_file_dir=%s", _tone_dir)
    logger.info("Voice Manager API (v3) listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
