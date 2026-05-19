"""
音色管理路由模块 (v5)：将管理逻辑封装为 APIRouter，供 v5 主服务引用。

与 v4 功能一致，为 v5 服务提供独立的音色管理 API，避免版本间依赖。
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import shutil
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.responses import JSONResponse

from voice_registry_v5 import (
    VoiceRegistryV5,
    audio_path_for_version,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice Management"])

# 运行时由主服务注入
_registry: Optional[VoiceRegistryV5] = None
_tone_dir: str = ""
_allowed_formats: List[str] = []
_max_file_mb: float = 20.0
_prime_fn: Optional[Callable[[str], Any]] = None
_model_loaded_fn: Optional[Callable[[], bool]] = None


def init_router(
    registry: VoiceRegistryV5,
    tone_dir: str,
    allowed_formats: List[str],
    max_file_mb: float,
    prime_fn: Callable[[str], Any],
    model_loaded_fn: Optional[Callable[[], bool]] = None,
):
    """初始化路由所需的全局状态。"""
    global _registry, _tone_dir, _allowed_formats, _max_file_mb, _prime_fn, _model_loaded_fn
    _registry = registry
    _tone_dir = tone_dir
    _allowed_formats = allowed_formats
    _max_file_mb = max_file_mb
    _prime_fn = prime_fn
    _model_loaded_fn = model_loaded_fn


def _validate_audio_file(file: UploadFile) -> str:
    """校验上传音频。"""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="未上传音频文件")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in _allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式 '{ext}'，允许：{_allowed_formats}",
        )
    return ext


async def _save_upload_async(file: UploadFile, dest: str) -> None:
    """异步保存上传文件并校验大小。"""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    loop = asyncio.get_event_loop()
    
    def _sync_save():
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_mb = os.path.getsize(dest) / (1024 * 1024)
        if file_mb > _max_file_mb:
            os.remove(dest)
            raise ValueError(f"文件大小 {file_mb:.1f}MB 超出上限 {_max_file_mb}MB")

    try:
        await loop.run_in_executor(None, _sync_save)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _remove_voice_audio_files_async(voice_id: str) -> None:
    """异步清理音色文件。"""
    if not _tone_dir or not os.path.isdir(_tone_dir):
        return
    loop = asyncio.get_event_loop()

    def _sync_remove():
        patterns = [
            os.path.join(_tone_dir, f"{voice_id}.*"),
            os.path.join(_tone_dir, f"{voice_id}_v*.*"),
        ]
        seen = set()
        for pat in patterns:
            for p in glob.glob(pat):
                ap = os.path.abspath(p)
                if ap in seen: continue
                seen.add(ap)
                try:
                    if os.path.isfile(ap):
                        os.remove(ap)
                        logger.info("已清理音频文件: %s", ap)
                except OSError as exc:
                    logger.warning("清理音频文件失败: %s %s", ap, exc)
    
    await loop.run_in_executor(None, _sync_remove)


@router.get("/health")
async def health():
    """健康检查：注册表可读 + 音频目录可写。"""
    registry_readable = False
    tone_dir_writable = False
    detail: Dict[str, Any] = {}
    try:
        assert _registry is not None
        _registry.read_all()
        registry_readable = True
    except Exception as exc:
        detail["registry_error"] = str(exc)
    try:
        os.makedirs(_tone_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="health_", dir=_tone_dir)
        os.close(fd)
        os.remove(tmp)
        tone_dir_writable = True
    except Exception as exc:
        detail["tone_dir_error"] = str(exc)
    ok = registry_readable and tone_dir_writable
    body: Dict[str, Any] = {
        "status": "ok" if ok else "degraded",
        "registry_readable": registry_readable,
        "tone_dir_writable": tone_dir_writable,
        **detail,
    }
    if _model_loaded_fn is not None:
        body["model_loaded"] = bool(_model_loaded_fn())
    return body


@router.get("/voices")
async def list_voices(status: str = Query("active")):
    """列出音色。"""
    assert _registry is not None
    return {"count": len(voices := _registry.list_voices(status=status)), "voices": voices}


@router.get("/voices/{voice_id}")
async def get_voice(voice_id: str):
    """查询单个 active 音色。"""
    assert _registry is not None
    if not (cfg := _registry.get_voice_active(voice_id)):
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")
    return cfg


@router.post("/voices")
async def add_voice(
    audio_file: UploadFile = File(...),
    voice_id: Optional[str] = Form(None),
    ref_text: str = Form(...),
    language: str = Form("Auto"),
):
    """新增音色。"""
    assert _registry is not None
    ext = _validate_audio_file(audio_file)
    vid = voice_id or str(uuid.uuid4())

    if _registry.get_entry_raw(vid) is not None:
        raise HTTPException(status_code=409, detail=f"音色 '{vid}' 已存在")

    dest = audio_path_for_version(_tone_dir, vid, ext, 1)
    await _save_upload_async(audio_file, dest)
    ref_audio_path = os.path.abspath(dest)

    def _fn(data: Dict[str, Any]):
        if vid in data: raise ValueError("exists")
        data[vid] = {
            "ref_audio": ref_audio_path, "ref_text": ref_text, "language": language,
            "status": "active", "version": 1, "updated_at": utc_now_iso(),
        }

    try:
        _registry.mutate(_fn)   # 在独占锁保护下修改注册表
    except Exception as exc:
        if os.path.isfile(ref_audio_path): os.remove(ref_audio_path)
        raise HTTPException(status_code=500, detail=str(exc))

    if _prime_fn:
        asyncio.create_task(_prime_fn(vid))

    return JSONResponse(status_code=201, content={"success": True, "voice_id": vid, "voice": _registry.get_voice_active(vid)})


@router.put("/voices/{voice_id}")
async def update_voice(
    voice_id: str,
    audio_file: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """更新音色。"""
    assert _registry is not None
    if not (existing := _registry.get_voice_active(voice_id)):
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")

    raw = _registry.get_entry_raw(voice_id) or {}
    new_path, old_path_to_remove = None, None
    update_kwargs: Dict[str, Any] = {}

    audio_changing = bool(audio_file and audio_file.filename)
    text_changing = bool(ref_text is not None and ref_text != existing.get("ref_text"))

    if audio_changing or text_changing:
        new_version = int(raw.get("version", 1)) + 1
        update_kwargs["version"] = new_version
        if audio_changing:
            old_path_to_remove = existing.get("ref_audio")
            ext = _validate_audio_file(audio_file)
            dest = audio_path_for_version(_tone_dir, voice_id, ext, new_version)
            await _save_upload_async(audio_file, dest)
            new_path = os.path.abspath(dest)
            update_kwargs["ref_audio"] = new_path
        if text_changing:
            update_kwargs["ref_text"] = ref_text

    if language is not None and language != existing.get("language"):
        update_kwargs["language"] = language

    if not update_kwargs:
        return {"success": True, "voice_id": voice_id, "voice": existing, "message": "无变更"}

    def _fn(data: Dict[str, Any]):
        e = data.get(voice_id)
        if not e or e.get("status", "active") == "disabled": raise KeyError("missing")
        for k, v in update_kwargs.items(): e[k] = v
        e["updated_at"] = utc_now_iso()

    try:
        _registry.mutate(_fn)
    except Exception as exc:
        if new_path and os.path.isfile(new_path): os.remove(new_path)
        raise HTTPException(status_code=500, detail=str(exc))

    if old_path_to_remove and new_path and os.path.isfile(old_path_to_remove):
        if os.path.abspath(old_path_to_remove) != os.path.abspath(new_path):
            try: os.remove(old_path_to_remove)
            except: pass

    if "version" in update_kwargs and _prime_fn:
        asyncio.create_task(_prime_fn(voice_id))

    return {"success": True, "voice_id": voice_id, "voice": _registry.get_voice_active(voice_id)}


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str, hard: bool = Query(False)):
    """删除音色。"""
    assert _registry is not None
    if not hard:
        if not _registry.get_voice_active(voice_id):
            raise HTTPException(status_code=404, detail="不存在或已禁用")
        _registry.mutate(lambda d: d[voice_id].update({"status": "disabled", "updated_at": utc_now_iso()}))
        return {"success": True, "voice_id": voice_id, "hard": False}

    if not _registry.get_entry_raw(voice_id):
        raise HTTPException(status_code=404, detail="不存在")
    _registry.mutate(lambda d: d.pop(voice_id, None))
    await _remove_voice_audio_files_async(voice_id)
    return {"success": True, "voice_id": voice_id, "hard": True}


@router.post("/voices/{voice_id}/restore")
async def restore_voice(voice_id: str):
    """恢复软删除。"""
    assert _registry is not None
    def _fn(d):
        if (e := d.get(voice_id)) and e.get("status") == "disabled":
            e.update({"status": "active", "updated_at": utc_now_iso()})
        else: raise ValueError("not_disabled")
    try: _registry.mutate(_fn)
    except: raise HTTPException(status_code=404, detail="无法恢复")
    return {"success": True, "voice_id": voice_id, "voice": _registry.get_voice_active(voice_id)}
