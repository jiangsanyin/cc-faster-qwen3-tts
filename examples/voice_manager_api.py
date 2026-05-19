#!/usr/bin/env python3
"""
音色管理 API —— 供平台组调用，负责音色的增删改查与参考音频上传。

启动方式：
    python examples/voice_manager_api.py --config config.json --port 8001

接口列表：
    POST   /voices              新增音色（multipart 上传音频 + 元数据）
    GET    /voices              列出所有 active 音色
    GET    /voices/{voice_id}   查询单个音色
    PUT    /voices/{voice_id}   更新音色（可选上传新音频）
    DELETE /voices/{voice_id}   删除音色（默认软删除）
    POST   /voices/{voice_id}/restore  恢复软删除的音色
    GET    /health              健康检查
"""

import argparse
import json
import logging
import os
import shutil
import sys
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from voice_db import VoiceDB, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS Voice Manager API")
db: Optional[VoiceDB] = None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _validate_audio_file(file: UploadFile) -> str:
    """校验上传文件的格式与大小，返回后缀名（不含点）。"""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="未上传音频文件")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in db.allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式 '{ext}'，允许：{db.allowed_formats}",
        )
    return ext


def _save_audio(file: UploadFile, voice_id: str, ext: str) -> str:
    """将上传的音频保存到磁盘，返回绝对路径。"""
    os.makedirs(db.tone_wav_dir, exist_ok=True)
    filename = f"{voice_id}.{ext}"
    dest = os.path.join(db.tone_wav_dir, filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file_mb = os.path.getsize(dest) / (1024 * 1024)
    if file_mb > db.max_file_mb:
        os.remove(dest)
        raise HTTPException(
            status_code=400,
            detail=f"文件大小 {file_mb:.1f}MB 超出上限 {db.max_file_mb}MB",
        )
    return dest


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    status = db.check_connections()
    return {"status": "ok" if all(status.values()) else "degraded", **status}


@app.get("/voices")
async def list_voices(status: str = Query("active", description="筛选状态")):
    """列出音色。"""
    voices = db.list_voices(status=status)
    return {"count": len(voices), "voices": voices}


@app.get("/voices/{voice_id}")
async def get_voice(voice_id: str):
    """查询单个音色。"""
    cfg = db.get_voice(voice_id)
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
    """新增音色：上传参考音频 + 元数据。"""
    ext = _validate_audio_file(audio_file)

    if not voice_id:
        # 完整 UUID 字符串（带连字符）；勿用裸 uuid.uuid4()，否则 JSON/部分驱动序列化易出问题
        voice_id = str(uuid.uuid4())

    existing = db.get_voice(voice_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"音色 '{voice_id}' 已存在")

    ref_audio_path = _save_audio(audio_file, voice_id, ext)

    try:
        cfg = db.add_voice(
            voice_id=voice_id,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            language=language,
        )
    except Exception as exc:
        if os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
        logger.error("新增音色失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"新增失败: {exc}")

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
    """更新音色：可选更换音频、修改文本或语言。"""
    existing = db.get_voice(voice_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")

    update_kwargs = {}

    if audio_file and audio_file.filename:
        ext = _validate_audio_file(audio_file)
        new_path = _save_audio(audio_file, voice_id, ext)
        update_kwargs["ref_audio_path"] = new_path

    if ref_text is not None:
        update_kwargs["ref_text"] = ref_text
    if language is not None:
        update_kwargs["language"] = language

    if not update_kwargs:
        return {"success": True, "voice_id": voice_id, "voice": existing, "message": "无变更"}

    cfg = db.update_voice(voice_id, **update_kwargs)
    if cfg is None:
        raise HTTPException(status_code=500, detail="更新失败")

    logger.info("更新音色: voice_id=%s fields=%s", voice_id, list(update_kwargs.keys()))
    return {"success": True, "voice_id": voice_id, "voice": cfg}


@app.delete("/voices/{voice_id}")
async def delete_voice(
    voice_id: str,
    hard: bool = Query(False, description="是否硬删除（物理删除行+文件）"),
):
    """删除音色。默认软删除（status='disabled'），hard=true 物理删除。"""
    existing = db.get_voice(voice_id)
    if not existing and not hard:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在或已禁用")

    ok = db.delete_voice(voice_id, hard=hard)

    if hard and existing:
        audio_path = existing.get("ref_audio", "")
        if audio_path and os.path.isfile(audio_path):
            try:
                os.remove(audio_path)
                logger.info("已删除音频文件: %s", audio_path)
            except Exception as exc:
                logger.warning("删除音频文件失败: %s", exc)

    logger.info("删除音色: voice_id=%s hard=%s ok=%s", voice_id, hard, ok)
    return {"success": ok, "voice_id": voice_id, "hard": hard}


@app.post("/voices/{voice_id}/restore")
async def restore_voice(voice_id: str):
    """将软删除的音色恢复为 active（PUT 无法恢复：get_voice 只查 active）。"""
    cfg = db.restore_voice(voice_id)
    if not cfg:
        raise HTTPException(
            status_code=404,
            detail=f"音色 '{voice_id}' 不存在、已是 active，或从未被软删除",
        )
    logger.info("恢复音色: voice_id=%s", voice_id)
    return {"success": True, "voice_id": voice_id, "voice": cfg}


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="TTS Voice Manager API")
    p.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.json"),
        help="配置文件路径（默认: ../config.json）",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    return p.parse_args()


def main():
    global db
    args = _parse_args()

    config = load_config(args.config)
    db = VoiceDB(config)
    db.ensure_table()

    status = db.check_connections()
    logger.info("连接检测: MySQL=%s Redis=%s", status["mysql"], status["redis"])
    if not status["mysql"]:
        logger.error("MySQL 不可用，退出")
        sys.exit(1)

    logger.info("Voice Manager API listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
