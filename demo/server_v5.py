#!/usr/bin/env python3
"""
§6.7.1：v5 OpenAI 合成服务的独立试用壳（静态页 + /config）。

不加载 TTS 模型；浏览器通过 CORS 访问下游 `openai_server_v5.py`。

用法：
  cd faster-qwen3-tts
  python demo/server_v5.py
  python demo/server_v5.py --port 7860 --api-base http://127.0.0.1:8000

环境变量：
  TTS_V5_API   下游 OpenAI v5 基址（与 --api-base 二选一，命令行优先）
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.responses import Response

BASE_DIR = Path(__file__).resolve().parent


def _build_app(api_base: str) -> FastAPI:
    api_base = api_base.rstrip("/")
    app = FastAPI(title="Qwen3-TTS v5 trial shell", version="0.1")

    @app.middleware("http")
    async def quiet_optional_assets(request: Request, call_next):
        """IDE / 扩展 / DevTools 常见探测（非页面引用），返回 204 以免刷屏 404。"""
        if request.method == "GET":
            path = request.url.path
            if path == "/favicon.ico":
                return Response(status_code=204)
            if path.endswith(".css.map") or path.endswith(".js.map"):
                return Response(status_code=204)
        return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "role": "v5_trial_shell"}

    @app.get("/config")
    def config() -> dict:
        return {"api_base": api_base}

    @app.get("/")
    def index() -> FileResponse:
        p = BASE_DIR / "index_v5.html"
        if not p.is_file():
            return JSONResponse(
                status_code=500,
                content={"detail": f"missing {p.name} next to server_v5.py"},
            )
        return FileResponse(p, media_type="text/html; charset=utf-8")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="v5 Clone trial static server (§6.7.1)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--api-base",
        default=None,
        help="OpenAI v5 HTTP 基址，默认 TTS_V5_API 或 http://127.0.0.1:8000",
    )
    args = p.parse_args()
    api = (args.api_base or os.environ.get("TTS_V5_API") or "http://127.0.0.1:8000").rstrip(
        "/"
    )
    app = _build_app(api)
    print(f"[server_v5] api_base={api}  listen http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
