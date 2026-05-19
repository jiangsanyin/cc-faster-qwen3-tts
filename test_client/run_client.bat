@echo off
REM 流式TTS测试客户端启动脚本 (Windows)

echo ========================================
echo 流式TTS测试客户端
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 安装依赖
echo 正在检查/安装依赖...
pip install PyQt5 pygame requests numpy -q

echo.
echo 启动客户端...
echo.

python tts_streaming_client.py

pause
