#!/bin/bash
# 流式TTS测试客户端启动脚本 (Linux/Mac)

echo "========================================"
echo "流式TTS测试客户端"
echo "========================================"
echo

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 安装依赖
echo "正在检查/安装依赖..."
pip3 install PyQt5 pygame requests numpy -q

echo
echo "启动客户端..."
echo

python3 tts_streaming_client.py
