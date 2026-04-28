FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl vim net-tools && apt-get install -y --no-install-recommends sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先把宿主机当前目录的文件拷贝进去，用于安装依赖
COPY . /app

# 使用 -e (editable) 模式安装。这非常关键！
# 这样安装后，Python 会直接运行 /app 目录下的源码，配合后续的目录挂载，实现代码热更新
RUN pip install --no-cache-dir -e ".[demo]" -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 7860 8000 8001

CMD ["python", "demo/server.py"]