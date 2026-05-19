# Locust 压测说明（`openai_server_v4` 合并服务）

本目录提供针对 **`examples/openai_server_v4.py`** 的 HTTP 并发压测脚本，可在任意安装 Locust 的机器（如 Windows 本机）上运行，向远端或本机已映射端口的 TTS 服务发压。另含 **`remote_benchmark_orchestrator.py`**：在压测机上按《faster-qwen3-tts修改后4090上并发测试.md》自动串联 **远端容器内启停服务** 与 **本机 Locust**，详见 **§2.8**。

---

## 1. 设计方案

### 1.1 目标与范围

- **被测服务**：`faster-qwen3-tts` 的 **v4 合并单进程**入口（OpenAI 兼容 **`POST /v1/audio/speech`**，以及与推理同端口的 **`GET /health`**、**`GET /voices`** 等）。
- **压测客户端**：Locust **`HttpUser`**，使用标准 **`requests`** 会话发 HTTP；**不**嵌入 GPU 或模型代码，仅模拟真实调用方行为。

### 1.2 虚拟用户模型：`TTSMergedServiceUser`

| 项目 | 说明 |
|------|------|
| **基址 `host`** | 优先使用 Locust 命令行 **`--host`** 或 Web UI 中填写的 Host；否则取环境变量 **`LOCUST_TTS_HOST`**；再否则使用文件内 **`_DEFAULT_HOST`**（需按部署改成你的 `http://IP:端口`）。 |
| **请求间隔 `wait_time`** | `between(1, 2)`：每完成一个 task 后随机休眠 1～2 秒再执行下一个，避免把客户端自身打满导致失真（可按需改小以加大请求密度）。 |
| **启动钩子** | `events.init` 中打印 **短/长句字数**、**有效 `TTS_VOICE`**；若仍为占位符则 **warning**。 |

### 1.3 任务（`@task`）与权重

Locust 按 **权重** 随机挑选下一个 task（权重为 **0** 的任务**不会执行**）。

| 任务名 | 权重 | HTTP | 说明 |
|--------|------|------|------|
| `speech_wav_stream` | **8** | `POST /v1/audio/speech`，`response_format=wav` | **主负载**：**流式**拉取完整响应体（`stream=True` + `iter_content`），避免连接悬挂。正文从 **`_INPUT_POOL`** 中 **`random.choice`** 取 **短句或长句**。 |
| `speech_mp3_non_stream` | **0** | `POST ...`，`mp3` | 默认关闭；将权重改为大于 0 可启用整包 MP3 场景。 |
| `health` | **0** | `GET /health` | 默认关闭；权重 > 0 时可混合健康检查。 |
| `list_voices` | **0** | `GET /voices?status=active` | 默认关闭。 |

当前默认配置下，**仅流式 WAV 合成**在循环执行，便于专注 GPU/并发槽位与 **`METRICS`** 对齐。

### 1.4 待合成文本池 `_INPUT_POOL`

- 固定为 **两条**：**`[短句, 长句]`**，模拟 **导医导语** 与 **助手较长说明** 两类长度分布。
- **默认文案**：内置中文 **医疗导诊 / 医疗助手** 相关短句与长句（见 `locustfile.py` 中 `_DEFAULT_SHORT_MEDICAL`、`_DEFAULT_LONG_MEDICAL`）。
- **覆盖方式**：见下文环境变量 **`TTS_INPUT_SHORT` / `TTS_INPUT_LONG`**，并兼容旧变量 **`TTS_INPUT` / `TTS_INPUT_ALT`**。

### 1.5 请求体约定

与 OpenAI 形态一致，由 **`_speech_json`** 构造：

- `model`: `"tts-1"`（占位；实际权重以服务端 **`--model`** 为准）
- `input`: 来自池子的随机一条
- `voice`: 环境变量 **`TTS_VOICE`**，须与 **`voices_registry_v4.json`** 顶层 **`voice_id`** 一致
- `response_format`: `wav` 或 `mp3`

### 1.6 超时、失败与可观测性

- 所有请求使用 **`TTS_REQUEST_TIMEOUT`**（默认 **600** 秒），流式长句需足够大。
- 非 200 或体过小会 **`resp.failure(...)`**，并在失败信息中附带响应体片段 **`_resp_error_snippet`**，便于在 Locust **Failures** 页看到 FastAPI **`detail`**。
- 可选 **`TTS_LOG_TTFB=1`**：约 **5%** 的合成请求在控制台打印 **客户端首字节耗时（ms）**，可与服务端 **`ttfa_wall_ms`** 对照（含网络）。

### 1.7 环境变量安全读取

- **`_env_str`**：去首尾空白，并去除 CMD 误写的 **成对英文引号**（避免 `set VAR="uuid"` 导致 `voice` 错误）。

---

## 2. 使用指南

### 2.1 依赖安装

在独立 venv/conda 中：

```bash
pip install -r requirements.txt
```

（当前为 **`locust>=2.17.0`**，与主工程 TTS 依赖分离。）

### 2.2 环境变量一览

| 变量 | 必填 | 默认值 / 说明 |
|------|------|----------------|
| **`TTS_VOICE`** | **是**（生产压测） | 默认占位 `REPLACE_WITH_YOUR_VOICE_ID`，必须改为注册表真实 `voice_id`。 |
| **`TTS_INPUT_SHORT`** | 否 | 自定义短句；不设则用内置导医短句。 |
| **`TTS_INPUT_LONG`** | 否 | 自定义长句；不设则用内置医疗助手长段。 |
| **`TTS_INPUT`** | 否 | **兼容旧版**：未设 `TTS_INPUT_SHORT` 时作为短句来源。 |
| **`TTS_INPUT_ALT`** | 否 | **兼容旧版**：未设 `TTS_INPUT_LONG` 时作为长句来源。 |
| **`TTS_REQUEST_TIMEOUT`** | 否 | 默认 `600`（秒）。 |
| **`TTS_LOG_TTFB`** | 否 | `1` / `true` 等开启抽样客户端 TTFB 日志。 |
| **`LOCUST_TTS_HOST`** | 否 | 类属性默认 host；**`--host` 优先**。 |

### 2.3 Windows CMD 示例

**注意**：`set` 的值**不要**加引号；注释用 **`REM`**，不要用 `#`。

```bat
cd path\to\test_using_locust

set TTS_VOICE=你的voice_id
set TTS_LOG_TTFB=1

locust -f locustfile.py
```

浏览器打开 **`http://localhost:8089`**，在界面中填写 **Host**（例如 `http://10.251.11.55:10018`，与容器映射端口一致），再设置用户数、Ramp-up、运行时长后 **Start**。

### 2.4 无 Web UI（Headless）示例

```bat
locust -f locustfile.py --headless -u 10 -r 2 -t 5m --host http://10.251.11.55:10018
```

### 2.5 Locust Web 界面常用项（简要）

- **Number of users**：峰值并发虚拟用户数。  
- **Ramp up**：每秒新增用户数。  
- **Host**：被测服务基址（含协议与端口）。  
- **Run time**：如 `5m`、`10m`；留空则手动 Stop。  
- **Profile**：可选标签，仅便于区分多次试验。

### 2.6 与服务端配置的关系

- 服务端 **`--concurrency`** 限制同时在途合成数；Locust 用户数过高时会在服务端 **排队**，**`ttfa_wall_ms`** 会上升，属预期现象。  
- 短/长句交替用于接近真实 **问诊 + 长回复** 的文本长度分布；若只测容量，也可临时把短、长设为同一句（通过环境变量）。

### 2.7 常见问题

1. **`/health` 成功、`/v1/audio/speech` 全失败且体积极小**  
   多为 **`voice` 无效** 或 **`set TTS_VOICE="..."` 带引号**。检查启动日志中的 **`Effective TTS_VOICE`** 与 Failures 里的 **`detail`**。

2. **大量约 5s 失败**  
   多为客户端默认超时过短；本脚本已把 **`timeout`** 设为 **`TTS_REQUEST_TIMEOUT`**，请确认未使用旧版 `locustfile`。

3. **只想压合成、不想压管理接口**  
   保持 **`health` / `list_voices` / `mp3` 的 `@task(0)`** 即可；若需混合负载，将对应权重改为正整数。

### 2.8 远程串联编排（`remote_benchmark_orchestrator.py`）

在 **压测机**（例如 `10.251.11.27`，与推理机分离）上运行 **`remote_benchmark_orchestrator.py`**，可自动完成：

1. 解析仓库根目录下 **`faster-qwen3-tts修改后4090上并发测试.md`**（各 `## 1.x.` 节：远端启动命令 + 四级标题 **`4090_u*r*t*m`** 对应的 Locust 参数）；  
2. **SSH** 到推理宿主机（默认 `10.251.11.55`），**`docker exec`** 进入容器（默认 **`faster-qwen3-tts-jiangsy`**），停旧进程、按 md 在 **`/data/tts/faster-qwen3-tts`** 下 **nohup** 启动 **`openai_server_v4.py`**；  
3. 在本机轮询 **`/health`**（默认 `http://10.251.11.55:10018/health`）直至 **`status`** 为 **`ok`**；  
4. 仅在容器内 **`inference.log`**（默认路径见下表）追加 **`ORCH_MARKER`** 分隔行（新启服务 / 每条 Locust 前），便于与 **`METRICS`** 对齐做日志切片；  
5. 在 **`--workdir`**（默认本脚本所在目录）下 **无界面执行 Locust**，与 **§2.2～2.4** 相同依赖 **`TTS_VOICE`** 等环境变量。

**设计说明与边界**以同目录 **`remote_benchmark_orchestrator_plan.md`** 为准（SSH 形态、就绪判定、与 `inference.log` 的约定等）。

#### 前置条件

| 项 | 说明 |
|----|------|
| **本机** | 已安装 **Python 3**、**`locust`**（与 §2.1 一致）、**`ssh`**；能访问 **`--health-url`** / **`--locust-host`** 所指服务。 |
| **远端** | 当前 SSH 用户对目标主机有登录权限（建议免密）；可 **`docker exec`** 目标容器；容器内已有 **`/data/tts/faster-qwen3-tts`** 与启动命令所需依赖。 |
| **`TTS_VOICE`** | 正式跑前在 shell 中 **`export TTS_VOICE=…`**（或 Windows 下 **`set`**），与 **`locustfile.py`**、注册表一致。 |
| **服务端 `inference.log`** | 若需把 **`ORCH_MARKER`** 与推理摘要写在**同一固定文件**，请在服务端 **`config_v4.json`** 的 **`inference_logging`** 中设置 **`add_timestamp`: false**，且 **`file`** 指向编排器 **`--inference-log`** 所用路径（默认见下表）。否则服务可能写入带时间缀的文件名，与编排器追加目标不一致。 |

#### 命令行参数（常用）

| 参数 | 默认 / 说明 |
|------|-------------|
| **`--markdown`** | 测试方案 md；未传时先试环境变量 **`ORCH_BENCHMARK_MD`**，再自本脚本目录**向上查找** `faster-qwen3-tts修改后4090上并发测试.md`。 |
| **`--ssh-user`** | SSH 用户名；默认 **`ORCH_SSH_USER`** 或本机 **`USER`** / **`root`**。 |
| **`--ssh-host`** | SSH 主机，默认 **`10.251.11.55`**。 |
| **`--ssh-identity`** | 私钥路径；默认 **`ORCH_SSH_IDENTITY`**，可留空使用 ssh-agent。 |
| **`--docker-container`** | 容器名，默认 **`faster-qwen3-tts-jiangsy`**。 |
| **`--health-url`** | 就绪轮询完整 URL，默认 **`http://10.251.11.55:10018/health`**；可用 **`ORCH_HEALTH_URL`** 覆盖。 |
| **`--locust-host`** | 传给 **`locust --host`**；未传时由 **`--health-url`** 去掉 **`/health`** 推导；可用 **`ORCH_LOCUST_HOST`** 覆盖（经网关时建议显式指定）。 |
| **`--inference-log`** | 容器内 **`inference.log`** 绝对路径，默认 **`/data/tts/faster-qwen3-tts-assets/logs/inference.log`**（**`ORCH_INFERENCE_LOG`**）。 |
| **`--workdir`** | Locust **`cwd`**，默认本脚本所在目录（与 **`locustfile.py`** 同目录）。 |
| **`--locust-file`** | 默认 **`locustfile.py`**。 |
| **`--post-service-sleep`** / **`--post-locust-sleep`** | 每轮服务就绪后、每条 Locust 后的休眠秒数，默认各 **30**。 |
| **`--health-timeout`** | 等待 **`/health`** 就绪的最长时间（秒），默认 **600**。 |
| **`--dry-run`** | 只解析 md 并打印计划，**不**执行 SSH / Locust。 |
| **`--continue-on-error`** | Locust 非零退出时仍继续后续用例（默认**遇错中止**）。 |
| **`-v` / `--verbose`** | 更详细日志。 |

#### 使用示例（压测机为 Linux / macOS 类环境）

```bash
cd /path/to/test_using_locust
export TTS_VOICE=你的voice_id
    # 可选：export ORCH_BENCHMARK_MD=/path/to/faster-qwen3-tts修改后4090上并发测试.md

python3 remote_benchmark_orchestrator.py --dry-run -v    # 校验解析
python3 remote_benchmark_orchestrator.py --ssh-user myuser   # 正式执行
```

**Windows**：在 **CMD** 或 **PowerShell** 中同样先设置 **`TTS_VOICE`**（CMD 勿给值加引号），再 **`python remote_benchmark_orchestrator.py ...`**；若本机无 **`python3`** 命令，可改用 **`python`**。

---

## 3. 文件清单

| 文件 | 说明 |
|------|------|
| `locustfile.py` | Locust 入口与用户定义 |
| `requirements.txt` | 压测端 Python 依赖 |
| `remote_benchmark_orchestrator.py` | 解析测试 md、SSH + 容器内启停服务、写 **`ORCH_MARKER`**、本机 headless Locust 串联编排 |
| `remote_benchmark_orchestrator_plan.md` | 上述编排器的方案说明（架构、日志约定、与 md 的对应关系） |
| `locust使用指南.md` | 本文档（Locust 设计与使用、含 §2.8 编排说明） |

更完整的架构与 **`METRICS`** 口径见仓库根目录 **`faster-qwen3-tts服务使用与优化.md`** 第五章、第六章。
