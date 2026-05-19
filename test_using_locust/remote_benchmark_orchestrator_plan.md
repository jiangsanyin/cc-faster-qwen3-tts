# 远程 TTS 服务 + Locust 串联自动化方案

本文说明：在 **10.251.11.27** 上运行编排程序，通过 **SSH** 登录 **10.251.11.55**，在容器 **`faster-qwen3-tts-jiangsy`** 内按《faster-qwen3-tts修改后4090上并发测试.md》逐轮启动 `openai_server_v4.py`，确认就绪后在 **27** 上执行 Locust，并**仅**在服务端 **`inference.log`** 中写入可检索的分隔标记（**不**理会历史 `inference_*.log`）。测试文档中 **三级标题编号已统一、不重复**，编排逻辑以 **解析该 md** 为主。**可以实现**，以下为推荐实现方案与注意事项。

---

## 1. 可行性结论

| 能力 | 结论 |
|------|------|
| 27 → 55 SSH 免密 | 可行（依赖你已配置好的 `ssh user@10.251.11.55`） |
| 进入容器执行命令 | 可行：`ssh ... 'docker exec <容器名> bash -lc "..."'`（非交互加 `-lc`） |
| 启动服务并等待就绪 | 可行：后台启动进程 + 轮询 **HTTP 健康检查**（推荐）或 **容器内日志** 中出现 `Server ready` |
| 在 27 上跑 Locust 压 55 上映射端口 | 可行：`--host http://10.251.11.55:10018`（端口以你实际映射为准） |
| 向 `inference.log` 追加分隔行 | 可行：`docker exec` 内对 **唯一目标文件** `inference.log` 执行 `printf`/`echo >>`（**不处理**历史 `inference_*.log`） |

---

## 2. 总体架构

```text
┌─────────────────────────────┐     SSH + docker exec      ┌──────────────────────────────┐
│ 10.251.11.27（客户端）       │ ─────────────────────────► │ 10.251.11.55                 │
│ 编排脚本 + locustfile.py    │                            │ 容器 faster-qwen3-tts-jiangsy │
│                             │     HTTP 压测               │ :8000 → 宿主机 :10018        │
│ subprocess: locust ...      │ ─────────────────────────► │                              │
└─────────────────────────────┘                            └──────────────────────────────┘
```

- **编排程序**建议用 **Python 3**（单文件即可），放在与 `locustfile.py` 同目录或项目内固定路径，便于 `cwd` 指向 `test_using_locust` 执行 Locust。
- **不要在编排脚本里依赖交互式 `docker attach`**；一律用 `docker exec` + 非交互 shell。

---

## 3. 服务端（容器内）生命周期

### 3.1 工作目录与启动命令

每次在容器内：

1. `cd /data/tts/faster-qwen3-tts`
2. 执行与《faster-qwen3-tts修改后4090上并发测试.md》各二级标题中一致的 `python examples/openai_server_v4.py ...`（`--concurrency` / `--replicas` 随章节变化）。

### 3.2 换一轮配置前：停止旧进程

新一轮启动前需在容器内 **结束占用 8000 端口的旧服务**，避免端口冲突。可选做法（实现时任选其一，需与容器内环境一致）：

- 通过 `fuser -k 8000/tcp` 或 `lsof -t -i:8000 | xargs kill`；
- 或 `pkill -f 'openai_server_v4.py'`（注意勿误杀无关 Python 进程时，优先按端口杀）。

### 3.3 后台启动与就绪判定

- 使用 **`nohup ... &`** 或 **`setsid`** 在容器内后台启动，**stdout/stderr** 可重定向到固定文件（如 `/tmp/openai_server_v4.log`）便于排错。
- **就绪判定（推荐双保险）**：
  1. **从 27 本机**轮询：`GET http://10.251.11.55:<映射端口>/health`（或你文档中的 Host），返回 `200` 且 JSON 中 `status` 为 `ok`；
  2. 可选：同步 `docker exec` **`tail -F`** 或周期性 `grep` 容器内日志，直到出现 `Server ready. Listening on http://0.0.0.0:8000`。
- 设置 **最大等待时间**（例如 600s），超时则中止本轮并打印容器内最近日志尾部。

### 3.4 启动后休眠

按你的要求：每次新启动服务成功后 **`sleep 30`**（可在就绪判定通过后执行）。

---

## 4. 推理日志分隔行写入

### 4.1 日志路径说明（仅 `inference.log`）

- **编排器只向固定文件 `inference.log` 追加分隔标记**，与当前服务配置的推理汇总日志一致即可。
- 容器内使用 **绝对路径**，例如：`/data/tts/faster-qwen3-tts-assets/logs/inference.log`（与 `../faster-qwen3-tts-assets/logs/inference.log` 相对项目根等价，以容器内实际挂载为准）。
- **历史测试里曾出现的按日期命名文件 `inference_*.log` 不再纳入本方案**；实现与文档分析均以 **`inference.log`** 为唯一目标文件，无需对齐或理会 `inference_*.log`。

### 4.2 分隔格式（建议）

为便于 `grep`/脚本切片，每段标记建议包含 **ISO 时间**、**阶段关键字**、**关键参数**，再跟装饰线：

**每次新启动服务后**（在容器内、服务已启动并开始写日志之后追加，避免与启动前状态混淆；也可在杀旧进程之后、新启动之前写「上一轮结束」标记，按你分析习惯二选一或两种都要）：

```text
YYYY-mm-dd HH:MM:SS,fff INFO ORCH_MARKER 新启动服务 replicas=1 concurrency=1 section=1.1
########################################
########################################
########################################
```

（装饰用 `#` 行 **两三行即可**，不必刷几十行。）

**每次执行一条 Locust 命令前或后**（建议 **命令开始前** 写一条，便于与紧随其后的 METRICS 对齐）：

```text
YYYY-mm-dd HH:MM:SS,fff INFO ORCH_MARKER 新启动locust并发测试 u=2 r=1 t=3m host=http://10.251.11.55:10018 section=1.1.1.2
----------------------------------------
----------------------------------------
----------------------------------------
```

（装饰用 `-` 行同样 **两三行即可**。）

说明：前缀使用 `INFO` 与现有日志风格接近，关键字 `ORCH_MARKER` 便于一次性过滤编排器写入的行。

### 4.3 写入方式

通过 SSH 在 55 上执行：

```bash
docker exec faster-qwen3-tts-jiangsy bash -lc 'printf "%s\n" "..." >> /data/tts/faster-qwen3-tts-assets/logs/inference.log'
```

少量 `#` / `-` 行用 **`printf` 逐行追加** 或短 **heredoc** 即可，避免引号转义问题。

---

## 5. Locust（在 10.251.11.27 上执行）

### 5.1 工作目录与环境变量

- `cwd`：`test_using_locust`（与 `locustfile.py` 同目录）。
- 与线上一致：`TTS_VOICE`、`TTS_LOG_TTFB`、`TTS_REQUEST_TIMEOUT` 等由 shell 环境或编排脚本 `os.environ` 注入。

### 5.2 命令行参数来源

- **1.1.1** 节已写出完整命令；其余四级标题形如 **`4090_u2_r1_3m`**，可解析为：
  - `-u 2`、`-r 1`、`-t 3m`
- 文档中部分章节 **时长为 `5m`**（如 1.5～1.10 部分四级标题），解析规则需 **以四级标题中的 `uXX_rYY_ZZm` 为准**，不要一律写 `3m`。
- `--host`：与映射一致，例如 `http://10.251.11.55:10018`（若你实际使用 `10.251.11.27` 反代，则改为对应 URL；建议做成编排脚本 **可配置常量**）。

### 5.3 Locust 后休眠

每跑完一条 Locust：`sleep 30`。

---

## 6. 《faster-qwen3-tts修改后4090上并发测试.md》如何驱动自动化

### 6.1 推荐做法：解析 Markdown（主路径）

《faster-qwen3-tts修改后4090上并发测试.md》中 **各二级标题下三级标题编号已修正、不再重复**，编排脚本以 **直接解析 md** 为默认实现方式：

- 按顺序扫描每个 `## 1.x.` 章节。
- 在该章节内取第一个 **bash 代码块**（` ```bash ` …）作为 **容器内启动命令**（可整段交给 `bash -lc`，保留换行与反斜杠续行）。
- 在同一 `## 1.x.` 下，找到其 **唯一的三级标题**（如 `### 1.x.1. 测试结果`）之下的各级 **`####`** 四级标题；对每个形如 **`4090_u2_r1_3m`** 的标题，用正则提取 `u(\d+)_r(\d+)_(\d+)m`，映射为 Locust 的 `-u`、`-r`、`-t`（如 `-t 3m` / `-t 5m`）。
- 若某节四级标题与正文中的 locust 命令行不一致，**以四级标题中的 `u/r/时长` 为准**（与文档约定一致）。

**可选兜底**：若日后 md 结构再次混乱，可临时在代码中保留一份 **内置矩阵**（与下表一致）作为开关，不作为当前默认。

### 6.2 与文档一致的 10 组（replicas, concurrency）校验用表

| 章节 | replicas | concurrency |
|------|----------|-------------|
| 1.1 | 1 | 1 |
| 1.2 | 1 | 2 |
| 1.3 | 1 | 3 |
| 1.4 | 1 | 4 |
| 1.5 | 2 | 2 |
| 1.6 | 2 | 3 |
| 1.7 | 2 | 4 |
| 1.8 | 3 | 3 |
| 1.9 | 3 | 4 |
| 1.10 | 3 | 5 |

Locust 的 `-t` 以各节 **四级标题** 为准（1.1 多为 `3m`，1.5 起部分为 `5m`，以当前 md 为准）；解析脚本应对每节 **动态读取**，勿写死全局时长。

---

## 7. SSH / Docker 调用形式（示例）

编排脚本在 27 上执行本地命令与远程命令的推荐形态：

```bash
# 在 55 上、容器内执行单条命令（示例）
ssh user@10.251.11.55 "docker exec faster-qwen3-tts-jiangsy bash -lc 'cd /data/tts/faster-qwen3-tts && ...'"
```

- `user`：你在 55 上的登录用户（需具备 `docker exec` 权限，通常在 `docker` 组或 root）。
- 若免密未配置，需改用 **密钥路径** 或 **sshpass**（不推荐明文密码）。

Python 中可用 **`subprocess.run([...], check=True)`** 调用 `ssh`；也可用 **Paramiko** 减少 shell 注入风险（可选）。

---

## 8. 错误处理与可观测性

- 任一轮：**杀旧进程失败**、**启动超时**、**health 不通**、**locust 非 0 退出**：打印明确错误，可选 **是否继续下一轮**（建议默认中止并保留现场日志）。
- 将编排器自身日志同时输出到 **27 上本地文件**（含时间、执行的 ssh 命令摘要、locust 返回码），与 `inference.log` 中的 `ORCH_MARKER` 互证。

---

## 9. 后续实现时的交付物建议（供下一版开发）

| 文件 | 作用 |
|------|------|
| `remote_benchmark_orchestrator.py` | **已实现**：解析《faster-qwen3-tts修改后4090上并发测试.md》、SSH、`docker exec`、健康检查、仅向 **`inference.log`** 写 `ORCH_MARKER`、在本机调用 headless locust |
| 本文 `remote_benchmark_orchestrator_plan.md` | 方案与运维说明（当前文件） |

**运行示例（在 10.251.11.27 上、与 `locustfile.py` 同目录或指定 `--workdir`）**：

```bash
export TTS_VOICE=你的_voice_id
python3 remote_benchmark_orchestrator.py --dry-run -v          # 仅校验解析与计划
python3 remote_benchmark_orchestrator.py --ssh-user myuser     # 正式跑（需免密 SSH、远端 docker、本机 locust）
```

常用参数：`--ssh-host`、`--docker-container`、`--health-url`、`--locust-host`（默认同 health 推导）、`--inference-log`、`--markdown`、`--post-service-sleep` / `--post-locust-sleep`（默认各 30s）、`--continue-on-error`。

---

## 10. 小结

- **能实现**：SSH + `docker exec` 满足远程容器内启停服务；**27 上 Locust** 满足「客户端与推理机分离」；**仅向 `inference.log` 追加分隔行** 满足分段分析需求。  
- **实现要点**：旧服务清理、就绪判定（health + 超时）、**`inference.log` 绝对路径** 与 **ORCH_MARKER** 约定、Locust 参数 **按各节四级标题解析 u/r/时长**、**md 由解析器驱动**（三级标题已不重复，以解析为主）。  
- **休眠**：服务启动成功后 30s、每条 Locust 后 30s，按主循环执行即可。

编排脚本见同目录 **`remote_benchmark_orchestrator.py`**；首次联调建议 **`--dry-run -v`** 确认 10 节解析无误后再去掉 `--dry-run`。
