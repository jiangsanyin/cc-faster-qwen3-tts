# 一、openai_server.py使用

## 1) 先准备一个 `voices.json`

`--voices` 与 `--ref-audio` 二选一；传了 `--voices` 就走多音色映射逻辑。这边优先使用，因为可以把参考语音一次性都预加载以便后续使用，后续切换参考音频不需要重启服务。只有当新上传音频时才需要修改此 `--voices` 配置并重启服务。

`--voices` 要传的是一个 JSON 文件路径，格式是“音色名 -> 配置对象”。

```json
### 示例 /data/tts-jiang/faster-qwen3-tts/voices.json 文件如下：
{
  "ref_audio_3": {
    "ref_audio": "/data/tts/faster-qwen3-tts-assets/ref_audio_3.wav",
    "ref_text": "Don't be deceived by the name. There is nothing cuddly about this particular teddy bear. In fact, it's the most dangerous plant in the desert.",
    "language": "English"
  },
  "ref_audio_7": {
    "ref_audio": "/data/tts/faster-qwen3-tts-assets/ref_audio_7.wav",
    "ref_text": "变种病毒呢首次在内地的社区传播，天津市有两例的确诊，目前天津呢已经开始进行全市的核酸检测。",
    "language": "Chinese"
  },
  "ref_audio_8": {
    "ref_audio": "/data/tts/faster-qwen3-tts-assets/ref_audio_8.wav",
    "ref_text": "欢迎您使用硅基数字人，实时交互能够通过面对面对话，通过情感互动的数字人提供更好的客户服务。",
    "language": "Chinese"
  },
  ...
}
```

## 2) 启动服务（传 `--voices`）

在项目根目录执行：

```shell
python examples/openai_server.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --voices ./voices.json \
  --host 0.0.0.0 \
  --port 8000
```

## 3) 调用时按 `voice` 选择条目

请求里 `voice` 的值要对应 JSON 的 key（如 `alloy` / `echo`）：

```shell
### 请求里 voice 的值要对应 JSON 的 key（如 ref_audio_7 / ref_audio_8）：
# 在容器中执行如下命令：
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"你好，这是测试","voice":"ref_audio_7","response_format":"wav"}' \
  --output speech.wav

# 或在容器外其他服务器或客户端上执行如下命令：
curl http://localhost:10018/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"你好，这是测试","voice":"ref_audio_7","response_format":"wav"}' \
  --output speech.wav
```

- `response_format=wav` 或 `pcm`：流式返回（边生成边回传 chunk）
- `response_format=mp3`：非流式（先全量生成再转码返回）

# 二、openai_server_concurrent.py 并发版改造与使用

## 1) 改造目标与核心变化

`examples/openai_server.py` 原实现使用全局锁串行推理。为了服务多个客户端，新增并发版 `examples/openai_server_concurrent.py`，核心变化如下：

- 去掉全局串行锁（不再所有请求强制串行）。
- 引入并发控制信号量：`--concurrency`（最大同时处理中的请求数）。
- 引入模型副本池：`--replicas`（单进程加载多个模型副本，轮询分发请求）。
- 保留流式与非流式能力（重点优化流式路径）。
- 增加每请求监控日志（默认开启）：`TTFA`、`RTF`、总耗时、GPU利用率/显存占用。
- **启动阶段推理预热（降低首个真实请求的 TTFA 抖动）**：
  - 默认对每个副本调用 `FasterQwen3TTS._warmup()`，完成 **CUDA Graph** 捕获（与 `demo/server.py` 思路一致）。
  - 默认对 `voices.json` 中 **每个 voice** 做 **参考音频侧缓存预填**（内部等价于走一遍 `_prepare_generation`），将结果写入模型内的 `_voice_prompt_cache`。
  - 可选 **`--prime-stream-first-chunk`**：在预填后再在主线程上 **拉取一小段流式首块**，更贴近真实 HTTP 流式首包路径（启动更慢，但能消除「首条 HTTP 仍比后续慢一个数量级」的现象，**已在服务端实测验证**）。

## 2) 新增启动参数

**并发与观测**

- `--concurrency`：允许同时处理的请求上限（默认 `2`，也可用环境变量 `QWEN_TTS_CONCURRENCY`）。
- `--replicas`：同进程加载模型副本数（默认 `1`，也可用环境变量 `QWEN_TTS_REPLICAS`）。
- `--no-metrics-log`：关闭每请求指标日志（默认不传即开启）。

**CUDA Graph 预热（默认开启）**

- `--skip-warmup`：跳过启动时 `_warmup`；首个真实请求可能承担 Graph 捕获等一次性成本。
- `--warmup-prefill-len`：Talker 图预热用的 prefill 长度（默认 `100`）。

**Voice 与流式首包预热（参考音频缓存默认预填；流式首块为可选）**

- `--skip-prime-voices`：跳过对各个 voice 的启动预填（首个用到该 voice 的真实请求 TTFA 可能明显偏高）。
- `--prime-text`：预填/流式首块预热时使用的合成文本（默认 `.`）。若希望预热时的目标文本长度接近线上，可改为与业务相近的短句（**不宜过长**，否则启动变慢）。
- `--prime-stream-first-chunk`：在每个 voice 完成 `_prepare_generation` 后，再 **`next()` 一次流式生成器**，拉取首块音频对应的推理路径；**推荐在意首包一致性时与业务向的 `--prime-text` 联用**。
- `--prime-stream-max-new-tokens`：与上一项配合，限制预热阶段解码步数上限（默认 `48`）。

说明：

- `concurrency` 控制在飞请求数量。
- `replicas` 控制模型并行副本数（受 GPU 显存限制，副本越多占用越高）。
- **多副本时**：每个副本各自一份缓存，启动时会对 **每个副本 × 每个 voice** 执行预填（与代码一致）。

## 3) 推荐启动方式（并发版）

**基础写法**（与早期并发版相同；已含默认 `_warmup` + 各 voice 的 `_prepare_generation` 预填）：

```shell
python examples/openai_server_concurrent.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --voices ./voices.json \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --replicas 2
```

**压低首条 HTTP 的 `ttfa_ms`（生产推荐，已验证）**  

在仅做 Graph + 参考音频预填时，**第一条真实 HTTP 请求**仍可能比后续高约一个数量级，原因之一是：预填跑在 **主线程**，而请求内合成跑在 **后台线程**，首条在线程侧仍可能有一次性开销。开启 **`--prime-stream-first-chunk`** 可在启动阶段在主线程上先跑通「流式首块」路径。实测在 `Qwen3-TTS-12Hz-1.7B-Base`、单副本、`--concurrency 4` 下，**首条请求 `ttfa_ms` 可与第 2、3 条同为约 300ms 量级**（具体数值随 GPU/文本/音色变化）。

```shell
python examples/openai_server_concurrent.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --voices ./voices.json \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

**Shell 续行注意**：多行命令时，**每一行末尾（最后一行除外）必须有反斜杠 `\`**。若某一参数行（例如 `--concurrency 4`）后面没有 `\`，shell 会认为命令已结束，其后的 `--prime-stream-first-chunk` 等 **不会** 传给 Python，表现为预热参数未生效。

保守起步建议（RTX 4090 单卡）：

- 第一轮：`--replicas 1 --concurrency 2`
- 第二轮：`--replicas 2 --concurrency 3~4`
- 以 TTFA/RTF/总耗时与显存占用是否可接受作为依据逐步调整。

## 4) 启动预热与首包 TTFA：分工小结

| 阶段 | 作用 | 默认 |
|------|------|------|
| `_warmup` | Predictor / Talker 的 **CUDA Graph** 捕获 | 开启（`--skip-warmup` 可关） |
| `_prepare_generation` 预填 | 各 voice **参考音频编码**结果进入 `_voice_prompt_cache`，避免「每个 voice 第一次」重复算参考侧 | 开启（`--skip-prime-voices` 可关） |
| `--prime-stream-first-chunk` | 在主线程上跑 **流式首块**，贴近 HTTP 流式首包路径 | 关闭；需要时显式传入 |

若不使用 `--prime-stream-first-chunk`，仍可在服务启动后 **手动先发 5～10 个请求** 做暖机，效果与真实负载预热一致。

## 5) 指标日志说明（用于压测定并发）

并发版会输出类似日志：

```text
METRICS req_id=12 mode=stream ttfa_wall_ms=210.5 ttfa_ms=168.0 inter_chunk_max_ms=95.2 inter_chunk_p95_ms=82.0 rtf=3.95 audio_s=5.742 total_ms=1489.3 gpu_util=86 mem_util=72 vram_used_gb=14.2 vram_total_gb=24.0
```

字段含义（**`openai_server_v4.py` 合并服务**；旧版并发脚本字段可能略少）：

- **`ttfa_wall_ms`**：**墙钟**「校验通过后的请求锚点 → **首块 PCM 入队**」耗时（毫秒），与产品口径 **TTFA（Time To First Audio）** 对齐；含 **`--concurrency` 排队**时间。
- **`ttfa_ms`**（流式）：首块对应的模型侧累计 **`prefill_ms + decode_ms`**（毫秒），用于对照算力分解；**非 mp3** 下与 `ttfa_wall_ms` 并存。
- **`inter_chunk_max_ms` / `inter_chunk_p95_ms`**（仅 **mode=stream**）：相邻两音频块在服务端**入队时刻**之间的间隔（毫秒）的 **max** 与 **p95**，用于 proxy **流式播放连续性**（块间空档过大时客户端更易断粮）。
- `rtf`：**生成音频时长（秒）÷ 合成累计耗时（秒）**，数值越大表示算得越快（同样墙钟能吐出更多秒波形）
- `audio_s`：本次生成音频时长（秒）
- `total_ms`：整次请求总耗时（毫秒）
- `gpu_util`：GPU计算利用率（需安装 `pynvml` 才能读取）
- `mem_util`：GPU显存控制器利用率（需 `pynvml`）
- `vram_used_gb / vram_total_gb`：显存使用/总量

注：

- 若环境未安装 `pynvml`，仍会记录 `vram_used_gb`，但 `gpu_util/mem_util` 可能缺失。

## 6) 客户端调用方式不变

接口仍为 `POST /v1/audio/speech`，调用方式与 `openai_server.py` 一致：

- `response_format=wav` 或 `pcm`：流式返回（推荐）
- `response_format=mp3`：非流式返回

```shell
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"你好，这是并发测试","voice":"ref_audio_7","response_format":"wav"}' \
  --output speech.wav
```

# 三、openai_server_concurrent_v2.py 热加载版本

本章节内容主要实现在 `openai_server_concurrent_v2.py` 文件中，它是基于 `openai_server_concurrent.py` 的基础上进行改造、优化得到的。

## 3.1. 方案设计

### 3.1.1 总体架构

```
平台组 ──HTTP──▶ 算法组「音色管理 API」
                      │
                      ├─ 写/读 MySQL（唯一权威数据源）
                      ├─ 写库成功后清除 Redis 缓存（下次读取时从 MySQL 回填）
                      │
磁盘（Ubuntu 22.04 + 4090 单机）◀── 本机路径存放参考音频文件

平台组 / 客户端 ──HTTP──▶ TTS 推理服务（openai_server_concurrent_v2.py）
                      │
                      ├─ 解析 voice_id
                      ├─ 先查 Redis；未命中则查 MySQL，查到后回写 Redis
                      └─ 用 ref_audio 本机路径 + ref_text 调用模型推理
```

**三层分工**：

| 层 | 职责 |
|------|------|
| **MySQL** | 音色元数据的**唯一权威数据源**，所有增删改查以 MySQL 为准 |
| **Redis** | **只读加速缓存**；缓存丢失可随时从 MySQL 重建，不影响正确性 |
| **磁盘** | 仅在该 4090 服务器本机上存放参考音频文件（wav / mp3 等）；MySQL 中存储的是文件在本机上的路径 |

单机部署下不存在多机文件挂载一致性问题，只需 TTS 推理服务与音色管理 API **约定同一套根目录与命名规则**即可。

### 3.1.2 配置文件（`config.json`）

所有外部依赖（数据库、缓存、文件目录等）的连接信息与业务参数统一放在 `config.json` 中，**不硬编码在代码里**。已确定的配置如下：

```json
{
  "mysql": {
    "faster_qwen3_tts_tone": {
      "host": "10.251.11.25",
      "port": 3306,
      "user": "root",
      "password": "gmq#2020",
      "database": "meta_human_zt0003",
      "table_name": "faster_qwen3_tts_tone",
      "charset": "utf8mb4",
      "connect_timeout": 10,
      "retry": 3
    }
  },
  "redis": {
    "faster_qwen3_tts_tone": {
      "redis_ip": "10.255.33.13",
      "redis_port": 6379,
      "password": null,
      "db": 1,
      "key_prefix": "tts:voice:",
      "cache_ttl_seconds": 86400
    }
  },
  "tone_wav_file_dir": "/data/tts-jiang/faster-qwen3-tts-assets",
  "allowed_audio_formats": ["wav", "mp3"],
  "max_audio_file_size_mb": 20
}
```

各字段说明：

| 配置项 | 说明 |
|--------|------|
| `mysql.*.host/port/user/password` | MySQL 连接信息 |
| `mysql.*.database` | 数据库名 |
| `mysql.*.table_name` | 音色元数据表名 |
| `mysql.*.charset` | 连接字符集，`utf8mb4` 支持中文与特殊符号 |
| `mysql.*.connect_timeout` | 连接超时（秒） |
| `mysql.*.retry` | 连接失败重试次数 |
| `redis.*.redis_ip/redis_port` | Redis 连接地址与端口 |
| `redis.*.password` | Redis 密码（无密码填 `null`） |
| `redis.*.db` | Redis 数据库编号（设为 `1`，与其他业务隔离） |
| `redis.*.key_prefix` | 缓存 key 前缀，格式 `tts:voice:` |
| `redis.*.cache_ttl_seconds` | 缓存过期时间（秒），`86400` = 24 小时 |
| `tone_wav_file_dir` | 参考音频文件存放根目录 |
| `allowed_audio_formats` | 允许上传的音频格式白名单 |
| `max_audio_file_size_mb` | 单个音频文件大小上限（MB） |

**TTS 推理服务本身的运行时参数**（`--model`、`--host`、`--port`、`--concurrency`、`--replicas`、预热相关参数等）与音色数据管理无关，继续通过**命令行参数**传入。

### 3.1.3 MySQL 表设计

在「200 音色以下」规模，**一张主表**即可（如需审计可后续加日志表）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `voice_id` | VARCHAR / 主键 | 唯一音色标识（UUID 或业务自定义），与 TTS 请求的 `voice` 参数对应 |
| `ref_text` | TEXT | 参考音频对应的文本内容 |
| `language` | VARCHAR | 语言标识，如 `Chinese` / `English` / `Auto` |
| `ref_audio_path` | VARCHAR | 参考音频文件在服务器上的**绝对路径**（或 `tone_wav_file_dir` + 相对路径） |
| `status` | VARCHAR | 状态：`active`（可用）/ `disabled`（已禁用） |
| `version` | INT | 版本号，每次更新音色（换音频或改文本）时递增 |
| `created_at` | DATETIME | 创建时间 |
| `updated_at` | DATETIME | 最近更新时间 |

### 3.1.4 Redis 缓存策略

- **Key 格式**：`{key_prefix}{voice_id}`，例如 `tts:voice:abc123`。
- **Value**：JSON 字符串，包含 `ref_audio_path`、`ref_text`、`language`、`version` 等推理所需字段。
- **过期时间**：由 `cache_ttl_seconds` 控制（当前配置为 24h），作为兜底过期策略。
- **缓存更新机制**：管理 API 执行任何**增加 / 修改 / 删除**操作成功后，**立即删除 Redis 中对应的 key**；TTS 下次收到该 `voice_id` 的请求时会因缓存未命中而从 MySQL 重新读取，并将最新数据回写到 Redis。这种「写后删缓存」的方式可避免 Redis 与 MySQL 的双写不一致问题。
- **启动时预热（可选）**：TTS 启动时可从 MySQL 拉取全部 `active` 音色一次性写入 Redis；也可完全采用**懒加载**方式（首次请求时再从 MySQL 读取）。200 条以内数据量很小，两种方式均可。

### 3.1.5 磁盘与文件约定

- **根目录**：由 `config.json` 中的 `tone_wav_file_dir` 指定（当前为 `/data/tts-jiang/faster-qwen3-tts-assets`）。
- **文件命名**：例如 `{voice_id}.wav`，或按 `{voice_id}/{version}.wav` 组织（更新音色时写入新版本文件，DB 指向新路径，旧文件可异步清理）。
- **上传校验**：格式必须在 `allowed_audio_formats` 白名单内（当前 `wav` / `mp3`）；文件大小不超过 `max_audio_file_size_mb`（当前 20 MB）。
- **管理 API 处理流程**：接收上传的音频流 → 校验格式与大小 → 落盘到 `tone_wav_file_dir` → 文件路径写入 MySQL → 清除 Redis 中对应缓存。
- **TTS 推理时**：只信任从 Redis/MySQL 解析出来的路径，若文件不存在则返回 4xx 错误并打日志。

### 3.1.6 TTS 推理服务与音色解析

1. **请求参数**：客户端发送的 `voice` 字段即为 `voice_id`（与此前 `voices.json` 的 key 角色相同）。
2. **解析流程**：`Redis GET` → 若未命中则 `SELECT MySQL` → 查到后 `SET Redis`（带 TTL）→ 得到 `ref_audio_path` / `ref_text` / `language`。
3. **与现有代码衔接**：内存中仍可维护 `dict` 作本地缓存（200 条以内全量缓存也简单），或每次请求查 Redis（Redis 在同机或同机房时延迟极低，通常不会成为瓶颈）。
4. **模型内缓存（`_voice_prompt_cache`）**：该缓存的 key 由 `(ref_audio路径, ref_text, ...)` 组成——**更新音色后若路径或 ref_text 发生变化**，新的组合自然不会命中旧缓存，模型会自动重新编码参考音频。若仅修改了不参与 cache key 的字段（如 `language`），则需要清除该条缓存或重启 TTS 服务。
5. **并发控制**：继续沿用现有 `--concurrency` / `--replicas` 命令行参数；MySQL 和 Redis 各自使用连接池，注意线程安全。

### 3.1.7 四类操作与一致性保障

| 操作 | MySQL | 磁盘 | Redis | TTS 模型缓存 |
|------|-------|------|-------|-------------|
| **查询** | 读取 | — | 优先读缓存 | — |
| **增加** | INSERT | 写入新文件 | 可不操作（首次请求时懒加载） | 首次请求时自动加载 |
| **删除** | DELETE 或软删除 | 删除文件（异步亦可） | **DEL key** | 旧路径自然失效；若内存 dict 有残留需剔除 |
| **更新** | UPDATE | 写入新文件或覆盖旧文件 | **DEL key** | 新的 (path, text) 产生新 cache key，旧缓存自然不再命中 |

**更新操作的执行顺序**：**先将新文件写入磁盘** → **再提交 MySQL 更新** → **最后清除 Redis 缓存**。这样可避免 TTS 从 MySQL 读到新路径时文件尚未写完的问题。

### 3.1.8 部署与压测衔接

- **部署形态**：单机单卡（Ubuntu 22.04 + RTX 4090），平台组远程通过端口调用。
- **压测方式**：仍按本文档「四、单卡 4090 并发压测流程」中的方法，扫 `--concurrency` / `--replicas`，观察 TTFA、RTF、显存、错误率。
- **Redis/MySQL 延迟影响**：在同机或同机房部署时，查询延迟相对 GPU 推理时间通常可忽略不计。但需注意连接池配置与超时参数，避免阻塞推理线程。

### 3.1.9 安全与运维

- **敏感信息**：通过 `config.json` 集中管理，不硬编码在代码中。
- **权限分离**：音色管理 API 具备读写权限；TTS 推理服务对音色数据**只读**，不暴露写音色能力。
- **数据备份**：MySQL 定期备份；Redis 作为纯缓存可不做持久化（缓存丢失可从 MySQL 重建）。

### 3.1.10 方案小结

MySQL 存储音色权威元数据，本机磁盘存放参考音频文件；Redis 仅作为 `voice_id → 元数据` 的只读缓存。TTS 单机部署，收到请求后按 `voice_id` 先查 Redis、未命中再查 MySQL，用本机路径完成推理。音色管理 API 负责增删改查，写库成功后清除 Redis 中对应缓存，保证下次读取时拿到最新数据。音色规模在 200 以内，单表加规范路径即可满足需求。

### 3.1.11 编码实现顺序

1. **MySQL 建表 + 音色管理 API**（含上传文件落盘逻辑）
2. **Redis 缓存封装**（get / invalidate 方法）
3. **TTS 侧音色解析改造**（替换原有的 `voices.json` 读取逻辑）
4. **联调四种操作**（增加、删除、更新、查询）
5. **并发压测**（调整 `--concurrency` / `--replicas` 参数，确定上线配置）

## 3.2 使用指南

### 3.2.1 文件结构

```
examples/
├── voice_db.py                       # MySQL + Redis 数据访问封装（共用模块）
├── voice_manager_api.py              # 音色管理 API（增删改查 + 文件上传）
├── openai_server_concurrent_v2.py    # TTS 推理服务（v2 热加载版）
├── openai_server_concurrent.py       # TTS 推理服务（并发版，旧版保留）
└── openai_server.py                  # TTS 推理服务（单线程原版，保留）
config.json                           # 统一配置文件
```

- **`voice_db.py`**：被管理 API 和 TTS 推理服务**共同引用**，避免重复代码。
- **`voice_manager_api.py`**：管理面，负责写操作（增/删/改）和查询，**独立进程运行**。
- **`openai_server_concurrent_v2.py`**：推理面，**只读**音色数据，不暴露写能力。

### 3.2.2 依赖安装

在已有 `faster-qwen3-tts` 环境基础上，额外安装：

```bash
pip install pymysql redis
```

### 3.2.3 启动音色管理 API

```bash
python examples/voice_manager_api.py \
  --config config.json \
  --port 8001
```

启动后自动执行 `ensure_table()`，若 MySQL 中表 `faster_qwen3_tts_tone` 不存在则自动建表。

### 3.2.4 音色管理 API 接口说明

接口一览（默认服务地址 `http://localhost:8001`，端口以实际启动参数为准）：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/voices` | 新增音色（multipart 上传音频） |
| `GET` | `/voices` | 列出音色（默认 `active`，可用 `?status=` 筛选） |
| `GET` | `/voices/{voice_id}` | 查询单个 **active** 音色 |
| `PUT` | `/voices/{voice_id}` | 更新音频 / 文本 / 语言（仅对 **active** 有效） |
| `DELETE` | `/voices/{voice_id}` | 软删除或硬删除 |
| `POST` | `/voices/{voice_id}/restore` | **恢复软删除**：将 `disabled` 改回 `active` |
| `GET` | `/health` | 健康检查 |

#### 新增音色

**指定 `voice_id`（自定义主键，与 TTS 请求里 `voice` 字段一致）：**

```bash
curl -X POST http://localhost:8001/voices \
  -F "audio_file=@/path/to/ref_audio.wav" \
  -F "ref_text=欢迎您使用语音合成服务" \
  -F "language=Chinese" \
  -F "voice_id=my_voice_001"
```

**不传 `voice_id`（由服务端自动生成）：**

```bash
curl -X POST http://localhost:8001/voices \
  -F "audio_file=@/path/to/ref_audio.wav" \
  -F "ref_text=欢迎您使用语音合成服务" \
  -F "language=Chinese"
```

表单字段说明：

- `audio_file`（必填）：参考音频文件（wav / mp3）。
- `ref_text`（必填）：参考音频对应的文本。
- `language`（可选，默认 `Auto`）：语言标识。
- `voice_id`（可选）：
  - **有值**：原样写入 MySQL 表字段 `voice_id`，并用于落盘文件名 `{voice_id}.{扩展名}`（实现见 `examples/voice_manager_api.py` 中 `add_voice` → `_save_audio`）。
  - **不传或空字符串**：服务端生成 **`str(uuid.uuid4())`**，即 **标准 UUID 字符串（含连字符，36 字符）**，例如 `f47ac10b-58cc-4372-a567-0e02b2c3d479`。  
    说明：使用 `str(uuid.uuid4())` 而非裸 `UUID` 对象，便于 JSON 响应与数据库统一按字符串处理（代码位置：`voice_manager_api.py` 中 `if not voice_id:` 分支）。

返回示例（自定义 `voice_id`）：

```json
{"success": true, "voice_id": "my_voice_001", "voice": {"voice_id": "my_voice_001", "ref_audio": "/data/tts-jiang/faster-qwen3-tts-assets/my_voice_001.wav", "ref_text": "...", "language": "Chinese", "status": "active", "version": 1}}
```

返回示例（自动生成 `voice_id`，`voice_id` 为 UUID 字符串）：

```json
{
  "success": true,
  "voice_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "voice": {
    "voice_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "ref_audio": "/data/tts-jiang/faster-qwen3-tts-assets/f47ac10b-58cc-4372-a567-0e02b2c3d479.wav",
    "ref_text": "...",
    "language": "Chinese",
    "status": "active",
    "version": 1
  }
}
```

#### 查询所有音色

```bash
curl http://localhost:8001/voices
```

返回所有 `active` 状态的音色列表。可通过 `?status=disabled` 查询已禁用音色。

#### 查询单个音色

```bash
curl http://localhost:8001/voices/my_voice_001
```

#### 更新音色

只能更新status=active的音色。否则提示“不存在或已禁用”，此时对于被软件删除的音色需要先恢复。

```bash
# 仅更新文本
curl -X PUT http://localhost:8001/voices/my_voice_001 \
  -F "ref_text=新的参考文本内容"

# 更换音频 + 更新文本
curl -X PUT http://localhost:8001/voices/my_voice_001 \
  -F "audio_file=@/path/to/new_audio.wav" \
  -F "ref_text=新的参考文本内容"
```

更新成功后会自动清除 Redis 缓存，TTS 下次请求该 `voice_id` 时自动拉取最新数据。

#### 删除音色

```bash
# 软删除（默认，status 置为 disabled）
curl -X DELETE http://localhost:8001/voices/my_voice_001

# 硬删除（物理删除数据库行 + 磁盘音频文件）
curl -X DELETE "http://localhost:8001/voices/my_voice_001?hard=true"
```

#### 恢复软删除的音色

对执行过 **软删除**（`DELETE /voices/{voice_id}` 且未带 `hard=true`）的音色，数据库中 `status` 为 `disabled`，TTS 推理侧 `get_voice` 只查询 `active`，故该音色暂时不可用。

**为何不能用 `PUT /voices/{voice_id}` 恢复**：更新接口会先调用 `get_voice` 校验；已禁用音色查不到，接口返回 **404**，无法走更新逻辑。

**正确方式**：调用专用恢复接口，将 `status` 从 `disabled` 更新为 `active`，并清除 Redis 中该 `voice_id` 的缓存（`config.json` 里 `key_prefix` + `voice_id`，默认形如 `tts:voice:my_voice_001`）。

```bash
# 恢复软删除的音色
curl -X POST http://localhost:8001/voices/my_voice_001/restore
```

- **路径参数**：`voice_id` 与创建时一致（如 `my_voice_001`）。
- **请求体**：无。
- **成功（HTTP 200）**：返回 JSON，含 `success: true`、`voice_id` 及完整 `voice` 对象（`status` 为 `active`）。
- **失败（HTTP 404）**：该行不存在、该音色已是 `active`、或从未处于 `disabled`（无符合条件的更新行）。

返回示例：

```json
{
  "success": true,
  "voice_id": "my_voice_001",
  "voice": {
    "voice_id": "my_voice_001",
    "ref_audio": "/data/tts-jiang/faster-qwen3-tts-assets/my_voice_001.wav",
    "ref_text": "……",
    "language": "Chinese",
    "status": "active",
    "version": 2
  }
}
```

恢复后 **无需重启** TTS 推理服务（`openai_server_concurrent_v2.py`），下一次合成请求即可按 `voice_id` 正常拉取配置。

**手工恢复（无管理 API 或应急）**：在 MySQL 中执行（表名以 `config.json` 中 `table_name` 为准，示例为 `faster_qwen3_tts_tone`）：

```sql
UPDATE faster_qwen3_tts_tone
SET status = 'active', updated_at = NOW()
WHERE voice_id = 'my_voice_001' AND status = 'disabled';
```

并在 Redis 删除对应缓存键，避免仍读到旧状态，例如：

```text
DEL tts:voice:my_voice_001
```

（若配置了 `key_prefix`，请将前缀与 `voice_id` 拼接为实际 key。）

#### 健康检查

```bash
curl http://localhost:8001/health
```

返回 MySQL 和 Redis 的连通状态。

### 3.2.5 启动 TTS 推理服务（v2 热加载版）

```bash
python examples/openai_server_concurrent_v2.py \
  --config config.json \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

**与 v1（`openai_server_concurrent.py`）的区别**：

| 项目 | v1 并发版 | v2 热加载版 |
|------|-----------|-------------|
| 音色数据来源 | `--voices voices.json`（启动时一次性加载） | `--config config.json`（Redis + MySQL 动态查询） |
| 新增/修改音色 | 需修改 `voices.json` 并**重启服务** | 管理 API 操作后，**无需重启**，下次请求自动生效 |
| 启动时预热 | 遍历 `voices.json` 中所有 voice | 从 MySQL 拉取所有 `active` 音色进行预热 |

**v2 专有的 CLI 参数**：

- `--config`：指定 `config.json` 路径（默认 `../config.json`）。

**沿用的 CLI 参数**（与 v1 一致）：

- `--model`、`--host`、`--port`、`--device`
- `--concurrency`、`--replicas`
- `--skip-warmup`、`--warmup-prefill-len`
- `--skip-prime-voices`、`--prime-text`、`--prime-stream-first-chunk`、`--prime-stream-max-new-tokens`
- `--no-metrics-log`

### 3.2.6 客户端调用 TTS（与 v1 一致）

接口仍为 `POST /v1/audio/speech`，`voice` 字段传的是 MySQL 中的 `voice_id`（例如 `ffac0174-df90-4f85-8240-f7b56ea7c91a`）。

#### 服务端与 `--output` 的关系

**TTS 服务不要求、也不识别 curl 的 `--output`。** 服务端只按 HTTP 返回音频流；是否落盘、是否边下边播，全由**客户端**决定。

若使用 **curl**：未指定 `-o` / `--output` 时，curl 默认把响应体写到**当前终端**。`wav` / `pcm` / `mp3` 为二进制，curl 会提示 *Binary output can mess up your terminal* 并**拒绝写入**（退出码 **23**）。这不是接口报错，而是 curl 的保护行为。

| 需求 | curl 写法示例 |
|------|----------------|
| 保存为文件 | `--output speech.wav` 或 `-o speech.wav` |
| 不保存、把字节送到标准输出再交给播放器 | `-o -` 或 `--output -`，再 **管道** 给播放器，例如：`curl ... -o - \| ffplay -nodisp -autoexit -i -`（需本机已装 ffplay；其它播放器按其对 stdin 的支持调整） |
| 仅测连通、丢弃 body | `-o /dev/null`（Linux）或 `-o NUL`（Windows cmd） |

**业务程序**（浏览器、App、Python `requests`/`httpx` 等）没有 `--output` 概念：在代码里读取 `response` 的流式 body，边读边解码播放即可；需要存档时再写入文件。

```bash
# 保存为文件（流式 wav）
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，这是热加载版测试","voice":"ffac0174-df90-4f85-8240-f7b56ea7c91a","response_format":"wav"}' \
  -o speech.wav

# 非流式 mp3，保存为文件
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，这是热加载版测试","voice":"ffac0174-df90-4f85-8240-f7b56ea7c91a","response_format":"mp3"}' \
  -o speech2.mp3
```

- **`model` 字段**：请求体里可省略。`SpeechRequest` 默认 `model="tts-1"`，仅为与 OpenAI TTS 入参形式兼容；**当前服务不会根据该字段切换权重**，实际使用的始终是启动 TTS 时 `--model` 所加载的模型。
- 若客户端（如 OpenWebUI）仍传 `"model":"tts-1"`，行为与省略时一致。
- `response_format=wav` 或 `pcm`：流式返回（推荐）
- `response_format=mp3`：非流式返回
- 上述 `localhost:8000` 是在容器内调用时使用，如果在容器外或其他客户端使用：`10.251.11.27:10019`

### 3.2.7 典型部署流程

1. **准备配置**：确认 `config.json` 中 MySQL / Redis / 音频目录等信息正确。
2. **启动管理 API**：`python examples/voice_manager_api.py --config config.json --port 8001`（自动建表）。
3. **录入音色**：通过管理 API 的 `POST /voices` 上传参考音频和文本。
4. **启动 TTS 推理服务**：`python examples/openai_server_concurrent_v2.py --config config.json ...`（自动从 MySQL 拉取 active 音色做预热）。
5. **客户端调用**：`POST /v1/audio/speech`，`voice` 传入 `voice_id`。
6. **后续变更音色**：通过管理 API 增删改，**无需重启 TTS 推理服务**。

### 3.2.8 `voice_db.py` 模块说明

该模块为共用数据访问层，被管理 API 和 TTS 推理服务同时引用。核心类 `VoiceDB` 的主要方法：

| 方法 | 说明 | 调用方 |
|------|------|--------|
| `ensure_table()` | 自动建表（`CREATE TABLE IF NOT EXISTS`） | 管理 API 启动时 |
| `get_voice(voice_id)` | Redis → MySQL → 回写 Redis，返回 voice config | TTS 推理（每次请求） |
| `list_voices(status)` | 列出指定状态的所有音色 | 管理 API + TTS 启动预热 |
| `add_voice(...)` | INSERT MySQL | 管理 API |
| `update_voice(voice_id, ...)` | UPDATE MySQL + 清 Redis 缓存 | 管理 API |
| `delete_voice(voice_id, hard)` | 软/硬删除 + 清 Redis 缓存 | 管理 API |
| `restore_voice(voice_id)` | 将 `disabled` 恢复为 `active` + 清 Redis 缓存 | 管理 API |
| `invalidate_cache(voice_id)` | 删除 Redis 中指定 key | 管理 API（写后清缓存） |
| `check_connections()` | 检测 MySQL / Redis 连通性 | 健康检查 |

`get_voice()` 返回的 dict 格式与 TTS 推理代码期望的一致：

```python
{
    "voice_id": "my_voice_001",
    "ref_audio": "/data/tts-jiang/faster-qwen3-tts-assets/my_voice_001.wav",
    "ref_text": "参考文本",
    "language": "Chinese",
    "status": "active",
    "version": 1
}
```

其中 `ref_audio` 由 MySQL 的 `ref_audio_path` 字段映射而来，与 TTS 推理代码中 `voice_cfg["ref_audio"]` 的取值完全一致。

# 四、openai_server_concurrent_v3.py 热加载版本

本章节描述在**不使用 MySQL、不使用 Redis** 的前提下，用 **JSON 文件 + 磁盘音频** 实现音色元数据管理与 TTS 侧**无需重启即可感知变更**的方案；基线实现见 `openai_server_concurrent_v3.py` 等 **`_v3` 文件**。**上传后立即预填 `_voice_prompt_cache`** 的推荐做法见 **4.1.6.1**（可与基线分步落地）。

> **与第三章的关系**：第三章中的 **v2（`openai_server_concurrent_v2.py` + `voice_db.py` + MySQL + Redis）** 仍可保留，适用于已接入公司统一数据库与缓存的环境；**v3** 面向「多环境易迁移、音色量级数百以内、不依赖数据库」的部署形态。

## 4.1 方案设计

### 4.1.1 背景与目标

- **管理侧诉求**：产品经理与技术总监希望整套方案**便于迁移到不同环境**（开发 / 测试 / 生产、不同机房），减少对外部 MySQL、Redis 的依赖与运维成本。
- **规模**：音色数量预期在**几百条以内**，JSON 单文件承载元数据完全可行。
- **能力保留**：
  - 仍通过**管理 API**（或脚本）维护音色，避免手工改 JSON 易出错（可选，但推荐保留与 v2 类似的 HTTP 管理面）。
  - TTS 推理侧在音色**增删改**后**尽量不重启进程**即可生效（热加载）。
  - 参考音频本体仍落在**服务器磁盘**指定目录，JSON 只存路径与业务字段。

### 4.1.2 与 v2（MySQL + Redis）的对比

| 维度 | v2（MySQL + Redis） | v3（JSON + 磁盘，本方案） |
|------|---------------------|---------------------------|
| 权威数据源 | MySQL 表 | **一个（或分层）JSON 注册表文件** |
| 读加速 | Redis 缓存 | **进程内内存 + 文件 mtime / 版本号** 失效即可，无需 Redis |
| 环境迁移 | 需导出库表、同步 Redis、改连接配置 | **拷贝注册表 JSON + 音频目录 + 精简 `config`** 即可 |
| 并发写 | 数据库事务 | **文件锁 + 原子写临时文件再 rename** |
| 适用规模 | 任意（偏中大规模） | **数百音色** 舒适区 |

### 4.1.3 总体架构

```
平台组 ──HTTP──▶ 音色管理 API（可选独立进程，与 v2 接口形态尽量一致）
                      │
                      ├─ 读/写 音色注册表 JSON（加文件锁）
                      ├─ 上传/删除 磁盘上的参考音频文件
                      └─ （推荐，见 4.1.6.1）写库成功后 ──HTTP（内部）──▶ TTS：prime-voice

客户端 ──HTTP──▶ TTS 推理服务（openai_server_concurrent_v3.py）
                      │
                      ├─ resolve_voice：从注册表解析 ref_audio / ref_text / language
                      │   （带 mtime 检测或短 TTL 内存缓存，实现热加载）
                      ├─ （内部）prime-voice：按 voice_id 预填各副本 _voice_prompt_cache
                      └─ 调用现有 generate_voice_clone / streaming
```

- **不再部署** MySQL、Redis（v3 路径下 `config.json` 可去掉 `mysql` / `redis` 段，或保留其它非 DB 项如 `tone_wav_file_dir`）。
- **单一真相**：音色注册表 JSON；音频文件为二进制资产，路径由注册表引用。

### 4.1.4 音色注册表 JSON 结构（建议）

在兼容现有推理代码的前提下，**顶层仍为「voice_id → 配置对象」**，与早期 `voices.json` 一致，便于 `voice` 请求字段直接对应 key。建议在配置对象中**扩展**可选字段，便于管理与软删除：

```json
{
  "ref_audio_8": {
    "ref_audio": "/data/tts-jiang/faster-qwen3-tts-assets/ref_audio_8.wav",
    "ref_text": "欢迎您使用硅基数字人……",
    "language": "Chinese",
    "status": "active",
    "version": 3,
    "updated_at": "2026-04-08T12:00:00"
  }
}
```

- **`ref_audio` / `ref_text` / `language`**：与当前 TTS 推理使用字段一致（`openai_server_concurrent*.py` 中 `voice_cfg`）。
- **`status`**（可选）：`active` / `disabled`，软删除等价于从「有效集合」中排除或标记 `disabled`（实现二选一，文档与接口约定统一即可）。
- **`version` / `updated_at`**（可选）：便于排障、审计；也可用于触发进程内缓存失效。

**约定**：仅 `status=active`（或未写 status 视为 active）的条目参与 TTS 解析与启动预热。

### 4.1.5 磁盘与音频文件

- **根目录**：继续由配置项指定（可与现网 `tone_wav_file_dir` 一致）。
- **命名**：建议 `{voice_id}.wav` 或带版本后缀。**更新音频时**，服务端会自动递增版本号并保存为新路径（如 `{id}_v2.wav`），同时修改 JSON 中的引用。
- **管理 API**：上传校验格式与大小后落盘，再把 `ref_audio` 绝对路径写入 JSON。**更新音色成功后**，管理端会自动清理磁盘上的旧版本音频文件，确保磁盘空间不被浪费。

### 4.1.6 热加载策略（TTS 进程内）

目标：**不重启 uvicorn** 也能在管理端改完 JSON 后尽快生效。推荐以下组合（实现时择一或组合）：

1. **基于文件 mtime 的缓存失效（推荐）**  
   - 进程内维护：`last_mtime`、`cached_dict`。  
   - 每次 `resolve_voice`（或首次访问注册表时）若 `os.path.getmtime(registry_path) > last_mtime`，则重新 `json.load` 并更新缓存。  
   - 几百条 JSON 解析成本极低，远小于一次 GPU 推理。

2. **可选：定期后台线程刷新**  
   - 每 N 秒检查 mtime，降低高频请求下的 `stat` 次数（非必须）。

3. **模型侧缓存**  
   - `FasterQwen3TTS._voice_prompt_cache` 仍按 `(ref_audio, ref_text, …)` 工作；**路径或文本变化**会自然 miss；同路径覆盖文件时需通过换路径或进程级清缓存策略规避（见 4.1.5）。

**不推荐**在 v3 中重新引入 Redis；进程内缓存 + mtime 已足够。

### 4.1.6.1 上传后主动预热 `_voice_prompt_cache`（方案 A：管理端 HTTP → TTS 内部接口，推荐）

**背景与问题**

- 注册表 **mtime 热加载**（见 **4.1.6**）只保证 TTS 能**立刻解析**到新/变更后的 `ref_audio`、`ref_text` 等字段。
- 模型内的 **`_voice_prompt_cache`** 默认仍在**该音色第一次被真实合成请求命中**时，才完成参考侧编码并写入缓存，**首包 TTFA 可能明显偏高**（行为与未做启动预热时类似）。
- **启动阶段**的 `_prime_voice_caches` 只覆盖**当时**注册表中的 active 条目；**服务已运行后**再上传的音色，不会自动参与该次启动预热。

**目标**

在管理 API **已成功**完成磁盘落盘与注册表写入后，**尽快**让**本机** TTS 进程内**每一个**模型副本（`--replicas`）对指定音色执行与启动预热**等价**的 **prime**（如 `_prepare_generation`，可选再 `next()` 一次流式首块），使 `_voice_prompt_cache` 中**提前**存在对应 `(str(ref_audio), ref_text, xvec_only, append_silence)` 条目，从而降低**首个真实业务合成请求**的延迟。

**推荐方案 A（管理端 HTTP 调用 TTS 内部接口）**

| 要点 | 说明 |
|------|------|
| **为何用 HTTP** | 管理 API 与 TTS 为**不同进程**、不共享内存，无法用「直接改对方进程里的 dict」实现；HTTP 为单机部署下实现简单、可观测、易排障的通道。 |
| **谁调用谁** | **`voice_manager_api_v3`（管理端）** 在 **`POST /voices` 成功**、以及 **`PUT /voices/{id}` 成功且更新了 `audio_file` 或 `ref_text` 时**（见 **4.1.5**）→ 向 TTS 发 **prime 请求**。同时，如果更新了音频或文本，服务端会自动递增 **`version`**。 |
| **TTS 侧** | **`openai_server_concurrent_v3`** 增加**不对外开放平台组**使用的 **内部路由**（例如 `POST /internal/v1/prime-voice`），校验鉴权后，对当前进程内 **`tts_models` 每个副本**执行与 **`_prime_voice_caches` 单条音色相同的逻辑**。 |
| **请求体** | **推荐**仅传 **`voice_id`**：TTS 用已有 **`HotVoiceRegistryV3`** 从本机注册表解析 `ref_audio` / `ref_text` / `language`，避免管理端与 TTS **路径或字段不一致**；也可扩展为显式传路径（用于极端排障）。 |
| **失败策略** | prime HTTP **超时或 5xx**：管理 API **只打日志**，**不回滚**已提交的注册表与文件；首个真实合成请求仍会**懒加载**写入缓存，**正确性不受影响**。 |
| **与合成 API 关系** | prime 为**旁路**；客户端仍只调 `POST /v1/audio/speech`，无需感知。 |

**部署范围（与当前选型一致）**

- **单机、单 TTS 进程**（进程内可有多个 replica）：配置**一个** TTS 基址（如 `http://127.0.0.1:8000`）即可；一次 prime 请求在 TTS 内**轮询所有副本**分别写入各自 `_voice_prompt_cache`。
- **多机 TTS**（多台机器各跑一份推理）：每台有**独立**缓存与注册表视图，方案 A 需对**每台**推理机各发 prime，或改用共享存储 + 各机轮询注册表增量 prime（复杂度高）；**本方案正文以单机为主**，多机见 **4.1.11**。

**多人同时上传（并发）**

- **注册表**：已由 **4.1.7** 所述 **独占锁 + 原子写** 保证多人同时保存时**串行**改写 JSON，**不会损坏文件**；不同 `voice_id` 的提交顺序一般**无业务语义要求**。
- **prime 请求**：多人同时上传会导致**短时间内多条** prime HTTP **并发**抵达 TTS；不同音色之间**无顺序依赖**。建议在 TTS 内部对 prime 处理增加**并发控制**（如 `asyncio.Lock` **串行**执行，或**限制并行数**），避免多条 prime 同时占满 GPU 显存导致抖动或 OOM——属**资源与体验**优化，**非**功能正确性前提。

**安全**

- 内部路由**禁止**无鉴权暴露在公网；优先 **仅监听 `127.0.0.1`** 的 admin 端口，或与主服务同端口但校验 **`Authorization: Bearer <token>`** / **`X-Internal-Token`** 等与 **`config_v3.json`（或管理端配置）一致**的共享密钥。
- 平台组调用的仍是**管理 API**；**prime URL 与 token 仅运维/配置文件可见**。

**配置项（建议，实现时落盘）**

- **管理端**：`tts_internal_base_url`、`enable_tts_prime_callback`（默认 `true`，便于关闭排障）、`tts_internal_prime_token`（可选，与 TTS 校验一致）。
- **TTS**：`internal_prime_enabled`、`internal_prime_token`（或与上项共用配置段）。

**与其余小节的关系**

- **4.1.6**：热加载解决「能看见新配置」；**本节**解决「尽量不等首条合成再填 `_voice_prompt_cache`」。
- **孤儿清理**：仍按推理结束后的阈值删除**孤儿** key；主动 prime **只增不减**，**不替代**孤儿清理。为最大程度保证业务请求性能，孤儿清理不再在每次普通合成请求结束后执行，而是改在 **4.1.6.1** 内部预热成功后顺带触发一次（因为此时注册表发生了变更，旧音色更可能变孤儿）。

### 4.1.7 管理 API 与并发写 JSON

- **建议保留** 独立管理 API 进程；实现文件建议为 **`voice_manager_api_v3.py`**（见 4.1.10），接口路径与 v2 **尽量一致**，降低平台组改造成本。写注册表与落盘成功后，**推荐**按 **4.1.6.1** 向本机 TTS 发内部 **prime** 请求（可配置开关）。
- **写文件必须**：  
  - **独占锁**：Linux 下 `fcntl.flock` / Windows 下 `msvcrt` 或 `portalocker`，保证读-写、写-写不交错。  
  - **原子替换**：写入 `voices.json.tmp` → `fsync` → `os.replace` 覆盖正式文件，避免写到一半进程崩溃导致 JSON 损坏。
- **删除音色**：软删除写 `status=disabled` 或从 JSON 移除 key；硬删除可删行 + 删音频文件。

### 4.1.8 配置与环境迁移

**精简 `config.json`（v3 示例形态）**：

- `voices_registry_path`：音色注册表 JSON 路径（绝对或相对工作目录）。
- `tone_wav_file_dir`：音频根目录。
- `allowed_audio_formats` / `max_audio_file_size_mb`：与管理端校验一致。
- **`enable_orphan_cache_cleanup`**（已定案）：是否启用 `_voice_prompt_cache` 孤儿清理逻辑，**默认 `true`**；设为 `false` 时仅依赖「新路径新 key」自然 miss，不主动删历史缓存条目（便于排障）。
- **`orphan_cache_cleanup_threshold`**（已定案）：触发孤儿清理的**启发式阈值**，**默认 `30`**。含义见 **4.1.9**。
- **（可选，见 4.1.6.1）管理端**：`tts_internal_base_url`、`enable_tts_prime_callback`、`tts_internal_prime_token` 等，用于写库成功后回调 TTS **prime**。
- **（可选，见 4.1.6.1）TTS 端**：内部 prime 路由开关与 token、prime **串行化**相关参数（实现时命名可与仓库统一）。

**迁移步骤**：在新环境放置同路径或使用配置指向新路径 → 拷贝注册表 JSON 与音频目录 → 启动管理 API（若有）与 TTS → 无需建库、配 Redis。

### 4.1.9 `_voice_prompt_cache` 孤儿缓存清理（已定案）

**背景**：参考音频更新若采用「新文件名 + 改注册表路径」，旧 `(ref_audio, ref_text, …)` 对应的缓存条目不再被访问，但仍占用 `FasterQwen3TTS._voice_prompt_cache` 内存；长期频繁更新可能使字典膨胀。

**实现层级（已定案）**：在 **TTS server 层**（如 `openai_server_concurrent_v3.py`），在 **4.1.6.1** 内部预热成功后执行清理逻辑；**遍历每个模型副本**上的 `model._voice_prompt_cache`，按规则删除**孤儿 key**，不在 `faster_qwen3_tts/model.py` 内改缓存结构（保持与现网 `model.py` 中 cache key 定义一致，与 `_resolve_voice_clone_prompt_from_reference` 使用的 `(str(ref_audio), ref_text, xvec_only, append_silence)` 对齐）。

**触发条件（启发式，已定案）**：

- 设 `N_cache = len(_voice_prompt_cache)`，`N_reg` = 当前注册表中 **`status=active`（或未写 status 视为 active）的音色条数**。
- 当 **`enable_orphan_cache_cleanup` 为 `true`** 且 **`N_cache - N_reg >= orphan_cache_cleanup_threshold`（默认 30）** 时，执行一次**孤儿清理**。

**清理规则（精确，已定案）**：

1. 根据**当前注册表快照**构造合法 key 集合 **`S`**：对每条 active 音色，按与 `model.py` **完全一致**的规则生成元组 `(str(ref_audio), ref_text, xvec_only, append_silence)`（其中 `xvec_only`、`append_silence` 与线上一致，通常为 `False`、`True`，以实际推理路径为准）。
2. 遍历 `_voice_prompt_cache` 的 key，**若 key ∉ `S`，则删除该条目**（仅删孤儿，不误删当前仍在注册表中的组合）。

**说明**：`N_cache - N_reg` 与「孤儿个数」不一定相等，仅作**低成本触发条件**；真正删除必须依赖 **`S` 集合比对**，避免误删。

**多副本**：若使用 `--replicas > 1`，每个副本各自一份 `_voice_prompt_cache`，**应对每个副本分别**做阈值判断与清理（或约定仅清理当前请求命中的副本，需在实现时与产品一致）。

### 4.1.10 代码与文件形态约定（`_v3`，已定案）

为保留 **v2（MySQL + Redis）** 可运行、可对照，**v3 相关实现不直接在现有 v2 源文件上原地修改**，约定如下：

- **新建文件**，或在现有文件上**复制一份后再改**，且 **v3 方案涉及的新增/派生文件，文件名末尾统一加 `_v3`**（示例：`openai_server_concurrent_v3.py`、`voice_manager_api_v3.py`、`voice_registry_v3.py`、`config_v3.json` 等；具体命名以实现时仓库结构为准）。
- **不在** `openai_server_concurrent_v2.py`、`voice_db.py`、`voice_manager_api.py`（无 `_v3` 后缀的 v2 版本）上直接改为 JSON 方案，避免影响已部署的 v2 环境。
- 若需与 `model.py` 交互，优先 **server 层** 访问 `model._voice_prompt_cache` 完成孤儿清理；**尽量不 fork** `faster_qwen3_tts/model.py`；确有必要扩展模型类时，同样建议**复制为** `…_v3` 或独立模块再引用，避免破坏包内默认行为。

### 4.1.11 风险与边界

- **多机 TTS**：若未来同一注册表挂载到多台推理机，需**共享存储**或**配置管理/下发**保证 JSON 与音频一致；单机场景无此问题。**若采用 4.1.6.1 的 prime**：每台推理机均需收到 prime 或使用等价机制，否则仅被通知到的机器具备热缓存。
- **内部 prime 接口暴露面**：必须配合 **127.0.0.1 / 防火墙 / Token**，避免未授权调用触发 GPU 负载或探测内网。
- **人工改 JSON**：允许，但需保证 UTF-8、合法 JSON；建议仍以管理 API 为主；手工改后若无管理端回调，TTS **不会**自动 prime，仍依赖首条合成懒加载。
- **超大规模**：若音色远超千级，再评估拆分为多文件或回到数据库方案。

### 4.1.12 后续实现清单（供排期）

1. 按 **4.1.10** 约定新增 **`_v3` 后缀**的注册表读写模块（锁 + 原子写），不修改 `voice_db.py` 行为（v2 专用）。  
2. **`voice_manager_api_v3.py`（示例名）**：CRUD 操作注册表 + 磁盘；健康检查改为「文件可读、目录可写」。  
3. **`openai_server_concurrent_v3.py`**：`resolve_voice` 使用 mtime 缓存加载注册表；预热遍历 `active` 条目；CLI 与 **`config_v3.json`（或统一 config 中 v3 段）** 对齐。  
4. 实现 **4.1.9** 孤儿清理：`enable_orphan_cache_cleanup`、`orphan_cache_cleanup_threshold`，在 server 层遍历各副本 `model._voice_prompt_cache`。  
5. （推荐）实现 **4.1.6.1**：管理 API 在成功新增/更新音色后 **HTTP 调用 TTS 内部 prime 接口**；TTS 对每副本预填 `_voice_prompt_cache`；配置 `tts_internal_base_url`、鉴权 token、TTS 侧 prime 并发控制。  
6. 文档与部署脚本：标注 v2 / v3 选型说明；**第三章保留为 v2 参考**，**第四章为 v3 目标方案**。

---

**小结**：v3 以 **JSON 注册表为唯一权威元数据**、**磁盘存音频**，用 **文件 mtime + 进程内缓存** 实现热加载，**去掉 MySQL 与 Redis**；**`_voice_prompt_cache` 孤儿清理**在 **server 层**、按 **注册表推导合法 key 集合** 精确删除，由 **`enable_orphan_cache_cleanup`（默认 true）** 与 **`orphan_cache_cleanup_threshold`（默认 30）** 控制触发；**推荐（4.1.6.1）** 在管理端写库成功后通过 **内部 HTTP** 触发 TTS **prime**，降低新音色**首条合成** TTFA，并在 TTS 侧对并发 prime 做**资源向**保护；**实现一律走 `_v3` 新文件/复制改**，不原地改 v2 现有文件。

## 4.2 使用指南

本节与 **3.2 使用指南** 结构对应，便于在同一仓库内对照 **v2（MySQL + Redis）** 与 **v3（JSON + 磁盘）** 的落地方式。

### 4.2.1 新增文件及功能说明

v3 不修改 `voice_db.py`、`voice_manager_api.py`、`openai_server_concurrent_v2.py`；以下为**新增或 v3 专用**文件及其职责。

| 路径（相对 `faster-qwen3-tts/`） | 类型 | 功能说明 |
|----------------------------------|------|----------|
| `config_v3.json` | 配置 | v3 专用：注册表路径、音频根目录、上传白名单与大小、孤儿缓存清理开关与阈值（见 **4.1.8**）。 |
| `voices_registry_v3.json` | 数据 | 音色注册表本体：顶层 `voice_id → { ref_audio, ref_text, language, status, version, updated_at }`；可由管理 API 自动创建/更新。 |
| `examples/voice_registry_v3.py` | 模块 | 注册表读写与热加载：`VoiceRegistryV3`（文件锁 + 原子 `os.replace` 写）、`HotVoiceRegistryV3`（按 mtime 失效缓存）、`maybe_cleanup_voice_prompt_cache`（与 `model.py` 一致的 cache key 集合比对）、`audio_path_for_version`（首版 `{id}.ext`，更新后 `{id}_vN.ext`）等。 |
| `examples/voice_manager_api_v3.py` | 服务 | 音色管理 HTTP API：与 v2 **同路径**（`/voices`、`/health` 等），底层改为 JSON `mutate`；健康检查为**注册表可读 + 音频目录可写**（无 MySQL/Redis）。 |
| `examples/openai_server_concurrent_v3.py` | 服务 | OpenAI 风格 TTS：由 v2 派生，`resolve_voice` 走热注册表；启动预热 `list_active_voice_cfgs()`；请求结束后按配置对各副本做 **4.1.9** 孤儿清理。 |

**运行期辅助文件**：

- 与注册表同目录会生成 **`voices_registry_v3.json.lock`**（与 `voices_registry_path` 配置一致），供 Linux 下 `flock` 协调读写；Windows 无 `fcntl` 时锁降级，多进程并发写注册表需谨慎。

### 4.2.2 文件结构

```
faster-qwen3-tts/
├── config_v3.json                    # v3 统一配置（无 mysql/redis 段）
├── voices_registry_v3.json           # 音色注册表（权威元数据）
├── examples/
│   ├── voice_registry_v3.py          # 注册表 + 热加载 + 孤儿清理工具（共用模块）
│   ├── voice_manager_api_v3.py       # 音色管理 API（v3）
│   ├── openai_server_concurrent_v3.py # TTS 推理服务（v3 热加载版）
│   ├── voice_db.py                   # 仅 v2 使用（v3 不引用）
│   ├── voice_manager_api.py          # v2 管理 API
│   └── openai_server_concurrent_v2.py
└── …
```

- **`voice_registry_v3.py`**：被 **管理 API v3** 与 **TTS v3** 共同引用，避免重复实现锁与 JSON 解析。
- **`voice_manager_api_v3.py`**：管理面，独立进程；负责写注册表与落盘音频。
- **`openai_server_concurrent_v3.py`**：推理面，**只读**注册表（通过 `HotVoiceRegistryV3`），不暴露写音色能力。

### 4.2.3 依赖安装

在已有 `faster-qwen3-tts` 推理环境（含 `fastapi`、`uvicorn`、`torch` 等）基础上，**无需**为 v3 额外安装 `pymysql`、`redis`。

若仍需使用 **v2** 与 **v3** 并存于同一虚拟环境，保留第三章中的数据库依赖安装即可；**仅部署 v3** 时可以不装上述二者。

### 4.2.4 配置文件与注册表

#### `config_v3.json`

与 **4.1.8** 一致，建议放在 `faster-qwen3-tts` 根目录，或通过 `--config` 指定任意路径。**`voices_registry_path`、`tone_wav_file_dir` 若为相对路径，均相对于「配置文件所在目录」解析**（与 v3 代码中 `_resolve_cfg_path` 行为一致）。

| 字段 | 说明 |
|------|------|
| `voices_registry_path` | 音色注册表 JSON 路径（如 `./voices_registry_v3.json`）。 |
| `tone_wav_file_dir` | 参考音频根目录；管理 API 上传落盘、注册表内 `ref_audio` 一般为该目录下绝对路径。 |
| `allowed_audio_formats` | 允许上传的扩展名白名单（小写，不含点亦可）。 |
| `max_audio_file_size_mb` | 单个上传文件大小上限（MB）。 |
| `enable_orphan_cache_cleanup` | 是否启用 `_voice_prompt_cache` 孤儿清理，默认 `true`（见 **4.1.9**）。 |
| `orphan_cache_cleanup_threshold` | 启发式触发条件：`len(cache) - N_active_registry >= 阈值`，默认 `30`；真正删除仍按注册表推导的合法 key 集合 **精确**比对。 |

仓库内示例：`tts-jiang/faster-qwen3-tts/config_v3.json`。

#### `voices_registry_v3.json`

- 顶层为对象：**键 = `voice_id`**（与请求体里 `voice` 一致），值见 **4.1.4**。
- 可为空对象 `{}`；首次启动管理 API 或 TTS v3 时会 `ensure_file_exists` 创建合法空文件。
- **手工编辑**需保证 UTF-8、合法 JSON；生产环境建议以管理 API 为主。

### 4.2.5 启动音色管理 API（v3）

```bash
cd faster-qwen3-tts
python examples/voice_manager_api_v3.py \
  --config config_v3.json \
  --host 0.0.0.0 \
  --port 8001
```

- 默认配置文件为 `examples/../config_v3.json`，也可用 `--config` 指向其他路径。
- 启动时会 `ensure_file_exists()` 注册表，并创建 `tone_wav_file_dir`（若不存在）。

### 4.2.6 音色管理 API 接口说明（v3）

接口一览（默认 `http://localhost:8001`，端口以实际参数为准）。**URL 与 HTTP 方法与 v2 对齐**，便于平台只改「基址 + 配置文件形态」。

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/voices` | 新增音色（multipart 上传音频） |
| `GET` | `/voices` | 列出音色（`?status=active` / `disabled`） |
| `GET` | `/voices/{voice_id}` | 查询单个 **active** 音色 |
| `PUT` | `/voices/{voice_id}` | 更新；**换音频时 `version` 递增并写入新文件名**（`audio_path_for_version`） |
| `DELETE` | `/voices/{voice_id}` | 软删除（`status=disabled`）或 `?hard=true` 硬删除 |
| `POST` | `/voices/{voice_id}/restore` | 软删除恢复为 `active` |
| `GET` | `/health` | `registry_readable`、`tone_dir_writable`（无 MySQL/Redis 字段） |

**与 v2 行为差异摘要**：

| 维度 | v2 | v3 |
|------|----|----|
| 写后缓存 | 需 `DEL` Redis key | 无 Redis；TTS 通过注册表 **mtime** 热加载 |
| 更新音频 | 落盘 + 更新 DB | 落盘到新路径（`v1` 为 `{id}.ext`，`v≥2` 为 `{id}_vN.ext`）+ 更新 JSON |
| 健康检查 | MySQL + Redis | 注册表可读 + 音频目录可写 |

#### 新增音色（示例）

```bash
curl -X POST http://localhost:8001/voices \
  -F "audio_file=@/path/to/ref_audio.wav" \
  -F "ref_text=欢迎您使用语音合成服务" \
  -F "language=Chinese" \
  -F "voice_id=my_voice_001"
```

表单字段与 **3.2.4** 相同：`audio_file`、`ref_text` 必填；`language` 默认 `Auto`；`voice_id` 可选（不传则生成 `str(uuid.uuid4())`）。

#### 更新音色

仅对 **active** 音色有效；更换 `audio_file` 时服务端 **自动递增 `version`** 并指向新路径，避免同路径覆盖导致模型 `_voice_prompt_cache` 仍命中旧内容（见 **4.1.5**）。

```bash
curl -X PUT http://localhost:8001/voices/my_voice_001 \
  -F "ref_text=新的参考文本内容"

curl -X PUT http://localhost:8001/voices/my_voice_001 \
  -F "audio_file=@/path/to/new_audio.wav" \
  -F "ref_text=新的参考文本内容"
```

#### 删除与恢复

```bash
# 软删除（仅标记禁用，保留文件）
curl -X DELETE http://localhost:8001/voices/my_voice_001

# 硬删除（彻底移除记录并清理磁盘文件）
curl -X DELETE "http://localhost:8001/voices/my_voice_001?hard=true"

# 恢复软删除
curl -X POST http://localhost:8001/voices/my_voice_001/restore
```

- **软删除**：注册表内 `status=disabled`；TTS 侧 `resolve_voice` 视为不可用。
- **硬删除**：移除注册表条目，并按 `{voice_id}.*`、`{voice_id}_v*.*` 在音频目录下尽量清理文件。**支持对已软删除的音色再次执行硬删除**，以实现从“禁用”到“彻底销毁”的二级管理。
- **恢复**：仅当当前为 `disabled` 时可恢复；**无需**像 v2 那样删除 Redis key。

#### 查询所有音色

```
curl -X GET http://localhost:8001/voices
```

#### 健康检查

```bash
curl http://localhost:8001/health
```

期望在注册表与目录均正常时返回 `status: ok`，并含 `registry_readable`、`tone_dir_writable`。

### 4.2.7 启动 TTS 推理服务（v3）

```bash
cd faster-qwen3-tts
python examples/openai_server_concurrent_v3.py \
  --config config_v3.json \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --replicas 1
```

**压低首包 TTFA（与 3.2.5 / 第二章一致，可选）**：

```bash
python examples/openai_server_concurrent_v3.py \
  --config config_v3.json \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

**v2 热加载版 与 v3 对比**：

| 项目 | v2（第三章） | v3（本章） |
|------|-------------|------------|
| 音色数据来源 | `config.json` → Redis + MySQL | `config_v3.json` → **JSON 注册表** + mtime 热加载 |
| 新增/修改音色 | 管理 API 写库 + 清 Redis | 管理 API 写 JSON；TTS **无需清 Redis**，下次请求或 mtime 变化即生效 |
| 启动预热 | MySQL `active` 列表 | `HotVoiceRegistryV3.list_active_voice_cfgs()` |
| 依赖 | `pymysql`、`redis`（管理端与推理端配置） | 无数据库驱动要求（仅 v3 路径） |

**CLI 说明**：

- **`--config`**：默认指向 `../config_v3.json`；环境变量 **`QWEN_TTS_CONFIG`** 可覆盖默认路径（与 v2 同名变量，部署 v3 时请指向 `config_v3.json`）。
- **与 v2 相同的参数**：`--model`、`--host`、`--port`、`--device`、`--concurrency`、`--replicas`、`--skip-warmup`、`--warmup-prefill-len`、`--skip-prime-voices`、`--prime-text`、`--prime-stream-first-chunk`、`--prime-stream-max-new-tokens`、`--no-metrics-log`。
- **孤儿清理**：在 **4.1.6.1** 内部预热成功后顺带触发一次清理（因为注册表可能发生了变更，旧音色变孤儿）；**不再**在每次普通合成请求结束后执行，以最大程度保证业务请求性能。

### 4.2.8 客户端调用 TTS（与 v2 / 并发版一致）

接口仍为 **`POST /v1/audio/speech`**，`voice` 填注册表中的 **`voice_id`**（JSON 顶层 key）。

**curl 与 `--output`**：与 **3.2.6** 相同——未指定 `-o`/`--output` 时，二进制响应可能触发 curl 退出码 23；保存文件请使用 `-o speech.wav` 等。

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，这是 v3 测试","voice":"my_voice_001","response_format":"wav"}' \
  -o speech.wav

curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"my_voice_001","response_format":"mp3"}' \
  -o speech.mp3
```

- `model` 字段兼容 OpenAI 形态；**实际权重以启动时 `--model` 为准**（说明同 3.2.6）。
- `response_format=wav` / `pcm`：**流式**；`mp3`：**非流式**（先合成再转码）。

### 4.2.9 典型部署流程（v3）

1. **准备配置**：编辑 `config_v3.json`（注册表路径、音频目录、格式与大小限制、孤儿清理参数）。
2. **放置数据**：拷贝 `voices_registry_v3.json` 与音频目录（或从空表开始仅拷贝配置）。
3. **启动管理 API**：`python examples/voice_manager_api_v3.py --config config_v3.json --port 8001`。
4. **录入音色**：`POST /voices` 上传参考音频与文本（除非注册表已由其他环境同步好）。
5. **启动 TTS**：`python examples/openai_server_concurrent_v3.py --config config_v3.json ...`；启动时会 `ensure_file_exists` 并对 active 条目预热（可用 `--skip-prime-voices` 关闭）。
6. **客户端调用**：`POST /v1/audio/speech`，`voice` = `voice_id`。
7. **后续变更**：仍通过管理 API 操作；**无需重启 TTS**，注册表变更经 mtime 被推理进程加载。

**环境迁移**：拷贝 **`config_v3.json` + `voices_registry_v3.json` + `tone_wav_file_dir` 下文件** 至新机器，修改配置中的绝对路径（若机器目录不同），启动两进程即可，**无需建库与 Redis**。

### 4.2.10 v2 与 v3 选型

| 场景 | 建议 |
|------|------|
| 已接公司 MySQL + Redis、多实例共享同一套音色 | **v2**（第三章） |
| 希望少依赖、单机或小规模、配置与数据「拷贝即走」 | **v3**（本章） |

### 4.2.11 `voice_registry_v3.py` 模块说明

管理 API 与 TTS **共用**本模块，核心符号如下。

| 符号 | 说明 | 典型调用方 |
|------|------|------------|
| `load_config_v3(path)` | 读取 JSON 配置文件 | 两服务 `main()` |
| `VoiceRegistryV3` | 独占/共享锁 + `read_all` / `mutate` / 原子写 | 管理 API |
| `HotVoiceRegistryV3` | 按注册表文件 mtime 重载；`resolve_active_voice`、`list_active_voice_cfgs`、`get_registry_copy` | TTS v3 |
| `valid_voice_prompt_cache_keys(registry)` | 由当前注册表构造合法 `(str(ref_audio), ref_text, False, True)` 集合 | 孤儿清理 |
| `maybe_cleanup_voice_prompt_cache(model, registry, …)` | 满足阈值时删除不在合法集合内的 `_voice_prompt_cache` 项 | TTS v3（每副本） |
| `audio_path_for_version(tone_dir, voice_id, ext, version)` | `v1` → `{id}.{ext}`，`v≥2` → `{id}_v{version}.{ext}` | 管理 API 更新音频 |
| `utc_now_iso()` | UTC ISO 时间戳（写入 `updated_at`） | 管理 API |

返回给推理逻辑的 `voice` 配置字段与 v2 / `voice_db` 一致：`voice_id`、`ref_audio`、`ref_text`、`language`、`status`、`version`，其中 `ref_audio` 为**本机绝对路径**字符串，对应 `voice_cfg["ref_audio"]`。

# 五、openai_server_v4.py 合并单进程版

在自己笔记本电脑上已经备份保存为`E:\Hdisk\学习\05-证通云上班\13_证通的AI相关项目\006_山东宁阳医院项目\002_tts流式与离线api\tts-jiang\faster-qwen3-tts_backup_20260414_171244`。

本章节描述将 **音色管理 API** 与 **TTS 推理服务** 合并为单一进程的方案（v4）。该方案旨在简化运维配置、节省系统资源，并利用进程内通信优化音色预热流程。

## 5.1 方案设计

### 5.1.1 背景与动机

- **运维简化**：平台组希望减少容器内运行的进程数量，仅需管理一个服务端口（如 `8000`），降低配置复杂度。
- **资源优化**：合并进程可微量节省 Python 解释器及基础库加载的内存开销。
- **业务匹配**：
  - 音色上传频率极低，通常仅在环境搭建或夜间低谷期进行。
  - 上传文件大小严格限制在 **20MB** 以内，对 CPU 和磁盘 I/O 的瞬时冲击在可控范围内。
  - 采用“单容器单 4090”部署模式，容器即为最小扩容单元，合并进程符合 Docker “一容器一进程”的哲学。

### 5.1.2 与 v3（多进程回调版）的对比

| 维度 | v3（多进程回调版） | v4（单进程合并版） |
|------|-------------------|-------------------|
| **进程数** | 2 个（TTS 服务 + 管理 API） | **1 个**（合并服务） |
| **监听端口** | 2 个（如 8000, 8001） | **1 个**（如 8000） |
| **预热通信** | 跨进程 HTTP 回调（需 Token） | **进程内函数直接调用** |
| **配置复杂度** | 需配置内部 Prime URL 与 Token | **无需** 内部通信配置 |
| **故障隔离** | 管理面与推理面物理隔离 | 一损俱损（但在单容器模式下影响较小） |

### 5.1.3 总体架构

```
平台组 / 客户端 ──HTTP (Port 8000)──▶ 合并服务进程 (openai_server_v4.py)
                                        │
                                        ├─ [APIRouter: 推理] /v1/audio/speech
                                        │
                                        └─ [APIRouter: 管理] /voices, /health, ...
                                                │
                                                └─ 成功更新注册表后 ──▶ [进程内调用] 触发模型预热 (Prime)
```

- **单端口多路径**：推理接口与管理接口共享同一个 FastAPI 实例，通过不同的 URL 路径区分功能。
- **单一真相**：依然以 JSON 注册表为权威数据源，由管理路由负责写，推理逻辑负责读。

### 5.1.4 逻辑组织与 APIRouter

为了保持代码整洁并兼容 v3 的开发成果，v4 采用 **模块化解耦** 策略：
- **`voice_manager_router_v4.py`**：将管理逻辑从独立服务改为 `APIRouter`。它不负责启动 Web 服务，仅定义路由。
- **`openai_server_v4.py`**：作为主入口，负责初始化沉重的 TTS 模型和 `HotVoiceRegistry`，并通过 `app.include_router()` 挂载管理路由。
- **状态共享**：管理路由通过 FastAPI 的依赖注入或全局变量访问主进程中的 `tts_models` 列表。

### 5.1.5 进程内预热 (Prime) 机制

- **触发流程**：管理路由在 `mutate` 成功写入 JSON 后，通过 `asyncio.create_task` 调用主进程注册的 **`prime_voice_v4(voice_id)`**（内部加锁、在线程池执行同步预热，再触发孤儿缓存清理）。
- **零网络开销**：彻底移除 `tts_internal_prime_url` 配置，消除回环网络请求的延迟与 Token 校验成本。
- **串行化保护**：即便在进程内，预热操作依然受 `asyncio.Lock` 保护，确保多个音色同时上传时 GPU 任务有序执行，不干扰正常推理。

### 5.1.6 稳定性与并发保护

- **文件限制**：严格执行 **20MB** 上传限制，防止大文件 I/O 导致进程卡顿。
- **异步非阻塞**：管理接口中的文件保存和清理操作必须在线程池（`run_in_executor`）中执行，确保不阻塞 FastAPI 的主事件循环，从而保证流式推理的平滑度。
- **避峰操作**：建议管理操作在业务低谷期进行，以最大程度降低对推理 TTFA 的潜在影响。

### 5.1.7 配置与文件形态约定 (_v4)

为确保版本兼容性，v4 方案的所有文件均在 v3 基础上复制并重命名，不直接修改 v3 文件。

| 文件名 | 职责 |
|------|------|
| `config_v4.json` | 统一配置文件。移除 `tts_internal_prime_url` 等网络回调配置。 |
| `openai_server_v4.py` | **主入口文件**。集成 FastAPI 应用、加载模型、挂载管理路由。 |
| `examples/voice_manager_router_v4.py` | **管理路由模块**。封装 `/voices` 等接口逻辑。 |
| `examples/voice_registry_v4.py` | 注册表操作模块（由 v3 复制）。 |

### 5.1.8 后续实现清单

1. **模块复制**：将 `voice_registry_v3.py` 复制为 `voice_registry_v4.py`。
2. **路由抽取**：将 `voice_manager_api_v3.py` 的逻辑重构为 `APIRouter` 形式的 `voice_manager_router_v4.py`。
3. **主服务整合**：创建 `openai_server_v4.py`，实现模型加载与路由挂载，并打通进程内 Prime 调用。
4. **配置更新**：提供精简后的 `config_v4.json` 示例。
5. **文档完善**：已补充紧随本方案设计之后的 **「使用指南」**小节（文件结构、配置表、启动命令与 API 说明）。

### 5.1.9 实现摘要与近期变更（维护说明）

本节汇总 v4 合并服务在落地过程中的**关键实现约定**与**已合入仓库的变更**，便于对照代码与配置。

#### 注册表模块命名（_v4）

- **`examples/voice_registry_v4.py`** 使用 **`load_config_v4`**、**`VoiceRegistryV4`**、**`HotVoiceRegistryV4`**，与文件名、导入方一致；避免在 v4 路径下仍使用 `_v3` 后缀符号。

#### 健康检查 `GET /health`

- 合并服务**仅通过管理路由**注册一条 **`GET /health`**（不在主应用重复注册），避免 FastAPI 下双路由冲突。
- **`voice_manager_router_v4.init_router`** 支持注入 **`model_loaded_fn`**（如 `lambda: tts_model is not None`），响应在 v3 管理端字段 **`registry_readable`、`tone_dir_writable`** 基础上增加 **`model_loaded`**，表示 TTS 权重是否已加载完成。

#### `config_v4.json` 扩展项（预热与首包体验）

除下列 **`config_v4.json` 基础字段**外，v4 还可选增与预热、日志相关的键：  
`voices_registry_path`（注册表 JSON 路径）、`tone_wav_file_dir`（参考音频根目录）、`allowed_audio_formats`（上传扩展名白名单）、`max_audio_file_size_mb`（单文件大小上限，单位 MB）、`enable_orphan_cache_cleanup`（是否启用 `_voice_prompt_cache` 孤儿清理，默认 `true`）、`orphan_cache_cleanup_threshold`（启发式触发阈值，默认 `30`；真正删除仍按注册表推导的合法 cache key 集合精确比对）。**相对路径**均相对于**配置文件所在目录**解析。

| 字段 | 说明 |
|------|------|
| **`prime_text`** | **代表性预热文本**。启动时对注册表内 **active** 音色批量预热、以及 **`POST/PUT` 写库成功后** `prime_voice_v4` 均使用该字符串调用 `_prepare_generation`；建议句长与业务常见请求接近，以缩小首次真实合成的 TTFA 与后续请求的落差。未配置或需极简开销时可为短句；未在 JSON 中提供该键时，回退为 `"."`。 |
| **`prime_stream_first_chunk`** | **`true`** 时，在 prepare 之后对同一文本再执行 **流式生成器首块**（`generate_voice_clone_streaming` 拉取一步），预热解码 / CUDA 路径；默认未写该键时为 `false`，与旧行为兼容。 |
| **`prime_stream_max_new_tokens`** | 流式首块预热时的 **`max_new_tokens`** 上限，默认 **`48`**。 |

#### `config_v4.json` 扩展项（推理摘要日志 `inference_logging`）

当配置 JSON **包含顶层键** `inference_logging` 时，`openai_server_v4.py` 在导入并加载 TTS 模型**之前**会调用 `_setup_inference_logging`，为专用 logger **`faster_qwen3_tts.inference`** 挂载处理器。以下输出走该通道（与模型加载、CUDA 图构建等 **`faster_qwen3_tts.model` 模块级日志**分离）：

- **`faster_qwen3_tts/model.py`**：每次非流式合成结束时的 **`Generated … ms/step, RTF: …`** 行，以及 **`Generation returned no tokens`** 等推理路径上的 **WARNING**。
- **`openai_server_v4.py`**：每请求 **`METRICS …`** 行（在未使用 **`--no-metrics-log`** 时），流式含 **`ttfa_wall_ms`**（产品口径 TTFA 墙钟）、**`ttfa_ms`**（首块模型 timing）、**`ttfa_cuda_graphs_ms`**（与 **`benchmarks/throughput.py`** 一致的 README CUDA Graphs TTFA 测法）、**`inter_chunk_max_ms` / `inter_chunk_p95_ms`**（块间入队间隔 proxy 连续性）、**`rtf`** 等；以及进程启动早期由 **`_log_service_startup_summary`** 写入的 **`STARTUP …`** 行（见下）。

`inference_logging` 为**对象**，常用子字段：

| 子字段 | 说明 |
|--------|------|
| **`console`** | `true` 时写入 **stderr**；默认 **`true`**。 |
| **`file`** | 非空字符串时写入磁盘日志；**`""`** 表示不写文件。**相对路径**相对于**配置文件所在目录**（与 `voices_registry_path` 等规则一致）。实际落盘路径是否带时间缀由 **`add_timestamp`** 决定；日志均以**追加**方式写入。 |
| **`add_timestamp`** | **`true`（默认）**：将 **`file`** 解析为绝对路径后，在**最后一级主文件名与扩展名之间**插入时间缀 **`_YYYY-M-D_HHMMSS`**（月、日不补零；时间为 6 位 `HHMMSS`），在**本次进程启动**调用 `_setup_inference_logging` 时计算一次，**每次重启**生成新文件。例如配置 `../logs/inference.log` 可能对应 `…/inference_2026-4-11_140346.log`。**`false`**：直接使用解析后的路径（如固定 **`inference.log`**），便于外部脚本向同一文件追加分隔标记或与旧有日志聚合；多实例共写同一文件时须自行注意并发与轮转。 |
| **`level`** | 如 `INFO`、`WARNING`；无法识别时回退 **`INFO`**。 |

**启动阶段写入推理摘要通道的内容**（在 **`_setup_inference_logging` 之后、挂载路由与加载模型之前**）：

- **`STARTUP cmdline=…`**：完整启动命令行（优先 **`shlex.join(sys.argv)`**，失败时退回空格拼接）。
- **若干行 `STARTUP --config=…`、`--model=…`、`--host=`、`--port=`、`--device=`、`--concurrency=`、`--replicas=`、`--prime-stream-first-chunk=`（含 CLI 与 **effective**）、`--prime-text(cli)=` / `(effective)=`、`prime_stream_max_new_tokens`（CLI 与 **effective**）、`--no-metrics-log`、`--skip-warmup`、`--warmup-prefill-len`、`--skip-prime-voices`**；若启用了文件日志，另有 **`STARTUP inference_log_file=`** 实际落盘绝对路径。
- 主模块 logger 另有一条 **`Inference log file: …`**（便于在常规控制台/容器日志中快速看到推理日志文件路径）。

**输出组合**：`console` 与 `file` 相互独立——可仅控制台、仅文件、或两者同时。若 **`console` 为 `false` 且 `file` 为空字符串**，则 **`faster_qwen3_tts.inference` 无 handler 且不向上传播**，运行期推理摘要（`Generated …`、`METRICS`）**不再输出**；但 **`STARTUP …` 启动摘要**会**回退**到 **`openai_server_v4` 模块 logger**（随 root 输出），避免启动信息被静默丢弃。METRICS 仍受 **`--no-metrics-log`** 约束（见下）。

**未配置该键时的兼容行为**：若 JSON 中**没有** `inference_logging` 顶层键，则**不**调用 `_setup_inference_logging`，`faster_qwen3_tts.inference` 保持默认传播；**`STARTUP …`、推理摘要（`Generated …`、`METRICS`）**仍随 **root** 输出（与合入该功能前的体验一致）。**`main()` 仍会调用** `_log_service_startup_summary`**，仅输出目标随上述传播链而定。

**与 `--no-metrics-log`**：命令行 **`--no-metrics-log`** 仍会**抑制整条 METRICS 日志**；**不**抑制 `Generated … RTF` 等模型侧推理摘要（后者仅受 `inference_logging` 与 `level` 影响）。

#### 命令行与配置文件的关系（`openai_server_v4.py`）

- **`--prime-text`**：若在命令行**显式传入**，则**覆盖** `config_v4.json` 中的 `prime_text`；**不传**则完全以配置文件为准（无键时用 `"."`）。
- **预热文本在进程内全局唯一**：`main()` 按上一条与配置解析出最终字符串后，写入代码中的全局 **`_v4_prime_text`**。**启动时**批量预热（`_prime_voice_caches`）与**服务运行期间**每次 **`POST /voices` / `PUT /voices` 写库成功后**触发的 **`prime_voice_v4`**，**共用这一份文本**（含通过 **`--prime-text xxx`** 传入的情形）。因此不会出现「仅启动用 CLI、上传仍读配置」的分裂；若需更换预热句长，应**修改 `config_v4.json` 或使用新的 `--prime-text` 后重启进程**。
- **`--prime-stream-first-chunk`**：与配置 **`prime_stream_first_chunk` 逻辑或**——任一为真即开启流式首块预热。
- **`--prime-stream-max-new-tokens`**：若指定则**覆盖**配置中的 `prime_stream_max_new_tokens`。

#### 预热代码结构（便于代码阅读）

- **`_run_prime_for_voice`**：单模型、单音色；**`_prepare_generation`** + 可选 **`generate_voice_clone_streaming` 首块**。
- **`_prime_voice_caches`**：服务启动时，对 **`hot_registry.list_active_voice_cfgs()`** 与每个模型副本批量调用上述逻辑。
- **`_prime_single_voice_sync` / `prime_voice_v4`**：管理端变更后异步触发；使用进程内全局 **`_v4_prime_text`、`_v4_prime_stream_first_chunk`、`_v4_prime_stream_max_tokens`**（在 `main()` 中由配置与 CLI 解析写入），与启动批量预热**参数一致**。

#### 源码注释风格

- **`examples/openai_server_v4.py`** 中主要函数、类与路由处理函数已采用与 **`voice_manager_api_v3.py`** 一致的多行 docstring：**标题句 + 空行 + 说明 + `Args` / `Returns` / `Raises` / `Yields`（异步生成器）** 等块。

### 5.1.10 指标口径：TTFA、RTF 与流式实时性（保障思路）

本节把**产品语义**与**服务端日志字段**对齐，并说明如何用日志保障「首包够快、流式够顺」；与 **§5.2.7 / 第六章 §5、§6** 中的阈值建议配套使用。

**TTFA（Time To First Audio，产品口径）**  
从客户端或网关**发起合成请求**（或等价地「系统已收到待合成文本、进入本服务处理路径」）到**第一段可播放音频可用**之间的**墙钟时间**。压测客户端常统计 **TTFB / 首包**；与服务端 **`ttfa_wall_ms`** 对比时，须约定双方**起点**是否一致（例如是否含网关排队、TLS）。

**服务端 METRICS 中的 `ttfa_wall_ms`**  

- **起点**：请求校验通过、分配 **`req_id`** 之后、**在进入合成槽位 `acquire()`（由 `--concurrency` 限制的在途合成数）之前**打点。因此该值**包含**本进程内因槽位占满导致的**排队等待**。若需观察「排除排队后的纯模型首包」，可对照 **`ttfa_ms`**（流式首块的模型侧 prefill + decode 累计，毫秒）。  
- **终点（流式）**：**首段 PCM 在服务端写入生产者队列（入队）**的时刻。**不是**「首字节已从网卡发出」，也**不是**在「先下发 WAV 头再下发 PCM」时以**WAV 头先出**作为首包——听感仍以**首段实质 PCM** 为准。

**RTF（本服务公式与解读）**  
$$
\mathrm{RTF} = \frac{\text{生成音频时长（秒）}}{\text{合成墙钟耗时（秒）}}
$$
**数值越大表示算得越快**（单位墙钟时间内产出更多秒音频）。日志中的 **`rtf`** 与模型侧 **`Generated … RTF`** 均按此含义理解。

**流式「连续 / 实时」：如何度量、局限与保障**  

- **服务端 proxy**：统计相邻音频块之间的**入队间隔**，聚合成 **`inter_chunk_max_ms`、`inter_chunk_p95_ms`**；**过大**往往更易感知卡顿。  
- **局限**：上述为**服务端入队间隔**，**不是**客户端 TCP 收包间隔或播放器消费间隔；网络缓冲与播放策略会改变听感。  
- **建议**：容量规划与回归以服务端 **max / p95** 为主，**最终听感**以客户端或业务侧实听（或播放器指标）校验。  
- **工程手段**：合理配置 **`prime_text`、`prime_stream_first_chunk`** 与 **`--concurrency` / `--replicas`**，并避免高峰期与同进程**大文件上传、批量 Prime** 与推理争抢 GPU（见 **5.1.5、5.1.6**），有利于压低 **`ttfa_wall_ms`** 并平滑块间间隔。

---

## 5.2 使用指南

以下列出 v4 合并服务所需的文件、配置项、启动命令、管理接口与客户端调用方式；**按本节即可独立完成部署与联调**，无需翻阅文档其他章节。

### 5.2.1 新增文件及功能说明

下列为 **`openai_server_v4.py` 合并单进程版** 落地时 **新增或修改** 的文件清单（相对 `faster-qwen3-tts/`）。**不改动** v3 既有入口文件（`openai_server_concurrent_v3.py`、`voice_manager_api_v3.py`、`voice_registry_v3.py` 等），v4 以独立路径并存。

| 路径 | 类型 | 功能说明 |
|------|------|----------|
| `config_v4.json` | **新增** 配置 | v4 专用：注册表路径、音频根目录、上传白名单与大小、孤儿缓存清理；另含 **`prime_text` / `prime_stream_first_chunk` / `prime_stream_max_new_tokens`**；可选 **`inference_logging`**，用于将 **`Generated … RTF`、`METRICS`、`STARTUP …`** 等输出到 stderr、日志文件或两者；其中 **`add_timestamp`** 控制是否在文件名中加 **`_YYYY-M-D_HHMMSS`**（默认加，规则见 **5.1.9**）。**不再使用** v3 双进程方案中的 **`tts_internal_prime_url`** 等内部 HTTP 回调项（预热改为进程内调用，见 **5.1.5**、**5.1.7**）。 |
| `voices_registry_v4.json` | **新增** 数据 | 音色注册表本体：顶层 `voice_id → { ref_audio, ref_text, language, status, version, updated_at }`；由挂载在同一进程内的管理路由创建/更新（可与示例一并提交，运行时由接口写入）。 |
| `examples/voice_registry_v4.py` | **新增** 模块 | 注册表与热加载：`VoiceRegistryV4`（文件锁 + 原子 `os.replace` 写）、`HotVoiceRegistryV4`（按注册表 mtime 失效缓存）、`load_config_v4`、`maybe_cleanup_voice_prompt_cache`（与 `model.py` 一致的 cache key 比对）、`audio_path_for_version`（首版 `{id}.ext`，更新后 `{id}_vN.ext`）等。 |
| `examples/voice_manager_router_v4.py` | **新增** 模块 | 音色管理逻辑以 **`APIRouter`** 形式提供，**不单独起服务**；由 `openai_server_v4.py` `include_router` 挂载。 |
| `examples/openai_server_v4.py` | **新增** 服务 | **唯一入口**：加载 TTS 模型、初始化注册表与 `HotVoiceRegistryV4`、挂载管理路由；于加载模型前**始终**写入 **`STARTUP`** 启动摘要（完整命令行与 `--config` / `--model` / 并发与预热相关等有效参数）；若配置了 **`inference_logging`**，该摘要与 **`Generated …` / METRICS** 走同一 logger（`faster_qwen3_tts.inference`），否则随 root 输出（见 **5.1.9**）。管理端写库成功后 **进程内** 调用 `prime_voice_v4` 预热，并在各模型副本上执行 **`maybe_cleanup_voice_prompt_cache`**：**仅在** `enable_orphan_cache_cleanup` 为真且 `len(_voice_prompt_cache) - N_active ≥ orphan_cache_cleanup_threshold` 等条件满足时，按注册表快照推导合法 key 集合并删除孤儿项；**不在**每条普通合成请求结束后执行。 |
| `faster_qwen3_tts/model.py` | **修改** 模块 | 增加专用 logger **`faster_qwen3_tts.inference`**（`_infer_log`），将 **`Generated … RTF`**、无 token 等 **WARNING** 与合并服务 **`inference_logging`** 配置对齐；未配置专用 handler 时仍随 root 传播，**v2/v3 入口行为保持兼容**。 |
| `docker-compose.yml` | **修改** 编排 | 在注释 **`command`** 中提供 **v4 合并服务** 启动示例（`python examples/openai_server_v4.py --config config_v4.json …`），便于容器内切换试验；默认 **`command` 未启用该示例** 时行为与改前一致。 |

**运行期辅助文件**：

- 与注册表同目录会生成 **`voices_registry_v4.json.lock`**（路径与 `voices_registry_path` 配置一致），供 Linux 下 `flock` 协调读写；Windows 无 `fcntl` 时锁降级，多进程并发写注册表需谨慎。

### 5.2.2 文件结构

```
faster-qwen3-tts/
├── config_v4.json                    # v4 统一配置（无内部 HTTP 回调项）
├── voices_registry_v4.json           # 音色注册表（权威元数据）
├── faster_qwen3_tts/
│   └── model.py                      # 配合 v4：推理摘要 logger（见 5.2.1 表）
├── docker-compose.yml                # 可选：注释中的 v4 启动示例（见 5.2.1）
├── examples/
│   ├── voice_registry_v4.py          # 注册表 + 热加载 + 孤儿清理工具（共用模块）
│   ├── voice_manager_router_v4.py    # 管理 APIRouter（v4）
│   ├── openai_server_v4.py           # 合并服务：推理 + 管理（v4）
│   ├── voice_registry_v3.py          # v3 仍保留，供双进程方案使用
│   ├── voice_manager_api_v3.py
│   └── openai_server_concurrent_v3.py
└── …
```

- **`voice_registry_v4.py`**：被 **管理路由 v4** 与 **合并服务** 共同引用。
- **`voice_manager_router_v4.py`**：只定义路由；**`openai_server_v4.py`** 负责进程级状态（模型列表、`HotVoiceRegistryV4`、`prime_voice_v4`）并通过 `init_router` 注入。

### 5.2.3 依赖安装

在已有 `faster-qwen3-tts` 推理环境（含 `fastapi`、`uvicorn`、`torch` 等）基础上，**无需**为 v4 额外安装 `pymysql`、`redis`。

### 5.2.4 配置文件与注册表

#### `config_v4.json`

建议放在 `faster-qwen3-tts` 根目录，或通过启动参数 **`--config`** 指向任意路径。

**基础字段**（与音色管理、孤儿清理、上传校验直接相关）：

| 字段 | 说明 |
|------|------|
| `voices_registry_path` | 音色注册表 JSON 路径（如 `./voices_registry_v4.json`）。 |
| `tone_wav_file_dir` | 参考音频根目录；管理接口上传落盘后，注册表内 `ref_audio` 一般为该目录下**绝对路径**。 |
| `allowed_audio_formats` | 允许上传的扩展名白名单（小写，可不含点）。 |
| `max_audio_file_size_mb` | 单个上传文件大小上限（MB）。 |
| `enable_orphan_cache_cleanup` | 是否启用 `_voice_prompt_cache` 孤儿清理，默认 `true`；`false` 时不主动删历史缓存条目（便于排障）。 |
| `orphan_cache_cleanup_threshold` | 启发式触发条件：`len(_voice_prompt_cache) - N_active_registry ≥ 阈值`，默认 `30`；**真正删除**仍须按注册表推导的合法 cache key 集合**精确**比对，避免误删。 |

**相对路径规则**：`voices_registry_path`、`tone_wav_file_dir` 及 `inference_logging.file` 等若为相对路径，均相对于**配置文件所在目录**解析（与代码中 `_resolve_cfg_path` 一致）。

**v4 额外字段**（预热与首包）：

| 字段 | 说明 |
|------|------|
| `prime_text` | 启动批量预热与上传/更新后 `prime_voice_v4` 使用的文本（进程内同一 `_v4_prime_text`）；建议贴近业务常见句长。未配置该键时回退为 `"."`。 |
| `prime_stream_first_chunk` | `true` 时在 `_prepare_generation` 之后对同一文本再拉取流式生成器**首块**，预热解码/CUDA 路径；默认未写为 `false`。 |
| `prime_stream_max_new_tokens` | 流式首块预热时的 `max_new_tokens` 上限，默认 `48`。 |

**可选对象 `inference_logging`**（仅当 JSON **存在顶层键** `inference_logging` 时生效；完整语义与边界情况见 **5.1.9**）：

| 子字段 | 说明 |
|--------|------|
| `console` | 是否将推理摘要（`Generated … RTF`、`METRICS`、`STARTUP …` 等）输出到 stderr，默认 `true`。 |
| `file` | 日志文件路径；空字符串表示不写磁盘；非空则**追加**写入（见 **5.1.9**）。相对路径规则同上文。 |
| `add_timestamp` | 是否为落盘文件名添加 **`_YYYY-M-D_HHMMSS`** 时间缀，默认 **`true`**；为 **`false`** 时使用配置路径原样落盘（如固定 `inference.log`）。 |
| `level` | 该 logger 级别，默认 `INFO`。 |

仓库内示例：`tts-jiang/faster-qwen3-tts/config_v4.json`。

#### `voices_registry_v4.json`

- **顶层结构**：JSON 对象，**键 = `voice_id`**（与 `POST /v1/audio/speech` 请求体中的 **`voice`** 一致），值为该音色的配置对象。
- **每个 `voice_id` 下建议字段**：`ref_audio`（本机绝对路径，指向参考音频）、`ref_text`（参考音频对应文本）、`language`（如 `Chinese` / `Auto`）、`status`（`active` / `disabled`，软删除）、`version`（整数，换音频时递增）、`updated_at`（ISO 时间字符串，可选）。仅 **`status=active`**（或未写 `status`，视为 active）的条目参与 TTS 解析与启动预热。
- 示例形态：

```json
{
  "my_voice_001": {
    "ref_audio": "/data/tts/tones/my_voice_001.wav",
    "ref_text": "欢迎您使用语音合成服务",
    "language": "Chinese",
    "status": "active",
    "version": 1,
    "updated_at": "2026-04-08T12:00:00"
  }
}
```

- 可为空对象 `{}`；启动时会 `ensure_file_exists` 创建合法空文件。
- 手工编辑须保证 UTF-8、合法 JSON；生产环境建议以 **`/voices`** 管理接口为主。

#### `METRICS` 行各字段含义（`openai_server_v4.py`）

在未使用 **`--no-metrics-log`** 时，每次合成结束会在 logger **`faster_qwen3_tts.inference`** 上打一条 **`INFO METRICS …`**（若配置了 **`inference_logging`**，则按 **5.1.9** 输出到 stderr 或落盘）。**`TTFA`、`RTF`、块间间隔的口径与边界**与 **§5.1.10** 一致；下表按字段解释单行日志中各项含义。

**示例（流式 `mode=stream`）**：

`INFO METRICS req_id=1 mode=stream ttfa_wall_ms=393.8 ttfa_ms=306.5 ttfa_cuda_graphs_ms=320.0 inter_chunk_max_ms=355.1 inter_chunk_p95_ms=354.9 rtf=3.366 audio_s=11.28 total_ms=3869.2 vram_used_gb=4.32 vram_total_gb=95.08`

| 字段 | 含义 |
|------|------|
| **`req_id`** | 本进程内为每次合成请求分配的递增编号，便于在多行日志中关联同一次调用。 |
| **`mode`** | 指标场景：`stream` 表示 **WAV/PCM 流式** 路径；非流式 MP3 为 **`non_stream_mp3`**（字段集合略少，见下）。 |
| **`ttfa_wall_ms`** | **墙钟 TTFA（服务端锚点）**：从「校验通过、已分配 `req_id`、尚未进入合成槽位 `acquire()`」到 **首段 PCM 在服务端入队** 的毫秒数；**含** `--concurrency` 排队；**不是**网卡首字节时刻，**不是**以 WAV 头为先。 |
| **`ttfa_ms`** | **首块模型耗时**：流式路径下首段有效输出时，累计的 **prefill + decode**（毫秒），用于与 `ttfa_wall_ms` 对照以区分排队与纯推理。 |
| **`ttfa_cuda_graphs_ms`** | （**仅 `mode=stream`**）与 **`README.md` / `benchmarks/throughput.py`** 中 **CUDA Graphs 流式 TTFA** 同一测法：**`torch.cuda.synchronize()` → `perf_counter` 起点 → `generate_voice_clone_streaming(...)`（`chunk_size` 与配置 **`stream_chunk_size`** 一致）→ 取首块 → **`torch.cuda.synchronize()`** → 再 `perf_counter` 得毫秒。用于与官方 benchmark 表格对比；**不含** HTTP 层排队（起点在生产者线程内、调用流式 API 之前）。 |
| **`inter_chunk_max_ms`** | （**仅 `mode=stream`**）相邻音频块 **服务端入队** 间隔的 **最大值**（毫秒）；过大往往更易感知卡顿。 |
| **`inter_chunk_p95_ms`** | （**仅 `mode=stream`**）上述间隔的 **95 分位**（毫秒）。 |
| **`rtf`** | **实时率**：本实现为 **生成音频时长（秒）÷ 模型侧累计合成耗时（秒）**（流式用各步 `prefill_ms`/`decode_ms` 之和换算）；**数值越大表示算得越快**。 |
| **`audio_s`** | 本段合成对应的 **音频时长（秒）**（由输出块长度与采样率折算累加）。 |
| **`total_ms`** | **流式**：流式生产者线程从开始进入生成循环到 **`finally` 打点** 的墙钟毫秒（与纯模型累计毫秒不完全等同）。**非流式 MP3**：近似整段合成墙钟。 |
| **`vram_used_gb` / `vram_total_gb`** | 当前 GPU **已用 / 总显存（GB）**（优先 NVML，不可用时部分字段可能仅来自 PyTorch 等，见实现中 `_gpu_stats()`）。 |

##### `ttfa_wall_ms`、`ttfa_ms`、`ttfa_cuda_graphs_ms`：白话说法与对比

下面三句都出现在同一条流式 **`METRICS`** 里，但**回答的问题不一样**；读懂差别后，排障时就不容易把「排队慢」和「GPU 慢」混在一起。

**`ttfa_wall_ms`（墙钟、偏「用户在这条请求上等了多久」）**  
从服务里「这条合成请求已经验过、也领到号了，但**还没抢到合成槽位**（还没进 `acquire()`）」开始计时，一直到「**第一段能拿去下发的 PCM 数据在服务端已经算好并进队**」为止，一共多少毫秒。**别人占满槽位时你在外面排队的时间，会计进这个数。** 它**不是**「包已经到你电脑网卡」的时间；若把「能播」理解成「用户耳机里已经出声」，那通常还要再加网络和播放器，**比这个数更晚**。

**`ttfa_ms`（模型自己报的、偏「首段算力花了多少」）**  
流式里**第一次产出**那一段时，把模型返回的 **`prefill_ms` 与 `decode_ms` 加起来**得到的毫秒数。你可以把它理解成：**首段输出在「模型 timing 账本」里记了多久**。它**一般不含**你在 **`--concurrency` 上排队**的时间；若 **`ttfa_wall_ms` 明显大于 `ttfa_ms`**，多半要怀疑**排队**或**进生成循环之前**还有别的工作。

**`ttfa_cuda_graphs_ms`（与官方 benchmark 同一种掐表方式）**  
在**后台生产者线程里**，按 **`benchmarks/throughput.py`** 的做法：先 **`cuda.synchronize`**，再掐表，再调用 **`generate_voice_clone_streaming`**，拿到**第一块**后再 **`cuda.synchronize`**，再掐表。得到的毫秒数用来和 **`README.md` 里 CUDA Graphs TTFA 表格**对齐；**起点在「开始调流式 API」之前**，**不含** HTTP 里 **`acquire()` 之前的排队**（那段体现在 **`ttfa_wall_ms`** 里）。**`chunk_size`** 与配置 **`stream_chunk_size`** 一致；要和 README 表中 **PRIMARY_CHUNK_SIZE=8** 那一列严格可比时，宜将 **`stream_chunk_size` 设为 8** 并在相近输入与 GPU 上对比。

**一句话对比（谁大谁小常见吗？）**

| 指标 | 更像在回答 |
|------|------------|
| **`ttfa_wall_ms`** | 「从**能进队合成**到**首段 PCM 已备好**，用户侧在服务端要等多久？」——**含排队**。 |
| **`ttfa_ms`** | 「**首段**在模型 timing 里 **prefill+decode** 记了多少？」——**不含排队**。 |
| **`ttfa_cuda_graphs_ms`** | 「按 **README/benchmark** 同一种掐表，**首块流式输出**用了多少墙钟毫秒？」——**对齐表格用**；**不含 `acquire()` 前排队**。 |

常见关系（非定理，仅供直觉）：**`ttfa_wall_ms` ≥ `ttfa_cuda_graphs_ms`** 的情况很多（墙钟起点更早）；**`ttfa_cuda_graphs_ms` 与 `ttfa_ms`** 同一「首段」相关，但前者是 **CUDA 同步后的墙钟包一圈**，后者是 **timing 字典累加**，**数值不必相等**。

**`mode=non_stream_mp3`** 时，同一行含 **`ttfa_wall_ms`、`ttfa_ms`、`rtf`、`audio_s`、`total_ms`** 及 **`vram_*`**，**不含** **`inter_chunk_*`、`ttfa_cuda_graphs_ms`**（无分块流式测法）。

**实现位置（便于对照代码）**：合并服务在 **`examples/openai_server_v4.py`** 的 **`_log_metrics`** 中拼接并 **`logging.info`** 输出；流式场景由 **`_stream_chunks`** 内生产者线程的 **`finally`** 在整段流结束后调用 **`_log_metrics(..., "stream", {...})`**；**`ttfa_cuda_graphs_ms`** 在同一生产者内按 **`benchmarks/throughput.py`**（约 50–61 行）顺序计时。

### 5.2.5 启动合并服务（v4）

**仅需一个进程、一个端口**，同时提供 **`POST /v1/audio/speech`**（OpenAI 兼容合成）与 **`/voices`、`/health`** 等管理路径（HTTP 路径与方法与常见独立管理 API 实现对齐，便于客户端只改基址与端口）。

```bash
cd faster-qwen3-tts
python examples/openai_server_v4.py \
  --config config_v4.json \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --replicas 1
```

#### `--concurrency` 与 `--replicas`：合成槽位与模型副本

二者都影响「能扛多少合成」，但层次不同：**`--concurrency` 限制同一时刻有多少条请求在跑合成逻辑**；**`--replicas` 决定在进程里加载几份完整模型、合成请求轮流打到哪一份上**。

**`--concurrency`（并发槽位）**

- 实现上对应进程内一个 **`asyncio.Semaphore`**，仅夹在 **`POST /v1/audio/speech`** 进入真正合成（如 `generate_voice_clone` / 流式生成）的路径上。
- **当前正在合成的条数少于 `concurrency`**：新请求会立刻拿到许可，**并行**执行（最多同时 `concurrency` 条在跑合成）。
- **已达到 `concurrency` 条都在合成**：后续请求在 **`await ... acquire()`** 处**等待**，相当于排队；某条请求在 `finally` 里 **`release()`** 后，下一个等待者才继续。
- **统计范围**：是**整台该服务进程**、**所有客户端加在一起**的在途合成数，**不是**「每个客户端各 `concurrency` 条」。
- **管理类接口**（如 **`/voices`、`/health`**）**不走**该信号量，不受 `--concurrency` 限制。

**`--replicas`（模型副本数）**

- 启动时用 `args.replicas` 创建 **`replicas` 个** 独立的 **`FasterQwen3TTS`** 实例（每个都是完整 `from_pretrained`），并分别做 CUDA 预热、启动批量 prime 等；推理时通过 **`_pick_model()`** 在副本间**轮询**选取本次请求使用的实例。
- **作用**：把「已通过 `--concurrency` 放行、正在合成」的请求，**分散到多个模型对象上**执行，而不是永远只打在一个实例上；是否带来吞吐或时延收益，与实现、驱动、单卡多上下文等因素有关，**需以本机压测为准**。
- **显存**：当前实现中各副本默认共用启动参数 **`--device`**（例如同一张 GPU），一般可按 **约 `replicas` 倍单模型显存** 估算占用（是否完全线性取决于框架与共享情况，以 `nvidia-smi` 为准）。
- **与 `--concurrency` 的关系**：信号量**仍然只有一个**，**全进程同时在途合成条数上限仍是 `concurrency`**；`replicas` **不会**把该上限乘以副本数。可以理解为：**「最多允许多少条并行合成」由 `concurrency` 决定；「这些并行合成轮流使用哪几个模型对象」由 `replicas` 决定。**

- **`--config`**：默认 `examples/../config_v4.json`（即仓库根下 `config_v4.json`）；可指向任意路径。
- **常用启动参数**：`--model`（HuggingFace 模型名或本地路径）、`--host`、`--port`、`--device`、`--concurrency`（同时在途合成上限）、`--replicas`（进程内模型副本数）、`--skip-warmup`、`--warmup-prefill-len`、`--skip-prime-voices`、`--prime-text`（**未传**则使用 `config_v4.json` 的 `prime_text`，无键时为 `"."`；**一旦在 `main()` 中解析确定**，启动批量预热与后续 **`POST/PUT /voices`** 成功后的进程内 **`prime_voice_v4` 共用该字符串**，直至重启）、`--prime-stream-first-chunk`（与配置项 **`prime_stream_first_chunk` 逻辑或**，任一为真即开启流式首块预热）、`--prime-stream-max-new-tokens`（指定则覆盖配置中的 `prime_stream_max_new_tokens`）、`--no-metrics-log`（**仅**关闭每请求 **`METRICS …`** 行；**不**关闭 **`Generated … RTF`**，后者由是否存在 **`inference_logging`** 及 `console`/`file`/`level` 决定，见 **5.1.9**）。
- **推理摘要日志**：在 `config_v4.json` 中配置对象 **`inference_logging`** 后，可将 **`Generated … RTF`**、**METRICS**、启动 **`STARTUP …`** 按 **`console` / `file` / `level`** 输出到 stderr、日志文件或两者；**`add_timestamp`**（默认 **`true`**）决定落盘文件名是否带 **`_YYYY-M-D_HHMMSS`**；**不含该顶层键**时不改专用 logger，上述内容随 root 默认行为输出（见 **5.1.9**）。
- **启动顺序（摘要）**：解析 **`--config`** 并 `ensure_file_exists` 注册表、构建 **`HotVoiceRegistryV4`** 快照、合并 **`prime_*`** 与 CLI 后，若存在 **`inference_logging`** 则初始化推理 logger（含可选落盘路径）；随后写入 **`STARTUP cmdline=`** 与关键参数行；再 **`init_router` / `include_router`**，加载模型、CUDA 预热与可选批量 **prime**；**无需**再单独启动 `voice_manager_api_v3.py`。

**压低首包 TTFA（推荐）**：

1. **优先在 `config_v4.json` 配置** `prime_text`（代表性长句）、`prime_stream_first_chunk: true`、`prime_stream_max_new_tokens`（可选），使启动批量预热与上传后 **`prime_voice_v4`** 与业务句长一致。
2. 或在命令行**临时覆盖**，例如：

```bash
python examples/openai_server_v4.py \
  --config config_v4.json \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

**孤儿清理**：在 **进程内预热**（新增/更新音色后的 `prime_voice_v4`）结束时对**各模型副本**调用 **`maybe_cleanup_voice_prompt_cache`**；**不在**每条普通合成请求结束后执行，以免拖慢在线合成路径。

### 5.2.6 音色管理 API 接口说明（v4）

接口一览（默认 **`http://localhost:8000`**，端口与 `--port` 一致）。**URL 与 HTTP 方法**与常见「独立进程管理 API」实现对齐（如 `POST/GET/PUT/DELETE /voices`、`GET /health` 等）。

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/voices` | 新增音色（multipart 上传音频） |
| `GET` | `/voices` | 列出音色（`?status=active` / `disabled`） |
| `GET` | `/voices/{voice_id}` | 查询单个 **active** 音色 |
| `PUT` | `/voices/{voice_id}` | 更新；换音频时 `version` 递增并写入新文件名 |
| `DELETE` | `/voices/{voice_id}` | 软删除或 `?hard=true` 硬删除 |
| `POST` | `/voices/{voice_id}/restore` | 软删除恢复为 `active` |
| `GET` | `/health` | `registry_readable`、`tone_dir_writable`；合并服务另含 **`model_loaded`**（TTS 模型是否已就绪） |

#### 各接口正常 HTTP 状态码（成功路径）

下列为 **`openai_server_v4.py` 合并服务**在**业务成功、无异常**时返回的 **HTTP 状态码**及含义（与 **`voice_manager_router_v4.py` + `create_speech`** 实现一致）。**校验失败、资源不存在、模型未就绪**等会返回 **4xx / 5xx**，见各接口实现中的 `HTTPException`，本节不展开。

| 方法 | 路径 | 正常状态码 | 说明 |
|------|------|------------|------|
| `GET` | `/health` | **200** | **HTTP 始终为 200**（路由不根据健康度改状态码）。响应 JSON 中 **`status`**：`"ok"` 表示注册表可读且音频根目录可写；**`"degraded"`** 表示至少一项失败，详见同级的 **`registry_error` / `tone_dir_error`** 等字段。**`model_loaded`**（布尔）表示 TTS 权重是否已加载，供编排与探活参考。 |
| `GET` | `/voices` | **200** | 成功返回列表 JSON：`count`、`voices`。 |
| `GET` | `/voices/{voice_id}` | **200** | 成功返回该 **active** 音色的配置对象。 |
| `POST` | `/voices` | **201 Created** | 成功新增音色；响应体为 JSON（如 **`success`、`voice_id`、`voice`**）。与其它返回 JSON 的管理接口区分，采用 **201** 表示**资源已创建**。 |
| `PUT` | `/voices/{voice_id}` | **200** | 成功更新，或**无字段实际变更**时同样 **200**（响应中含 **`message": "无变更"`** 等）。 |
| `DELETE` | `/voices/{voice_id}` | **200** | 软删除（`hard` 默认 `false`）或硬删除（**`?hard=true`**）成功；响应体标明 **`hard`: true/false**。 |
| `POST` | `/voices/{voice_id}/restore` | **200** | 软删除条目成功恢复为 **active**。 |
| `POST` | `/v1/audio/speech` | **200** | 合成成功：**`response_format` 为 `wav` / `pcm`** 时为 **`StreamingResponse`**（**200**），**`Content-Type`** 分别为 **`audio/wav`** / **`audio/pcm`**，体为**流式**二进制（`wav` 先写头再 PCM）；**`mp3`** 时为**非流式**整段 **`Response`**（**200**，**`Content-Type`: `audio/mpeg`**）。 |

**说明**：

- 管理类接口成功时响应体均为 **JSON**（除上述语义外，字段以实际返回为准）。  
- **`GET /health`** 的 **「正常 TCP/HTTP 可达」**与 **「依赖项全部健康」**需区分：探活若要求「可合成」，应同时判断 **`model_loaded`** 与 JSON **`status == "ok"`**（按业务约定）。  
- 合成接口 **`POST /v1/audio/speech`** 的详细请求体与流式行为见 **§5.2.7**。

**与 v3 管理进程的差异**：

| 维度 | v3（独立管理 API） | v4（合并服务） |
|------|-------------------|----------------|
| 基址 | 常为 `http://host:8001` | **与推理同端口**（如 `http://host:8000`） |
| 预热 | 可选 HTTP 回调 TTS | **进程内** `prime_voice_v4`，无 Token / 无回环 URL |

#### 新增音色（示例）

```bash
curl -X POST http://localhost:8000/voices \
  -F "audio_file=@/path/to/ref_audio.wav" \
  -F "ref_text=欢迎您使用语音合成服务" \
  -F "language=Chinese" \
  -F "voice_id=my_voice_001"
```

**`POST /voices` 表单字段**：`audio_file`、`ref_text` **必填**；`language` 默认 `Auto`；`voice_id` 可选（不传则服务端生成 UUID）。**`PUT /voices/{voice_id}`** 可只更新文本，或同时上传新 `audio_file`（换音频时 **`version` 递增**并写入新文件名，避免同路径覆盖导致缓存仍命中旧内容）。

#### 更新音色

```bash
curl -X PUT http://localhost:8000/voices/my_voice_001 \
  -F "ref_text=新的参考文本内容"

curl -X PUT http://localhost:8000/voices/my_voice_001 \
  -F "audio_file=@/path/to/new_audio.wav" \
  -F "ref_text=新的参考文本内容"
```

#### 删除与恢复

```bash
curl -X DELETE http://localhost:8000/voices/my_voice_001
curl -X DELETE "http://localhost:8000/voices/my_voice_001?hard=true"
curl -X POST http://localhost:8000/voices/my_voice_001/restore
```

#### 查询与健康检查

```bash
curl -X GET http://localhost:8000/voices | jq
curl http://localhost:8000/health | jq
```

期望在注册表、音频目录与模型均正常时：`status` 为 `ok`，且含 `registry_readable`、`tone_dir_writable`、`model_loaded: true`。

### 5.2.7 客户端调用 TTS（v4）

- **正常 HTTP 状态码**：合成成功时为 **200 OK**（`wav`/`pcm` **流式**、`mp3` **整包**均为 200）。与 **`GET /health`**、管理类接口的状态码一览见 **§5.2.6「各接口正常 HTTP 状态码」**。
- **接口**：`POST /v1/audio/speech`，`Content-Type: application/json`。
- **请求体**：`input`（待合成文本）、`voice`（须为 **`voices_registry_v4.json` 顶层键**，即 **`voice_id`**）、`response_format`（`wav` / `pcm` 为**流式**，`mp3` 为**非流式**整包）；`model` 字段兼容 OpenAI 形态，**实际权重以启动时 `--model` 为准**。
- **基址**：v4 下合成与管理**同端口**（例如 `http://host:8000`），无需再配置第二管理端口。

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，这是 v4 测试","voice":"my_voice_001","response_format":"wav"}' \
  -o speech.wav

curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"my_voice_001","response_format":"mp3"}' \
  -o speech.mp3
```

- `model` 字段兼容 OpenAI 形态；**实际权重以启动时 `--model` 为准**。
- `response_format=wav` / `pcm`：**流式**；`mp3`：**非流式**。
- 使用 **curl** 保存音频二进制时请加 **`-o 输出文件路径`**，避免把二进制打印到终端。

### 5.2.8 典型部署流程（v4）

1. **准备配置**：编辑 `config_v4.json`（填写上文 **`#### config_v4.json`** 中的基础字段表，并按需增加 **`prime_*` / `inference_logging`**）；若需将 **`STARTUP` / METRICS / RTF** 写入磁盘，在 **`inference_logging`** 中设置非空 **`file`**、按需 **`console`** / **`level`** / **`add_timestamp`**（默认 **`true`** 时在文件名中加 **`_YYYY-M-D_HHMMSS`**；设为 **`false`** 可固定写入如 **`inference.log`**），并确保进程对日志目录有写权限。
2. **放置数据**：准备 `voices_registry_v4.json` 与音频目录，或从空表开始。
3. **启动合并服务**：`python examples/openai_server_v4.py --config config_v4.json ...`。
4. **录入音色**：`POST /voices`（multipart：**`audio_file`、`ref_text` 必填**；`language` 默认 `Auto`；`voice_id` 可选；**端口与合成接口相同**）。
5. **客户端调用**：`POST /v1/audio/speech`，`voice` = `voice_id`。
6. **后续变更**：继续通过 **`/voices`** 管理；**无需重启**，注册表变更经 mtime 被 `HotVoiceRegistryV4` 加载；写库成功后会 **异步进程内预热** 对应音色。

**环境迁移**：拷贝 **`config_v4.json` + `voices_registry_v4.json` + `tone_wav_file_dir` 下文件**，调整路径后**只启动** `openai_server_v4.py` 即可。

### 5.2.9 v3 与 v4 选型

| 场景 | 建议 |
|------|------|
| 希望管理面与推理面**进程隔离**、不同端口或不同扩容策略 | **v3**：独立 **`voice_manager_api_v3.py`** + **`openai_server_concurrent_v3.py`**，通常两个监听端口 |
| 单容器单卡、**单端口**、少配置、接受管理操作与推理同进程 | **v4**：本章 **`openai_server_v4.py` 合并服务** |

**生产压测约定**：单卡上线前须做**逐步提并发**压测（固定模型、`voice` 与文本长度，观察 TTFA、p95 总耗时、RTF、错误率与显存），再选定 **`--replicas` / `--concurrency`** 与客户端并发上限；压测对象应为 **`openai_server_v4.py` + `config_v4.json`** 合并服务，启动方式与上文 **「启动合并服务（v4）」** 中的 `python examples/openai_server_v4.py` 多行示例命令一致。

### 5.2.10 `voice_registry_v4.py` 模块说明

合并服务内的管理路由与推理逻辑 **共用** 本模块，核心符号如下。

| 符号 | 说明 | 典型调用方 |
|------|------|------------|
| `load_config_v4(path)` | 读取 v4 JSON 配置文件（可含 `prime_text`、`inference_logging` 等扩展键） | `openai_server_v4.py` `main()` |
| `VoiceRegistryV4` | 锁 + `read_all` / `mutate` / 原子写 | 管理路由、`main()` 中 `ensure_file_exists` |
| `HotVoiceRegistryV4` | 按注册表 mtime 重载；`resolve_active_voice`、`list_active_voice_cfgs`、`get_registry_copy` | 合并服务推理路径 |
| `valid_voice_prompt_cache_keys(registry)` | 合法 `_voice_prompt_cache` key 集合 | 孤儿清理 |
| `maybe_cleanup_voice_prompt_cache(model, registry, …)` | 阈值满足时清理孤儿缓存项 | `prime_voice_v4` 之后（各副本） |
| `audio_path_for_version(...)` | 版本化音频路径规则 | 管理路由 |
| `utc_now_iso()` | UTC ISO 时间戳 | 管理路由 |

解析 **`voice_id`** 后，返回给推理逻辑的 `voice` 配置字典至少包含：`voice_id`、`ref_audio`（**本机绝对路径** 字符串）、`ref_text`、`language`、`status`、`version` 等，供 `voice_cfg["ref_audio"]` 等字段使用。

# 六、分时复用让多路 decode 交错在GPU上执行

单卡 **RTX 4090** 上，当前 **`openai_server_v4.py`** 用 **`--concurrency`** 允许多条请求**同时进入合成路径**（流式为**多线程**各自跑 **`generate_voice_clone_streaming`**），但 GPU 上多为**算力与显存带宽争用**，**并非**「按 decode 步精细分时、多会话公平穿插」的调度。结果是：**并发抬高时 `ttfa_wall_ms`（含排队）、`rtf`（往往下降）与块间间隔（`inter_chunk_*`）往往变差**，与 **§5.1.10** 中「单卡多上下文」的讨论一致。

本章给出在**同时照顾三类指标**——**TTFA**、**RTF（本服务定义下越大越快）**、**流式 chunk 间隔（不宜过大）**——的前提下，提高可承载会话数或改善调度的**可行方案设计**（偏架构与迭代路线）；**落地需改推理栈或服务编排**，与当前仓库实现并非一一对应，实施前须做原型验证。

## 6.1 方案设计

### 6.1.1 目标与约束

| 维度 | 说明 |
|------|------|
| **业务目标（多指标）** | **① TTFA**：**`ttfa_wall_ms`**（及可选 **`ttfa_ms` / `ttfa_cuda_graphs_ms`**）控制在产品阈值内（如 p95 ≤ 1000ms，以 **§5.1.10 / §七** 为准）。**② RTF**：本实现中 **RTF = 生成音频时长（秒）÷ 合成墙钟（秒），数值越大表示算得越快**；在既定并发与调度下希望 **RTF 尽可能大**（并与 **§七** 经验阈值如平均 ≥ 10 等对齐）。**③ 流式连续性**：相邻块在服务端**入队**间隔不宜过大，以 **`inter_chunk_max_ms`、`inter_chunk_p95_ms`**（**METRICS**）为 proxy，并与客户端听感交叉验证。 |
| **指标张力** | **提高并发**或**强分时穿插**常伴随 **RTF 下降**、**chunk 间隔拉大**（多路争用 GPU）；**保 RTF 与块间密**又往往要**限并发**或接近**独占 decode**。须在**三维上联合验收**，不能只优化其一。 |
| **「分时交错」含义** | 理想情况：多路自回归 decode 在 GPU 时间轴上**穿插推进**（或**等价的批处理吞吐**），避免**一路长跑占满**导致其它路首包长期饿死；且穿插下仍尽量**维持每路解码节奏**，避免 **inter_chunk** 失控。 |
| **硬约束** | **自回归**同一序列内 **step 依赖 step**；两路不同文本**不能**在无批处理/无状态拆分的前提下「任意交错到同一组算子的一次前向里」。交错要么靠 **多 CUDA stream 上的 kernel 重叠**，要么靠 **显式多步调度**，要么靠 **batch 维合并多序列**。 |
| **与本项目实现相关** | **`faster-qwen3_tts`** 的 CUDA Graphs 快路径面向**单序列流式**优化；**`parity`/`dynamic`** 路径更灵活但更慢。任何「多会话穿插」若破坏 Graph 捕获条件，需评估 **回退路径** 或 **重写捕获粒度**。**`config_v4.json` 中 `stream_chunk_size`** 影响每块音频时长与块数，与 **块间间隔统计** 强相关（见 **§5.1.9 / §5.1.10**）。 |

### 6.1.2 现状归纳（为何「看起来像独占」）

- **服务层**：**`asyncio.Semaphore(concurrency)`** 只限制**同时在途合成条数**，不调度 GPU。  
- **执行层**：流式为**每请求一线程**（或线程池）驱动生成器；多请求向**同一张卡**提交 kernel。  
- **设备层**：若多线程共享**默认流**或 Graph 绑定流导致**强顺序**，GPU 上易呈**排队**；即便多 stream，**4090 单卡算力与 HBM 带宽**仍为共享瓶颈，**TTFA、RTF、块间间隔**未必随「并发路数」线性受益，**往往一并恶化**。

因此：**要「多客户端 + 仍控 TTFA + 仍保 RTF 与块间间隔」**，需要**调度策略**（何时让谁跑几步）、**批处理**（一次前向带多路）、或**多卡分流**，而不是仅调大 **`--concurrency`**；且每调一档 **`concurrency` / `replicas` / `stream_chunk_size`**，都应**同时看 METRICS 中 TTFA、RTF、inter_chunk 三项**是否仍达标。

### 6.1.3 可行路线分级（由易到难）

下列路线可**组合**使用；建议按阶段推进，每阶段用 **§七** 的压测矩阵复核 **`ttfa_wall_ms`、`rtf`、`inter_chunk_max_ms` / `inter_chunk_p95_ms`** 与 **OOM**。

**路线 A — 接入与排队层（不改模型，优先落地）**

- **网关 / 业务侧并发上限**：按压测得到的 **`concurrency` 安全值**限制同时进入合成的连接数，**超额排队**并返回 **429 / 队列深度**（避免进程内无限等待导致客户端超时）。  
- **优先级队列**：导医/通话类 **interactive** 与批量任务分队列；同卡算力有限时**保低延迟队列**。  
- **多进程多端口 × 单卡（谨慎）**：多 worker 仍共享 GPU，**总吞吐不必然倍增**，且易 OOM；一般**不如**单进程内可控调度。  

**价值**：不解决「GPU 内交错」，但**把 TTFA 可控性收回产品侧**，实施成本低。

**路线 B — 服务内「decode 量子」调度（中等改动，依赖可拆步 API）**

- **思路**：将「一路从头跑到尾」改为：**全局调度器**维护多路 **session**，每路有 **KV/缓存句柄**；调度器在**时间片或步数预算**内对多路 **轮流执行 `decode_step`**（或「小批量步数」），再让出给其它路。  
- **前提**：底层需暴露**细粒度步进接口**（或可在不破坏正确性的前提下 **暂停/恢复** 生成器状态）。当前 **`generate_voice_clone_streaming`** 为**迭代器抽象**，是否可安全拆步、与 **CUDA Graph** 是否兼容，需阅读 **`faster_qwen3_tts/streaming.py`** 与上游 **`qwen_tts`** 后做**可行性实验**。  
- **风险**：步长过小 → kernel 启动开销上升、Graph 失效 → **单路 RTF 下降、块间抖动**；分时过「碎」还会导致**同一路两次入队间隔拉大**；需 **TTFA + RTF + inter_chunk** 联合验收。

**价值**：最接近用户所说的「**分时交错**」；**工程难度与模型耦合度最高**。

**路线 C — 动态批处理（Dynamic Batching）**

- **思路**：多路处于 **prefill 后 decode 阶段**时，将**相同结构**的步合并为 **batch 维 > 1** 的一次前向（或上游支持的 **grouped query** 变体），提高 **SM 利用率**。  
- **前提**：**Talker / Predictor** 在 **codec 步进**上支持 batched tensor；需改 **`qwen_tts` / `FasterQwen3TTS`**，并处理**不同路不同剩余步长**的 **padding / early finish**。  
- **与三指标关系**：排队凑批可能**略增首包（TTFA）**；批内若等待不齐可能**拉长某些会话的块间隔**；成功合批则常**提高 GPU 利用率、有利于 RTF**。可通过 **最大等待时间（微秒～毫秒级）与最大 batch** 折中，并以 **METRICS** 同时盯 **TTFA / RTF / inter_chunk**。

**价值**：工业界Serving常见路径；**长期吞吐**通常优于「多线程各跑各的」。

**路线 D — CUDA 多 Stream + 与 Graph 解耦**

- **思路**：每会话 **独立 `torch.cuda.Stream`**，在**非 Graph** 或 **每流独立 Graph** 下提交 kernel，争取 **计算与拷贝重叠**。  
- **局限**：自回归 **步间依赖** 仍使单序列呈链式；**两序列**的重叠度受 **算子融合与显存** 限制；**4090** 上实测增益需 **nsys / Nsight** 验证。

**价值**：作为 **B/C** 的补充优化；**单独使用**往往不足以同时满足 **TTFA + RTF + inter_chunk** 的全面达标。

**路线 E — 多卡 / 多实例物理分流**

- **思路**：每 GPU（或每容器）**独立进程 + 单（或少量）replica**，上游 **负载均衡** 将会话分到不同卡；**单请求 TTFA** 接近单卡单会话，**总 QPS** 随卡数扩展。  
- **适用**：机房有多张卡时**最稳**的「多客户端 + 控 TTFA + 保单路 RTF/块间」横向扩展手段之一。

### 6.1.4 推荐落地路线（分阶段）

建议按以下顺序投入，避免一次性改 Graph + 调度 + 批处理导致不可控回归。

1. **阶段 0（立即）**  
   - 用 **§七** 压测确定 **`--concurrency` / `--replicas`**、（按需）**`stream_chunk_size`** 与 **网关并发上限**；**`inference_logging` + METRICS** 同时监控 **`ttfa_wall_ms`、`rtf`、`inter_chunk_p95_ms` / `inter_chunk_max_ms`** 分位数。  
   - 落实 **路线 A**（排队、优先级、超时与降级文案）。

2. **阶段 1（评估）**  
   - 在 **开发分支** 验证：**parity / 非 Graph** 路径下，能否对单会话 **单步前向** 封装为可调用原语；若能，再试 **双会话轮流步进**，测 **TTFA、RTF、`inter_chunk_*`**（**路线 B** 原型）。  
   - 若单步 API **不可行**或 Graph **无法保留**，在文档中明确 **「当前架构下 GPU 内精细分时成本」**，避免无效投入。

3. **阶段 2（若阶段 1 证明可拆步或上游支持 batch）**  
   - 设计 **micro-batch 策略**（最大 batch、最大等待时间、同音色是否更易合批）（**路线 C**）。  
   - 与 **路线 D** 做 **Profiling** 驱动的叠加优化。

4. **阶段 3（资源允许）**  
   - **多卡部署（路线 E）** 作为横向扩展主路径；单卡方案保留给边缘或演示环境。

### 6.1.5 验收与风险摘要

- **验收（建议列为压测/上线检查表）**  
  - **TTFA**：**`ttfa_wall_ms`** p50/p95/p99（及与 **`ttfa_ms`** 对照区分排队与模型首包）。  
  - **RTF**：单请求或分位数 **平均 `rtf`**，目标为**在既定并发下仍尽可能大**（并与 **§七** 经验阈值对齐）。  
  - **流式块间隔**：**`inter_chunk_p95_ms`、`inter_chunk_max_ms`** 不超过业务可接受上限（听感与播放器策略需交叉验证）。  
  - **稳定性**：错误率、超时、**OOM**；同等硬件下**总产出音频秒数 / 墙钟**（系统吞吐）。  

- **风险**  
  - **批处理**与 **分时调度** 可能使 **CUDA Graphs 失效** → **单路 RTF 下降**；分时过碎或一路饥饿 → **inter_chunk 恶化**。  
  - **仅压 TTFA** 而忽略 **RTF / inter_chunk**，易出现「首包尚可、播放断续或整体变慢」的体验问题。  

- **结论定位**：**单卡 4090** 上 **TTFA、RTF、块间间隔** 与 **并发路数** 之间存在**物理与实现双重约束**；**工程上**应 **A 必做、E 能多则多、B/C 按原型结果决定是否深做**，且**任何改动都以三指标联合达标为门禁**。

## 6.2 路线B方案细化与分析

本节把 **§6.1.3 路线 B** 里「依赖可拆步 API」这句话**摊开说明**：当前仓库里**逻辑上能不能拆**、**拆到什么粒度**、**CUDA Graph 卡在哪里**、**和 `qwen_tts` 是什么关系**、**第一步该做什么实验**。表述尽量用**日常语言**，便于和 **§6.1** 对照阅读。

### 6.2.1 路线 B 想多管一层什么

今天 **v4 服务**是：每条请求在自己的线程里，从 **`generate_voice_clone_streaming`** **头跑到尾**，多路请求只是**同时**往同一张 GPU 上挤。路线 B 想改成：有一个**全局调度器**，像「轮流发言」一样——**这条路跑几步 decode，再换另一条路跑几步**，避免某一路长时间占满、其它路首包或块间隔失控。

要做到这一点，程序不能只知道「一整段合成函数」；必须能**反复调用**类似 **「请把某条 session 再往前推一小步」** 的接口，且**换 session 时状态不丢、不串台**。

### 6.2.2 前提到底要求什么（用人话）

**前提可以概括成三句话：**

1. **能拆步**：合成过程不能只封装成一个黑盒 `for ... in gen` 跑到结束；要能把「一步 decode」变成**可重复调用的函数**（或等价的状态机），调度器才能在多路之间来回切。  
2. **每路一份状态**：每条 session 自己保存 **KV / 隐藏状态 / 当前 token / 攒块用的 buffer** 等；切走时存好，切回来接着用。  
3. **和现有实现一致或明确回退**：拆步后**数值与听感**仍要对，或**明确**某条路径（例如不用 Graph）接受变慢换穿插。

**注意**：「暂停 Python 生成器再恢复」听起来省事，但**多路穿插**时不能指望**两路共用一个生成器**；工程上更现实的是：**显式的 session 对象 + `advance_*()`**，而不是对现有迭代器「插一两行 sleep / yield」。

### 6.2.3 当前流式代码在 `streaming.py` 里长什么样

**快路径 `fast_generate_streaming`**（带 **CUDA Graph**）大致是：**先 prefill**，再进入一个 **`for` 循环**，每一轮做「预测 codec → talker 一步 → 采样」，把结果放进 **`chunk_buffer`**，凑满 **`chunk_size`** 才 **`yield` 一块 codec**。循环在遇到 **EOS** 或**序列过长**时结束；**最后可能还有不足 `chunk_size` 的尾巴**，再 **`yield` 一次**（与 **§5** 里「末块可能更短」一致）。

这些步骤里用到的 **`token`、`past_hidden`、`gen_step`、`chunk_buffer`** 等，今天全是**函数里的局部变量**，靠「生成器在两次 `yield` 之间暂停」来保存。也就是说：**不是「数学上不能拆步」**，而是**没有把「一步」对外暴露成 API**，路线 B 需要**重构**（把状态收进 session），**不是改一两行就能调度**。

**调度粒度（快路径上很重要的一点）**：**Predictor** 一侧的 **CUDA Graph** 把「多码本那一小段循环」收成 **一次 `run()`**。所以在**不重做图捕获**的前提下，你在 GPU 上能插队的**最细粒度**通常是：

> **一整次 predictor 的图回放 + 一次 talker 图步**（一整包），而不是 predictor 内部再拆得更碎。

若调度器说的「一步」比这个还细，就要动 **Graph 的捕获粒度** 或 **放弃这段 Graph**。

### 6.2.4 为什么「单副本 + CUDA Graph + 多 session 轮流」会顶牛

**TalkerGraph / PredictorGraph** 可以理解为：为**一条正在跑的序列**准备了一套路——**固定的输入输出缓冲区、StaticCache 等**，**`run()` 是在这块内存上反复回放**。

因此：

- **同一条 `FasterQwen3TTS`、同一对 Graph**，在 **A 路跑到一半**时，若 **B 路也来 `run()`**，会**覆盖** A 路留在 buffer / cache 里的内容，**除非**给 B 路**另准备一整套** Graph 与缓存（**显存和工程成本都会上去**）。  
- 换句话说：**「步级穿插」和「多路共用一对 Graph 缓冲区」在默认设计下是冲突的**；要么 **一路 session 独占一对图**（资源 ×路数），要么 **穿插走不用这对图的路径**（通常更慢），要么 **多 replica 时每路绑定不同模型副本**（穿插粒度变成「实例级」而不是「同实例内让步」）。

这是路线 B 里 **CUDA Graph 兼容性**的核心结论，比「会不会写 `await`」更底层。

### 6.2.5 `parity`（非 Graph）路径为什么更「好做穿插」

**`parity_generate_streaming`** 用 **`talker.forward(..., past_key_values=...)`** 这类**动态 KV**，状态主要在**张量**里，**不绑死**「全进程只有一份 Graph 缓冲区」。从**架构形状**上看，更容易做成：**每路 session 一套自己的 KV / hidden**，调度器 **A 前进一步 → B 前进一步**。

代价也直观：

- **显存**随并发路数涨（每路一套缓存）；  
- **速度**通常不如 Graph 快路径（与 **§6.1.1**「parity 更灵活但更慢」一致）；  
- 仍要实测 Talker 在**交替 forward** 时有没有隐藏的全局假设（一般应无，但应用测试锁死）。

**路线 B 的常见落地策略**往往是：**先用 parity 证明「轮流步进」有价值**，再决定要不要为每路复制 Graph、或接受「穿插只走慢路径」。

### 6.2.6 只改 `streaming.py` 还不够：`model.py` 外面还有一截

**`generate_voice_clone_streaming`** 在 **`streaming.py` 产 codec 块之后**，还会在 **`model.py`** 里做 **码流累积、参考段拼接、tokenizer 解码成波形、校准与滑窗** 等，再 **`yield` 给 HTTP 的音频块**。

所以 session 状态不能只记 **`streaming` 内部**；还要包含例如 **已生成的 codec 序列、校准用变量、上一段波形切分位置** 等。否则「拆步」拆到一半，**声音块**会对不齐或和现有 **METRICS** 语义不一致。

### 6.2.7 和上游 `qwen_tts` 的关系（避免找错改动点）

本仓库里 **`from qwen_tts import Qwen3TTSModel`** 多用于**基线 / 对比**；**线上 faster 流式**的主路径是 **`faster_qwen3_tts/streaming.py` + 本地 Talker / Graph**。**路线 B 的拆步与调度**，**首要精读与改动的是 `faster_qwen3_tts`这一支**；不必先把问题归因为「必须改 HuggingFace 上的 qwen_tts 包才能做 B」，除非你的产品明确要锁官方实现行为。

### 6.2.8 建议的原型顺序（降低白做风险）

| 顺序 | 做什么 | 目的 |
|------|--------|------|
| 1 | **parity** 下 **单 session**：把内层循环收成 **`session.step()`**，结果与当前流式 **逐块对齐**（**实现见 §6.3**） | 证明「拆步」不破坏正确性 |
| 2 | **parity** 下 **双 session 轮流 `step()`**，看 **TTFA / RTF / `inter_chunk_*`** | 证明「穿插」是否改善公平性、代价多大 |
| 3 | 再评估 **Graph 路径**：**每 session 一套图**是否可承受，或 **Graph 仅单路、穿插走 parity** 的混合策略 | 在 RTF 与公平性之间选产品可接受的点 |

### 6.2.9 小结（对照 §6.1.3 前提）

| 问题 | 结论（通俗版） |
|------|----------------|
| 能不能拆步？ | **能**，但要 **重构** 成 **session + 可调用步进**，不是「暂停一下生成器」那么简单。 |
| 最小调度量子（Graph 快路径） | 至少 **一整次 predictor 图 + 一次 talker 图步**；更细要 **改图** 或 **不用图**。 |
| 单 replica、多 session、还要 Graph | **需要策略**：**一对 Graph 缓冲区**不适合两路轮流各跑半步；但实测 1.7B 模型 + Graph 显存仅约 **5.8GB**（并非此前估算的 20GB+），24GB 显卡上仍有余量做「单路 Graph + 并发 parity」混合策略或「多份 Graph」方案。纯 parity 多路穿插已验证可行（§6.3.6）。 |
| `qwen_tts` | **不是** faster 流式拆步的第一现场；重点在 **`faster_qwen3_tts`**。 |
| 改一两行行吗？ | **不行**；属于 **中等规模改动 + 实测**，与 **§6.1.4 阶段 1** 的「开发分支验证」一致。 |

**落地代码**：**§6.3** 已落地 **§6.2.8 顺序 1**（`ParityStreamSession`）与 **顺序 2 原型**（双 session 轮询 `step()` + 基准脚本，见下）；**服务进程级**多路穿插（如 **`openai_server_v5`**）仍可按项目约定 **复制为 `*_v5` 等** 再改，避免与 **`openai_server_v4` / `config_v4`** 混线难以回滚。

## 6.3 路线B方案实现与使用指南

本节对应 **§6.2.8 顺序 1～3**。**顺序 1**：在 **parity（`parity_mode=True`，非 CUDA Graph）** 路径上，将原 **`parity_generate_streaming`** 的内层 decode 循环收成 **`ParityStreamSession.step()`**，**逐块语义**与改前 **保持一致**。**顺序 2（原型）**：提供 **`iter_round_robin_codec_chunks`** 与示例基准脚本，在两条 parity session 上 **严格轮询 `step()`**，并复用 **`model._iter_voice_clone_audio_from_codec_stream`** 按路解码，便于观察 **TTFA / `inter_chunk_*` / RTF**。**顺序 3（原型）**：评估 **CUDA Graph 路径**在多路并发下的取舍——包括 **Graph vs Parity 单路基线对比**（`graph_vs_parity_benchmark.py`）与 **混合策略**（一路 Graph + 一路 parity 并行，`hybrid_graph_parity_benchmark.py`），为产品决策提供量化数据。

### 6.3.1 已实现内容一览

**§6.2.8 顺序 1～2 在本仓库中的新增 / 修改文件**（代码路径相对 **`tts-jiang/faster-qwen3-tts/`** 工程根；文档路径相对与本 **`.md` 同级** 的项目根 **`002_tts流式与离线api/`**）：

| 类型 | 文件 |
|------|------|
| **新增** | **`tts-jiang/faster-qwen3-tts/faster_qwen3_tts/parity_stream_session.py`**（`ParityStreamSession` 及中文 docstring；**`__init__` 与 `step()` 均加 `@torch.inference_mode()` 装饰**，防止梯度图泄漏导致 OOM） |
| **新增** | **`tts-jiang/faster-qwen3-tts/faster_qwen3_tts/parity_dual_round_robin.py`**（**`iter_round_robin_codec_chunks`**：多 session 每轮各 `step()` 一次） |
| **新增** | **`tts-jiang/faster-qwen3-tts/examples/parity_dual_session_benchmark.py`**（顺序 2 原型：双路轮询 + **TTFA / inter_chunk / RTF** 打印；含 `disable_cuda_graph=True` 与 `torch.inference_mode()` 保护） |
| **修改** | **`tts-jiang/faster-qwen3-tts/faster_qwen3_tts/streaming.py`**（`parity_generate_streaming` 改为基于 `ParityStreamSession` 的薄包装） |
| **新增** | **`tts-jiang/faster-qwen3-tts/examples/graph_vs_parity_benchmark.py`**（顺序 3 基线对比：**Graph vs Parity** 单路 **TTFA / inter_chunk / RTF** 加速比） |
| **新增** | **`tts-jiang/faster-qwen3-tts/examples/hybrid_graph_parity_benchmark.py`**（顺序 3 混合策略原型：一路 **Graph** + 一路 **parity** 并行，量化混合调度下的指标） |
| **修改** | **`tts-jiang/faster-qwen3-tts/examples/openai_server_v4.py`**（① 新增 **`--disable-cuda-graph / --skip-warmup`** 命令行参数；② **`parity_mode`** 从硬编码 `True` 改为动态判断 `model.talker_graph is None`） |
| **修改** | **`tts-jiang/faster-qwen3-tts/faster_qwen3_tts/model.py`**（① **`_iter_voice_clone_audio_from_codec_stream`**：将 **`(codec_chunk, timing)`** 可迭代流转为 PCM 流；② **`from_pretrained`** 新增 **`disable_cuda_graph`** 参数；③ **`_warmup`** 在 Graph 缺失时跳过；④ **`generate_voice_clone_streaming`** 在 Graph 缺失但 `parity_mode=False` 时自动降级） |
| **修改** | **`faster-qwen3-tts服务使用与优化.md`**（**§6.3** 全文；**§6.2.8 / §6.2.9** 与 §6.3 交叉说明） |

**说明**：**`openai_server_v4.py`**、**`config_v4.json`** 等 **未因顺序 1～2 原型必改**；顺序 2 基准脚本与 **`parity_mode=True`** 的 **`model.generate_voice_clone_streaming`** 同属 parity 推理路径，用于离线对比穿插调度下的指标。

| 项 | 说明 |
|------|------|
| **新模块** | **`faster_qwen3_tts/parity_stream_session.py`**：类 **`ParityStreamSession`** |
| **衔接** | **`faster_qwen3_tts/streaming.py`** 中 **`parity_generate_streaming(...)`** 内部构造 **`ParityStreamSession`**，在 **`while not session.finished:`** 中反复 **`session.step()`**，对非空结果 **`yield`**，与 **`generate_voice_clone_streaming(..., parity_mode=True)`** 及既有测试/压测调用方式 **兼容** |
| **未改** | **`openai_server_v4.py`** 默认仍 **`parity_mode=False`（Graph 快路径）**；不强制升级服务入口即可使用本实现 |

### 6.3.2 `ParityStreamSession` 行为说明（使用前先读）

- **`ParityStreamSession(...)`**（构造）：完成与原 **`parity_generate_streaming`** 相同的 **prefill**、首 token 采样、**`attention_mask` 克隆** 等。**`__init__` 和 `step()` 均加了 `@torch.inference_mode()` 装饰**，保证即使在无上下文的外部循环中调用，PyTorch 也不会为 Transformer 前向过程保留梯度计算图（否则每步 forward 都会在显存里累积完整的激活历史，导致双路运行几十秒后 OOM，详见 §6.3.6 避坑记录）。
- **`step() -> Optional[Tuple[torch.Tensor, dict]]`**：推进 **至多一次** 原 **`for _ in range(max_new_tokens):`** 中的逻辑（含首部 **EOS** 判断、decode **forward**、**chunk_buffer** 累积、采样下一 **token**）。  
  - 若本步 **未凑满** **`chunk_size`** 且未结束，返回 **`None`**（仅内部状态前进）；  
  - 若凑满一块 codec，返回 **`(codec_chunk, timing_dict)`**，字段含义与原 **`yield`** 一致（含 **`chunk_index`、`chunk_steps`、`prefill_ms`、`decode_ms`、`total_steps_so_far`、`is_final`**）；  
  - **EOS**、**达到 `max_new_tokens`**、**`hidden_states[1]` 为 `None`** 等结束时，进入与原实现一致的 **尾块 flush**（**`is_final=True`** 或无可发块）。
- **`finished`**（属性）：会话是否已彻底结束（**无更多 `step()` 有效产出**）。

**注意**：`step()` 内部在需 flush 时可能对 **`torch.cuda.synchronize()`** 的语义与原 **`parity_generate_streaming`** 一致；**无 CUDA** 环境下需自行评估是否替换同步调用（本仓库主线假定 GPU 推理）。

### 6.3.3 直接使用 `ParityStreamSession`（自定义调度）

若要在 **不经过** **`parity_generate_streaming` 生成器** 的情况下 **手动穿插** 多路，需自行与 **`FasterQwen3TTS._prepare_generation`** 或等价逻辑对接，拿到 **`talker`、`tie`、`tam`、`tth`、`tpe`、`config`** 后构造 session：

```python
from faster_qwen3_tts.parity_stream_session import ParityStreamSession

session = ParityStreamSession(
    talker=talker,
    talker_input_embeds=tie,
    attention_mask=tam,
    trailing_text_hiddens=tth,
    tts_pad_embed=tpe,
    config=config,
    max_new_tokens=2048,
    min_new_tokens=2,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    repetition_penalty=1.05,
    chunk_size=8,  # 与 config_v4.json 中 stream_chunk_size / 业务约定一致
)
while not session.finished:
    item = session.step()
    if item is not None:
        codec_chunk, timing = item
        # 此处仅为 codec 块；接流式音频需复用 model.py 中
        # generate_voice_clone_streaming 对 codec 的累积与 speech_tokenizer.decode 逻辑
```

**重要**：**`generate_voice_clone_streaming`** 在 **`streaming.py` 之后**仍有 **`model.py`** 内 **码流累积、参考码拼接、滑窗 decode、波形切分** 等；**仅持有 `ParityStreamSession` 只得到 codec 块**。**顺序 2 原型** 已把上述 decode 逻辑抽成 **`FasterQwen3TTS._iter_voice_clone_audio_from_codec_stream`**，可按 session 把收集到的 **`(codec_chunk, timing)`** 序列喂入以得到与线上一致的 PCM 流（详见 **§6.3.6**）。**若要在 HTTP 服务内做多 session 穿插**，仍须把 **调度 + 每路 ref_codes / 会话句柄** 纳入进程内设计（参见 **§6.2.6**），基准脚本仅用于离线验证指标。

### 6.3.4 与现有 API 的调用关系

- **业务侧 / 服务侧**仍推荐通过 **`model.generate_voice_clone_streaming(..., parity_mode=True)`** 获取 **(audio_chunk, sr, timing)**；其内部仍会调用 **`parity_generate_streaming`**，因而 **自动使用 `ParityStreamSession`**，**无需改调用参数**。
- **仅当你要实现路线 B 的调度器**时，才需要 **直接** 使用 **`ParityStreamSession`** 并拼接 **model 层 decode**。

### 6.3.5 性能与验证建议

- **parity** 相对 **Graph 快路径** 往往 **慢数倍**（**TTFA / RTF** 均可能明显变差），见前文 **§6.2** 与实测；**顺序 1** 的目标是 **正确拆步、行为对齐**，不是优化单路速度。
- **回归验证**：对同一段文本、同一 **`chunk_size`**、**`parity_mode=True`**，对比 **改动前后** 的 **METRICS**（**`ttfa_wall_ms`、`inter_chunk_*`、`rtf`**）与 **听感/波形**；若仅更换了 session 实现而 **未改采样种子与模型权重**，输出应与原 **parity 流式** 一致。

### 6.3.6 顺序 2 原型：parity 下双 session 轮流 `step()`

本节对应 **§6.2.8 顺序 2**：在 **顺序 1** 已把 parity 路径拆成可步进 **`ParityStreamSession`** 的基础上，加入 **严格轮询调度器** 与 **离线基准脚本**，用于观察 **双路穿插** 对 **TTFA / `inter_chunk_*` / RTF** 的影响，**不改动线上 `openai_server_v4.py`**。

**新增 / 修改（相对 §6.3.1 顺序 1 文件表补充）**：

| 类型 | 文件 | 作用 |
|------|------|------|
| **新增** | **`tts-jiang/faster-qwen3-tts/faster_qwen3_tts/parity_dual_round_robin.py`** | **`iter_round_robin_codec_chunks(sessions)`**：严格轮询，多轮；每轮对每个未 `finished` 的 session 各 `step()` **一次**；**有 codec 块时** `yield (session_index, codec_chunk, timing)`；所有 session 均无法再推进则退出 |
| **新增** | **`tts-jiang/faster-qwen3-tts/examples/parity_dual_session_benchmark.py`** | 双路基准脚本：两段文本 → 两个 **`ParityStreamSession`** → 轮询收 codec → 每路单独解码 → 打印 TTFA / inter_chunk / RTF |
| **修改** | **`tts-jiang/faster-qwen3-tts/faster_qwen3_tts/model.py`** | 抽出 **`_iter_voice_clone_audio_from_codec_stream(speech_tokenizer, ref_codes, chunk_size, codec_stream)`**：把 **`(codec_chunk, timing)` 可迭代流** 转成与 **`generate_voice_clone_streaming`** 相同的 **`(PCM, sr, timing)`**；**`generate_voice_clone_streaming`** 改为内部 `yield from` 同一函数，语义与先前一致 |

**轮询语义**（与 §6.2.8 表格一致）：

```python
while True:
    stepped = False
    for sid, sess in enumerate(sessions):
        if sess.finished:
            continue
        stepped = True
        item = sess.step()
        if item is not None:
            yield sid, *item  # (codec_chunk, timing)
    if not stepped:
        break
```

**每轮每路最多推进一次 `step()`**；由于 **`step()` 未凑满 `chunk_size` 时返回 `None`**，外层只在 **凑满** 时才看到 yield，这就把「两路交替推进 token、谁先凑满谁先发」这件事在 **parity 路径** 上如实复现。

**运行方式**（**需 GPU 与模型权重**，参数按本机路径修改）：

```shell
cd tts-jiang/faster-qwen3-tts
python examples/parity_dual_session_benchmark.py \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio /path/to/ref.wav \
  --ref-text "参考文本" \
  --text-a "第一段合成文本" \
  --text-b "第二段合成文本" \
  --chunk-size 8 \
  --max-new-tokens 2048
```

> **OOM 避坑**：脚本内部通过 `FasterQwen3TTS.from_pretrained(..., disable_cuda_graph=True)` 完全跳过 CUDA Graph 的构建与预热，同时主推理循环套在 `with torch.inference_mode():` 中。两者缺一不可，详见 §6.3.6.1。

**脚本流程**：对两段文本分别 **`_prepare_generation(... xvec_only=False, non_streaming_mode=False)`** 拿到 **`(m, talker, config, tie, tam, tth, tpe, ref_codes)`**，构造两个 **`ParityStreamSession`**，进入 **`iter_round_robin_codec_chunks([s0, s1])`**：**按 session 记录各 codec 块的墙钟**，同时按 session 保留 **`(chunk, timing)`** 序列；codec 阶段结束后，对每路调用 **`FasterQwen3TTS._iter_voice_clone_audio_from_codec_stream(speech_tokenizer, ref, chunk_size, pair_iter())`** 解码到 **PCM**，累计 **`audio_s`** 与 **`prefill_ms+decode_ms`**。

**打印项与 METRICS 对齐**（**§5.1.10**）：

| 打印字段 | 含义 | 与 METRICS 对应 |
|----------|------|-----------------|
| `ttfa_wall_ms(codec)` | 自进程 `t0` 到本路 **首块 codec** 产出的墙钟（ms） | **`ttfa_wall_ms` proxy**（未经 HTTP 队列/PCM 封装，仅到 codec） |
| `inter_chunk_max_ms` / `inter_chunk_p95_ms` | 同 session 相邻两次 **codec 产出** 的墙钟间隔 max / p95 | **`inter_chunk_max_ms` / `inter_chunk_p95_ms` proxy**（与服务端 PCM 入队间隔同阶，因两者之间只差一次 tokenizer.decode） |
| `ttfa_wall_ms(audio)` | 自进程 `t0` 到本路 **首块 PCM** 产出的墙钟（ms） | 更接近 **`ttfa_wall_ms`** 本身（但不含 HTTP 排队） |
| `rtf(model_timing)` | `audio_s / (Σ prefill_ms + Σ decode_ms) / 1000` | **RTF** 口径同向（仅模型计时，**不含** 服务器旁路开销） |
| `n_chunks` / `audio_s` / `decode_wall_ms` | 该路 codec 块数、总时长、解码阶段墙钟 | 辅助读数 |

**读数要点**：
- **parity 路径本身** 相对 **Graph 快路径** 往往 **慢数倍**（**§6.2** / **§6.3.5**）；本基准的目的是 **对比双路与单路** 在同一条件下的 **TTFA / `inter_chunk_*` 变化**，**不**宜把数值直接外推到 `openai_server_v4.py` 线上 Graph 路径。
- **TTFA(codec)** 增加量 ≈ 另一路在本路首块之前「占走」的 `step()` 次数 × 每步耗时；若 **增幅显著** 超过单路的一半，说明 **每步同步开销 / KV 交错惩罚** 非零，需要在 **§6.2.4 / §6.2.9** 的路径上做进一步评估。
- **`inter_chunk_*`** 的变化更重要：**严格轮询** 会把另一路 **每轮至多一次 `step()`** 的开销摊进本路相邻 codec 的间隔，**若 p95 接近单路 max 的 2 倍** 则视为「近似对等分时」；**显著 > 2 倍** 通常意味着 parity 路径某些步耗时不均（例如首步 prefill、尾部 flush）。
- **RTF(model_timing)** 由于只累计 **模型 forward 计时**，双路轮询下两路的 RTF **不会** 直接相加，但 **墙钟**（`decode_wall_ms` / `total_wall_ms`）会受影响，**端到端 RTF** 应以 `audio_s / 墙钟` 估算——脚本中两种口径都打印，方便对比。

**与 HTTP 服务的差距（仍未落地的部分）**：
- 本基准在 **同一进程、同一 Python 线程、无 HTTP** 下跑，**不涉及** `openai_server_v4.py` 的 **concurrency 信号量、PCM 编码、流式队列**；若要把轮询推进到线上，需要在服务入口里接 **`iter_round_robin_codec_chunks`** 并为每路维护 **`ref_codes` / speech_tokenizer 状态 / PCM 编码器 / 响应队列**（见 **§6.2.6**）。
- **ref_codes 按 session 区分**：**`_iter_voice_clone_audio_from_codec_stream`** 接收的 `ref_codes` 必须是 **对应这一路 `_prepare_generation` 返回的那条**；**不要** 跨路复用（否则滑窗/校准比例会错位）。
- **Graph 快路径** 下 **每 session 一套图** 的内存与首次编译代价仍未评估（**§6.2.4 / §6.2.9 / §6.3.7 顺序 3**）。

#### 6.3.6.1 避坑记录：OOM 三阶调试历程

在实际运行基准脚本时，先后触发了三次形态不同的 OOM，记录如下，供后续维护者参考。

##### 第一阶段：CUDA Graph 捕获 + 缺少 inference_mode 双重叠加

**症状**：启动命令后几秒内崩溃；日志显示先出现了 Graph 捕获（"Warming up predictor / Capturing CUDA graph / Talker CUDA graph captured"），随后在 `iter_round_robin_codec_chunks` 循环内崩溃，`20.46 GiB is allocated by PyTorch`，剩余仅 1.75 MiB。

**原因**：此阶段有两个问题叠加：
1. **CUDA Graph 被意外触发**：`FasterQwen3TTS.from_pretrained` 默认构建 Graph wrapper；`_prepare_generation` 首次调用时 `if not self._warmed_up: self._warmup()` 会自动捕获 Graph。Parity 路径本不需要 Graph，但基准脚本没有禁用。
2. **缺少 `torch.inference_mode()` 保护**（这是更根本的原因，见第三阶段详述）：即使禁用了 Graph，在无 `inference_mode()` 的情况下，PyTorch 默认保留每步 Transformer 的全部中间激活值（梯度计算图），显存随步数 $O(N)$ 累积到 20GB+。

两个问题同时存在时，Graph 的静态显存分配（实测约 5.8GB for 1.7B 模型，见下文）加上梯度计算图的泄漏，合力挤爆了 24GB 显存。

**修复**：在 `FasterQwen3TTS.from_pretrained` 中新增 **`disable_cuda_graph: bool = False`** 参数；设 `True` 时完全跳过 `PredictorGraph` / `TalkerGraph` 的构建，同时 `_prepare_generation` 中的 `skip_warmup` 也不再触发捕获。基准脚本调用 `from_pretrained(..., disable_cuda_graph=True)`。

> **CUDA Graph 实际显存占用（实测）**：在 RTX H20（98GB）上，**`Qwen3-TTS-12Hz-1.7B-Base`** 单路 Graph 路径下（`openai_server_v4.py`，`--replicas 1 --concurrency 4`），进程显存占用约 **5.8GB**（稳定值，含模型权重 + Graph 静态池），远小于 24GB 显存总量。因此 **一份 CUDA Graph 并不会吃满 20GB**——先前文档中「一份 Graph 就占 20GB」的估计是基于早期调试时把 **梯度图泄漏** 与 **Graph 分配** 混在一起的误判。修正后：在 `inference_mode()` 保护下，**1.7B 模型 + Graph** 总显存约 5.8GB，**0.6B 模型 + disable_cuda_graph** 总显存约 2.8GB，均远低于 24GB 上限。

##### 第二阶段：base_model 属性访问错误

**症状**：第一阶段修复（加 `disable_cuda_graph`）后执行报 `AttributeError: 'Qwen3TTSModel' object has no attribute 'modules'`。

**原因**：`qwen_tts.Qwen3TTSModel` 是对 HuggingFace 的封装类，**本身不继承 `torch.nn.Module`**；真正的 PyTorch Module 层在 `base_model.model` 中。代码里写的是 `base_model.modules()`，应改为 `base_model.model.modules()`。

**修复**：更正属性路径。此阶段还曾尝试通过遍历模块设置 `_use_cache = False` 来防止 HuggingFace 静态缓存预分配，但后续验证表明 **该 hack 不是 OOM 的根因（根因是缺少 inference_mode）且可能影响正常 KV Cache 行为**，已在最终版本中移除。

##### 第三阶段：梯度计算图泄漏——OOM 的根本原因

**症状**：前两阶段修复后（`disable_cuda_graph=True`，Graph 不再捕获），程序能正常启动并开始执行，但 **显存从约 2780MB 随生成步数线性攀升**，几十秒后吃满 24GB 再次 OOM，崩溃点仍在 `torch.cat([self.keys, key_states], dim=-2)`。此时已无 CUDA Graph 参与。

**原因**：这是最隐蔽也最根本的问题。

原版 `parity_generate_streaming` 函数带有 **`@torch.inference_mode()`** 装饰，整个生成过程在推理上下文中进行，PyTorch **不记录任何梯度计算图**。但在将其拆解为 `ParityStreamSession` 类之后，每次对 `session.step()` 的调用都脱离了原来的推理上下文保护。如果调用方（如基准脚本的 `for` 循环）外部没有套 `torch.inference_mode()`，PyTorch 会进入默认的"训练模式"：每一步 Transformer `forward` 的所有中间激活值（attention 权重、layer norm 状态等）全部保留在显存中，用于反向传播准备。这些张量随 token 步数的增加 $O(N)$ 累积，几十秒后彻底耗尽 24GB 显存。

**这解释了第一阶段日志中的 `20.46 GiB is allocated by PyTorch`**：并非 CUDA Graph 吃了 20GB，而是**梯度计算图泄漏**吃了 20GB。Graph 本身只占约 5.8GB（1.7B 模型实测），与梯度图叠加后才挤满了 24GB。

**修复**（双保险）：
1. 在 `ParityStreamSession.__init__` 和 `step()` 方法上均加 **`@torch.inference_mode()`** 装饰，保证类本身的每次调用都处于推理模式，即使调用方忘记套上下文也不会泄漏。
2. 在 `parity_dual_session_benchmark.py` 的主推理循环外套 **`with torch.inference_mode():`**：

```python
with torch.inference_mode():
    for sid, chunk, timing in iter_round_robin_codec_chunks([s0, s1]):
        ...
```

**修复后的显存表现**（实测，RTX 4090，`Qwen3-TTS-12Hz-0.6B-Base`）：从加载模型后的约 **2780MB** 起，整个双路生成过程（双路合计输出 73 秒音频）结束后显存仅涨至约 **2820MB**（涨幅仅约 40MB，为两路 KV Cache 动态累积的正常增量）。

##### 小结：三阶调试的关键教训

| 阶段 | 根因 | 修复 | 是否仍需保留 |
|------|------|------|------------|
| 1 Graph + 梯度图叠加 OOM | CUDA Graph 不必要触发 + 梯度图泄漏 | `disable_cuda_graph=True` + `skip_warmup=True` | **保留**：parity 路径确实不需要 Graph |
| 2 `AttributeError` | `base_model` vs `base_model.model` 属性层级 | 更正属性路径 | **保留**：代码中不再使用此 hack，但教训记录在此 |
| 3 梯度图泄漏（根本原因） | 缺少 `torch.inference_mode()` | `@torch.inference_mode()` 装饰 + 外层 `with` | **必须保留**：这是所有 OOM 的根本解法 |

##### RTF 指标口径说明（避免误读）

脚本中的 `rtf(model_timing)` 定义为：

```
rtf = audio_s / (total_gen_ms / 1000.0)
```

即 **`音频时长 / 生成耗时`**，这是「**速度比**」而非传统的「实时因子」：

| 口径 | 公式 | > 1.0 意味着 |
|------|------|------------|
| **本脚本 `rtf(model_timing)`** | `audio_s / 生成耗时_s` | 算得比说得**快**（越大越好） |
| 传统 RTF（`Real-Time Factor`）| `生成耗时_s / audio_s` | 算得比说得**慢**（越小越好） |

两者互为倒数，**本脚本口径与 §5.1.10 METRICS 的 RTF 定义同向（越大越好）**。

#### 6.3.6.2 实测基准结果（Qwen3-TTS-12Hz-0.6B-Base，双路 parity，RTX 4090）

**运行配置**：`chunk_size=8`，`max_new_tokens=2048`，parity 模式（无 CUDA Graph），双路严格轮询。

```text
=== Codec 阶段（轮流 step，墙钟）===
session A: ttfa_wall_ms(codec)=1569.4  inter_chunk_max_ms=1993.9  inter_chunk_p95_ms=1693.4  n_chunks=67
session B: ttfa_wall_ms(codec)=1671.9  inter_chunk_max_ms=2072.2  inter_chunk_p95_ms=1703.1  n_chunks=50

=== 波形阶段（每路单独 decode，与线上一致）===
session A: ttfa_wall_ms(audio)=96721.8  audio_s=42.32  rtf(model_timing)=0.437  decode_wall_ms=1650.0
session B: ttfa_wall_ms(audio)=98302.1  audio_s=31.60  rtf(model_timing)=0.383  decode_wall_ms=1101.3

total_wall_ms=99374.6   （codec_phase_wall_ms=96622.4）
```

**解读**：

| 指标 | 值 | 解读 |
|------|-----|------|
| **TTFA 公平性** | A: 1.57s / B: 1.67s | 两路首块差距仅约 100ms，严格轮询下起跑近乎同步 |
| **inter_chunk_p95（codec）** | A/B 约 1.70s | chunk_size=8 对应约 0.63s 音频；每块间隔约 1.70s，**当前 parity 双路为慢于实时（生成 0.63s 需等 1.70s）** |
| **rtf(model_timing)** | A: 0.437 / B: 0.383 | 按**本脚本口径（越大越好）**，0.4 < 1.0，即纯模型 forward 累计耗时已慢于实时；**parity 路径本身在非 Graph 下速度限制所致** |
| **端到端 RTF（两路合计）** | `(42.32+31.60) / (99.37)=0.744` | 两路共产出约 73.9 秒音频，总墙钟约 99 秒，端到端速度比约 0.74，未达实时 |
| **显存占用增幅** | 约 +40MB | 两路 KV Cache 动态累积的正常量，**无梯度图泄漏** |

**关键结论**：

1. **两路 TTFA 公平**：严格轮询确实让两条请求的首包几乎同步，这是路线 B 核心价值的验证。
2. **当前速度慢于实时**：parity（非 Graph）路径本身 RTF 约 0.4，叠加双路分时后每路 inter_chunk 约 1.7s（对应 0.63s 音频），流式播放会卡顿。这是预期内的，parity 测试目的是**验证行为正确性与公平性**，速度问题在考量 Graph 路径（顺序 3）时再处理。
3. **显存行为正常**：`disable_cuda_graph=True` + `@torch.inference_mode()` 双重保护下，显存完全可控。

### 6.3.8 顺序 3 原型：Graph 路径评估与混合策略

本节对应 **§6.2.8 顺序 3**：在 **顺序 1～2** 已验证 parity 路径的拆步与穿插可行性的基础上，进一步评估 **CUDA Graph 路径**在多路并发下的取舍——量化 **Graph vs Parity** 的单路加速比，并验证 **「一路 Graph + 并发 parity 回退」的混合策略**是否在 RTF 与公平性之间取得可接受的平衡。

#### 6.3.8.1 CUDA Graph 显存实测数据（修正此前误判）

**此前文档曾估算「一份 Graph 就占 20GB+」**，这是在 OOM 三阶调试过程中把 **梯度计算图泄漏** 与 **Graph 静态池** 混在一起的误判。修正后的实测数据：

| 环境 | 模型 | Graph 状态 | 进程显存占用 |
|------|------|-----------|------------|
| RTX H20（98GB） | 1.7B | Graph 启用（`--replicas 1 --concurrency 4`） | **约 5.8GB**（稳定值，含模型权重 + Graph 静态池） |
| RTX 4090（24GB） | 0.6B | Graph 禁用（`disable_cuda_graph=True`） | **约 2.8GB**（模型权重 + parity 动态 KV） |

**TalkerGraph** 的静态 KV Cache 内存估算（1.7B 模型，28 层，max_seq_len=2048）：
- 28 层 × 2（K+V） × `[1, kv_heads, 2048, head_dim]` × bfloat16 ≈ **56MB**
- PredictorGraph（5 层，max_seq=17）：≈ **0.2MB**
- **一份 Graph + PredictorGraph 合计 ≈ 60MB**（不含模型权重）

在 24GB 4090 上：
- **1.7B 模型权重** ≈ 3.4GB（bfloat16）
- **一份 Graph 静态池** ≈ 60MB
- **一份 parity 动态 KV Cache（最大时）** ≈ 56MB
- **合计约 3.5GB + 动态增量**，远低于 24GB 上限

**结论**：**一份 Graph 不会吃满 20GB**；此前 OOM 的根本原因是缺少 `torch.inference_mode()` 导致的梯度计算图泄漏（§6.3.6.1）。在 24GB 显卡上，**一份 Graph + 一份 parity 动态 KV** 合计约 3.6GB，仍有约 20GB 余量。

#### 6.3.8.2 Graph vs Parity 单路基线对比

**脚本**：**`examples/graph_vs_parity_benchmark.py`**

对同一段文本分别用 **Graph 快路径**（`parity_mode=False`）和 **Parity 动态路径**（`parity_mode=True`，`disable_cuda_graph=True`）跑流式合成，对比 **TTFA / inter_chunk / RTF**。

**运行方式**（**需 GPU 与模型权重**）：

```shell
cd tts-jiang/faster-qwen3-tts

# 1. 先跑 Graph 路径（模型加载时自动捕获 CUDA Graph）
python examples/graph_vs_parity_benchmark.py \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio /data/tts/faster-qwen3-tts-assets/ref_audio_8.wav \
  --ref-text "欢迎您使用硅基数字人，实时交互能够通过面对面对话，通过情感互动的数字人提供更好的客户服务。" \
  --text "建议您尽快预约消化内科门诊。" \
  --chunk-size 8 \
  --max-new-tokens 2048
```

脚本内部依次加载两个模型实例（Graph 版和 Parity 版），各跑一次流式合成后对比输出加速比。**打印项**：

| 打印字段 | 含义 |
|----------|------|
| `ttfa_wall_ms` | 自进程 `t0` 到 **首块 PCM** 的墙钟（ms） |
| `inter_chunk_max_ms / p95_ms` | 相邻 PCM 块产出的墙钟间隔 |
| `rtf(model_timing)` | `audio_s / (Σ prefill_ms+decode_ms)/1000`，**越大越好** |
| `rtf(wall_clock)` | `audio_s / 墙钟总耗时_s`，**越大越好** |
| 加速比 | Graph 相对 Parity 的 **TTFA 加速倍数 / RTF 加速倍数 / 总耗时加速倍数** |

#### 6.3.8.3 混合策略原型：一路 Graph + 一路 parity 并行

**脚本**：**`examples/hybrid_graph_parity_benchmark.py`**

在同一 GPU 上，路 A 走 **CUDA Graph 快路径**（独占 TalkerGraph 缓冲区），路 B 跳 **parity 动态路径**（ParityStreamSession，动态 KV Cache），通过**块级轮询**交替推进两路 decode，观察混合调度下的 **TTFA / inter_chunk / RTF**。

**架构关键决策**：

- **路 A（Graph）** 的 decode 步通过 `fast_generate_streaming` 的生成器 `next()` 拿出一个完整 chunk（`chunk_size` 个步一次性跑完），**不可拆成单步**——因为 CUDA Graph 的核心优势是消除 Python 端的逐步开销，拆成单步调用会回退到 parity 速度。
- **路 B（parity）** 的 decode 步通过 `ParityStreamSession.step()` 推进，**可拆成单步**。
- **轮询策略**：每轮先拉路 A 的一个 chunk（Graph 一次性跑完 chunk_size 步），再拉路 B 的一个 chunk（parity 连续 step 直到攒够 chunk_size），直至两路全部完成。
- **PredictorGraph** 是**单实例共享**的（每步 reset 后 replay，无持久 KV 状态），路 A 和路 B 各自的 predictor 都用它。TalkerGraph 是路 A **独占**的（含路 A 的 StaticCache），路 B 用 DynamicCache。

**运行方式**：

```shell
cd tts-jiang/faster-qwen3-tts
python examples/hybrid_graph_parity_benchmark.py \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio /data/tts/faster-qwen3-tts-assets/ref_audio_8.wav \
  --ref-text "欢迎您使用硅基数字人，实时交互能够通过面对面对话，通过情感互动的数字人提供更好的客户服务。" \
  --text-a "建议您尽快预约消化内科门诊。" \
  --text-b "建议您尽快预约消化内科门诊。" \
  --chunk-size 8 \
  --max-new-tokens 2048
```

**打印项**与 §6.3.6 双路 parity 基准一致，按路标记 **A (Graph)** / **B (parity)**。

**读数要点**：

- **路 A（Graph）TTFA**：应接近单路 Graph 的 TTFA（因为路 A 独占 TalkerGraph 缓冲区，每轮跑完 chunk_size 步后才让路 B 推进，路 A 不被 parity 速度拖慢）。
- **路 B（parity）TTFA**：会比单路 parity 的 TTFA **增加**（因为路 A 的每个 chunk 占走 chunk_size × 单步耗时的墙钟时间，路 B 的首步必须等到路 A 的首块完成才被调度）。增幅 ≈ 路 A 首 chunk 墙钟 / 路 B parity 单步耗时比 × chunk_size。
- **路 A inter_chunk**：应接近单路 Graph 的 inter_chunk（路 A 独占缓冲区，每轮跑完自己的 chunk 后才让路 B 占 GPU）。
- **路 B inter_chunk**：会比单路 parity 的 inter_chunk **增加**约路 A 的一个 chunk 产出时间——因为每轮路 B 要等路 A 先跑完一个 chunk 才能推进自己的 chunk。
- **路 A RTF(model_timing)**：应与单路 Graph RTF 一致（模型 forward 累计计时不受轮询影响）。
- **路 B RTF(model_timing)**：应与单路 parity RTF 一致（模型 forward 累计计时不受轮询影响）。
- **两路总墙钟**：比单路任一路都长（两路共享 GPU 时间），但比纯双路 parity 可能更优（路 A 用 Graph 快路径，比 parity 快数倍）。

**与 §6.2.4 的对照**：

§6.2.4 指出「一对 Graph 缓冲区不适合两路轮流各跑半步」。混合策略的解法是：**路 A 独占 Graph 缓冲区跑完整 chunk，路 B 完全不走 Graph 而用 parity 动态路径**——两路不争抢 Graph 缓冲区，而是在不同路径上各跑各的。路 A 享有 Graph 的速度优势，路 B 享有 parity 的灵活性（可以步级穿插，但速度慢）。

**与纯双路 parity 的对照（§6.3.6）**：

混合策略下路 B 的 TTFA / inter_chunk 可能比纯 parity 双路更差（因为路 A 每轮要先占走 chunk 级的 GPU 时间），但路 A 的 TTFA / inter_chunk 可能比纯 parity 双路更好（路 A 独占 Graph，不被 parity 拖慢）。**产品决策的核心**：是否接受「第一路快、第二路慢」的不公平性，以换取至少一路的 Graph 级速度？

#### 6.3.8.4 实测基线数据（Qwen3-TTS-12Hz-0.6B-Base，RTX 4090）

```bash
# 单路基线对比：
python examples/graph_vs_parity_benchmark.py \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio /data/tts/faster-qwen3-tts-assets/ref_audio_8.wav \
  --ref-text "欢迎您使用硅基数字人，实时交互能够通过面对面对话，通过情感互动的数字人提供更好的客户服务。" \
  --text "建议您尽快预约消化内科门诊。" \
  --chunk-size 8 --max-new-tokens 2048
```

**运行配置**：`chunk_size=8`，`max_new_tokens=2048`，参考音频 `ref_audio_8.wav`，参考文本 "欢迎您使用硅基数字人，实时交互能够通过面对面对话，通过情感互动的数字人提供更好的客户服务。"，合成文本 "建议您尽快预约消化内科门诊。"（约 12 字，产出约 1.7~2.4 秒音频）。

**实测结果**（2026-04-22，容器内实测，脚本已加入预热排除 capture 开销）：

| 指标 | Graph 快路径 | Parity 动态路径 | 加速比 | 解读 |
|------|-------------|----------------|--------|------|
| **TTFA（首包墙钟）** | 223 ms | 852 ms | **3.83x** | Graph 首包快近 4 倍 |
| **inter_chunk_p95（块间间隔）** | 181 ms | 859 ms | **4.76x** | Graph 块间紧凑近 5 倍 |
| **RTF(model_timing)** | **4.45** | 0.77 | **5.75x** | Graph 已实时（>1.0），Parity 未实时（<1.0） |
| **RTF(wall_clock)** | **3.18** | 0.76 | **4.18x** | Graph 端到端也是实时的 |
| **总耗时** | 554 ms | 3163 ms | **5.71x** | 同一段音频 Graph 只需 0.55s，Parity 需 3.16s |
| **n_chunks** | 3 | 4 | — | 产出音频量略有差异（随机采样导致） |

**关键结论**：

1. **CUDA Graph 路径已实时**：RTF=4.45（模型计时）/ 3.18（墙钟），远大于 1.0，说明 0.6B 模型在 Graph 路径下**算得比说得快**。
2. **Parity 路径未实时**：RTF=0.77，小于 1.0，说明 0.6B 模型在 Parity 路径下**算得比说得慢**（需要约 1.3 倍音频时长才能算完）。
3. **TTFA 体验**：Graph 首包 223ms，Parity 首包 852ms，Graph 首包体验显著更好。
4. **加速比一致性**：所有指标（TTFA、inter_chunk、RTF、总耗时）一致指向 **Graph 比 Parity 快约 4~6 倍**。

#### 6.3.8.5 多份 Graph 方案评估（理论，尚未原型验证）

另一种策略是为 **每路 session 分配一对 TalkerGraph + PredictorGraph**。根据 §6.3.8.1 的实测数据：

- **每对 TalkerGraph + PredictorGraph ≈ 60MB**（不含模型权重）
- **模型权重 ≈ 3.4GB**（共享，不重复）
- **在 24GB 4090 上**：模型权重 3.4GB + 2 × Graph 静态池 120MB + 2 × 动态 KV ~120MB ≈ **3.7GB**，仍有约 20GB 余量

**理论可行性**：在 24GB 显卡上为 2~3 路各分配一对 Graph 是 **显存层面可行的**。但有以下工程挑战：

1. **TalkerGraph 的 `prefill_kv()` + `capture()` 耗时约 2~3 秒**（每路首次使用时需捕获），启动预热时间会随路数增长。
2. **CUDA Graph 的 `run()` 是在捕获时的 CUDA stream 上 replay**；多路共用同一个默认 stream 时，两路的 replay 会**串行排队**，不会真正并行。
3. **TalkerGraph 内部的 `StaticCache` 在 `run()` 时通过 `index_copy_` 写入**——如果两路在同一个 stream 上交替 `run()`，**后路会覆盖前路刚写入的 KV**（因为 `StaticCache` 的内存地址是 Graph 捕获时固定的）。
4. **解决方案**：要么为每路分配 **独立的 CUDA stream** 并在调度时做 stream synchronization；要么确保两路不在**同一步**同时 replay——即「块级轮询」与混合策略类似，只是两路都用 Graph。

**当前结论**：多份 Graph 方案在 **显存层面可行**，但 **工程实现复杂度高于混合策略**（需要独立 CUDA stream、多份 StaticCache、跨 stream 同步），且收益不确定（两路 Graph 在同一 GPU 上仍是串行排队，不会比块级轮询更快）。§6.3.8.4 的实测数据表明混合策略已能提供「一路实时 + 一路非实时」的方案，建议 **先跑混合策略基准确认是否满足产品需求**，再决定是否投入多份 Graph 方案。

#### 6.3.8.6 产品决策建议

基于 §6.3.8.4 的实测基线数据（0.6B 模型，RTX 4090）：

| 方案 | TTFA | RTF | 显存 | 适用场景 |
|------|------|-----|------|---------|
| **纯 Graph 单路** | 最优（~223ms） | **实时**（RTF=4.45） | ~3.5 GB | **推荐**：单用户独占、追求极致低延迟 |
| **纯 parity 双路** | 中等（~852ms） | **非实时**（RTF=0.77） | ~2.8 GB | 多用户并发、对延迟不敏感、无 Graph 显存 |
| **混合（1 Graph + 1 parity）** | Graph路优 / parity路劣 | Graph路实时 / parity路非实时 | ~3.5 GB + 增量 | **实验性**：1 个快用户 + 1 个慢用户共享 GPU |
| **多份 Graph**（理论） | 均优 | 均实时 | ~3.5 GB + N×60 MB | 需额外工程改造（独立 CUDA stream、步级 Session） |

**推荐**：
- **线上服务默认走 Graph 路径**（`parity_mode=False`，即 `openai_server_v4.py` 默认行为），获得实时 RTF 和低 TTFA。
- **当并发路数超过 1 且需要公平性时**：
  - 方案 A：走 **纯 parity 多路**（已实现，顺序 2），牺牲速度换取公平性。
  - 方案 B（实验性）：走 **混合策略**（1 Graph + N-1 parity），保证至少 1 路实时，其余路非实时但可穿插。
  - 方案 C（未来）：**多份 Graph**，每路各一对 TalkerGraph，需评估独立 CUDA stream 的工程成本。

### 6.3.9 后续工作（对照 §6.2.8）

| 顺序 | 状态（截至本文） |
|------|------------------|
| 1 单 session **`session.step()`** | **已落地**：**`ParityStreamSession`** + **`parity_generate_streaming` 薄包装**（**§6.3.2 / §6.3.3**） |
| 2 双 session 轮流 **`step()`** | **原型已落地**：**`parity_dual_round_robin.iter_round_robin_codec_chunks`** + **`examples/parity_dual_session_benchmark.py`** + **`_iter_voice_clone_audio_from_codec_stream`**（**§6.3.6**）；**服务端多路 HTTP / `openai_server_v5`** 仍待接入 |
| 3 **Graph 路径**评估 | **原型已落地 + 实测数据已采集**：**`examples/graph_vs_parity_benchmark.py`**（单路基线：Graph 比 Parity 快约 **4~6 倍**）+ **`examples/hybrid_graph_parity_benchmark.py`**（混合策略原型）；多份 Graph 方案仍待评估 |

### 6.3.10 多路 Graph 并发基准测试：确定单 GPU 最大并发数

本节对应 **§6.2.8 顺序 3 的扩展**：在已确认 **CUDA Graph 快路径**（`parity_mode=False`）单路性能优异的基础上（**§6.3.8.4** 实测 RTF≈4.7，算得比说得快），进一步测试**多路纯 Graph 并发**（2/3/4 路均走 Graph 快路径）的性能表现，目的是**在 TTFA、RTF、流式连续性（inter_chunk）之间取得平衡**，**确定单 RTX 4090 GPU 上纯 Graph 路径的最大可行并发路数**。

与 **§6.3.8.3 混合策略**（1 Graph + 1 parity）不同，本节测试**所有并发路均使用 Graph 快路径**，调度方式为**块级轮询**（每轮各路各产一个 PCM chunk），以评估「全速多路」而非「快慢混合」的可行性。

#### 6.3.10.1 新增基准脚本一览

**本节新增 / 修改（相对 §6.3.8）**：

| 类型 | 文件 | 作用 |
|------|------|------|
| **新增** | **`examples/dual_graph_streaming_benchmark.py`** | **双路 Graph** 流式基准：两路均使用 CUDA Graph 快路径，块级轮询，输出 TTFA / inter_chunk / RTF |
| **新增** | **`examples/triple_graph_streaming_benchmark.py`** | **三路 Graph** 流式基准：三路均使用 CUDA Graph 快路径，块级轮询 |
| **新增** | **`examples/quad_graph_streaming_benchmark.py`** | **四路 Graph** 流式基准：四路均使用 CUDA Graph 快路径，块级轮询，含播放流畅性分析 |
| **新增** | **`examples/hybrid_graph_parity_streaming_benchmark.py`** | **混合策略流式基准**（1 Graph + 1 parity），与 **§6.3.8.3** 的 `hybrid_graph_parity_benchmark.py` 类似，但采用更贴近真实 HTTP 流式的 PCM chunk 级轮询 |

**设计要点**：

- **调度顺序**：各脚本均采用 **A → B → C → D**（或按实现略有调整）的**同步轮询**，每轮各路各 `next(generator)` 一次，产生一个 PCM chunk 后切换到下一字路。
- **预热策略**：CUDA Graph capture 完成后，对**每路分别做首包预热**（`next()` 一次），尽量消除首包时的 prefill 耗时差异对正式测试的影响。
- **指标口径**：与 **§6.3.6 / §6.3.8** 一致——`ttfa_wall_ms(audio)` 为自 `t0` 到首块 PCM 产出的墙钟；`inter_chunk_p95_ms` 为相邻 PCM 块产出间隔的 p95；`rtf(model_timing)` 为 `audio_s / (total_gen_ms / 1000)`（越大越快）。

#### 6.3.10.2 运行方式与参数建议

**双路 Graph 测试**：

```shell
cd tts-jiang/faster-qwen3-tts
python examples/dual_graph_streaming_benchmark.py \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio /data/tts/faster-qwen3-tts-assets/ref_audio_8.wav \
  --ref-text "欢迎您使用硅基数字人，实时交互能够通过面对面对话，通过情感互动的数字人提供更好的客户服务。" \
  --text-a "根据您刚才描述的反复胃痛两周、进食后加重，偶尔反酸，建议您尽快预约消化内科门诊，必要时医生可能会安排幽门螺杆菌检测或胃镜检查。就诊前请清淡饮食，避免饮酒和辛辣刺激；若出现呕血、黑便或剧烈腹痛，请立即前往急诊。您可以在问诊时补充既往胃病用药史与过敏史，并带好近期化验单，方便医生综合判断是否需要调整治疗方案或进一步随访。" \
  --text-b "根据您刚才描述的反复胃痛两周、进食后加重，偶尔反酸，建议您尽快预约消化内科门诊，必要时医生可能会安排幽门螺杆菌检测或胃镜检查。就诊前请清淡饮食，避免饮酒和辛辣刺激；若出现呕血、黑便或剧烈腹痛，请立即前往急诊。您可以在问诊时补充既往胃病用药史与过敏史，并带好近期化验单，方便医生综合判断是否需要调整治疗方案或进一步随访。" \
  --chunk-size 8 \
  --max-new-tokens 2048
```

**三路 / 四路**：将脚本换为 `triple_graph_streaming_benchmark.py` / `quad_graph_streaming_benchmark.py`，并增加 `--text-c`、`--text-d` 参数。

**chunk_size 调参**：四路并发时 `chunk_size=8` 可能接近卡顿临界点，可尝试 `10` 或 `12` 增加每块音频时长、改善流畅性。

#### 6.3.10.3 实测数据汇总（Qwen3-TTS-12Hz-0.6B-Base，RTX 4090）

**基础参数**：`chunk_size=8`（每块音频 ≈ 8/12×1000 = **667ms**），`max_new_tokens=2048`，codec 帧率 **12Hz**。

| 并发路数 | 路A TTFA（第1路） | 路D TTFA（最后一路） | inter_chunk_p95 | 缓冲余量（667 - inter_chunk_p95） | slowdown | 播放流畅性 |
|---------|---------|---------|----------------|---------|---------|-----------|
| **2 路** | ~205ms | ~408ms | **~350ms** | **+317ms** | 1.96x | 非常流畅 |
| **3 路** | ~205ms | ~617ms | **~515ms** | **+152ms** | 2.86x | 流畅 |
| **4 路** | ~202ms | ~808ms | **~677ms** | **-10ms** | 3.76x | 临界/有风险 |

> **缓冲余量** = 每块音频时长(667ms) - inter_chunk_p95；正值表示播放流畅，负值可能出现卡顿。

**详细测试数据（三次运行平均值）**：

**双路 Graph（`dual_graph_streaming_benchmark.py`，chunk_size=8）**：

```text
=== 双路 Graph 模式统计 ===
  路 A (Graph): ttfa_wall_ms(audio)=205.4  inter_chunk_max_ms=350.5  inter_chunk_p95_ms=349.9  n_chunks=44  audio_s=28.08  total_gen_ms=5914.33  rtf(model_timing)=4.748
  路 B (Graph): ttfa_wall_ms(audio)=411.1  inter_chunk_max_ms=369.4  inter_chunk_p95_ms=339.1  n_chunks=44  audio_s=27.76  total_gen_ms=5847.08  rtf(model_timing)=4.748

=== 两路 Graph 对比 ===
  TTFA 差距        = 205.7ms
  inter_chunk_p95 差距 = 10.8ms
  RTF 差距         = 0.000
  相对单路 slowdown = 1.88x
```

**三路 Graph（`triple_graph_streaming_benchmark.py`，chunk_size=8，调度顺序 A→B→C）**：

```text
=== 三路 Graph 模式统计 ===
  路 A (Graph): ttfa_wall_ms(audio)=205.4  inter_chunk_max_ms=596.2  inter_chunk_p95_ms=515.3  n_chunks=47  audio_s=29.84  total_gen_ms=6285.89  rtf(model_timing)=4.747
  路 B (Graph): ttfa_wall_ms(audio)=410.8  inter_chunk_max_ms=555.3  inter_chunk_p95_ms=511.7  n_chunks=45  audio_s=28.80  total_gen_ms=6063.41  rtf(model_timing)=4.750
  路 C (Graph): ttfa_wall_ms(audio)=617.1  inter_chunk_max_ms=518.4  inter_chunk_p95_ms=508.8  n_chunks=47  audio_s=29.52  total_gen_ms=6214.92  rtf(model_timing)=4.750

=== 三路 Graph 对比 ===
  TTFA（按调度顺序）: 路A=205ms / 路B=411ms / 路C=617ms
  TTFA 最大差距    = 411.7ms
  inter_chunk_p95 差距 = 6.5ms
  RTF 差距         = 0.003
  相对单路 slowdown = 2.86x
  理论 3路 slowdown ≈ 3.00x
```

**四路 Graph（`quad_graph_streaming_benchmark.py`，chunk_size=8，调度顺序 A→B→C→D）**：

```text
=== 四路 Graph 模式统计 ===
  路 A (Graph): ttfa_wall_ms(audio)=202.1  inter_chunk_max_ms=788.7  inter_chunk_p95_ms=677.1  n_chunks=47  audio_s=29.84  total_gen_ms=6256.17  rtf(model_timing)=4.770
  路 B (Graph): ttfa_wall_ms(audio)=403.7  inter_chunk_max_ms=750.3  inter_chunk_p95_ms=674.6  n_chunks=47  audio_s=29.76  total_gen_ms=6241.07  rtf(model_timing)=4.768
  路 C (Graph): ttfa_wall_ms(audio)=605.7  inter_chunk_max_ms=710.8  inter_chunk_p95_ms=673.2  n_chunks=47  audio_s=29.76  total_gen_ms=6239.70  rtf(model_timing)=4.769
  路 D (Graph): ttfa_wall_ms(audio)=807.9  inter_chunk_max_ms=681.4  inter_chunk_p95_ms=665.0  n_chunks=47  audio_s=30.00  total_gen_ms=6289.04  rtf(model_timing)=4.770

=== 四路 Graph 对比 ===
  TTFA（按调度顺序）: 路A=202ms / 路B=404ms / 路C=606ms / 路D=808ms
  TTFA 最大差距    = 605.8ms
  inter_chunk_p95 差距 = 12.1ms
  RTF 差距         = 0.002
  相对单路 slowdown = 3.76x
  理论 4路 slowdown ≈ 4.00x

  播放流畅性分析:
    每 chunk 音频时长 = 666.7ms
    inter_chunk_p95   = 677.1ms
    缓冲余量          = -10.4ms (可能有卡顿风险)
```

**不同 chunk_size 四路对比**（同一脚本 `quad_graph_streaming_benchmark.py`）：

| chunk_size | 音频时长/chunk | inter_chunk_p95 | 缓冲余量 | 路D TTFA |
|-----------|---------------|----------------|---------|---------|
| 8 | 667ms | 677ms | **-10ms** | 808ms |
| 10 | 833ms | 809ms | **+24ms** | 945ms |
| 12 | 1000ms | 949ms | **+51ms** | 1079ms |

**四路不同 chunk_size 详细数据**：

**chunk_size=10（缓冲余量 +24ms，基本流畅）**：

```text
=== 四路 Graph 模式统计（chunk_size=10）===
  路 A (Graph): ttfa_wall_ms(audio)=237.0  inter_chunk_max_ms=925.2  inter_chunk_p95_ms=808.9  n_chunks=39  audio_s=30.48  total_gen_ms=6396.00  rtf(model_timing)=4.765
  路 B (Graph): ttfa_wall_ms(audio)=472.5  inter_chunk_max_ms=887.1  inter_chunk_p95_ms=803.7  n_chunks=38  audio_s=30.40  total_gen_ms=6378.11  rtf(model_timing)=4.766
  路 C (Graph): ttfa_wall_ms(audio)=708.4  inter_chunk_max_ms=847.9  inter_chunk_p95_ms=794.8  n_chunks=38  audio_s=30.24  total_gen_ms=6342.46  rtf(model_timing)=4.768
  路 D (Graph): ttfa_wall_ms(audio)=945.3  inter_chunk_max_ms=813.2  inter_chunk_p95_ms=791.8  n_chunks=38  audio_s=30.32  total_gen_ms=6358.40  rtf(model_timing)=4.768

  播放流畅性分析:
    每 chunk 音频时长 = 833.3ms
    inter_chunk_p95   = 808.9ms
    缓冲余量          = 24.5ms (播放流畅)
```

**chunk_size=12（缓冲余量 +51ms，非常流畅）**：

```text
=== 四路 Graph 模式统计（chunk_size=12）===
  路 A (Graph): ttfa_wall_ms(audio)=269.7  inter_chunk_max_ms=1061.0  inter_chunk_p95_ms=949.0  n_chunks=31  audio_s=29.68  total_gen_ms=6222.64  rtf(model_timing)=4.770
  路 B (Graph): ttfa_wall_ms(audio)=539.2  inter_chunk_max_ms=1023.4  inter_chunk_p95_ms=942.9  n_chunks=32  audio_s=29.84  total_gen_ms=6256.73  rtf(model_timing)=4.769
  路 C (Graph): ttfa_wall_ms(audio)=809.0  inter_chunk_max_ms=985.4  inter_chunk_p95_ms=937.2  n_chunks=32  audio_s=30.32  total_gen_ms=6355.54  rtf(model_timing)=4.771
  路 D (Graph): ttfa_wall_ms(audio)=1078.7  inter_chunk_max_ms=952.0  inter_chunk_p95_ms=931.0  n_chunks=31  audio_s=29.76  total_gen_ms=6238.24  rtf(model_timing)=4.771

  播放流畅性分析:
    每 chunk 音频时长 = 1000.0ms
    inter_chunk_p95   = 949.0ms
    缓冲余量          = 51.0ms (播放流畅)
```

**关键读数**：

1. **TTFA 随调度顺序线性增长**：每增加一路，后续路 TTFA 增加约 **200ms**（prefill 串行执行导致）。
2. **inter_chunk 随路数线性增长**：N 路时 inter_chunk ≈ 单路 180ms × N × 0.93~0.98（实测 slowdown 略低于理论 N 倍，GPU 利用率优化所致）。
3. **播放临界点**：**4 路 + chunk_size=8** 时 inter_chunk (677ms) 已接近音频时长 (667ms)，缓冲余量仅 -10ms，可能出现偶发卡顿；增大 chunk_size 到 10/12 可改善。

#### 6.3.10.4 产品决策建议

| 目标 | 推荐配置 | 理由 |
|------|---------|------|
| **追求最低 TTFA（首包快）** | **2 路**，chunk_size=8 | 路D TTFA 仅 ~400ms，inter_chunk 350ms 有充足缓冲 |
| **平衡 TTFA 与流畅度（推荐）** | **3 路**，chunk_size=8 | 路D TTFA ~600ms 可接受，inter_chunk 515ms 仍有余量 152ms，播放流畅 |
| **最大化并发数（可接受轻微卡顿）** | **4 路**，chunk_size=10 | chunk_size=10 时缓冲余量 24ms，基本流畅；路D TTFA ~950ms |
| **极致流畅优先** | **4 路**，chunk_size=12 | 缓冲余量 51ms，非常流畅；但路D TTFA 达 ~1080ms，首包较慢 |

**综合结论**：

- **单 GPU 4090 纯 Graph 路径的舒适区：3 路并发**
  - TTFA：首路 200ms，末路 600ms，差距可接受
  - RTF：各路均维持 4.75+（算得比说得快 4 倍以上）
  - 播放流畅：inter_chunk 515ms < 音频时长 667ms，缓冲 152ms

- **4 路是临界点**：需根据业务对 TTFA 和流畅性的容忍度选择 chunk_size（8 有风险，10 基本可用，12 流畅但首包慢）。

- **与混合策略（§6.3.8.3）对比**：
  - 混合策略：1 路快（Graph，RTF 4.7）+ 1 路慢（parity，RTF 0.8），不公平
  - 纯 Graph 多路：各路性能对称（RTF 均 4.7+），调度公平，但并发数受限（3~4 路）

**下一步建议**：
- 若业务需支持 >4 路并发且保持流畅：考虑**多 GPU 部署**（路线 E）或**动态批处理**（路线 C，待原型验证）
- 若 3~4 路已满足业务峰值：采用 **chunk_size=8（3 路）或 chunk_size=10（4 路）**，配置 `--concurrency` 限制并发上限，避免超发导致卡顿

#### 6.3.10.5 与 §6.3.9 的衔接（更新状态表）

更新 **§6.2.8 / §6.3.9** 状态表，补充本节原型：

| 顺序 | 状态（更新后） |
|------|------------------|
| 1 单 session **`session.step()`** | **已落地**：`ParityStreamSession` + `parity_generate_streaming`（**§6.3.2 / §6.3.3**） |
| 2 双 session 轮流 **`step()`** | **原型已落地**：`parity_dual_session_benchmark.py`（**§6.3.6**） |
| 3 **Graph 路径评估** | **原型已落地 + 实测数据已采集**：单路基线（**§6.3.8.4**）+ 混合策略（**§6.3.8.3**）+ **多路 Graph 并发基准（本节 §6.3.10）** |

---

## 6.4 路线B的综合方案与实现

### 6.4.1 路线B的综合方案

本节基于 **§6.2.8 顺序 1～3** 已全部实施并验证的基础上，提出**服务进程级多路穿插调度**的完整方案——即 **`openai_server_v5.py`** 的架构设计。

#### 6.4.1.1 方案可行性分析（基于现有实现与测试数据）

**§6.2.8 原型验证状态**：

| 顺序 | 原型 | 测试结论 | 可行性 |
|------|------|----------|--------|
| 1 单 session `step()` | `ParityStreamSession` | 拆步不破坏正确性，输出与流式逐块对齐 | ✅ |
| 2 双 session 轮流 `step()` | `parity_dual_session_benchmark.py` | 两路TTFA公平（差距~100ms），但parity路径RTF<1（慢于实时） | ⚠️ 仅parity路径 |
| 3 Graph 路径评估 | `graph_vs_parity_benchmark.py` + `hybrid_graph_parity_benchmark.py` + `dual/triple/quad_graph_streaming_benchmark.py` | **Graph比Parity快4~6倍**，RTF=4.7+（算得比说得快）；多路Graph并发时TTFA线性增长（每路+~200ms），inter_chunk≈180ms×N | ✅ 3路以内流畅 |

**核心结论**：

1. **CUDA Graph 路径是 RTX 4090 的必选项**：单路RTF=4.7，3路并发RTF仍维持4.7+（各路计算效率不下降），只是inter_chunk增长到~515ms（音频时长667ms，仍有缓冲）。
2. **多路并发的瓶颈是 inter_chunk**：不是计算速度问题，而是调度频率问题。N路时每路每轮只能产1个chunk，inter_chunk≈N×单路chunk耗时。
3. **单 GPU 4090 舒适区：3路**，临界点：4路（需调大chunk_size）。

#### 6.4.1.2 openai_server_v5 架构设计

**核心思想**：

- 用一个 **并发槽位信号量**（`threading.Semaphore(concurrency)`）限制**同时在途**的流式请求数量
- 拿到槽位的请求被登记为一个 **`TTSWorker`** 对象（dataclass，不是独立线程），加入全局 **active_workers** 字典
- 真正在 GPU 上推进生成的只有**一个独立的「全局轮询调度器线程」**：每轮遍历所有活跃 worker，**各调用一次 `step()` 产一个 PCM chunk**，然后进下一轮——从而把 GPU 按 chunk 公平切给各路
- 拿不到槽位的请求进入 **FIFO 等待队列**；另一个「队列管理线程」在槽位空出时负责晋升
- MP3 非流式请求也要先获取**同一个槽位**再整段合成，避免它绕过 v5 的公平性边界去抢 GPU

**架构图**：

```
┌─────────────────────────────────────────────────────────────────┐
│                     openai_server_v5.py                          │
├─────────────────────────────────────────────────────────────────┤
│  HTTP Layer (FastAPI, 事件循环)                                  │
│  └── POST /v1/audio/speech                                       │
│         ├── 流式 wav/pcm：创建 TTSWorker，尝试拿并发槽位         │
│         │     拿到 → 加入 active_workers（交给轮询调度器）       │
│         │     拿不到 → 入 FIFO 等待队列                          │
│         └── 非流式 mp3：进入 **非流式 FIFO 队列** → 独占 GPU →   │
│              整段合成返回（期间暂停流式轮询，见读写锁）           │
├─────────────────────────────────────────────────────────────────┤
│  Concurrency Gate: threading.Semaphore(N = --concurrency)        │
│  ├── 流式请求完成（或失败）时由调度器线程 release 槽位           │
│  └── MP3 请求在 finally 中 release                               │
├─────────────────────────────────────────────────────────────────┤
│  Queue Manager Thread（等待队列晋升者）                           │
│  └── while True: _submit_to_scheduler(queue.get())               │
│         尝试拿槽位 → 成功则加入 active_workers                    │
├─────────────────────────────────────────────────────────────────┤
│  Global Round-Robin Scheduler Thread（唯一的 GPU 使用者）         │
│  while running:                                                  │
│      workers = snapshot(active_workers)                          │
│      for w in workers:                                           │
│          w.step()           # 首次 step 时 init_generator()      │
│                             # 之后每次 next(gen) → 一个 PCM chunk │
│          if w.finished: 标记待清理                                │
│      清理已完成 worker → 给响应队列投 None（结束）→ release 槽位  │
├─────────────────────────────────────────────────────────────────┤
│  Response Streaming                                              │
│  └── 每个 worker 有自己的 response_queue；HTTP handler 用         │
│      run_in_executor 异步从队列取 bytes → chunked streaming 下行 │
└─────────────────────────────────────────────────────────────────┘
```

**说明**：
- **只有一个调度器线程在跑 GPU 推理**，所以不存在多线程在 GPU 上互相争抢的情况；多路之间的"公平"来自于**每轮各给一 chunk**这条硬约束。
- `TTSWorker` 本身是纯数据与状态容器；**首次 `step()` 时**才懒加载底层生成器（`init_generator`），这样 HTTP 层可以尽早返回 `StreamingResponse` 开始下行。
- 客户端如果中途断连，`audio_stream()` 捕获 `CancelledError / GeneratorExit` 后会把 `worker.client_cancelled=True`；调度器之后 `step()` 仍会推进生成（避免底层状态不一致），只是不再把 chunk 塞入响应队列，也不阻塞其它路。

**与 `openai_server_v4` 的关键区别**：

| 维度 | v4 | v5（本方案） |
|------|-----|-------------|
| 并发控制 | `asyncio.Semaphore` 限制同时在途请求数 | `threading.Semaphore` 槽位 + FIFO 等待队列晋升 |
| GPU 调度 | 每个请求在自己的 producer 线程内跑 `generate_voice_clone_streaming`，多线程在 GPU 上**随机争抢** | **单一调度器线程**按路轮询，每路每轮产一个 chunk，**GPU 时间按 chunk 切片公平分配** |
| 首包公平性 | 先到先服务，后到者排队等前面生成完或 `concurrency` 释放 | 多路同时进入 active 后，首包时间差距≈每路一个 chunk 耗时，可控 |
| 块间间隔 | 单路独占时最优；多路争抢时抖动大 | 多路时 inter_chunk≈N×单路，但**各路一致** |
| MP3 并发 | 与流式共享 `asyncio.Semaphore`（并发受控，会交叠跑在 GPU 上） | **非流式 FIFO 串行独占 GPU**，不参与轮询时分复用，流式轮询与非流式整段通过读写锁互斥（见 §6.4.1.3 第 5 条） |
| 适用场景 | 单路追求极致速度 | **多路并发追求公平性+流畅性平衡** |

#### 6.4.1.3 关键技术决策

**1. Graph 路径还是 Parity 路径？**

- **推荐：全部使用 Graph 快路径**（`parity_mode=False`）
- **理由**：§6.3.10 实测，即使4路并发，Graph RTF仍维持4.7+，远高于Parity的0.8
- **代价**：inter_chunk随路数增长，但3路以内（舒适区）仍有152ms缓冲，播放流畅
- **v5 服务（`openai_server_v5.py`）流式与 CUDA Graph（重要）**：进程用 `--concurrency` 限制**同时在途流式合成条数**，用 `--replicas` 决定进程内加载几份 `FasterQwen3TTS`。仅在 **未** `--disable-cuda-graph`、各副本已捕获 `predictor_graph` / `talker_graph`，且 **`--replicas` ≥ `--concurrency`** 时，才会置 `_v5_stream_use_cuda_graph=True`，于是 `TTSWorker.init_generator` 里 **`parity_mode=False`**，`generate_voice_clone_streaming` 走 **CUDA Graph 快路径**（与本文其它章节所述 Graph / `fast_generate_streaming` 语义一致）。若 **`replicas < concurrency`**，启动时会 **WARNING**「流式会话数可能超过 CUDA Graph 安全副本」并 **强制流式 Parity**（`parity_mode=True`），避免多路并发共抢单份 Graph 静态缓冲导致异常块长或「卡碟」杂音。可按启动行 **`STARTUP stream_cuda_graph=`** / **`parity_stream_fallback=`** 对照（见 **§6.4.3.5**）。

**2. 调度粒度：chunk 级还是 step 级？**

- **推荐：chunk 级**（每调度一次产一个PCM chunk）
- **理由**：
  - Graph 路径的最小调度量子是 **一整次 predictor 图 + 一次 talker 图步**（§6.2.9）
  - 拆到step级需要放弃Graph或重做捕获粒度，RTF会大幅下降
  - chunk级已在 `quad_graph_streaming_benchmark.py` 验证可行

**3. chunk_size 选择？**

| 并发数 | 推荐 chunk_size | 理由 |
|--------|----------------|------|
| 2路 | 8（667ms音频） | 缓冲317ms，非常流畅 |
| 3路 | 8（667ms音频） | 缓冲152ms，舒适区 |
| 4路 | 10（833ms音频） | 缓冲24ms，基本流畅；若追求更流畅可用12 |

> **`stream_chunk_size` 配置优先级**（实现细节）：
> CLI `--stream-chunk-size` > `config_v5.json.stream_chunk_size` > 默认 8。
> 建议把稳定值写在 `config_v5.json` 里，调参阶段才用 CLI 覆盖。

**4. 最大并发数（`--concurrency`）建议？**

- **保守值：3**（chunk_size=8，舒适区）
- **激进值：4**（chunk_size=10，临界点，需业务可接受偶发卡顿或稍慢首包）
- **超过4路**：不推荐单GPU，应考虑多GPU部署（路线E）

**5. MP3 非流式请求如何参与调度？**

**设计声明（语义）**：

- **非流式请求不参与轮询时分复用**。也就是说，MP3 这类整段合成请求，不会被调度器拆成 chunk 和流式请求交替推进，而是**拿到 GPU 后整段一次性算完**。
- **非流式请求之间按 FIFO 先来先处理**。即多个 MP3 请求并发到达时，严格按到达顺序依次完成，不互相切片。
- **非流式与流式请求之间彼此互斥**。当有非流式请求正在合成时，调度器会**暂停**对流式 worker 的 `step()` 推进，直到该非流式请求完成；反之，流式调度器正在切片推进时，新到达的非流式请求需要**等到当前轮询空窗**才能独占 GPU。

**这样设计的理由**：

- MP3 整段合成本身就是**非实时语义**（客户端在等 HTTP 响应结束才会播放），对"首包时间 / 块间间隔"不敏感；反而对**总时长 `total_ms`** 敏感——如果和流式共享 GPU 时分复用，单条 MP3 的 `total_ms` 会被显著拉长。
- 让非流式 FIFO 串行处理，既能保证每条 MP3 的端到端时延可预期，也能让流式请求在非 MP3 时段完全按 §6.4.1.2 的公平轮询走，不必承担 MP3 整段独占 GPU 带来的大空窗。

**代价**：

- 多条 MP3 堆积时，排在后面的会等待较久；可接受，因为 MP3 路径本身就是"牺牲首包时延、换取一次性稳定交付"的场景。
- 如果某些业务真的对 MP3 时延敏感，应在网关层把它改走 `response_format=wav` 流式路径，而不是让它参与轮询。

> **实现要点**（代码层面，已落地于 `openai_server_v5.py`）：
> - **`_v5_nonstream_fifo_lock: asyncio.Lock`**：非流式请求进入 handler 后先 `async with` 这把锁。CPython 的 `asyncio.Lock` 是 **FIFO** 的，天然保证"先来先处理"。
> - **`_v5_gpu_rwlock: _WriterPreferringRWLock`**：自定义**写者优先**读写锁。
>   - 流式调度器每次调用 `worker.step()` **前后**成对 `acquire_read / release_read`；
>   - 非流式 MP3 在拿到 FIFO 锁后 `acquire_write`，**完成整段合成**后 `release_write`；
>   - 写者优先意味着：MP3 一旦进入等待队列，调度器下一次 `acquire_read` 就会被阻塞，等 MP3 结束再继续。因此非流式请求不会被流式的轮询饿死，流式请求只会被非流式**成段挂起**一次（挂起时长≈一条 MP3 的 `total_ms`）。
> - **流式并发槽位 `_v5_thread_pool_sem` 仅限流式使用**，非流式不占用它；这样非流式在空载时可以无等待直达 GPU，不会被"槽位已满但都是流式"的场景卡住。
> - 综合上面三项，就同时实现了"**非流式 FIFO、非流式独占、流式公平轮询**"三条语义。

#### 6.4.1.4 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 调度器线程与 HTTP 事件循环之间的队列同步开销 | 低 | 通过 `queue.Queue` + `run_in_executor` 桥接，v4 已验证模式 |
| 轮询调度导致单路 TTFA 相对 v4 升高 | 必然 | 这是公平性的代价；多路场景下对应的是"末路 TTFA 也受控"，综合验收 |
| chunk_size 增大导致首包变慢 | 中 | 4 路以上才需调大 chunk_size，首包每档约+150ms，仍在 1s 内 |
| 客户端中途断连，worker 残留在 active_workers | 中 | HTTP 层捕获 `CancelledError/GeneratorExit` 置 `client_cancelled=True`，调度器继续推进直至 `StopIteration`，但不再入队，也不阻塞其它路 |
| 单一调度器线程变成瓶颈 | 低 | 调度器只负责"叫 `next()`"，真正耗时在 GPU；实测 Graph 路径 3 路时 CPU 基本不忙 |
| OOM | 低 | `torch.inference_mode()` + 单/多模型副本共享；显存主要由 CUDA Graph 静态占用，可控 |
| 与 v4 行为差异导致回归 | 中 | 独立为 v5 文件，不改动 v4，可 A/B 测试对比 |

#### 6.4.1.5 验收指标（建议上线检查项）

> 指标定义（`ttfa_wall_ms`、`inter_chunk_*`、`rtf`）以 **§5.1.10** 为准；其中 **`ttfa_wall_ms` 含槽位等待时间**，多路压测时"路末 TTFA"的组成 = 槽位等待 + 自身首包耗时，需一并达标。

| 指标 | 阈值（3路，chunk_size=8） | 阈值（4路，chunk_size=10） |
|------|-------------------------|--------------------------|
| TTFA（路A/路末，`ttfa_wall_ms`） | ~200ms / ~600ms | ~250ms / ~950ms |
| TTFA 差距（末路 - 首路） | ≤ 450ms | ≤ 750ms |
| `inter_chunk_p95_ms` | ≤ 550ms | ≤ 850ms |
| RTF | ≥ 4.5 | ≥ 4.5 |
| 播放卡顿感知 | 无 | 偶发（可接受） |
| 错误率（5xx + 断连） | < 0.5% | < 0.5% |

### 6.4.2 路线B的综合方案实现

本节把 §6.4.1 的设计逐一**落到代码**，说明 `examples/openai_server_v5.py` 的每一块对应 §6.4.1 的哪一条决策、如何启动、如何观察与压测。代码已在仓库中落地，直接按本节的命令即可跑起来。

#### 6.4.2.1 涉及文件与职责

| 文件 | 作用 | 与 v4 的关系 |
|------|------|-------------|
| `examples/openai_server_v5.py` | v5 主服务入口。HTTP 路由、TTSWorker、轮询调度器线程、队列管理线程、读写锁、FIFO 锁、指标日志、启动摘要 | 由 `openai_server_v4.py` 复制改造，保留 v4 的日志/预热/配置解析风格 |
| `examples/voice_registry_v5.py` | v5 音色注册表（JSON + 文件锁 + 原子替换） | 与 `voice_registry_v4.py` 结构一致，仅类名带 `V5` 后缀，便于隔离 |
| `examples/voice_manager_router_v5.py` | v5 音色管理 REST 路由（`/voices` 增删改查、`/health` 健康检查） | 与 `voice_manager_router_v4.py` 结构一致 |
| `config_v5.json` | v5 配置（注册表路径、音色目录、`stream_chunk_size`、预热文本、推理日志等） | 新增 `stream_chunk_size`、更新 `voices_registry_path` |

> 三个 `*_v5.py` 都是**版本隔离复制**（§六通用约定），不改 v4 文件，保留 A/B 回滚能力。

#### 6.4.2.2 模块分层与关键符号

`openai_server_v5.py` 自上而下按 §6.4.1.2 架构图分五层：

1. **HTTP 层**（FastAPI）：`POST /v1/audio/speech` 入口 `create_speech`，以及挂载的 `voice_manager_router_v5.router`（音色管理 API）。
2. **并发闸门**：两把独立的"门"
   - `_v5_thread_pool_sem: threading.Semaphore(concurrency)` —— **流式**请求的在途槽位
   - `_v5_nonstream_fifo_lock: asyncio.Lock` —— **非流式** MP3 之间 FIFO 串行
   - `_v5_gpu_rwlock: _WriterPreferringRWLock` —— **流式↔非流式**互斥（写者优先）
3. **队列**：
   - `_v5_request_queue: queue.Queue` —— 流式超出槽位时的 FIFO 等待队列
   - `_v5_active_workers: Dict[int, TTSWorker]` —— 持有槽位、等待调度器推进的活跃集合
4. **调度器 + 队列管理**：两个独立线程
   - `_v5_scheduler_thread` 跑 `_scheduler_loop()`：遍历 `_v5_active_workers` 逐个 `step()`
   - `_v5_queue_manager_thread` 跑 `_queue_manager_loop()`：有空槽时把等待队列里的请求推进 active
5. **响应流**：每个 `TTSWorker` 有自己的 `response_queue`，HTTP 层用 `run_in_executor` 异步从队列取字节并通过 `StreamingResponse` 下发。

#### 6.4.2.3 `TTSWorker` —— 每个流式请求的状态载体

**关键字段**（对应 §6.4.1.2 架构图里的 worker 节点）：

```python
@dataclass
class TTSWorker:
    req_id: int
    voice_cfg: dict
    text: str
    generator: Optional[Iterator] = None            # 懒加载的底层生成器
    response_queue: queue.Queue                     # 调度器产出 → HTTP 消费
    t_start / t_ttfa0 / t_bench0                    # 三个时间锚点
    ttfa_wall_ms / ttfa_ms / ttfa_cuda_graphs_ms    # 三类 TTFA（与 v4 口径一致）
    chunk_gaps_ms: List[float]                      # 块间间隔样本
    total_gen_ms / total_audio_s / n_chunks         # RTF 所需累计量
    finished / error / client_cancelled             # 结束/异常/客户端断连标志
```

**两个关键方法**：

- `init_generator()`：**调度器线程首次调用时**懒加载底层生成器；之前 HTTP handler 先行返回 `StreamingResponse`。这样客户端连接建立无需等模型准备。
- `step()`：从生成器拉一个 PCM chunk → 更新计时指标 → `response_queue.put(_to_pcm16(chunk))`。
  - 首块：`cuda.synchronize()` 后记录 `ttfa_cuda_graphs_ms`；同时冻结 `ttfa_ms`（累计 `prefill_ms+decode_ms` 首次 >0 的值，与 v4 `_stream_chunks` 同口径）
  - 客户端断连：`client_cancelled=True` 时 `step()` 继续推进生成器（避免底层状态不一致），但**不再入队**——避免队列无限膨胀
  - 结束/异常：`_log_final_metrics()` 落一条 `METRICS mode=stream` 行

#### 6.4.2.4 `_WriterPreferringRWLock` —— 流式 ↔ 非流式互斥

§6.4.1.3 第 5 条的实现核心。标准条件变量实现：

```python
class _WriterPreferringRWLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self):
        with self._cond:
            while self._writer or self._writers_waiting > 0:   # 写者优先
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self):
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._writers_waiting -= 1

    def release_write(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()
```

**使用点**（两处，成对出现）：

- 调度器 `_scheduler_loop`：每次调用 `worker.step()` 前后 `acquire_read / release_read`
- MP3 handler：拿到 `_v5_nonstream_fifo_lock` 后 `await loop.run_in_executor(None, _v5_gpu_rwlock.acquire_write)`，合成结束 `release_write`

#### 6.4.2.5 `_scheduler_loop` —— 单调度线程轮询

```python
while _v5_scheduler_running:
    with _v5_scheduler_lock:
        active_workers = list(_v5_active_workers.values())
    if not active_workers:
        time.sleep(0.01); continue
    finished_ids = []
    for worker in active_workers:
        if worker.finished or worker.error:
            finished_ids.append(worker.req_id); continue
        _v5_gpu_rwlock.acquire_read()           # ① 每步进入读锁
        try:
            worker.step()                       # ② 产一个 PCM chunk
            if worker.finished or worker.error:
                finished_ids.append(worker.req_id)
        finally:
            _v5_gpu_rwlock.release_read()       # ③ 立即释放——给 MP3 写者可插入的窗口
    if finished_ids:                            # ④ 清理 + 释放流式槽位
        with _v5_scheduler_lock:
            for rid in finished_ids:
                w = _v5_active_workers.pop(rid, None)
                if w is not None:
                    w.response_queue.put(None)  # 结束标记
        for _ in finished_ids:
            if _v5_thread_pool_sem:
                _v5_thread_pool_sem.release()
    time.sleep(0.001)
```

**要点**：

- **读锁粒度 = 一次 `step()`**，不是一轮。这是 MP3 能在很短时间内切入的前提——最长只等一次 `step()`（≈`chunk_size×12Hz`，百来毫秒量级）。
- **调度器从不直接调用 GPU/模型 API**，它只负责"叫 `next()`"。GPU 真正的计算发生在 `worker.step()` 内部。
- **finished worker 清理后立刻 `release()` 流式槽位**——与此同时 `_queue_manager_loop` 会把等待队列里的请求推进 active。

#### 6.4.2.6 HTTP 入口 `create_speech` —— 三条分支

```python
@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    # 1) 基本校验：模型就绪、文本非空、音色解析、格式合法
    ...
    req_id = _next_req_id()

    # 2) 非流式 MP3 分支：FIFO + 独占
    if fmt == "mp3":
        async with _v5_nonstream_fifo_lock:                         # FIFO 串行
            await loop.run_in_executor(None, _v5_gpu_rwlock.acquire_write)   # 独占 GPU
            try:
                audio_arrays, sr = await loop.run_in_executor(
                    None, lambda: _pick_model().generate_voice_clone(...))
                _log_metrics(..., mode="non_stream_mp3")
                return Response(content=_to_mp3_bytes(...), media_type="audio/mpeg")
            finally:
                _v5_gpu_rwlock.release_write()

    # 3) 流式 wav/pcm 分支：拿槽位 → 进 active 或等待队列
    worker = TTSWorker(req_id, voice_cfg, req.input, t_ttfa0=time.perf_counter())
    if not _submit_to_scheduler(worker):            # 非阻塞尝试占槽
        _v5_request_queue.put(worker)                # 满则入 FIFO 等待
    async def audio_stream():
        try:
            async for raw in _stream_chunks_v5(worker, fmt):
                yield raw
        except (asyncio.CancelledError, GeneratorExit):
            worker.client_cancelled = True          # 断连后调度器不再入队
            raise
    return StreamingResponse(audio_stream(), media_type=...)
```

**请注意三条分支的时间锚点差异**：

| 分支 | `ttfa_wall_ms` 起点 | 含哪些等待 |
|------|--------------------|-----------|
| 流式 | `t_ttfa0 = time.perf_counter()`（`TTSWorker` 创建时） | 流式槽位等待 + 调度轮询首块耗时 |
| 非流式 MP3 | `t_ttfa0 = time.perf_counter()`（进 handler 时） | FIFO 锁等待 + 写锁等待 + 整段合成 |

#### 6.4.2.7 配置与参数优先级

`config_v5.json` 中的字段：

```json
{
  "voices_registry_path": "./voices_registry_v5.json",
  "tone_wav_file_dir": "../faster-qwen3-tts-assets/tone_wav_files",
  "allowed_audio_formats": ["wav", "mp3", "flac", "m4a"],
  "max_audio_file_size_mb": 20,
  "enable_orphan_cache_cleanup": true,
  "orphan_cache_cleanup_threshold": 30,
  "prime_text": "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。",
  "prime_stream_first_chunk": true,
  "prime_stream_max_new_tokens": 48,
  "stream_chunk_size": 8,
  "metrics_log_mode": "DEBUG",
  "inference_logging": { ... }
}
```

**CLI 参数覆盖优先级**（高 → 低）：

| 参数 | CLI | config | 默认 |
|------|-----|--------|------|
| `stream_chunk_size` | `--stream-chunk-size N` | `stream_chunk_size` | 8 |
| `prime_text` | `--prime-text "..."` | `prime_text` | `"."` |
| `prime_stream_first_chunk` | `--prime-stream-first-chunk` (or) | `prime_stream_first_chunk` | `false` |
| `prime_stream_max_new_tokens` | `--prime-stream-max-new-tokens N` | `prime_stream_max_new_tokens` | 48 |
| `metrics_log_mode` | `--metrics-log-mode DEBUG/DEV` | `metrics_log_mode` 或环境变量 `TTS_METRICS_LOG_MODE` | `DEBUG` |
| 并发数 | `--concurrency N` | — | 3 |
| 副本数 | `--replicas N` | — | 1 |

#### 6.4.2.8 启动命令

**最小启动**（使用 `config_v5.json` 里的默认值）：

```bash
cd tts-jiang/faster-qwen3-tts
python examples/openai_server_v5.py \
  --config config_v5.json \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --host 0.0.0.0 --port 8000 \
  --concurrency 3 \
  --stream-chunk-size 12 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

**四路临界区**（需要稍大 chunk 缓冲流畅性）：

```bash
python examples/openai_server_v5.py \
  --config config_v5.json \
  --concurrency 4 \
  --stream-chunk-size 10
```

**启动后你应当依次看到**（截取关键行）：

```
STARTUP cmdline=python examples/openai_server_v5.py --concurrency 3 ...
STARTUP --concurrency=3 --replicas=1
STARTUP stream_chunk_size(effective)=8
STARTUP metrics_log_mode=DEBUG
Loading model: Qwen/Qwen3-TTS-12Hz-0.6B-Base (replicas=1)
Warming up replica 1/1...
Priming: voice='zhentong_v1' replica=1/1
Scheduler started
Queue manager thread started
Server v5 ready: host=0.0.0.0 port=8000 concurrency=3 chunk_size=8
```

#### 6.4.2.9 指标字段与 v4 的对应

v5 的 `_log_metrics` 支持两种字段模式：

- **`DEBUG`（默认）**：输出完整诊断字段，适合排查 `ttfa_wall_ms`、prefill、首块 decode、ICL 等问题。
- **`DEV`**：只输出线上常看核心字段，减少日志宽度，适合日常开发/联调或给运维看板消费。

配置优先级为：

```text
CLI --metrics-log-mode > config_v5.json.metrics_log_mode > 环境变量 TTS_METRICS_LOG_MODE > DEBUG
```

**DEV 模式输出字段**：

```text
METRICS mode=stream ttfa_wall_ms=... ttfa_cuda_graphs_ms=... inter_chunk_max_ms=... inter_chunk_p95_ms=... rtf=... audio_s=... total_gen_ms=... total_ms=... n_chunks=...
```

**DEBUG 模式输出字段**（流式 `mode=stream`，当前排障版）：

```text
METRICS req_id=... mode=stream ttfa_wall_ms=... ttfa_ms=... ttfa_cuda_graphs_ms=... first_prefill_ms=... first_decode_ms=... first_overhead_ms=... prefill_len=... attention_mask_shape=... trailing_text_len=... icl=... parity_mode=... inter_chunk_max_ms=... inter_chunk_p95_ms=... rtf=... audio_s=... total_gen_ms=... total_ms=... n_chunks=...
```

**当前所有字段的通俗解释**：

| 字段 | 通俗解释 | 常见出现 |
|------|----------|----------|
| `req_id` | 本服务进程给每条请求分配的流水号。主要用于排障时把同一条请求的日志串起来看。**DEV 模式不输出**。 | `DEBUG/stream` |
| `mode` | 这条指标对应哪种合成路径。`stream` 表示 `wav/pcm` 流式；`non_stream_mp3` 表示 MP3 整段合成。 | 全部 |
| `ttfa_wall_ms` | 用户侧最关心的“服务端首段音频等了多久”。从请求通过校验、进入本服务合成路径开始，到第一段 PCM 音频在服务端准备好并入队为止。**流式下包含槽位排队/等待调度的时间**。 | 全部 |
| `ttfa_ms` | 模型 timing 里首块音频对应的 `prefill_ms + decode_ms`。更像“模型自己账本里首块算了多久”，常用于和 `ttfa_wall_ms` 对比，判断慢在排队还是慢在模型。**DEV 模式不输出**。 | `DEBUG/stream`，MP3 为整段耗时近似 |
| `ttfa_cuda_graphs_ms` | 按 README / benchmark 口径测的首块 CUDA Graph 墙钟耗时：`cuda synchronize → 开表 → 调流式生成器取首块 → cuda synchronize → 停表`。更适合和官方/离线 benchmark 对齐。 | `stream` |
| `first_prefill_ms` | 首块里 prefill 阶段花了多久。可以理解为模型先读完整输入、参考音频提示、音色上下文并建立初始 KV 的时间。**只用于 DEBUG 排障**。 | `DEBUG/stream` |
| `first_decode_ms` | 首块里 decode 阶段花了多久。可以理解为凑出第一块 codec/audio 所需的自回归生成时间。**只用于 DEBUG 排障**。 | `DEBUG/stream` |
| `first_overhead_ms` | `ttfa_cuda_graphs_ms - ttfa_ms`。粗略表示首块模型 timing 之外的包裹开销，例如 Python 调用、codec 到 PCM、同步与计时口径差异等。**不是越小越绝对正确，只用于相对比较**。 | `DEBUG/stream` |
| `prefill_len` | prefill 输入序列长度。排查“是不是这条输入变长导致 prefill 慢”时看它。 | `DEBUG/stream` |
| `attention_mask_shape` | attention mask 的形状，例如 `1x112`。用于确认 batch/序列长度是否和预期一致。 | `DEBUG/stream` |
| `trailing_text_len` | 还要在后续 decode 阶段持续注入/对齐的文本隐藏状态长度。短文本通常小，长文本更大。 | `DEBUG/stream` |
| `icl` | 是否走 ICL 音色克隆路径。`1` 表示带参考音频 codec 上下文，`0` 表示不是 ICL。 | `DEBUG/stream` |
| `parity_mode` | 是否走 parity 非 Graph 路径。`0` 表示 CUDA Graph 快路径，`1` 表示 parity 慢路径。线上压测通常希望它是 `0`。 | `DEBUG/stream` |
| `inter_chunk_max_ms` | 同一条流式请求里，相邻两段音频块在服务端入队之间的最大间隔。越大越容易听到卡顿或客户端缓冲见底。 | `stream` |
| `inter_chunk_p95_ms` | 相邻音频块间隔的 95 分位，比 max 更能代表大多数块的连续性。压测时主要盯这个值。 | `stream` |
| `rtf` | 实时率，本服务按“生成出来的音频秒数 ÷ 模型累计生成耗时秒数”计算。**越大表示算得越快**；例如 `4.7` 表示 1 秒墙钟模型时间大约能生成 4.7 秒音频。 | 全部 |
| `audio_s` | 本次请求最终生成的音频总时长，单位秒。若明显超过文本预期很多，可能是自回归 runaway 续说，见 §6.6。 | 全部 |
| `total_gen_ms` | 模型 timing 里累计的生成耗时，约等于各 chunk 的 `prefill_ms/decode_ms` 累加。它是计算 `rtf` 的分母之一。 | `stream` |
| `total_ms` | 服务端从开始处理这条合成到最终打 METRICS 的总墙钟时间。流式下通常包含整个流生成与下发队列过程；MP3 下接近整段合成和转码总时间。 | 全部 |
| `n_chunks` | 本次流式请求一共产出了多少个 PCM chunk。和 `audio_s` 一起看，能帮助发现异常超长生成。 | `stream` |

**两种模式怎么选**：

- 排查性能差异、首包慢、ICL/prefill 问题时，用 `DEBUG`。
- 日常开发、压测看板、日志留存时，用 `DEV`，重点看 `ttfa_wall_ms`、`inter_chunk_p95_ms`、`rtf`、`audio_s`、`total_ms`。

#### 6.4.2.10 压测（最小可复现）

**3 路舒适区**：

```bash
# 在另一个 shell 执行
python examples/parity_dual_session_benchmark.py \
  --host http://127.0.0.1:8000 \
  --voice zhentong_v1 \
  --concurrency 3 \
  --texts examples/bench_texts_zh.txt
```

核对 `inference_v5_*.log` 里每条 `mode=stream` 是否满足 §6.4.1.5 阈值表的 3 路一行：

- `ttfa_wall_ms` 首路 ~200ms / 末路 ~600ms、差距 ≤ 450ms
- `inter_chunk_p95_ms` ≤ 550ms
- `rtf` ≥ 4.5

**混合场景**（流式 + MP3 并发）：同时对 `/v1/audio/speech` 发 2 路流式 + 1 路 `response_format=mp3`，验证：

- MP3 的 `total_ms` 与单独发一条 MP3 接近（说明确实独占 GPU，没有被流式时分复用拖慢）
- 流式的 `inter_chunk_max_ms` 会出现一次 ≈ MP3 `total_ms` 的大空窗（正是 §6.4.1.3 第 5 条预告的"成段挂起"）
- MP3 完成后，流式恢复原有 `inter_chunk_p95_ms`

#### 6.4.2.11 与 §6.4.1 的逐条对应

| §6.4.1 决策 | §6.4.2 落地位置 |
|-------------|-----------------|
| 架构图（HTTP/闸门/队列/调度/响应流五层） | `openai_server_v5.py` 模块内五段分区（见 §6.4.2.2） |
| 全路 Graph 快路径 | `TTSWorker.init_generator` 中 `parity_mode = (model.talker_graph is None)` |
| chunk 级调度 | `_scheduler_loop` → `worker.step()` 每次产一个 chunk |
| `stream_chunk_size` 配置优先级 | `main()` 中 CLI > config > 默认 8 |
| 流式并发槽位 | `_v5_thread_pool_sem = threading.Semaphore(args.concurrency)` |
| 流式等待队列 | `_v5_request_queue` + `_queue_manager_loop` |
| **非流式 FIFO**（§6.4.1.3 第 5 条 ①） | `_v5_nonstream_fifo_lock: asyncio.Lock`（CPython FIFO） |
| **非流式独占 GPU**（§6.4.1.3 第 5 条 ②） | `_v5_gpu_rwlock.acquire_write()` 前后包住 `generate_voice_clone` |
| **流式↔非流式互斥 + 写者优先**（§6.4.1.3 第 5 条 ③） | `_WriterPreferringRWLock` + 调度器每步 `acquire_read/release_read` |
| 客户端断连处理（§6.4.1.4） | `audio_stream()` 捕获 `CancelledError/GeneratorExit` → `worker.client_cancelled=True` → `step()` 继续推进但不入队 |
| 指标字段模式 | `_log_metrics` 按 `_metrics_log_mode` 输出；`DEBUG` 保留完整诊断字段，`DEV` 只保留线上核心字段，详见 §6.4.2.9 |

#### 6.4.2.12 已知限制与后续工作

- **非流式"成段挂起"对流式的体感**：当 MP3 的 `total_ms` 较大（例如合成长文本）时，流式的 `inter_chunk_max_ms` 会对应出现一次大空窗；网关若希望"永不挂起"，需要在 v5 之上再做一层"**分片式非流式**"（把 MP3 也拆成 chunk 跑轮询，但代价是端到端时长变长）——此项不在 §6.4 范围内，作为 §6.5 候选。
- **多副本模型**：`--replicas N` 时 `_pick_model` 会轮询各副本，但**调度器仍是单线程**——真正的多副本并行需要每个副本一个调度线程（`§6.5` 的多 GPU / 多副本扩展）。
- **严格 FIFO 仅限 MP3 之间**：流式在槽位满时入 `queue.Queue`，本身是 FIFO；但"流式先到先服务"和"非流式先到先服务"是两条独立线，混排序时以"先非流式独占 → 再流式轮询"的优先级为准。

### 6.4.3 使用指南

以下列出 v5 多路 Graph 轮询调度版所需的文件、配置项、启动命令、管理接口与客户端调用方式；**按本节即可独立完成部署与联调**，无需翻阅文档其他章节。与 **§5.2（v4 使用指南）** 重合处会直接引用，只描述 v5 的差异。

#### 6.4.3.1 新增文件及功能说明

下列为 **`openai_server_v5.py` 多路 Graph 轮询调度版** 落地时 **新增或修改** 的文件清单（相对 `faster-qwen3-tts/`）。**不改动** v3/v4 既有入口，v5 以独立路径并存，保留 A/B 回滚能力（§6.4.2.1）。

| 路径 | 类型 | 功能说明 |
|------|------|----------|
| `config_v5.json` | **新增** 配置 | v5 专用：注册表路径、音频根目录、上传白名单与大小、孤儿缓存清理；在 v4 基础上新增 **`stream_chunk_size`**（流式 chunk 大小，3 路舒适区建议 **8**、4 路临界区建议 **10 或 12**，参见 **§6.4.1.3 第 3 点**）与 **`metrics_log_mode`**（`DEBUG` 完整诊断字段 / `DEV` 线上核心字段）；保留 **`prime_text` / `prime_stream_first_chunk` / `prime_stream_max_new_tokens`** 与可选对象 **`inference_logging`**，语义与 v4 相同（见 **§5.2.4**）。 |
| `voices_registry_v5.json` | **新增** 数据 | 音色注册表本体：结构与 v4 完全一致（顶层 `voice_id → { ref_audio, ref_text, language, status, version, updated_at }`）。与 v4 **互不共享**——通过独立文件避免 v4/v5 并存时串扰；如需迁移可直接复制 v4 注册表并改名。 |
| `examples/voice_registry_v5.py` | **新增** 模块 | 注册表与热加载：`VoiceRegistryV5`、`HotVoiceRegistryV5`、`load_config_v5`、`maybe_cleanup_voice_prompt_cache`、`audio_path_for_version` 等。**与 `voice_registry_v4.py` 结构一致**，仅类名带 `V5` 后缀（§6.4.2.1）。 |
| `examples/voice_manager_router_v5.py` | **新增** 模块 | 音色管理 `APIRouter`，**不单独起服务**；由 `openai_server_v5.py` `include_router` 挂载。路径与方法与 v4 完全一致（`/voices`、`/health` 等，见 **§6.4.3.6**）。 |
| `examples/openai_server_v5.py` | **新增** 服务 | **唯一入口**：加载 TTS 模型、初始化注册表与 `HotVoiceRegistryV5`、挂载管理路由、启动 **轮询调度器线程** 与 **队列管理线程**；管理端写库成功后 **进程内** 调用 `prime_voice_v5` 预热并做 `maybe_cleanup_voice_prompt_cache`。**v5 特有**：内部维护 `_v5_thread_pool_sem`（流式并发槽位）+ `_v5_request_queue`（FIFO 等待队列）+ `_v5_nonstream_fifo_lock`（非流式 FIFO 串行）+ `_v5_gpu_rwlock`（流式↔非流式互斥读写锁）；实现细节见 **§6.4.2**。 |

**运行期辅助文件**：

- 与注册表同目录会生成 **`voices_registry_v5.json.lock`**（路径与 `voices_registry_path` 一致），用于 Linux 下 `flock` 协调读写；Windows 无 `fcntl` 时锁降级，多进程并发写注册表需谨慎。
- 若配置了 `inference_logging.file`（`add_timestamp` 默认 `true`），推理摘要日志会写到 `…/inference_v5_YYYY-M-D_HHMMSS.log`。

#### 6.4.3.2 文件结构

```
faster-qwen3-tts/
├── config_v5.json                    # v5 统一配置（含 stream_chunk_size / metrics_log_mode）
├── voices_registry_v5.json           # 音色注册表（v5 专用）
├── faster_qwen3_tts/
│   └── model.py                      # 共用：推理摘要 logger faster_qwen3_tts.inference
├── examples/
│   ├── voice_registry_v5.py          # 注册表 + 热加载 + 孤儿清理工具
│   ├── voice_manager_router_v5.py    # 管理 APIRouter（v5）
│   ├── openai_server_v5.py           # v5 主服务：推理 + 管理 + 轮询调度
│   ├── voice_registry_v4.py          # v4 仍保留
│   ├── voice_manager_router_v4.py
│   └── openai_server_v4.py
└── …
```

- **`voice_registry_v5.py`**：被 **管理路由 v5** 与 **主服务** 共同引用。
- **`voice_manager_router_v5.py`**：只定义路由；**`openai_server_v5.py`** 负责进程级状态（模型副本、`HotVoiceRegistryV5`、`prime_voice_v5`）并通过 `init_router` 注入。

#### 6.4.3.3 依赖安装

与 v4 相同，见 **§5.2.3**。v5 在 v4 运行环境之上**无需新增依赖**——`threading.Condition / Semaphore` 与 `asyncio.Lock` 都是 Python 标准库。

#### 6.4.3.4 配置文件与注册表

##### `config_v5.json`

建议放在 `faster-qwen3-tts` 根目录，或通过 **`--config`** 指向任意路径。**相对路径规则**（`voices_registry_path`、`tone_wav_file_dir` 及 `inference_logging.file`）与 v4 一致：相对于**配置文件所在目录**解析。

**基础字段**（与音色管理、孤儿清理、上传校验直接相关）：**与 v4 同名同义，见 §5.2.4**。

**v5 新增 / 调整字段**：

| 字段 | 说明 |
|------|------|
| `stream_chunk_size` | 流式 chunk 大小（codec 帧数）。3 路舒适区 **8**（每块 ~667ms 音频），4 路临界区建议 **10**（833ms）或 **12**（1000ms）以换取 inter_chunk 缓冲。优先级：**CLI `--stream-chunk-size` > config > 默认 8**。 |
| `metrics_log_mode` | METRICS 字段模式。`DEBUG` 输出完整诊断字段（含 `ttfa_ms`、`first_prefill_ms`、`prefill_len` 等）；`DEV` 只输出线上核心字段（`mode`、`ttfa_wall_ms`、`ttfa_cuda_graphs_ms`、`inter_chunk_*`、`rtf`、`audio_s`、`total_gen_ms`、`total_ms`、`n_chunks`）。优先级：**CLI `--metrics-log-mode` > config > 环境变量 `TTS_METRICS_LOG_MODE` > 默认 `DEBUG`**。 |

**v5 继续沿用**（语义同 v4，见 §5.2.4）：

| 字段 | 说明 |
|------|------|
| `prime_text` | 启动批量预热与上传/更新后 `prime_voice_v5` 使用的文本；未配置时回退为 `"."`。 |
| `prime_stream_first_chunk` | `true` 时预热阶段额外拉取流式首块；默认 `false`。 |
| `prime_stream_max_new_tokens` | 流式首块预热的 `max_new_tokens`，默认 `48`。 |
| `inference_logging` | 仅当 JSON 存在该顶层键时生效；把 `Generated … RTF` / `METRICS` / `STARTUP …` 输出到 stderr 或落盘，完整语义见 **§5.1.9**。 |

**仓库内示例**：`tts-jiang/faster-qwen3-tts/config_v5.json`。

##### `voices_registry_v5.json`

**结构与 §5.2.4 的 v4 注册表完全一致**。v5 与 v4 通过独立文件隔离，不串扰；若需从 v4 迁移，直接复制 `voices_registry_v4.json` 重命名即可。

##### `METRICS` 行各字段含义（v5）

v5 的 `_log_metrics` 字段口径见 **§6.4.2.9**。日常只需要记住：

- **`DEBUG` 模式**：字段最多，适合定位问题；包含 `req_id`、`ttfa_ms`、`first_prefill_ms`、`first_decode_ms`、`prefill_len`、`attention_mask_shape`、`icl`、`parity_mode` 等诊断字段。
- **`DEV` 模式**：字段较少，适合开发/压测留档；只保留 `mode`、`ttfa_wall_ms`、`ttfa_cuda_graphs_ms`、`inter_chunk_max_ms`、`inter_chunk_p95_ms`、`rtf`、`audio_s`、`total_gen_ms`、`total_ms`、`n_chunks`。
- 压测脚本如用正则抽取字段，迁移到 v5 时至少要支持 **`n_chunks`**；如果使用 `DEV` 模式，还要注意日志行里不再输出 `req_id`。

#### 6.4.3.5 启动服务（v5）

**仅需一个进程、一个端口**，同时提供 **`POST /v1/audio/speech`**（OpenAI 兼容合成）与 **`/voices`、`/health`** 等管理路径。

##### 3 路舒适区（推荐默认）

```bash
cd faster-qwen3-tts
python examples/openai_server_v5.py \
  --config config_v5.json \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 3 \
  --stream-chunk-size 12 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

`stream_chunk_size` 未在 CLI 指定时会读取 `config_v5.json` 的 `stream_chunk_size`（默认 8）。

##### 4 路临界区（需加大 chunk 缓冲）

```bash
python examples/openai_server_v5.py \
  --config config_v5.json \
  --concurrency 4 \
  --stream-chunk-size 10
```

##### `--concurrency` 与 `--replicas`（v5 语义）

二者都影响"能扛多少合成"，但层次与 v4 不同：

**`--concurrency`（流式并发槽位，默认 3）**

- 对应 **`_v5_thread_pool_sem: threading.Semaphore`**，**仅**夹在**流式 `wav/pcm`** 路径上：拿到槽位才进入 `_v5_active_workers`，由 **全局轮询调度器线程** 逐个 `step()` 推进；满则入 **`_v5_request_queue`**（FIFO 等待队列），由队列管理线程在空槽时晋升。
- **非流式 `mp3` 不占用该槽位**（见 §6.4.1.3 第 5 条）——它走自己的 `_v5_nonstream_fifo_lock` + 写锁通道。
- **推荐值**：**3**（舒适区）、**4**（临界点，需配合 `stream_chunk_size≥10`）；单卡 4090 超过 4 路请转多 GPU 部署。

**`--replicas`（模型副本数，默认 1）**

- 启动时用 `args.replicas` 创建 N 份 `FasterQwen3TTS` 实例，合成时通过 **`_pick_model()`** **轮询** 分发。
- **v5 当前限制**（§6.4.2.12）：调度器线程只有一个，多副本实际仍按"一个大池子"串行被调度器消费；显存会按 **约 N 倍**增长。单卡单调度器的场景下 **`--replicas=1` 即可**，除非你希望配合多卡多进程编排。
- **流式 CUDA Graph 与副本数**：即便各副本已完成 Graph 预热，只要 **`replicas < concurrency`**，**流式**仍会退回 **Parity**（`parity_mode=True`），不会在多路并发下假定「每路独占一份 Graph 缓冲区」——细则见 **§6.4.1.3**。若希望 **多客户端并发流式**仍稳定走 **CUDA Graph 快路径**，请保证 **`--replicas` ≥ `--concurrency`**（并留出显存），使启动后出现 **`STARTUP stream_cuda_graph=True`**、**`parity_stream_fallback=False`**。

##### 其它常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | `examples/../config_v5.json` | 配置文件路径 |
| `--model` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | HuggingFace 模型名或本地路径 |
| `--host` / `--port` | `0.0.0.0` / `8000` | 监听地址 |
| `--device` | `cuda` | 推理设备 |
| `--stream-chunk-size` | 无 | 覆盖 `config_v5.json` 的 `stream_chunk_size` |
| `--metrics-log-mode` | 无 | 覆盖 `config_v5.json` 的 `metrics_log_mode`；可选 `DEBUG` / `DEV` |
| `--disable-cuda-graph` | `false` | **不推荐**：禁用后 RTF 会从 ~4.7 降到 ~0.8（§6.4.1.3 第 1 点） |
| `--skip-warmup` / `--warmup-prefill-len` | `false` / `100` | 启动 CUDA 预热 |
| `--skip-prime-voices` | `false` | 跳过启动阶段批量预热所有 active 音色 |
| `--prime-text` | 无 | 覆盖 `config.prime_text`；**在 `main()` 中一次性解析后**供启动批量预热与运行时 `prime_voice_v5` 共用 |
| `--prime-stream-first-chunk` | `false` | 与配置 `prime_stream_first_chunk` **逻辑或**；任一为真即开启流式首块预热 |
| `--prime-stream-max-new-tokens` | 无 | 覆盖 `config.prime_stream_max_new_tokens` |
| `--no-metrics-log` | `false` | 仅关闭每请求 `METRICS …` 行；**不**关闭 `Generated … RTF`（后者由 `inference_logging` 控制） |

##### 启动顺序（摘要）

`main()` 同步入口按以下顺序执行（代码位置见 §6.4.2.6 启动流程）：

1. 解析 `--config` → `load_config_v5` → `VoiceRegistryV5.ensure_file_exists()`
2. 构建 `HotVoiceRegistryV5` 快照
3. 合并 `prime_*` / `stream_chunk_size` / `metrics_log_mode` 的优先级（其中 `metrics_log_mode` 还支持环境变量 `TTS_METRICS_LOG_MODE`）
4. 若存在 `inference_logging` 则初始化推理 logger（含可选落盘路径，`add_timestamp` 默认 `true`）
5. 写入 `STARTUP cmdline=…` 等关键参数行
6. `init_router` + `app.include_router(voice_router)`（挂载音色管理 API）
7. 加载模型副本 → CUDA 预热（`_warmup(prefill_len=…)`）
8. 可选批量 `_prime_voice_caches(...)`：逐副本 × 逐音色做 `_prepare_generation` + 首块预热
9. 启动 **`_v5_queue_manager_thread`** 与 **`_v5_scheduler_thread`**（都是守护线程）
10. `uvicorn.run(app, host, port)`；退出时 `finally: _stop_scheduler()` 给两条线程投递停止信号

##### 预期启动日志（v5 关键行）

```
STARTUP cmdline=python examples/openai_server_v5.py --concurrency 3 ...
STARTUP --concurrency=3 --replicas=1
STARTUP stream_chunk_size(effective)=8
STARTUP metrics_log_mode=DEBUG
Loading model: Qwen/Qwen3-TTS-12Hz-0.6B-Base (replicas=1)
Warming up replica 1/1...
Priming: voice='xxx' replica=1/1
Queue manager thread started
Scheduler started
Server v5 ready: host=0.0.0.0 port=8000 concurrency=3 chunk_size=8
STARTUP stream_cuda_graph=False parity_stream_fallback=True replicas=1 concurrency=3
```

> **说明**：上例为 **`--replicas 1` + `--concurrency 3`**（`replicas < concurrency`），故 **`stream_cuda_graph=False`**、**`parity_stream_fallback=True`**——流式走 **Parity**。若需多路并发仍用 Graph 快路径，应令 **`--replicas` ≥ `--concurrency`**（如 `--replicas 4 --concurrency 4`，且显存足够），则通常 **`stream_cuda_graph=True`**、**`parity_stream_fallback=False`**。与 **§6.4.1.3** 条目 1 所述条件一致。

##### 压低首包 TTFA（推荐）

与 v4 思路一致（见 §5.2.5）：在 `config_v5.json` 中把 `prime_text` 设为代表性长句、`prime_stream_first_chunk: true`、`prime_stream_max_new_tokens` 按业务句长设，使启动批量预热与上传后 `prime_voice_v5` 的语义与在线合成对齐。也可在命令行临时覆盖：

```bash
python examples/openai_server_v5.py \
  --config config_v5.json \
  --concurrency 3 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

##### METRICS 字段模式选择

排查性能问题时保持默认 `DEBUG`：

```bash
python examples/openai_server_v5.py \
  --config config_v5.json \
  --concurrency 3 \
  --metrics-log-mode DEBUG
```

日常开发或压测留档时可切到 `DEV`，日志更短：

```bash
python examples/openai_server_v5.py \
  --config config_v5.json \
  --concurrency 3 \
  --metrics-log-mode DEV
```

也可以直接写入 `config_v5.json`：

```json
"metrics_log_mode": "DEV"
```

##### 孤儿清理

在**进程内预热**（新增/更新音色后的 `prime_voice_v5`）结束时，对各模型副本调用 `maybe_cleanup_voice_prompt_cache`；**不在**每条普通合成请求结束后执行，避免影响在线合成路径。

#### 6.4.3.6 音色管理 API 接口说明

**v5 的管理路由与 v4 完全一致**（`voice_manager_router_v5.py` 与 `voice_manager_router_v4.py` 同构）。

- **URL 与方法、成功状态码、请求体、响应 JSON 字段**：**完全参照 §5.2.6**，将 `v4` 替换为 `v5` 即可（注册表文件名为 `voices_registry_v5.json`、注册表模块为 `VoiceRegistryV5`）。
- **基址与 v3 管理进程的差异**：v5 同样是**管理与推理同进程同端口**（如 `http://host:8000`），预热走**进程内** `prime_voice_v5`，无 HTTP 回环。

v5 特有：**`GET /health`** 的响应 JSON 仍含 **`registry_readable` / `tone_dir_writable` / `model_loaded`**，**与 v4 完全一致**；**不**暴露 `_v5_active_workers` / `_v5_request_queue` 深度等内部调度状态（如需可自行扩展）。

##### 常用 curl 示例（仅基址端口与 v4 相同，路径可直接复制）

```bash
# 新增音色
curl -X POST http://localhost:8000/voices \
  -F "audio_file=@/path/to/ref_audio.wav" \
  -F "ref_text=欢迎您使用语音合成服务" \
  -F "language=Chinese" \
  -F "voice_id=my_voice_001"

# 列出音色 / 查询单个 / 健康检查
curl -X GET http://localhost:8000/voices | jq
curl -X GET http://localhost:8000/voices/my_voice_001 | jq
curl http://localhost:8000/health | jq

# 更新（仅文本 / 换音频）
curl -X PUT http://localhost:8000/voices/my_voice_001 -F "ref_text=新的参考文本内容"
curl -X PUT http://localhost:8000/voices/my_voice_001 \
  -F "audio_file=@/path/to/new_audio.wav" -F "ref_text=新的参考文本内容"

# 软删 / 硬删 / 恢复
curl -X DELETE http://localhost:8000/voices/my_voice_001
curl -X DELETE "http://localhost:8000/voices/my_voice_001?hard=true"
curl -X POST http://localhost:8000/voices/my_voice_001/restore
```

#### 6.4.3.7 客户端调用 TTS（v5）

- **接口**：`POST /v1/audio/speech`，`Content-Type: application/json`。
- **请求体字段**：`input`（待合成文本）、`voice`（须为 `voices_registry_v5.json` 顶层键，即 `voice_id`）、`response_format`（`wav` / `pcm` 为**流式**，`mp3` 为**非流式**整包）；`model` 字段兼容 OpenAI 形态，**实际权重以启动时 `--model` 为准**。
- **正常 HTTP 状态码**：合成成功均为 **200 OK**（`wav/pcm` 流式、`mp3` 整包均 200）。`Content-Type` 分别为 `audio/wav` / `audio/pcm` / `audio/mpeg`。

##### v5 的并发行为（调用方必须知道）

| 请求类型 | 到达服务后的行为 |
|---------|------------------|
| **流式 `wav` / `pcm`** | 拿流式槽位（`_v5_thread_pool_sem`，上限 = `--concurrency`），加入轮询调度器 active 集合；满则入 FIFO 等待队列。多路并发时**各路公平穿插**：每轮各产 1 个 PCM chunk。 |
| **非流式 `mp3`** | 进入独立的**非流式 FIFO 锁**（`asyncio.Lock`）串行排队；拿到 FIFO 锁后取**写锁**独占 GPU 整段合成，**期间所有流式 worker 的 `step()` 被挂起**。不占用流式槽位（即流式满了，MP3 仍能按到达顺序被处理）。 |

**调用方建议**：

- **对时延敏感的业务优先用 `wav/pcm` 流式**；`mp3` 适合"能等完整文件到手再播"的异步/批量场景。
- **如果同一客户端同时发多条 `mp3`**，它们在服务端严格按到达顺序 FIFO 处理，不会并发。
- **混发 `wav + mp3` 时**，注意：`mp3` 一旦开始合成，此刻在途的流式请求会出现一次 ≈ MP3 `total_ms` 的**大空窗**（`inter_chunk_max_ms` 体现），这是 §6.4.1.3 第 5 条明面上的设计取舍。

##### 最小示例

```bash
# 流式 wav（浏览器/播放器可边下边播）
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，这是 v5 测试","voice":"my_voice_001","response_format":"wav"}' \
  -o speech.wav

# 非流式 mp3（需等待整段合成完成）
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"my_voice_001","response_format":"mp3"}' \
  -o speech.mp3
```

> 用 **curl** 保存音频二进制时务必加 **`-o 输出文件`**，避免把二进制打印到终端。

#### 6.4.3.8 典型部署流程（v5）

1. **准备配置**：编辑 `config_v5.json`（填写 §6.4.3.4 中的基础字段 + **`stream_chunk_size`**，按需增加 `prime_*` / `inference_logging`）；若需将 `STARTUP / METRICS / RTF` 写入磁盘，在 `inference_logging` 中设置非空 `file` 并按需 `add_timestamp`，确保进程对日志目录有写权限。
2. **放置数据**：准备 `voices_registry_v5.json` 与音频目录，或从空表开始（服务启动时 `ensure_file_exists` 会创建）；若从 v4 迁移，直接复制 `voices_registry_v4.json` 重命名为 `voices_registry_v5.json`。
3. **启动服务**：`python examples/openai_server_v5.py --config config_v5.json --concurrency 3 ...`。
4. **录入音色**：`POST /voices`（multipart：**`audio_file`、`ref_text` 必填**；`language` 默认 `Auto`；`voice_id` 可选）；写入注册表成功后服务会**异步进程内**调用 `prime_voice_v5` 预热。
5. **客户端调用**：`POST /v1/audio/speech`，`voice = voice_id`，流式优先用 `wav`，批处理可用 `mp3`。
6. **后续变更**：继续通过 `/voices` 管理；**无需重启**——注册表变更经 mtime 被 `HotVoiceRegistryV5` 加载。

**环境迁移**：拷贝 **`config_v5.json` + `voices_registry_v5.json` + `tone_wav_file_dir` 下文件**，调整路径后**只启动** `openai_server_v5.py` 即可。

#### 6.4.3.9 v4 与 v5 选型

| 场景 | 建议 |
|------|------|
| 主要是**单路或低并发**（≤ 2 路）追求极致速度，偶发排队可接受 | **v4**（§5.2）：单路独占 GPU，TTFA 最低 |
| **多路流式并发**（2～4 路）在线合成，要求各路**首包差距可控、块间间隔一致** | **v5**（本章）：轮询调度 + 流式↔非流式 FIFO 互斥 |
| 同时需要"在线流式"与"批量整段 MP3"，且希望两者互不干扰 | **v5**：非流式独占写锁，不挤占流式槽位（§6.4.1.3 第 5 条） |
| 需要 **进程级管理面/推理面隔离** 或不同端口/不同扩容策略 | **v3** 双进程方案（与 v4/v5 无冲突，可并存） |

**生产压测约定**：与 §5.2.9 一致——单卡上线前须做**逐步提并发**压测（固定模型、`voice` 与文本长度，观察 TTFA、`inter_chunk_p95_ms`、RTF、错误率与显存），再按 §6.4.1.5 表核对阈值，最终选定 **`--concurrency` / `--stream-chunk-size`** 与客户端并发上限。压测流程与脚本复用 **§七**，压测对象为 **`openai_server_v5.py` + `config_v5.json`**，启动方式与本节 §6.4.3.5 一致。

#### 6.4.3.10 `voice_registry_v5.py` 模块说明

v5 管理路由与主服务**共用**本模块，核心符号与 v4 一一对应（只是类名带 `V5` 后缀），便于 v4 压测脚本/运维工具按正则替换迁移。

| 符号 | 说明 | 典型调用方 |
|------|------|------------|
| `load_config_v5(path)` | 读取 v5 JSON 配置（可含 `prime_text`、`stream_chunk_size`、`inference_logging` 等） | `openai_server_v5.py` `main()` |
| `VoiceRegistryV5` | 文件锁 + `read_all` / `mutate` / 原子写 | 管理路由、`main()` 中 `ensure_file_exists` |
| `HotVoiceRegistryV5` | 按注册表 mtime 重载；`resolve_active_voice`、`list_active_voice_cfgs`、`get_registry_copy` | 主服务推理路径 |
| `valid_voice_prompt_cache_keys(registry)` | 合法 `_voice_prompt_cache` key 集合 | 孤儿清理 |
| `maybe_cleanup_voice_prompt_cache(model, registry, …)` | 阈值满足时清理孤儿缓存项 | `prime_voice_v5` 之后（各副本） |
| `audio_path_for_version(...)` | 版本化音频路径规则 | 管理路由 |
| `utc_now_iso()` | UTC ISO 时间戳 | 管理路由 |

解析 `voice_id` 后返回给推理逻辑的 `voice` 配置字典至少包含：`voice_id`、`ref_audio`（**本机绝对路径**字符串）、`ref_text`、`language`、`status`、`version` 等，供 `voice_cfg["ref_audio"]` 等字段使用（与 v4 完全一致）。

#### 6.4.3.11 常见问题排查（FAQ）

| 症状 | 可能原因 | 定位点 |
|------|---------|--------|
| 启动时 `Server v5 ready` 之前就退出 | 配置字段缺失（`voices_registry_path` / `tone_wav_file_dir`），或模型下载失败 | 先跑 `python -c "import json; print(json.load(open('config_v5.json')))"` 校验 JSON；再查 `STARTUP` 日志的 `--model` 路径 |
| 首条请求 `ttfa_wall_ms` 特别大（> 3 秒） | 未预热或预热文本与业务差距过大 | 启动时加 `--prime-stream-first-chunk` 与代表性 `--prime-text`；确认启动日志出现 `Priming: voice=...` |
| 多路 `inter_chunk_p95_ms` 远超 §6.4.1.5 阈值 | `stream_chunk_size` 过小、或有 MP3 并发在跑 | 提高 `stream_chunk_size`（3 路 → 8、4 路 → 10）；错峰 MP3；结合客户端日志确认是否存在断连噪声 |
| MP3 请求 `ttfa_wall_ms` 持续增长 | MP3 FIFO 排队堆积 | 网关层限流；或改走 `wav` 流式 |
| `RTF` 明显低于 4.5（单路空载场景） | CUDA Graph 未启用，走了 parity 路径 | 启动日志核对 `--disable-cuda-graph=False`；`ttfa_cuda_graphs_ms` 字段是否出现 |
| 音色注册表"看起来对"但请求 400 "Voice '…' 不存在或已禁用" | `status=disabled` 或大小写/下划线错位 | `GET /voices?status=disabled` 查确认；或 `POST /voices/{id}/restore` |
| Windows 下多进程并发写 `/voices` 行为异常 | 无 `fcntl`，文件锁降级 | 生产环境建议 Linux；Windows 仅作开发环境 |

## 6.5 v4/v5性能差异对比与尝试修改

本节记录 2026-04-25 对 **v4 请求级流式服务**（`tts-jiang/faster-qwen3-tts_v4_请求级别并发/examples/openai_server_v4.py`）与 **v5 多路 Graph 轮询调度服务**（`tts-jiang/faster-qwen3-tts/examples/openai_server_v5.py`）在同一服务器、同一 GPU、同一模型与同一音色下的初步性能对比。当前只聚焦 **`ttfa_wall_ms` 首块墙钟 TTFA** 的差异，作为后续优化 v5 首包耗时的排查记录。

### 6.5.1 已观察到的现象

在 v4 未拆步、无并发、单客户端请求场景中，流式 `METRICS` 的稳定态大致为：

```text
ttfa_wall_ms≈225~240
ttfa_ms≈163~176
ttfa_cuda_graphs_ms≈223~239
inter_chunk_p95_ms≈178~181
```

在 v5 拆步后，初始使用 `--stream-chunk-size 10` 时，单请求稳定态大致为：

```text
ttfa_wall_ms≈290~330
ttfa_cuda_graphs_ms≈289~328
inter_chunk_p95_ms≈192~216
```

将 v5 改为 `--stream-chunk-size 8` 后，稳定态回落为：

```text
ttfa_wall_ms≈265~292
ttfa_cuda_graphs_ms≈262~285
inter_chunk_p95_ms≈163~167
```

其中首条正式请求偶尔偏高（例如 `ttfa_wall_ms≈365ms`），通常可视作启动后第一次进入该音色/该路径的残余预热、kernel 或 codec 解码校准成本，不宜直接纳入稳定态结论。

### 6.5.2 初步判断

第一阶段差异（`chunk_size=10` 时从 200 多毫秒到 300 多毫秒）主要来自 **首块大小变大**：Qwen3-TTS 为 12Hz codec，`stream_chunk_size` 表示首块需要攒多少个 codec step；从 8 改为 10 后，首包必须多生成 2 个 codec step，因此 `ttfa_cuda_graphs_ms` 与 `ttfa_wall_ms` 都同步变大。

第二阶段差异（v5 改回 `chunk_size=8` 后仍比 v4 高约 35~55ms）更可能发生在 **首个 `next(generate_voice_clone_streaming(...))` 内部**，而不是 HTTP 层、队列或调度器层。证据是 v5 日志中：

```text
ttfa_wall_ms≈ttfa_cuda_graphs_ms
```

两者通常只差 1~10ms，说明 v5 单请求场景下调度器、`response_queue`、读写锁、线程切换等额外路径开销较小，不是 40ms 级别差异的主要来源。

同时，v5 在 `chunk_size=8` 下的 `inter_chunk_p95_ms≈163~167ms`，反而低于 v4 的 `178~181ms`。这说明底层稳态 decode 并非整体变慢，差异更集中在首块特有路径：

- `_prepare_generation()` 与 prompt/cache 准备；
- 首次 prefill；
- 首块 codec chunk 到 PCM 的 `speech_tokenizer.decode()`；
- 首块后的 `torch.cuda.synchronize()` 与 numpy 转换；
- prime 是否真正覆盖正式请求的首块路径。

### 6.5.3 已排除或基本一致的项

对比当前文件后，以下项暂未发现足以解释差异的明显不一致：

- `config_v4.json` 与 `config_v5.json` 的关键项一致：`prime_text`、`prime_stream_first_chunk=true`、`prime_stream_max_new_tokens=48`、`stream_chunk_size=8`。
- `voices_registry_v4.json` 与 `voices_registry_v5.json` 中目标音色一致：同一 `voice_id`、`ref_audio`、`ref_text`、`language=Chinese`。
- Locust 压测脚本中的短句/长句输入池一致。
- `faster_qwen3_tts/streaming.py` 的 `fast_generate_streaming()` 首块核心循环基本一致。
- v4/v5 正式请求调用 `generate_voice_clone_streaming()` 的核心入参基本一致：`chunk_size`、`non_streaming_mode=False`、`ref_audio`、`ref_text`、`language` 等一致。

### 6.5.4 追加诊断字段后的结论（2026-04-25）

为进一步定位 v5 首包偏慢的原因，已在 v4 与 v5 的流式 `METRICS` 中临时加入首块诊断字段：

```text
first_prefill_ms
first_decode_ms
first_overhead_ms = ttfa_cuda_graphs_ms - ttfa_ms
prefill_len = tie.shape[1]
attention_mask_shape = attention_mask.shape
trailing_text_len = trailing_text_hiddens.shape[1]
icl = int(ref_codes is not None)
parity_mode = int(parity_mode)
```

在相同服务器、相同 GPU、相同模型、相同音色、相同 Locust 短/长文本池、`stream_chunk_size=8` 的测试下，关键结果如下。

**v4 稳定态（去掉前几条偶发预热/抖动样本后）**：

```text
ttfa_wall_ms≈225~240
ttfa_ms≈164~176
first_prefill_ms≈30~35
first_decode_ms≈133~140
first_overhead_ms≈60~62
prefill_len=112
attention_mask_shape=1x112
trailing_text_len=1(短文本) / 30(长文本)
icl=1
parity_mode=0
inter_chunk_p95_ms≈178~181
```

**v5 稳定态（去掉首条请求与偶发抖动后）**：

```text
ttfa_wall_ms≈260~280
ttfa_ms≈195~213
first_prefill_ms≈60~75
first_decode_ms≈134~142
first_overhead_ms≈62~64
prefill_len=112
attention_mask_shape=1x112
trailing_text_len=1(短文本) / 30(长文本)
icl=1
parity_mode=0
inter_chunk_p95_ms≈161~167
```

因此，当前结论从“首块内部路径可疑”进一步收敛为：**v4 与 v5 的 prefill 输入形态完全一致，但 v5 的 `first_prefill_ms` 明显更高**。

可以明确排除的方向：

- **不是输入长度差异**：两边 `prefill_len=112`、`attention_mask_shape=1x112` 完全一致。
- **不是短/长文本路径错位**：两边短文本 `trailing_text_len=1`、长文本 `trailing_text_len=30` 一致。
- **不是 ICL 模式差异**：两边 `icl=1`。
- **不是 CUDA Graph/parity 路径差异**：两边 `parity_mode=0`，均走 Graph fast path。
- **不是首块 PCM 解码/同步/numpy 转换新增开销**：两边 `first_overhead_ms` 都约 60~64ms。
- **不是稳态 decode 整体变慢**：两边 `first_decode_ms` 基本一致，且 v5 的 `inter_chunk_p95_ms` 反而更低。
- **不是 v5 调度器主导的单请求开销**：单路场景下 `ttfa_wall_ms` 与 `ttfa_cuda_graphs_ms` 差距很小。

剩余核心差异集中在：

```text
v4 first_prefill_ms≈30~35ms
v5 first_prefill_ms≈60~75ms
```

也就是：**同样的 prefill 输入，v5 运行时的 `talker.forward(...)` prefill 阶段比 v4 慢约 30~40ms**。因此后续优化应从 `openai_server_v5.py` 的调度器/队列层转向底层推理包差异排查，重点是 `faster_qwen3_tts/model.py`、`streaming.py` 及 graph 相关实现。

### 6.5.5 后续尝试修改与测试计划

建议按“先隔离底层差异，再决定是否回退/合并”的顺序推进。当前不建议继续优先修改 v5 调度器，因为诊断字段已经显示瓶颈在首块 prefill，而不是调度路径。

**第一步：文件级 diff，对齐底层推理包差异**

优先比对以下文件：

```text
tts-jiang/faster-qwen3-tts_v4_请求级别并发/faster_qwen3_tts/model.py
tts-jiang/faster-qwen3-tts/faster_qwen3_tts/model.py

tts-jiang/faster-qwen3-tts_v4_请求级别并发/faster_qwen3_tts/streaming.py
tts-jiang/faster-qwen3-tts/faster_qwen3_tts/streaming.py

tts-jiang/faster-qwen3-tts_v4_请求级别并发/faster_qwen3_tts/talker_graph.py
tts-jiang/faster-qwen3-tts/faster_qwen3_tts/talker_graph.py

tts-jiang/faster-qwen3-tts_v4_请求级别并发/faster_qwen3_tts/predictor_graph.py
tts-jiang/faster-qwen3-tts/faster_qwen3_tts/predictor_graph.py
```

重点关注：

- `model.py`：`_prepare_generation()`、`_build_talker_inputs_local()`、`generate_voice_clone_streaming()`、`from_pretrained()`、`_warmup()`。
- `streaming.py`：首块 `out = talker.forward(...)` prefill 段、`torch.cuda.synchronize()` 位置、`timing["prefill_ms"]` 口径。
- `talker_graph.py` / `predictor_graph.py`：capture 参数、静态 cache、`prefill_kv()`、`set_generation_state()` 是否有差异。

**第二步：做最小交叉验证，隔离“入口层”与“底层包”**

推荐做两个小实验（二选一也可以）：

- **实验 A：v5 入口 + v4 底层推理包**  
  保留 `examples/openai_server_v5.py`、`voice_registry_v5.py`、`voice_manager_router_v5.py` 与 v5 调度器，只临时将 v5 目录下 `faster_qwen3_tts/` 中与推理相关的文件替换/回退为 v4 对应版本（先从 `model.py`、`streaming.py` 开始）。若 `first_prefill_ms` 回到 30~35ms，则证明 v5 入口层没有问题，差异来自底层推理包。

- **实验 B：v4 入口 + v5 底层推理包**  
  在 v4 请求级服务中临时使用 v5 的 `faster_qwen3_tts/` 对应文件。若 v4 的 `first_prefill_ms` 也升到 60~75ms，则进一步证明底层包差异是主因。

做交叉验证时，保持：

- 同一模型路径；
- 同一 `voices_registry` 音色内容；
- `stream_chunk_size=8`；
- `prime_stream_first_chunk=true`；
- 同一 Locust 输入池；
- 先丢弃首条或前 2~3 条预热样本，再看稳定态。

**已执行的交叉验证结果（2026-04-25）**

已先执行 **实验 A 的简化版**：将 v4 版本的 `faster_qwen3_tts/model.py` 临时替换到 v5 目录中，并在 `examples/openai_server_v5.py` 的 `FasterQwen3TTS.from_pretrained(...)` 调用处临时去掉 `disable_cuda_graph` 参数，以保证 v5 入口能够继续启动；`streaming.py` 暂未替换。测试条件与前述 v5 测试保持一致，日志中 `parity_mode=0`，说明仍走 CUDA Graph fast path。

本次结果没有让 `first_prefill_ms` 回到 v4 先前观察到的 30~35ms 水平。除第 1 条请求可能包含启动/预热扰动外，稳定态大致为：

```text
first_prefill_ms   ≈ 59~74ms
first_decode_ms    ≈ 134~142ms
first_overhead_ms  ≈ 61~64ms
ttfa_wall_ms       ≈ 256~283ms（多数请求）
ttfa_ms            ≈ 193~215ms
ttfa_cuda_graphs_ms≈ 255~278ms
prefill_len        = 112
attention_mask     = 1x112
icl                = 1
parity_mode        = 0
```

典型日志片段如下：

```text
req_id=26 ttfa_wall_ms=256.3 ttfa_ms=192.7 ttfa_cuda_graphs_ms=255.0 first_prefill_ms=59.0 first_decode_ms=133.7 first_overhead_ms=62.3 prefill_len=112 attention_mask_shape=1x112 trailing_text_len=30 icl=1 parity_mode=0
req_id=29 ttfa_wall_ms=279.0 ttfa_ms=208.1 ttfa_cuda_graphs_ms=270.1 first_prefill_ms=73.2 first_decode_ms=134.8 first_overhead_ms=62.1 prefill_len=112 attention_mask_shape=1x112 trailing_text_len=30 icl=1 parity_mode=0
req_id=31 ttfa_wall_ms=281.3 ttfa_ms=209.7 ttfa_cuda_graphs_ms=272.8 first_prefill_ms=73.9 first_decode_ms=135.8 first_overhead_ms=63.1 prefill_len=112 attention_mask_shape=1x112 trailing_text_len=1 icl=1 parity_mode=0
```

阶段性判断：**v5 的 `model.py` 相对 v4 的新增逻辑，不太可能是 `first_prefill_ms` 从 30~35ms 升到 60~75ms 的主因**。因此短期内不建议继续围绕 `model.py` 做回退式修改。若后续重新排查，优先级可调整为：

1. 恢复 v5 原始 `model.py`，避免长期混用临时替换版本。
2. 如需把底层包差异完全排干净，再临时替换 v4 `streaming.py` 到 v5 测一次；但因当前 `fast_generate_streaming()` 快路径已基本一致，预期收益有限。
3. 更有价值的是编写 standalone 对照脚本，绕开 FastAPI/openai_server，分别在 v4/v5 目录下直接加载同一模型、同一 voice、同一文本，并只拉取 `generate_voice_clone_streaming()` 首块，观察 `first_prefill_ms` 是否仍有 30~40ms 差距。

**第三步：如果定位到具体差异，优先回退非必要改动**

若 diff 发现 v5 的 `model.py` / `streaming.py` 中有与 v5 调度器无关、但会影响 prefill 的改动，优先尝试回退这些改动，同时保留 v5 服务层能力：

- 保留 v5 的 `TTSWorker`、轮询调度器、流式队列、MP3 FIFO/RWLock。
- 尽量让 `FasterQwen3TTS.generate_voice_clone_streaming()` 的首块 fast path 与 v4 保持一致。
- 回退后用 `first_prefill_ms`、`ttfa_wall_ms`、`inter_chunk_p95_ms`、RTF 四类指标同时验收。

**第四步：若底层代码完全一致仍慢，再查运行环境与 warmup 状态**

如果 diff 显示关键路径一致，但 v5 仍慢，继续检查：

- `from_pretrained()` 参数是否一致（`dtype`、`attn_implementation`、`disable_cuda_graph`）。
- CUDA Graph capture 日志是否一致，是否均完成 `predictor_graph.capture()` 与 `talker_graph.capture(prefill_len=100)`。
- 是否存在不同的安装包/依赖版本（`qwen_tts`、`torch`、CUDA、驱动等）。
- v5 是否在正式请求前经历了额外模型状态改动（例如管理端 prime、cache cleanup 或其他请求混入）。

**第五步：最后再考虑业务性优化**

若最终 v5 单路 `ttfa_wall_ms` 仍稳定比 v4 高 20~30ms，但 3 路并发的 `inter_chunk_p95_ms`、RTF、排队与取消行为明显更稳定，可接受作为 v5 多路公平调度的工程权衡。若业务强要求单路 230ms，则再评估：

- 单路请求直通 v4 fast path 的可选模式；
- 代表性业务短/长文本的额外 prime；
- 更激进的首包 fast path，但必须同步做听感回归，避免首块缺音、截断或 pops。


## 6.6 极少见日志中 `audio_s` 指标值异常偏大

### 6.6.1 现象

在一次 v5 并发压测中，出现过如下极少见日志：

```text
2026-04-24 09:28:44,744 INFO METRICS req_id=6 mode=stream ttfa_wall_ms=1117.3 ttfa_cuda_graphs_ms=289.8 inter_chunk_max_ms=677.5 inter_chunk_p95_ms=593.4 rtf=4.752 audio_s=154.88 total_gen_ms=32592.6 total_ms=91711.7 n_chunks=194
```

其中最异常的是 **`audio_s=154.88`**，明显远高于同一批测试中长文本正常约 28~30 秒的音频时长。`n_chunks=194` 也同步异常，约为正常长文本 35~50 个 chunk 的数倍。

相比之下，其他指标并不指向 GPU 或调度器故障：

- **`ttfa_wall_ms=1117.3`**：在并发 3 路、请求排队进入调度器时可以出现，属于排在后面的请求累计等待，不是本次异常的核心。
- **`ttfa_cuda_graphs_ms=289.8`**：首块 CUDA Graph 路径耗时仍在正常区间。
- **`inter_chunk_p95_ms=593.4` / `inter_chunk_max_ms=677.5`**：块间隔与当时三路轮转调度下的预期接近，说明流式调度仍在稳定推进。
- **`rtf=4.752`**：生成效率正常，说明不是 GPU 卡顿导致总时长变长。
- **`total_gen_ms=32592.6`、`total_ms=91711.7`**：模型持续生成了很久，且请求长时间占用一个流式合成槽位。

### 6.6.2 判断：自回归 TTS 的 runaway 长尾事件

该日志更符合 **Qwen3-TTS 自回归生成没有及时输出 EOS / end-of-speech token，导致持续续说** 的现象，而不是服务端调度、网络传输或 GPU 性能异常。

判断依据如下：

- `rtf` 正常，代表单位时间生成音频的速度没有变慢；
- `inter_chunk_*` 正常，代表服务端仍按稳定节奏产出 chunk；
- `ttfa_cuda_graphs_ms` 正常，代表首块推理路径没有异常；
- 只有 `audio_s`、`n_chunks`、`total_ms` 明显放大，说明模型“多生成了很多内容”，而不是“生成同样内容变慢了”。

这类现象可称为 **runaway generation（失控续说）**。在 Qwen3-TTS、CosyVoice、XTTS、Fish-Speech 等自回归式 TTS/LLM-TTS 中都可能偶发出现：同样输入大多数请求正常结束，极少数采样路径会错过合适的结束点，后续概率分布继续滚动，最终生成远超预期的音频。

### 6.6.3 可能原因

可能触发因素通常不是单一项，而是多项叠加：

- **随机采样长尾**：采样时偶发抽到低概率但非 EOS 的 token，一旦错过合适停顿点，后续更难自然回到 EOS。
- **文本结束边界不够强**：输入末尾缺少 `。？！.?!` 等强停顿标点，或测试文本结尾语义上容易被模型理解为“还可以继续说”。
- **参考音频 prompt 的影响**：参考音频结尾如果存在拖音、未完整收尾或说话习惯偏连续，模型可能更倾向继续生成。
- **长文本采样路径更复杂**：长文本本身有更多生成步，遇到长尾采样路径的机会也更高。
- **并发并非根因**：并发会放大这类请求对整体服务的占用影响，但从该日志看，runaway 的直接原因仍是单条自回归生成没有及时结束。

如果一批几十条请求中只出现 1 条，通常应按低概率长尾事件处理，而不是直接判断为服务端 bug。

### 6.6.4 建议防护方案

**方案 A：在服务端增加音频时长看门狗（推荐优先）**

在 `openai_server_v5.py` 的 `TTSWorker.step()` 中累计已经产出的音频秒数。若超过设定阈值，则主动结束当前 worker，并写入 WARNING 或 METRICS 标记，例如 `runaway=1`、`stop_reason=audio_cap`。

建议阈值可以先采用固定值，例如：

```text
max_audio_seconds = 60.0
```

也可以后续改成动态阈值，例如：

```text
max_audio_seconds = max(60.0, estimated_audio_seconds * 1.8)
```

优点是实现简单、不需要改模型、不影响正常短文本，并且能防止单条 runaway 请求长时间占用流式槽位。缺点是 runaway 请求会被截断，客户端收到的是前半段有效音频。

**方案 B：限制 `max_new_tokens`**

在调用 `generate_voice_clone_streaming()` 时传入更明确的 `max_new_tokens` 上限，或根据输入文本长度估算上限。

该方案能从模型生成步数层面截断 runaway，但需要谨慎设定阈值。阈值过小会误伤正常长文本；阈值过大则防护效果有限。更适合作为方案 A 的补充。

**方案 C：输入文本做轻量规整**

在 `/v1/audio/speech` 入口对文本做低风险清理：

- 如果末尾不是强停顿标点，自动补一个 `。`；
- 去除明显异常的超长空白、重复符号；
- 对极端重复文本做长度或重复次数限制。

该方案不能完全消除 runaway，但能降低因为文本边界不清晰导致的续说概率。

**方案 D：压测统计中单列 runaway 样本**

性能评估时，不建议把 runaway 样本直接混入普通请求的 `total_ms`、`audio_s`、`n_chunks` 分位数中，否则会污染常规性能结论。可以按规则单列，例如：

```text
audio_s > expected_audio_s * 2
或 audio_s > 60
```

满足条件的样本记录为 `runaway_count` / `runaway_rate`，作为模型稳定性指标单独报告；常规 TTFA、RTF、inter_chunk 统计则可同时给出“含 runaway”和“剔除 runaway”两组口径。

### 6.6.5 当前建议

短期内若业务更关注服务稳定性，建议优先实现 **方案 A：音频时长看门狗**。这类防线不解决模型为何偶发续说，但能把最坏占用时间控制在可接受范围内，避免单条异常请求拖慢整体并发服务。

若后续继续优化，可组合采用：

1. 服务端 `max_audio_seconds` 看门狗；
2. 输入末尾强停顿标点规整；
3. `max_new_tokens` 动态上限；
4. METRICS 中增加 `runaway` / `stop_reason` 字段，便于压测和线上观测。



暂时未进行任何修改。

## 6.7 web界面使用示例

**本节结构**：**§6.7.1** 为方案设计；**§6.7.2** 为**已落地实现**的使用说明。**已定稿的部署方式**为：**试用 Web** 由 **`demo/server_v5.py` + `index_v5.html`** 单独进程、**默认端口 7860**；**不再**在 **`openai_server_v5.py` 与合成/管理「同端口」挂 `GET /ui`**。**合成、`/docs`、方案 α 的 `/v1/ui/*` 旁路** 仍属 **`openai_server_v5`**。

### 6.7.1 方案设计

本小节为**设计说明**：**与现有 `demo/server.py` + `demo/index.html` 类似**，新增独立 **`demo/server_v5.py`**（及静态资源 **`index_v5.html` 等**），在**单独监听端口**上提供 **仅语音克隆试用** Web UI；**合成 API、音色路由、`METRICS` 旁路（方案 α）仍落在 `examples/openai_server_v5.py`** 进程。两进程**默认不同端口**（试用 UI 默认 **7860**，OpenAI v5 服务沿用如 **8000**）；下文为实施前约束与选型。

#### （1）现状对照：`demo/server.py` 与 `openai_server_v5.py`

| 维度 | `demo/server.py` + `demo/index.html`（参考） | `openai_server_v5.py`（合成与 METRICS 宿主） |
|------|-----------------------------------------------|-----------------------------------|
| 入口语义 | Demo：单进程内 **`/load`** 换模型，`/generate/stream` SSE 推送 **base64 WAV 分块** JSON；`/generate` 非流式。 | **生产 v5**：启动时加载若干 **`FasterQwen3TTS` 副本**；合成走 **`POST /v1/audio/speech`**（**OpenAI 形态**，`response_format=wav/pcm/mp3`）；音频体为 **标准流式 WAV 头 + raw PCM**，**不是** SSE JSON。 |
| 并发/GPU | 全局 **`_generation_lock`**：**严格串行**一次完整生成。 | **轮询调度 + 读写锁**：多路 **`stream`** 可读锁交错 step；**非流式 MP3** 写锁优先 + asyncio FIFO。**指标（`ttfa_wall_ms`、`inter_chunk_*` 等）含排队与调度语义**，与 demo 单笔串行完全不同。 |
| 音色 | 预设 + 上传临时参考音，`ref_preset`/`ref_audio`。 | **`voices_registry_v5.json`** 注册 **`voice`**；已由 **`voice_manager_router_v5`** 提供 **`/voices*`** CRUD。**试用页必须用已注册 `voice_id`**（与压测/OpenAI API 一致），避免在页面内再走「单次上传未见于注册表」而与生产路径分裂（若未来要支持上传，应走 **`POST /voices`** 再在合成里选对 `voice`）。 |
| 指标口径 | SSE 每条 `chunk`/`done` 里 **`ttfa_ms`、`rtf`、`total_audio_s`、`elapsed_ms`** 等——由 **服务端在合成循环内**算出。 | 生产就绪指标在 **`TTSWorker._log_final_metrics` → `_log_metrics`**：`mode=stream` 时 **`ttfa_wall_ms`、`ttfa_cuda_graphs_ms`、`inter_chunk_*`、`rtf`、`audio_s`、`total_gen_ms`、`total_ms`、`n_chunks`**；**DEV 模式再屏蔽部分字段**。当前 **不向 HTTP 客户端回传**，只入 **`INFERENCE_LOGGER_NAME`**（及可选 inference log file）。 |

**结论：** 不能直接「照搬 demo 前端去调 `/generate/stream`」接上 v5；也**不合适**把整个生产合成改成 SSE JSON。**试用页（`demo/server_v5.py` + `index_v5.html`）仅作静态壳 + 配置 API 根地址**：**实际调用**仍为 **`${API_BASE}/v1/audio/speech`**、**`${API_BASE}/voices`** 等（**`API_BASE`** 指向 OpenAI v5）。**方案 α** 的 **仅读 METRICS 路由**（如 **`GET …/last-metrics`**）**加在 `openai_server_v5.py`**；浏览器从 **7860** 页访问合成服务端口（如 **8000**）。若浏览器拦截，须在 **OpenAI 侧** 对试用页 Origin 配置 **CORS**（见 §（3）（7））。


#### （2）功能范围（按需求裁剪）

仅保留 **「Clone — match a voice from a reference clip」** 在 **v5/OpenAI** 语义下的等价能力：

- **不做**：**Custom built-in speaker**、**Voice Design**（不调用 `generate_custom_voice`、`generate_voice_design`；前端不提供模式切换）。
- **必做**：**文本输入**、`voice_id` **选择（下拉）**，与 **`SpeechRequest`** 对齐；合成格式 **`wav`（首选，便于浏览器 `MediaSource`/`Audio` decode）**，可选 pcm/mp3（若前端实现成本高可一开始只放开 wav）。
- **保留**：**设置**面板中**与 v5 合成路径仍有关**的子集（详见下文「参数映射」）。
- **保留**：合成结束后 **总时长展示 + 播放控件**。
- **保留**：底部 **评测/诊断信息区**——**字段必须与 `_log_metrics` 输出一致**（见下一节对齐策略）。

**不推荐**在首期实现 demo 自带的 **`/transcribe`**（Parakeet 转写）；若需要「一键填参考文」，可走 **离线准备**或使用 **管理侧已填好的 `ref_text`**（**`GET ${API_BASE}/voices/{id}`** 可查，`API_BASE` 指向 **`openai_server_v5`**）。若需页面内转写可作为**后续扩展小节**另行规划。


#### （3）进程拆分、端口与静态资源（已定稿）

与现有 **`demo/server.py` + `index.html`** 模式一致，**新建**：

| 组件 | 路径约定（建议） | 默认端口 | 职责 |
|------|------------------|----------|------|
| **试用 Web 壳** | **`tts-jiang/faster-qwen3-tts/demo/server_v5.py`** + **`demo/index_v5.html`**（及 **`demo/assets_v5/`** 等拆分静态资源，命名可随仓库规范微调） | **`7860`**（`uvicorn` **`--port 7860`**，可用环境变量或 argparse 覆盖） | **只做**：挂载首页/静态文件、可选 **`GET /health`**、**不加载 TTS 模型**；页内 **`fetch`** 访问下游 API。 |
| **合成与 METRICS 旁路** | **`examples/openai_server_v5.py`** | **与现有一致（例：8000）** | **`/v1/audio/speech`**、**`/voices*`**、**`_log_metrics` + 方案 α 的只读查询路由**、`/docs` |

**API 根地址配置**：`server_v5` 通过 **`--api-base` / 环境变量 `TTS_V5_API`（如 `http://127.0.0.1:8000`）** 注入给 **`index_v5.html`**（构建时写死、或 **`GET /config` 返回 JSON`**），保证 **`fetch(`${API_BASE}/v1/audio/speech`)`**、**`fetch(`${API_BASE}/voices`)`** 指向 **`openai_server_v5`**。


**OpenAPI 文档（已定稿）**：**保留** 合成服务上的 **`/docs`**（Swagger UI），由 **`openai_server_v5`** 提供；**试用站点不要求重复挂载 `/docs`**，页面可提供「打开 API 文档」外链至 **`${API_BASE}/docs`**。

**CORS（跨端口必查）**：浏览器从 **`http://host:7860`** 访问 **`http://host:8000`** 属 **跨域**，须在 **`openai_server_v5`** 上对 **`/v1/audio/speech`**、**`/voices*`**、**方案 α 的 metrics 路由** 配置 **`CORSMiddleware`**（允许 Origin 为试用页，或开发期 `*` 仅限内网），否则 **`fetch` 与指标轮询会失败**。内网联调下同机部署时也可用 **反向代理** 把两端口统一切到同 Origin（可选，非必须）。


#### （4）音色与业务流程

1. **启动前/后**：用户使用 **`voices_registry_v5`** + **`/voices`**（**OpenAI 服务端口**）维护 **`voice_id`、reference、ref_text、language**。  
2. **页面**：**`GET ${API_BASE}/voices`** → 下拉 **`voice`**；**`input`** 文本。  
3. **合成**：**`POST ${API_BASE}/v1/audio/speech`**，body 形如 `{"model":"tts-1","input":"…","voice":"<voice_id>","response_format":"wav"}`。合成请求可带 **`X-Try-Correlation-Id`**（或与 `req_id` 协议一致），便于 **方案 α** 取回 METRICS。  
4. **流**：读 **`ReadableStream`**，首字节可作 **浏览器侧** TTFA/TTFB 估计（与服务端 **`ttfa_wall_ms`** 含排队可能不一致）；拼 **Blob** 播放并展示时长。  


#### （5）参数映射：**设置**面板 vs 当前 OpenAI Handler

当前 **`SpeechRequest`**（见 `openai_server_v5.py`）主要为 **`model, input, voice, response_format`**，**不包含** **`temperature`、`top_k`、`chunk_size`、`xvec_only`** 等。

| 用户需求（类 demo 「设置」） | v5 现状 | 方案 |
|-----------------------------|---------|------|
| **`chunk_size`/流式粒度** | 由 **`--stream-chunk-size` / config_v5 `stream_chunk_size`** **全局生效**，合成 API **未暴露**。 | **首期**：在 **`openai_server_v5`** 提供只读 **`GET /v1/ui/effective-config`**（或等价）返回 **`stream_chunk_size` 等**；**`server_v5` 侧**也可在启动时 **`GET` 一次**缓存到页面展示。**远期**（可选）：试用专用 **`SpeechRequest` 可选字段** / Header，仅在显式开发开关下启用。 |
| **采样解码温度等** | 未从 HTTP 透出。 | **首期**：不写或灰显；若必须对齐 demo，再开 **restricted** Query/Header，**文档声明与生产网关策略一致**。 |

**原则：** 首期 **不要为了 UI 大范围改 `_log_metrics` 语义或调度器**。


#### （6）服务端指标与页面展示的**对齐策略**（关键难点，**已定稿：方案 α**）

`_log_metrics` 在 **`TTSWorker._log_final_metrics`** **流结束时**写入日志；HTTP **WAV 流**无法再附加 JSON trailer。本方案**唯一采用**：

**方案 α：「最终指标旁路寄存」（实现于 `openai_server_v5.py`）**

- 在 **`_log_metrics` 调用前**（或与合并后的 **`merged` 字典同一内容**）：将本条指标（**与日志中 METRICS 行一致**）连同 **`req_id`、`mode`、时间戳**，写入 **`threading.Lock`** 保护的 **`collections.deque(maxlen=N)`**（或 **按 `req_id` 索引的 LRU map**，避免并发试听串单）。
- 新增只读 HTTP 接口（路径示例，实现时可微调）：**`GET /v1/ui/last-metrics?req_id=<id>`** 或 **`GET /v1/ui/metrics-tail?limit=`**；返回 **JSON**，字段与 **`_log_metrics` DEV/DEBUG 裁剪规则**一致。
- **`create_speech`** 在响应 **`StreamingResponse`** 上已能拿到 **`req_id`**：合成流结束后由 **`Worker._log_final_metrics`** 打 METRICS；**浏览器**读完 **`fetch` body** 后用 **`req_id`**（或由服务端在 response header **`X-Req-Id`** 回传）**poll 一次**旁路接口，填充页面底部表格。
- **DEBUG 扩展字段**：仅当 **`inference_logging.level` 为 DEBUG**（与 `_metrics_log_mode` 一致）时，旁路 JSON **含** `ttfa_ms`、`first_prefill_*` 等；否则与 **DEV** 子集一致。

**不再采用**「方案 β：仅人工对日志」作为本 § 的交付路径；若极端环境无法开通 CORS，再个案处理。

**前端展示字段**（与 **`_DEV_METRIC_KEYS`** + **`mode`** + **DEBUG-only** 对齐）：

- **总是展示（与 DEV 对齐）**：`mode`、`ttfa_wall_ms`、`ttfa_cuda_graphs_ms`、`inter_chunk_max_ms`、`inter_chunk_p95_ms`、`rtf`、`audio_s`、`total_gen_ms`、`total_ms`、`n_chunks`。  
- **`inference_logging.level=DEBUG`** 且旁路带出完整字典时：再展示 **`ttfa_ms`、`first_prefill_ms`、`first_decode_ms`、`first_overhead_ms`、`prefill_len`、`attention_mask_shape`、`trailing_text_len`、`icl`、`parity_mode`**。  
可选 **`GET ${API_BASE}/v1/ui/metrics-schema`**（实现于 **openai_server_v5**）返回当前应展示的 key 列表。

**非流式 MP3**（若 UI 可选 **mp3**）：`_log_metrics` 可能为 **`non_stream_mp3`**，字段少于 **stream**；前端 **按 key 存在性渲染**。


#### （7）与其它组件的兼容性

- **反向代理**：可按需将 **`/v1/audio/speech`**、**`/docs`**、试用站 **同源**反代；指标旁路路径一并纳入。  
- **安全（已定稿）**：试用页 **`server_v5`（如 7860）** 与 **`openai_server_v5`（如 8000）** 均按 **内网联调** 使用，**不做鉴权**（不设 Basic/OAuth）；若未来暴露公网，须另立网关策略（不在本 § 约束内）。  
- **性能**：**`server_v5`** 无 GPU 逻辑，负载可忽略；**deque 旁路** **O(1)**，对合成路径影响可忽略。


#### （8）交付物清单（后续实现时用）

| 交付物 | 说明 |
|--------|------|
| **`examples/openai_server_v5.py` 增量** | **方案 α**：`_log_metrics` 前写入 **线程安全寄存**；**`GET /v1/ui/last-metrics`（或 tail）**、**可选 `GET /v1/ui/metrics-schema`、`GET /v1/ui/effective-config`**；**`CORSMiddleware`**（对 **`7860` Origin** 等）；合成响应头 **`X-Req-Id`**（或与现有 **`req_id`** 对齐）；**`/docs` 保留**。 |
| **`demo/server_v5.py` + `demo/index_v5.html`（+ `assets_v5/`）** | **默认端口 7860**；**`GET /`** 返回静态页；**`--api-base`** 指向 OpenAI v5；页面仅 **Clone**、**`fetch` 至 `${API_BASE}`**；底部绑定 **metrics 轮询**。 |
| **运维说明** | **双进程启动顺序**、**端口与环境变量**、**内网无鉴权**前提、**CORS** 检查项。 |
| **文档** | **`config_v5`/`--stream-chunk-size`**、**浏览器 TTFA ≠ `ttfa_wall_ms`**。 |


以上为 **§6.7.1 方案设计**（**双进程：7860 试用壳 + 原端口合成；指标用方案 α；`/docs` 保留；内网联调不设鉴权**）。实现阶段按上表交付即可。

### 6.7.2 使用指南

以下为 **`demo/server_v5.py` + `index_v5.html`** 与 **`examples/openai_server_v5.py`** 联调的**操作步骤**与**环境变量**说明（与 §6.7.1 已定稿一致）。

#### （1）前置条件

- 在仓库 **`faster-qwen3-tts`** 根目录下已安装依赖（含 **`fastapi`**、**`uvicorn`**、**`faster_qwen3_tts`** 等），且 **`config_v5.json`**、**`voices_registry_v5.json`** 与 **`tone_wav_file_dir`** 路径有效（参见 **§6.4.3**）。
- **`openai_server_v5`** 启动完成前，试用页拉 **`/voices`** 会失败；至少需有 **一条 `active` 音色**（可通过 **`POST /voices`** 或事先维护注册表）。

#### （2）双进程启动（推荐顺序）

**① 合成与管理 API（OpenAI v5）**

在 `faster-qwen3-tts` 目录下（示例监听 **8000**，可按需修改）：

```bash
python examples/openai_server_v5.py \
  --config config_v5.json \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --host 0.0.0.0 --port 8000 \
  --concurrency 3 \
  --stream-chunk-size 10 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。" \
  --cors-origins http://10.251.11.55:10017 \
  --replicas 3
```

- **OpenAPI 文档**：浏览器打开 **`http://<主机>:8000/docs`**（**保留**，与 §6.7.1 一致）。
- **健康/音色**：**`GET /health`**、**`GET /voices`** 等同端口。
- **多窗口 / 多会话并发流式与 Graph**：若期望合成端在 **`POST /v1/audio/speech` 流式**上走 **CUDA Graph 快路径**（`parity_mode=False`，见 **§6.4.1.3**），启动 **`openai_server_v5`** 时需满足 **`--replicas` ≥ `--concurrency`**；否则进程会 **自动流式 Parity**（`parity_mode=True`）。排障时看合成服务日志中的 **`STARTUP stream_cuda_graph=`**、**`parity_stream_fallback=`**，参数含义与示例见 **§6.4.3.5**。

**② 试用静态壳（默认 7860）**

**另开一个终端**，仍在 `faster-qwen3-tts` 目录下：

```bash
# python demo/server_v5.py --host 0.0.0.0 --port 7860 --api-base http://127.0.0.1:8000
# 此处--api-base 参数值代表 openai_server_v5.py 服务的监听地址如 http://10.251.11.55:10018
# --host 0.0.0.0 表示监听来自任何服务器的申请
# --port 7860 表示demo/server_v5.py 服务监听在7860 端口（此处是容器内，实际一般使用宿主机的对应映射端口）

python demo/server_v5.py --host 0.0.0.0 --port 7860 --api-base http://10.251.11.55:10018
```

- **`--api-base`**：必须与 **①** 中 **`openai_server_v5` 对外 HTTP 基址**一致（含 **主机与端口**；若服务在远端，写 **`http://<ip>:8000`**）。
- **环境变量**（与 **`--api-base` 二选一，命令行优先）：**`TTS_V5_API`**，例如 `set TTS_V5_API=http://127.0.0.1:8000`（Windows）或 `export TTS_V5_API=...`（Linux/macOS）。

**③ 打开页面**

浏览器访问 **`http://<运行 server_v5 的主机>:7860/`**（本机一般为 **`http://127.0.0.1:7860/`**）。

页面会通过 **同源** **`GET /config`** 获知 **`api_base`**，再 **`fetch` 至下游** 的 **`/voices`**、**`/v1/audio/speech`**、**`/v1/ui/last-metrics`** 等。

#### （3）页面操作说明（Clone 试用）

| 步骤 | 说明 |
|------|------|
| **刷新音色** | 点击 **「刷新音色列表」**，等价于 **`GET ${api_base}/voices?status=active`**；下拉框项为注册表中的 **`voice_id`**。 |
| **选择格式** | **`wav` / `pcm`**：走 v5 **流式**合成；**`mp3`**：走 **非流式**整段再返回（与 **`openai_server_v5`** 实现一致）。 |
| **合成并播放** | **`POST ${api_base}/v1/audio/speech`**，body 与 OpenAI 形态一致；全部收完后 **Blob 播放**。 |
| **METRICS 表格** | 响应头 **`X-Req-Id`** 与日志 **`req_id` 一致**；结束后 **`GET /v1/ui/last-metrics?req_id=`** 拉取与 **`_log_metrics` 同结构的旁路**。 |

**说明**：浏览器侧感知的首包/总耗时与日志中的 **`ttfa_wall_ms`（含排队）** 不在同一语义上，对照时以 **服务端 METRICS** 为准（§6.7.1 已述）。

#### （4）`openai_server_v5` 侧环境变量（与试用页联调）

Cross-origin fetch 依赖 **`Access-Control-Allow-Origin`**：**必须以浏览器地址栏的完整 Origin** 为准（协议 + 主机 + 端口）。例如：试用页映射为 **`http://10.251.11.55:10017`**，而后端 OpenAPI 在 **`http://10.251.11.55:10018`** 时，白名单必须包含 **`http://10.251.11.55:10017`**——仅配置 **`localhost:7860` 或与容器内端口 7860 字面一致是不够的**。

实现规则（简要）：

- **`TTS_V5_CORS_ORIGINS` 不设**：在 **`127.0.0.1:7860`/`localhost:7860`** 默认两项基础上，再合并命令行 **`--cors-origins`**（见下）。
- **`TTS_V5_CORS_ORIGINS` 设为逗号分隔列表**：与上述**默认两项**、**`--cors-origins`** **去重合并**（同一 OpenAPI 可同时给本机与局域网页面用）。
- **`TTS_V5_CORS_ORIGINS=*` 或 `all`（不区分大小写）**：允许**任意** Origin（ **`allow_credentials=false`** ，符合浏览器对 `*` 的限制）。**仅建议在隔离内网联调**；**勿对公网直连开放**。
- **`TTS_V5_CORS_ORIGIN_REGEX`**：可选，POSIX 风格正则；若请求 **`Origin`** **整串匹配**正则，则放行（可与白名单并用，适合「某内网 IP + 任意宿主映射端口」）。

| 变量 | 作用 |
|------|------|
| **`TTS_V5_CORS_ORIGINS`** | 见上。典型：`http://10.251.11.55:10017`。**与默认 `localhost:7860` 合并**，不必删默认项。 |
| **`TTS_V5_CORS_ORIGIN_REGEX`** | 例：仅用某宿主机 IP 时，`^http://10\.251\.11\.55:\d+$` 可匹配该机任意端口上的试用页 Origin。 |
| **`TTS_METRICS_LOG_MODE`** | **`DEBUG`** 或 **`DEV`**：与 **`--metrics-log-mode`、config 中 `metrics_log_mode`** 择一生效（优先级见实现），控制 **`_log_metrics` 与旁路** 是否含扩展诊断字段。 |

**示例（任选其一；生效后须重启 `openai_server_v5`）**

下面假定浏览器地址栏 Origin 为 **`http://10.251.11.55:10017`**（典型：**容器内试用页监听 7860，映射到宿主机 `10017`**）。**须替换为你实际访问试用页的完整 Origin**（协议 + 主机 + 端口，缺一不可）。

**示例 A：`TTS_V5_CORS_ORIGINS` 环境变量（推荐）**

Linux / macOS（在启动同一终端先 export，再启动进程）：

```bash
export TTS_V5_CORS_ORIGINS=http://10.251.11.55:10017
# 不加限制：export TTS_V5_CORS_ORIGINS=*
python examples/openai_server_v5.py \
  --config config_v5.json \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --host 0.0.0.0 --port 8000 \
  --concurrency 3 \
  --stream-chunk-size 10 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。" \
  --replicas 3
```

Windows **CMD**：

```bat
set TTS_V5_CORS_ORIGINS=http://10.251.11.55:10017
python examples\openai_server_v5.py --config config_v5.json --host 0.0.0.0 --port 8000 --concurrency 3
```

Windows **PowerShell**：

```powershell
$env:TTS_V5_CORS_ORIGINS="http://10.251.11.55:10017"
python examples/openai_server_v5.py --config config_v5.json --host 0.0.0.0 --port 8000 --concurrency 3
```

容器 / **docker compose**（在 **`openai_server_v5` 对应服务**上）：

```yaml
environment:
  TTS_V5_CORS_ORIGINS: "http://10.251.11.55:10017"
```

**示例 B：启动参数 `--cors-origins`**

与 **`TTS_V5_CORS_ORIGINS`、默认 localhost:7860 白名单去重合并**；适合不想改环境变量的场景：

```bash
python examples/openai_server_v5.py --config config_v5.json --host 0.0.0.0 --port 8000 --concurrency 3 \
  --cors-origins http://10.251.11.55:10017
```

多个 Origin（逗号分隔，无空格或按你 shell 规则转义）：

```bash
python examples/openai_server_v5.py ... \
  --cors-origins http://10.251.11.55:10017,http://127.0.0.1:7860
```

**示例 C：内网临时允许任意 Origin**

```bash
export TTS_V5_CORS_ORIGINS=*
python examples/openai_server_v5.py --config config_v5.json --host 0.0.0.0 --port 8000
```

等价写法：**`TTS_V5_CORS_ORIGINS=all`**（不区分大小写的 **`all`** 亦可）。此时服务端 **`allow_credentials=false`**（浏览器与 `*` 的约束）；**仅建议在隔离内网联调**，**禁止对公网直连开放**。

**补充（可选）**：若同一内网 IP 下试用页端口经常变，可在设置 **`TTS_V5_CORS_ORIGINS`** 或 **`--cors-origins`** 的同时配置 **`TTS_V5_CORS_ORIGIN_REGEX`**，例如 **`^http://10\.251\.11\.55:\d+$`**（表示 **`Origin` 整串**匹配该正则即放行）。具体语义以实现为准。

反向代理须**透传 CORS 相关响应头链**；实现中已对 **`X-Req-Id`** 做 **`expose_headers`**，便于读 METRICS。

（**`TTS_V5_API`** 仅给 **`server_v5.py`** 使用，含义见 **（2）**。）

#### （5）辅助 HTTP 接口（均在 `openai_server_v5` 端口）

便于脚本或排障：

- **`GET /v1/ui/effective-config`**：当前 **`stream_chunk_size`、`sample_rate`、`metrics_log_mode`**。
- **`GET /v1/ui/metrics-schema`**：DEV / DEBUG 下 key 列表说明。
- **`GET /v1/ui/metrics-tail?limit=`**：最近若干条旁路记录（调试）。

**试用壳自身**：**`GET http://<7860>/config`** 返回 **`{"api_base": "..."}`**；**`GET /health`** 返回试用进程存活信息（**不表示**下游 GPU 服务正常）。

#### （6）常见问题

| 现象 | 可能原因与处理 |
|------|------------------|
| 下拉 **无音色** | 下游 **`/voices`** 为空或未启动；先 **`POST /voices`** 或检查 **`voices_registry_v5.json`**。 |
| **`fetch` 被 CORS 拦截**（控制台 **`No 'Access-Control-Allow-Origin'`**） | **Origin 必须与白名单逐字一致**。映射端口场景：页面是 **`http://<宿主机>:10017`** 就写死该项，不要只配 **`…:7860`**。**任选其一**：① `export TTS_V5_CORS_ORIGINS=http://10.251.11.55:10017`（与默认 localhost 合并）；② 启动加 **`--cors-origins http://10.251.11.55:10017`**；③ 内网临时放开 **`TTS_V5_CORS_ORIGINS=*`**；④ **`TTS_V5_CORS_ORIGIN_REGEX`** 批量匹配 IP。修改后须**重启 `openai_server_v5`**。 |
| **METRICS 提示缺少 `X-Req-Id`** | 下游未暴露响应头；实现中已配 **`expose_headers`**；若仍失败，核对是否走了**反向代理**且代理**剥头/未透传**。 |
| **`last-metrics` 404** | **`req_id` 错误**，或旁路 **LRU 已满被逐出**（高并发连打时）；可缩小并发或查日志 **`METRICS req_id=...`**。 |
| **无 METRICS 行/无旁路** | 使用了 **`--no-metrics-log`**；此时 **`_log_metrics`** 不写日志、**也不写旁路**（与实现一致）。 |


#### （7）小结

- **两进程**：**`openai_server_v5`**（合成、`/docs`、`/v1/ui/*`）+ **`demo/server_v5.py`**（**7860** 静态页）。
- **联调关键**：**`--api-base`/`TTS_V5_API`** 指对 **`openai_server_v5`**，**`TTS_V5_CORS_ORIGINS`** 覆盖浏览器 Origin。
- **指标对齐**：以 **`X-Req-Id` + `/v1/ui/last-metrics`** 为准，与 **§6.7.1 方案 α** 一致。



# 七、单卡4090并发压测流程（逐步提并发、观察指标、确定上线阈值）

本章压测对象默认为：**单卡 RTX 4090、单容器**，运行 **`examples/openai_server_v4.py`**，配置 **`config_v4.json`**：**一个进程、一个监听端口**同时提供 **`POST /v1/audio/speech`**（合成）、**`/voices`**（音色管理）与 **`GET /health`**（健康检查）。不要求另起独立管理进程；若你使用双进程 v3 或其他入口，须自行对照调整基址与端口，本文数值**不直接外推**到其他架构。

**TTFA、RTF、流式连续性**的**定义、日志锚点与边界**（含 **`ttfa_wall_ms` 含排队**、**首块为服务端 PCM 入队**、**块间间隔为入队 proxy** 等）以 **§5.1.10** 为准；本章 **§5、§6** 给出与 **METRICS** 字段的对应关系及**经验阈值**，便于压测记录与上线评审。

## 1) 压测目标

在单卡 RTX 4090 上，找到“稳定可用且延迟可接受”的最大并发配置，输出上线参数：

- **服务侧**：`--replicas`（进程内 **`FasterQwen3TTS` 实例个数**，请求轮询分配）、`--concurrency`（**同时在途合成**上限；超出时新请求在服务端等待，直到已有合成释放“槽位”）
- **业务侧建议**：对客户端或网关给出并发上限建议（例如 2 / 4 / 6 路）
- **联合验收（与 §5.1.10、§六一致）**：在每一档 **`--concurrency` / `--replicas`**（及 **`stream_chunk_size`** 若调整）下**同时**核对三类指标，不能只优化其一：**① TTFA**（如 **`ttfa_wall_ms` p95**）满足产品阈值；**② RTF**（本服务定义：**越大越快**）在可接受并发下**尽量大**；**③ 流式连续性**：**`inter_chunk_max_ms` / `inter_chunk_p95_ms`** 不宜过大，避免多 chunk 之间“空窗”过长。提并发若只改善 TTFA 表象而 RTF 暴跌或块间间隔失控，应视为**未达标**，回退或改调度/分批策略（见 **§6.1**）。

## 2) 压测前准备

**服务入口（合并服务）**

示例（在 `faster-qwen3-tts` 目录下）：

```bash
python examples/openai_server_v4.py \
  --config config_v4.json \
  --model /data/models/Qwen3-TTS-12Hz-0.6B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --concurrency 4 \
  --replicas 1 \
  --prime-stream-first-chunk \
  --prime-text "大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"
```

- 压测 HTTP 基址为 **`http://<host>:<port>`**（上例即 **`http://<host>:8000`**）；合成、管理、健康检查**共用该端口**。

**固定测试条件，避免结果漂移**

- 固定模型（例如 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`，与启动参数 **`--model`** 一致）。
- 固定 **`voice`**：须为 **`voices_registry_v4.json`** 中的 **`voice_id`**（顶层 key），或启动后 **`GET http://<host>:<port>/voices`** 查询确认。
- 固定输入文本长度（建议 2 组：短文本 20~40 字，长文本 80~150 字），与业务上线后主要分布接近。
- 固定 **`response_format=wav`**：接口以**流式**返回（先 WAV 头再 PCM 块），压测须统计**首包/首字节时延（TTFA/TTFB）**与**整请求完成耗时**。

**预热（与生产推荐一致）**

- **优先**在 **`config_v4.json`** 配置 **`prime_text`**（代表性长句）、**`prime_stream_first_chunk: true`**、按需 **`prime_stream_max_new_tokens`**，使启动批量预热与上传后 **`prime_voice_v4`** 与压测句长一致，缩小首包抖动。
- 或使用命令行 **`--prime-stream-first-chunk`**、**`--prime-text`**（显式传入时覆盖配置）。
- 正式压测前仍可再发 **5～10** 条 **`/v1/audio/speech`** 做队列与 GPU 状态预热。

**合并服务特别注意**

- 压测窗口内尽量**只打** **`POST /v1/audio/speech`**；避免同进程、同端口大量 **`POST /voices` / 大文件上传**，以免与合成 **争用 GPU、磁盘与事件循环**，使 TTFA 指标失真。

## 3) 推荐测试矩阵（逐步提并发）

按以下顺序测试，每组至少跑 3~5 分钟：

1. `replicas=1, concurrency=1`
2. `replicas=1, concurrency=2`
3. `replicas=1, concurrency=3`
4. `replicas=2, concurrency=2`
5. `replicas=2, concurrency=3`
6. `replicas=2, concurrency=4`

说明：

- 先增加 `concurrency`，观察排队和延迟变化。
- 再增加 `replicas`，观察吞吐提升与显存压力。
- 若出现 OOM、频繁超时或 TTFA 明显恶化，停止继续上探。

## 4) 压测执行方式

可用任意支持并发 HTTP 的压测工具（Locust/k6/JMeter/自写 asyncio 脚本），请求 URL 为 **`http://<host>:<port>/v1/audio/speech`**（**`<port>` 与合并服务 `--port` 一致**），例如：

```http
POST /v1/audio/speech
Content-Type: application/json
{
  "model": "tts-1",
  "input": "固定测试文本",
  "voice": "<注册表中的 voice_id>",
  "response_format": "wav"
}
```

备注：

- 压测客户端应尽量靠近服务端（同机房），避免网络抖动污染 TTFA。
- 流式接口要统计“首包时间”（TTFB/TTFA）与“请求完成总耗时”。

## 5) 重点观察指标（与日志对齐）

口径总述见 **§5.1.10**（**TTFA / `ttfa_wall_ms` / `ttfa_ms`**、**RTF 公式**、**块间间隔 proxy 与听感验证**）。

排障与对齐指标时，可查看合并服务日志中的 **推理摘要**（logger **`faster_qwen3_tts.inference`**）：包括启动时的 **`STARTUP …`**（完整命令行与有效参数）、每请求 **`METRICS …`**（在未加 **`--no-metrics-log`** 时）、以及非流式路径上模型打印的 **`Generated … RTF`**。若在 **`config_v4.json` 中配置了顶层对象 `inference_logging`**，则按其中的 **`console` / `file` / `level`** 将上述内容写到 stderr、日志文件或两者；**`add_timestamp`** 为 **`true`（默认）** 时落盘文件名为配置路径主名加 **`_YYYY-M-D_HHMMSS`** 时间缀，为 **`false`** 时使用配置路径原样。**未配置该键**时上述日志一般随 **root** 默认行为输出。落盘时文件内通常以 **`STARTUP`** 段起首，便于区分每次进程启动（固定文件名模式下需结合时间戳行区分重启）。

**METRICS** 行重点统计（**`openai_server_v4.py`**，与 **§5.1.10** 一致）：

- **`ttfa_wall_ms`**：**TTFA 墙钟（服务端锚点）**，从「校验通过、`req_id` 已分配、**尚未 `acquire()` 槽位**」到「**首段 PCM 服务端入队**」的毫秒数；**含** 本进程 **`--concurrency` 排队**。**产品阈值示例**：p95 ≤ **1000ms**（与总监对齐时常以该字段为准）。**不含** 网卡发出首字节、客户端解码；**不等于** 纯模型首包时请对照 **`ttfa_ms`**。
- **`ttfa_ms`**：流式首块的模型 timing 累计（毫秒），用于与 `ttfa_wall_ms` 对比；**mp3 非流式**下与整段 `total_ms` 同量级，与 `ttfa_wall_ms` 一并打出。
- **`ttfa_cuda_graphs_ms`**（**stream**）：与 **`benchmarks/throughput.py`** 一致的 **CUDA Graphs 流式 TTFA**（见 **§5.2.4** 表）；与 README 表格对比时用 **`stream_chunk_size=8`** 更接近表中 **PRIMARY_CHUNK_SIZE**。
- **`inter_chunk_max_ms` / `inter_chunk_p95_ms`**（**stream**）：**服务端**相邻块 **入队**间隔，作流式连续性 **proxy**；须与客户端听感交叉验证（见 **§5.1.10**）。无绝对阈值，**p95 过大**往往更易感知卡顿。
- **`rtf`**：**生成音频秒数 ÷ 合成墙钟秒数**，越大越快；**产品阈值示例**：平均 ≥ **10**（以本实现公式为准）。
- `total_ms`：流式为生产者线程墙钟总时长；`audio_s`：累计音频时长
- `gpu_util`、`mem_util`、`vram_*`：资源侧

建议同时统计聚合值：

- p50 / p90 / p95 / p99 **`ttfa_wall_ms`**（及可选 `ttfa_ms`）
- p95 **`inter_chunk_p95_ms`**、max **`inter_chunk_max_ms`**
- p95 `total_ms`、平均 **`rtf`**
- 错误率（HTTP 非200、超时、异常）

若需关闭 **METRICS** 行（减少日志量），启动时加 **`--no-metrics-log`**；压测阶段建议**不要**加，以便与本节指标对齐。**`--no-metrics-log` 只抑制 METRICS**，**不**抑制 **`Generated … RTF`**。若也要减少 RTF 类输出，请在 **`inference_logging`** 中关闭 **`console`**、将 **`file`** 置空，或提高 **`level`**（例如 `WARNING`）。

## 6) 上线阈值判定（建议）

可按如下经验阈值选“上线并发”（可与产品/技术总监约定合并）；**字段含义与边界**见 **§5.1.10**。

- 稳定性优先：错误率 < 0.5%，无 OOM
- **RTF**：**`rtf` = 生成音频秒数 ÷ 合成墙钟秒数**，平均 ≥ **10**（示例目标，以压测为准；越大越快）
- **TTFA**：**`ttfa_wall_ms`** 的 p95 ≤ **1000ms**（**首段 PCM 服务端入队**的墙钟，**含** `--concurrency` 排队；与 **`ttfa_ms`** 对照可区分排队与模型首包）
- **流式连续性（服务端 proxy）**：关注 **`inter_chunk_p95_ms` / `inter_chunk_max_ms`**（**入队**间隔，非客户端收包间隔），与客户端实际听感交叉验证
- 资源余量：显存长期占用不超过 85~90%，GPU 利用率不过度打满导致抖动

取满足上述条件的“最高一档并发”，再回退一档作为生产初始值（预留峰值余量）。

## 7) 示例：建议记录表

每组测试记录一行，便于横向比较：

- 配置：`replicas=2, concurrency=4`；服务入口注明 **`openai_server_v4.py`** 及 **`config_v4.json`** 路径（若与默认不同）
- 请求量：总请求数、成功数、失败数
- 时延：p50/p95 TTFA、p95 total_ms
- 性能：平均 RTF
- 资源：平均/峰值 GPU util、显存占用
- 结论：通过/不通过、原因（如 TTFA 超阈值/OOM）

## 8) 生产落地建议

- 首发用“保守值”（例如压测最优并发 -1 档）。
- 单进程单端口上线时建议纳入运维检查：**`config_v4.json`** 中 **`prime_text` / `prime_stream_first_chunk`**（与首包 TTFA 相关）；若启用落盘日志，检查 **`inference_logging.file`** 所在目录的**写权限**及磁盘空间（**`add_timestamp`** 为 **`true`（默认）** 时实际文件名带启动时间缀；为 **`false`** 时为固定名，更需关注单文件增长与轮转）；**`GET /health`** 应返回 **`registry_readable`**（注册表可读）、**`tone_dir_writable`**（音频根目录可写）、**`model_loaded`**（权重是否已加载）等字段，并与业务告警联动。本章压测结论适用于上述 **合并服务** 部署形态。
- 配置请求超时与最大排队长度。
- 监控告警：TTFA 抖动、错误率、显存逼近上限。
- 周期复测：模型版本、驱动版本、业务文本长度分布、**注册表音色变更**后宜重测。