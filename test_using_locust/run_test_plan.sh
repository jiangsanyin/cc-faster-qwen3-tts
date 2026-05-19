#!/usr/bin/env bash
# =============================================================
# run_test_plan.sh
# 按 test_plan.txt 自动化执行多组 TTS 服务压测
#
# 运行环境：10.251.11.27（本机）
#   - 本机已安装 locust，locustfile.py 与本脚本同目录
#   - root@10.251.11.55 已配置免密 SSH
#   - 目标服务运行在 10.251.11.55 的容器 faster-qwen3-tts-jiangsy 内
#
# 用法：bash run_test_plan.sh [test_plan.txt 路径（可选）]
# =============================================================
set -uo pipefail

# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PLAN="${1:-$SCRIPT_DIR/test_plan.txt}"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_TS"

# ------------------------------------------------------------
# 远端服务器 / 容器配置
# ------------------------------------------------------------
REMOTE_HOST="10.251.11.55"
CONTAINER="faster-qwen3-tts-jiangsy"
WORK_DIR="/data/tts/faster-qwen3-tts"
SERVER_LOG="/tmp/openai_server_v5_autotest.log"    # 容器内日志路径

# ------------------------------------------------------------
# TTS 服务固定启动参数
# ------------------------------------------------------------
# MODEL="/data/models/Qwen3-TTS-12Hz-0.6B-Base"
# MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PRIME_TEXT="大家好，今晚的水上芭蕾将为大家呈现最动人的表演。"

# ------------------------------------------------------------
# Locust 配置
# ------------------------------------------------------------
TTS_VOICE="ff609584-a1c3-4f29-837a-615871f9ecd7"
# TTS_VOICE="1"   # fish speech中使用的音色
LOCUST_HOST="http://10.251.11.55:10018"   # 与 curl 验证通过的地址保持一致
LOCUST_DURATION="10m"
LOCUST_RAMP_UP=1        # 每秒新增用户数（-r 参数）
# 与 locustfile.py 中 LOCUST_TOTAL_REQUESTS 一致：>0 时在该次数请求完成后结束（与 -t 谁先满足谁先停）；0=不按次数限制
# 说明：Locust 2.x 无官方 --total-requests/--num-requests，勿在命令行使用无效参数。
LOCUST_TOTAL_REQUESTS="${LOCUST_TOTAL_REQUESTS:-0}"

# ------------------------------------------------------------
# 服务启动超时（秒）
# ------------------------------------------------------------
STARTUP_TIMEOUT=180

# ============================================================
# 工具函数
# ============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_sep() {
    log "========================================================"
}

# 停止容器内所有 openai_server_v5.py 进程（幂等，失败忽略）
stop_server() {
    log "停止容器内 openai_server_v5.py ..."
    ssh -n root@"${REMOTE_HOST}" \
        "docker exec ${CONTAINER} bash -c \
            \"pkill -SIGTERM -f 'openai_server_v5.py' 2>/dev/null; \
              sleep 3; \
              pkill -SIGKILL -f 'openai_server_v5.py' 2>/dev/null; \
              true\"" 2>/dev/null || true
    log "停止完成"
}

# 脚本任意原因退出时都清理服务
cleanup() {
    log "脚本退出，清理残留服务进程..."
    stop_server
}
trap cleanup EXIT

# ============================================================
# 检查依赖
# ============================================================
if ! command -v locust &>/dev/null; then
    log "[ERROR] 未找到 locust 命令，请先安装：pip install locust"
    exit 1
fi

if [[ ! -f "$TEST_PLAN" ]]; then
    log "[ERROR] 测试计划文件不存在：$TEST_PLAN"
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/locustfile.py" ]]; then
    log "[ERROR] 未找到 locustfile.py：$SCRIPT_DIR/locustfile.py"
    exit 1
fi

# ============================================================
# 准备结果目录 & 统计有效行数
# ============================================================
mkdir -p "$RESULTS_DIR"

# 说明：test_plan 若在 Windows 上编辑，常为 CRLF（\r\n）或老 Mac 的仅 \r 换行。
# Bash 的 read 默认只以 \n 断行；若文件里完全没有 \n，整份内容会被读成「一行」，
# 导致只跑第一组、且 grep 的「行首^」也只会命中一次。
# 下面先把 \r 统一转成 \n，再统计与循环，可兼容 LF / CRLF / CR 三种情况。
# 有效行须与主循环一致：非空、非「整行注释」，且同时包含 concurrency= 与 stream-chunk-size=（顺序任意）。
count_plan_lines() {
    tr '\r' '\n' < "$1" \
        | grep -vE '^[[:space:]]*#' \
        | grep -vE '^[[:space:]]*$' \
        | grep -cE 'concurrency=[[:digit:]]+.*stream-chunk-size=[[:digit:]]+|stream-chunk-size=[[:digit:]]+.*concurrency=[[:digit:]]+' \
        || true
}

total_tests=$(count_plan_lines "$TEST_PLAN")
log "结果保存目录：$RESULTS_DIR"
log "共 $total_tests 组测试计划（已规范化换行后计数；若为 0 请确认非注释行形如 stream-chunk-size=N concurrency=M；仍异常可执行：dos2unix test_plan.txt）"
log ""

# ============================================================
# 主循环
# 从进程替换 < <(tr ... < test_plan) 读行时，循环体里凡会读「标准输入」
# 的子进程（openssh 默认会读 stdin 并尝试转发到远端、locust/python 等）
# 都会把管道里未读行一次性吃掉，下一轮 read 即 EOF。对策：循环内
# 全部 ssh 使用 -n；locust 显式接 </dev/null>。
# ============================================================
line_num=0
test_idx=0

while IFS= read -r line || [[ -n "$line" ]]; do
    line_num=$((line_num + 1))
    # 防御：行内残留 \r（混合换行时）
    line="${line//$'\r'/}"

    # 跳过空行与注释行
    trimmed="${line#"${line%%[![:space:]]*}"}"   # ltrim
    [[ -z "$trimmed" || "$trimmed" == '#'* ]] && continue

    test_idx=$((test_idx + 1))

    # ---------- 解析参数 ----------
    concurrency="$(echo "$line" | grep -oP '(?<=concurrency=)\d+' || true)"
    chunk_size="$(echo "$line"  | grep -oP '(?<=stream-chunk-size=)\d+' || true)"

    if [[ -z "$concurrency" || -z "$chunk_size" ]]; then
        log "[WARN] 第 $line_num 行格式无法解析，跳过：\"$line\""
        continue
    fi

    log_sep
    log "第 $test_idx / $total_tests 组  |  concurrency=$concurrency  stream-chunk-size=$chunk_size"
    log_sep

    # ---- 1. 确保无残留进程 ----
    stop_server

    # ---- 2. 在容器内后台启动服务 ----
    log "在容器内启动 openai_server_v5.py ..."
    ssh -n root@"${REMOTE_HOST}" \
        "docker exec ${CONTAINER} bash -c \
            'rm -f ${SERVER_LOG}; \
             cd ${WORK_DIR} && \
             nohup python examples/openai_server_v5.py \
               --config config_v5.json \
               --model ${MODEL} \
               --host 0.0.0.0 \
               --port 8000 \
               --concurrency ${concurrency} \
               --stream-chunk-size ${chunk_size} \
               --prime-stream-first-chunk \
               --prime-text \"${PRIME_TEXT}\" \
               --replicas ${concurrency} \
               > ${SERVER_LOG} 2>&1 &'"

    # ---- 3. 轮询等待 "Server v5 ready" ----
    log "等待服务就绪（超时 ${STARTUP_TIMEOUT}s）..."
    ready=0
    elapsed=0
    sleep 5   # 给进程一点启动缓冲

    while [[ $elapsed -lt $STARTUP_TIMEOUT ]]; do
        if ssh -n root@"${REMOTE_HOST}" \
               "docker exec ${CONTAINER} grep -q 'Server v5 ready' ${SERVER_LOG} 2>/dev/null"; then
            ready=1
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    if [[ $ready -eq 0 ]]; then
        log "[ERROR] 服务在 ${STARTUP_TIMEOUT}s 内未就绪，打印容器日志末尾并跳过本组："
        ssh -n root@"${REMOTE_HOST}" \
            "docker exec ${CONTAINER} tail -60 ${SERVER_LOG} 2>/dev/null || true"
        stop_server
        log ""
        continue
    fi

    log "服务已就绪（等待约 $((elapsed + 5))s）"

    # ---- 4. 运行 Locust ----
    # 重要：本循环用「done < <(tr ... < test_plan)」从管道读行；子进程会继承同一条 stdin。
    # locust / Python 可能读 stdin，会把管道里剩余行一次性读完，read 下轮即 EOF ——
    # 表现「共 9 组、却只跑第 1 组」。对 locust 显式 </dev/null> 可消除该问题。
    CSV_PREFIX="$RESULTS_DIR/c${concurrency}_cs${chunk_size}"
    # log "启动 Locust：-u $concurrency  -r $LOCUST_RAMP_UP  -t $LOCUST_DURATION  --host $LOCUST_HOST  LOCUST_TOTAL_REQUESTS=${LOCUST_TOTAL_REQUESTS}"
    log "启动 Locust：-u $concurrency  -r $LOCUST_RAMP_UP  --host $LOCUST_HOST  LOCUST_TOTAL_REQUESTS=${LOCUST_TOTAL_REQUESTS}"

    # 勿在续行中间写「# 注释」：从 # 起整行会被 shell 忽略，反斜杠续行会断掉。
    TTS_VOICE="$TTS_VOICE" \
    LOCUST_TOTAL_REQUESTS="$LOCUST_TOTAL_REQUESTS" \
    locust \
        -f "$SCRIPT_DIR/locustfile.py" \
        --headless \
        -u "$concurrency" \
        -r "$LOCUST_RAMP_UP" \
        --host "$LOCUST_HOST" \
        --csv "$CSV_PREFIX" \
        --logfile "${CSV_PREFIX}.locust.log" \
        </dev/null \
        2>&1 | tee -a "${CSV_PREFIX}.locust.log" \
        || log "[WARN] locust 退出码非零（可能有失败请求），继续下一组"

    log "Locust 测试完成"

    # 把这轮容器服务日志也保存一份，便于事后分析
    log "保存本组服务端日志 -> ${CSV_PREFIX}.server.log"
    ssh -n root@"${REMOTE_HOST}" \
        "docker exec ${CONTAINER} cat ${SERVER_LOG} 2>/dev/null || true" \
        > "${CSV_PREFIX}.server.log"

    # ---- 5. 停止服务，准备下一组 ----
    stop_server
    sleep 3

    log "第 $test_idx 组完成：concurrency=$concurrency  stream-chunk-size=$chunk_size"
    log ""

done < <(tr '\r' '\n' < "$TEST_PLAN")

log_sep
log "全部 $test_idx 组测试计划执行完毕！"
log "CSV / 日志结果保存在：$RESULTS_DIR"
log_sep
