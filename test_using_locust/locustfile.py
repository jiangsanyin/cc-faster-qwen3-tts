# -*- coding: utf-8 -*-
"""
Locust 压测脚本：针对 openai_server_v4 合并服务（OpenAI 兼容 TTS + 管理/健康接口）。

在 Windows（或任意安装了 Locust 的机器）上运行，指向远端容器映射端口即可。

基本用法（在 test_using_locust 目录下）::

    # Web UI（浏览器打开 http://localhost:8089），Host 填 http://10.251.11.55:10018
    locust -f locustfile.py

    # 无 UI：10 并发用户，每秒.spawn 2 个，压 5 分钟
    locust -f locustfile.py --headless -u 10 -r 2 -t 5m --host http://10.251.11.55:10018

环境变量（可选）::

    LOCUST_TTS_HOST     默认与下面 HOST 一致；也可用命令行 --host 覆盖
    TTS_VOICE           必填：须与 voices_registry_v4.json 中 voice_id 一致（默认占位，请改成你的音色）
    TTS_INPUT_SHORT     短句待合成（可选；不设则用内置医疗问诊短句）
    TTS_INPUT_LONG      长句待合成（可选；不设则用内置医疗助手长回复）
    TTS_INPUT           兼容旧版：未设 TTS_INPUT_SHORT 时作为短句来源
    TTS_INPUT_ALT       兼容旧版：未设 TTS_INPUT_LONG 时作为长句来源
    TTS_REQUEST_TIMEOUT 单请求超时秒数（默认 600，流式长文本需足够大）
    TTS_LOG_TTFB        设为 1 时在控制台打印部分请求的客户端 TTFB（首字节）毫秒值，便于与 METRICS 对照
    LOCUST_TOTAL_REQUESTS 设为大于 0 时，在完成该次数的请求（无论成功失败）后自动结束本轮压测；不设则仅靠 -t/--run-time 等控制时长（仅适合单机 Locust；分布式多 worker 时每个进程单独计数）。

流式任务在 **短句 / 长句** 间随机选用，模拟问诊导语与助手说明两类负载。

Windows CMD：值两侧不要加引号；不要用 # 作注释（应使用 REM）。正确示例::

    set TTS_VOICE=ff609584-a1c3-4f29-837a-615871f9ecd7
    set TTS_LOG_TTFB=1
    locust -f locustfile.py

若使用 set TTS_VOICE="uuid"，变量会带上引号，导致合成请求 voice 错误（GET /health 仍可能成功）。
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List

import gevent
from gevent.lock import Semaphore
from locust import HttpUser, between, constant, events, task

logger = logging.getLogger(__name__)

# 默认仅作兜底；实际以 Locust Web「Host」或环境变量 LOCUST_TTS_HOST / --host 为准
_DEFAULT_HOST = "http://10.251.11.55:10018"


def _env_str(name: str, default: str = "") -> str:
    """读取环境变量并去掉首尾空白；去除 CMD 误写的成对引号。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    if not v:
        return default
    return int(v)


def _resp_error_snippet(resp) -> str:
    """从失败响应中截取一段正文，便于在 Locust Failures 里看到服务端 detail。"""
    try:
        t = (resp.text or "").strip()
    except Exception:
        t = ""
    if not t:
        try:
            c = getattr(resp, "content", None) or b""
            t = c.decode("utf-8", errors="replace").strip()
        except Exception:
            t = ""
    one = t.replace("\r", " ").replace("\n", " ")
    return one[:500]


def _http_fail_msg(resp) -> str:
    """
    构造失败说明。Locust 在连不上服务时常报告 HTTP 0，易被误读为「协议错误」。
    """
    code = getattr(resp, "status_code", None)
    if code is None:
        code = 0
    bits: list[str] = [f"HTTP {code}"]
    err = getattr(resp, "error", None)
    if err is not None and str(err):
        bits.append(str(err))
    snip = _resp_error_snippet(resp)
    if snip:
        bits.append(snip)
    if code == 0:
        bits.append(
            "[提示] 无 HTTP 状态行：多为目标不可达、connect 超时或链路被重置。"
            f" 当前 TTS_CONNECT_TIMEOUT={_CONNECT_TIMEOUT}s（连不上时常约该时长失败）。"
            " 请在同机 curl 测试: curl -v --connect-timeout 5 \"<你的--host>/health\""
        )
    return " | ".join(bits)


# 须与注册表中 voice_id 完全一致（无多余引号）
_DEFAULT_VOICE = _env_str("TTS_VOICE", "REPLACE_WITH_YOUR_VOICE_ID")

_CONNECT_TIMEOUT = _env_int("TTS_CONNECT_TIMEOUT", 10)   # TCP 握手最长等待秒
_REQUEST_TIMEOUT = _env_int("TTS_REQUEST_TIMEOUT", 600)  # 读取（流式）最长等待秒
_TIMEOUT = (_CONNECT_TIMEOUT, _REQUEST_TIMEOUT)           # (connect, read) 元组
_LOG_TTFB = _env_str("TTS_LOG_TTFB").lower() in ("1", "true", "yes")
# 0 表示不限制；>0 时在本 Locust 进程内统计「请求完成次数」达该值后停止（含失败；与 -t 同时存在时先触发的先停）
_TOTAL_REQUEST_LIMIT = _env_int("LOCUST_TOTAL_REQUESTS", 100)


# 默认短句：分诊/导医常见一问（约二十余字）
_DEFAULT_SHORT_MEDICAL = (
    "您好，我是导诊助手。请问您今天哪里不舒服？有没有发热或胸闷？"
)
# 默认长句：随访与健康宣教式回复（约一百五十字级，便于压测长文本流式）
# _DEFAULT_LONG_MEDICAL = (
#     "根据您刚才描述的反复胃痛两周、进食后加重，偶尔反酸，建议您尽快预约消化内科门诊，"
#     "必要时医生可能会安排幽门螺杆菌检测或胃镜检查。就诊前请清淡饮食，避免饮酒和辛辣刺激；"
#     "若出现呕血、黑便或剧烈腹痛，请立即前往急诊。您可以在问诊时补充既往胃病用药史与过敏史，"
#     "并带好近期化验单，方便医生综合判断是否需要调整治疗方案或进一步随访。"
# )
# 默认长句：随访与健康宣教式回复（约90字级，便于压测长文本流式）
_DEFAULT_LONG_MEDICAL = (
    "根据您描述的反复胃痛、饭后加重、偶尔反酸，建议预约消化内科就诊。医生可能会建议幽门螺杆菌检测或胃镜检查。就诊前请清淡饮食，避免饮酒和辛辣食物。若出现呕血、黑便或剧烈腹痛，请立即前往急诊。"
)

def _inputs() -> List[str]:
    """返回 [短句, 长句]，任务中 random.choice 二选一。"""
    short = _env_str("TTS_INPUT_SHORT") or _env_str("TTS_INPUT") or _DEFAULT_SHORT_MEDICAL
    long_ = _env_str("TTS_INPUT_LONG") or _env_str("TTS_INPUT_ALT") or _DEFAULT_LONG_MEDICAL
    return [short, long_]


_INPUT_POOL = _inputs()


def _speech_json(response_format: str, text: str) -> Dict[str, Any]:
    return {
        "model": "tts-1",
        "input": text,
        "voice": _DEFAULT_VOICE,
        "response_format": response_format,
    }


@events.init.add_listener
def _on_locust_init(environment, **kwargs):
    """启动时打印解析后的 voice，便于发现 CMD 多写了引号等问题。"""
    logger.info(
        "HTTP timeouts: connect=%ds read=%ds (env TTS_CONNECT_TIMEOUT / TTS_REQUEST_TIMEOUT)",
        _CONNECT_TIMEOUT,
        _REQUEST_TIMEOUT,
    )
    logger.info("Effective TTS_VOICE for speech requests: %r", _DEFAULT_VOICE)
    if _TOTAL_REQUEST_LIMIT > 0:
        logger.info("LOCUST_TOTAL_REQUESTS=%d (stop after this many completed requests)", _TOTAL_REQUEST_LIMIT)
    if _DEFAULT_VOICE == "REPLACE_WITH_YOUR_VOICE_ID":
        logger.warning(
            "TTS_VOICE 仍为占位符 REPLACE_WITH_YOUR_VOICE_ID，"
            "请设置环境变量 TTS_VOICE 或在下方修改 locustfile 默认值，否则 /v1/audio/speech 将失败。"
        )
    elif _DEFAULT_VOICE.startswith('"') or _DEFAULT_VOICE.endswith('"'):
        logger.warning(
            "TTS_VOICE 似乎仍含引号，请检查 CMD 是否使用了 set VAR=\"value\"；应使用 set VAR=value"
        )

    if _TOTAL_REQUEST_LIMIT <= 0:
        return

    req_done = [0]
    req_lock = Semaphore()

    def on_request(**_kw) -> None:
        req_lock.acquire()
        try:
            req_done[0] += 1
            n = req_done[0]
            if n >= _TOTAL_REQUEST_LIMIT:

                def _quit() -> None:
                    r = environment.runner
                    if r is not None:
                        logger.info(
                            "LOCUST_TOTAL_REQUESTS reached (%d), stopping runner", _TOTAL_REQUEST_LIMIT
                        )
                        r.quit()

                gevent.spawn(_quit)
        finally:
            req_lock.release()

    environment.events.request.add_listener(on_request)

    def on_test_start(**_kw) -> None:
        req_done[0] = 0

    environment.events.test_start.add_listener(on_test_start)


class TTSMergedServiceUser(HttpUser):
    """
    模拟客户端：以合成接口为主，辅以轻量健康检查与音色列表（权重较低）。
    """

    # 命令行 --host 会覆盖此处；未传 --host 时用环境变量或默认 IP
    host = os.environ.get("LOCUST_TTS_HOST", _DEFAULT_HOST)
    # Locust 里「虚拟用户」两次任务之间的等待时间，随机 1～2 秒
    # wait_time = between(1, 2)
    wait_time = constant(1)

    def on_start(self):
        if _DEFAULT_VOICE == "REPLACE_WITH_YOUR_VOICE_ID":
            # 仍然允许压 health，但 speech 会失败；提前打日志
            pass

    @task(8)
    def speech_wav_stream(self):
        """流式 WAV：消费完整响应体，统计客户端首字节时间（TTFB）仅供对照服务端 ttfa_wall_ms。"""
        text = random.choice(_INPUT_POOL)
        payload = _speech_json("wav", text)
        start = time.perf_counter()
        ttfb_ms: float | None = None
        nbytes = 0

        with self.client.post(
            "/v1/audio/speech",
            json=payload,
            catch_response=True,
            stream=True,
            timeout=_TIMEOUT,
            name="/v1/audio/speech [wav stream]",
        ) as resp:
            try:
                if resp.status_code != 200:
                    resp.failure(_http_fail_msg(resp))
                    return
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - start) * 1000.0
                    nbytes += len(chunk)
                if nbytes < 1024:
                    resp.failure(
                        f"body too small: {nbytes} bytes (HTTP was 200; check Content-Type / proxy)"
                    )
                    return
                resp.success()
            except Exception as exc:
                resp.failure(str(exc))
                return

        if _LOG_TTFB and ttfb_ms is not None and random.random() < 0.05:
            logger.info(
                "client_ttfb_ms≈%.1f (first byte after POST), bytes=%d, voice=%s",
                ttfb_ms,
                nbytes,
                _DEFAULT_VOICE,
            )

    @task(0)
    def speech_mp3_non_stream(self):
        """非流式 MP3：整包下载，更接近「短句+整段延迟」场景。"""
        text = random.choice(_INPUT_POOL)
        payload = _speech_json("mp3", text)
        with self.client.post(
            "/v1/audio/speech",
            json=payload,
            catch_response=True,
            stream=True,
            timeout=_TIMEOUT,
            name="/v1/audio/speech [mp3]",
        ) as resp:
            try:
                if resp.status_code != 200:
                    resp.failure(_http_fail_msg(resp))
                    return
                total = sum(len(c) for c in resp.iter_content(chunk_size=65536) if c)
                if total < 64:
                    resp.failure(f"mp3 body too small: {total}")
                    return
                resp.success()
            except Exception as exc:
                resp.failure(str(exc))

    @task(0)
    def health(self):
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
            timeout=_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(_http_fail_msg(resp))
            else:
                try:
                    data = resp.json()
                    if data.get("status") != "ok":
                        # 仍算 HTTP 成功，标记为失败便于告警 degraded
                        resp.failure(f"status field={data.get('status')!r}")
                    else:
                        resp.success()
                except Exception as exc:
                    resp.failure(f"json: {exc}")

    @task(0)
    def list_voices(self):
        with self.client.get(
            "/voices?status=active",
            catch_response=True,
            name="/voices",
            timeout=_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(_http_fail_msg(resp))
            else:
                resp.success()
