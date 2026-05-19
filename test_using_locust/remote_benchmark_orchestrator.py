#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 remote_benchmark_orchestrator_plan.md：在客户端机（如 10.251.11.27）上解析《faster-qwen3-tts修改后4090上并发测试.md》，
SSH 到远端（如 10.251.11.55）在 Docker 容器内启停 openai_server_v4.py，向 inference.log 写入 ORCH_MARKER，
并在本机以 headless Locust 压测。

依赖：本机 ssh、docker（仅远端需要）、locust、Python 3.10+；远端需已配置免密 SSH 且用户可 docker exec。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("remote_benchmark_orchestrator")

# 默认与方案文档一致；可通过命令行覆盖
DEFAULT_SSH_HOST = "10.251.11.55"
DEFAULT_SSH_USER = os.environ.get("ORCH_SSH_USER", os.environ.get("USER", "root"))
DEFAULT_CONTAINER = "faster-qwen3-tts-jiangsy"
DEFAULT_HEALTH_URL = "http://10.251.11.55:10018/health"
DEFAULT_INFERENCE_LOG = "/data/tts/faster-qwen3-tts-assets/logs/inference.log"
DEFAULT_WORKDIR = Path(__file__).resolve().parent
DEFAULT_LOCUST_FILE = "locustfile.py"
DEFAULT_POST_SERVICE_SLEEP = 30
DEFAULT_POST_LOCUST_SLEEP = 30
DEFAULT_HEALTH_TIMEOUT_SEC = 600
DEFAULT_HEALTH_POLL_SEC = 2


def _default_markdown_path() -> str:
    """自本脚本目录向上查找《faster-qwen3-tts修改后4090上并发测试.md》（通常在仓库根与 tts-jiang 同级）。"""
    name = "faster-qwen3-tts修改后4090上并发测试.md"
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        cand = cur / name
        if cand.is_file():
            return str(cand)
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return str(Path(__file__).resolve().parents[2] / name)

# 按二级标题切分，避免 (.*?) 与「下一节」前瞻在复杂正文下的边界问题
SECTION_SPLIT = re.compile(r"(?m)^(?=##\s+1\.\d+\.\s)")
CODE_FENCE = re.compile(r"```(?:bash)?\s*\n(.*?)```", re.DOTALL)
LOCUST_CASE = re.compile(
    r"^####\s+(\S+)\s+4090_u(\d+)_r(\d+)_(\d+)m\s*$",
    re.MULTILINE,
)
REPLICAS_CCY = re.compile(r"replicas\s*=\s*(\d+).*?concurrency\s*=\s*(\d+)", re.IGNORECASE)


@dataclass
class LocustCase:
    section_label: str  # e.g. 1.1.1.1.
    users: int
    ramp: int
    duration_min: int

    @property
    def locust_time(self) -> str:
        return f"{self.duration_min}m"


@dataclass
class BenchmarkSection:
    headline: str  # e.g. 1.1.
    replicas: int
    concurrency: int
    server_bash: str  # multiline script body (cd + nohup python ... will be wrapped)
    locust_cases: List[LocustCase]


def _parse_replicas_concurrency(headline_line: str, body: str) -> Tuple[int, int]:
    m = REPLICAS_CCY.search(headline_line) or REPLICAS_CCY.search(body[:500])
    if not m:
        raise ValueError(f"无法在章节首行或正文前部解析 replicas/concurrency: {headline_line!r}")
    return int(m.group(1)), int(m.group(2))


def _extract_server_command(section_body: str) -> str:
    for m in CODE_FENCE.finditer(section_body):
        block = m.group(1).strip()
        if "openai_server_v4.py" in block and "python" in block:
            return block
    raise ValueError("未找到包含 openai_server_v4.py 的代码块")


def parse_benchmark_md(text: str) -> List[BenchmarkSection]:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    parts = SECTION_SPLIT.split(text)
    sections: List[BenchmarkSection] = []
    for part in parts[1:]:
        if not part.strip():
            continue
        head_line, _, rest = part.partition("\n")
        body = rest
        hm = re.match(r"^##\s+(1\.\d+\.)\s+", head_line)
        if not hm:
            continue
        headline = hm.group(1).strip()
        if not body.strip():
            continue
        replicas, concurrency = _parse_replicas_concurrency(head_line, body)
        server_bash = _extract_server_command(body)
        locust_cases: List[LocustCase] = []
        for lm in LOCUST_CASE.finditer(body):
            locust_cases.append(
                LocustCase(
                    section_label=lm.group(1).rstrip("."),
                    users=int(lm.group(2)),
                    ramp=int(lm.group(3)),
                    duration_min=int(lm.group(4)),
                )
            )
        if not locust_cases:
            raise ValueError(f"章节 {headline} 下未解析到 #### … 4090_u*_r*_**m 用例")
        sections.append(
            BenchmarkSection(
                headline=headline,
                replicas=replicas,
                concurrency=concurrency,
                server_bash=server_bash,
                locust_cases=locust_cases,
            )
        )
    if not sections:
        raise ValueError("未解析到任何 ## 1.x 章节，请检查 Markdown 结构")
    return sections


def _ssh_base(ssh_user: str, ssh_host: str, identity: Optional[str]) -> List[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if identity:
        cmd.extend(["-i", identity])
    cmd.append(f"{ssh_user}@{ssh_host}")
    return cmd


def remote_bash(
    ssh_user: str,
    ssh_host: str,
    container: str,
    inner_bash: str,
    *,
    identity: Optional[str] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """在远端通过 docker exec 执行一段 bash -lc（脚本经 base64 传入，避免多层引号问题）。"""
    inner_b64 = base64.b64encode(inner_bash.encode("utf-8")).decode("ascii")
    wrapped = (
        f"docker exec {shlex.quote(container)} "
        f"bash -lc {shlex.quote(f'echo {inner_b64} | base64 -d | bash')}"
    )
    argv = _ssh_base(ssh_user, ssh_host, identity) + [wrapped]
    log.debug("remote_bash: %s", argv[0:3] + ["..."] + [argv[-1][:80] + "..."])
    cp = subprocess.run(argv, capture_output=True, text=True, check=False)
    if check and cp.returncode != 0:
        log.error(
            "remote_bash 失败 rc=%s stdout=%r stderr=%r",
            cp.returncode,
            (cp.stdout or "")[:2000],
            (cp.stderr or "")[:2000],
        )
        cp.check_returncode()
    return cp


def append_inference_log_marker(
    ssh_user: str,
    ssh_host: str,
    container: str,
    inference_log: str,
    marker_lines: str,
    *,
    identity: Optional[str] = None,
) -> None:
    """向容器内 inference.log 追加 UTF-8 文本（经 base64，避免引号转义）。"""
    payload = marker_lines.rstrip("\n") + "\n"
    b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    inner = (
        f"mkdir -p $(dirname {shlex.quote(inference_log)}) && "
        f"echo {shlex.quote(b64)} | base64 -d >> {shlex.quote(inference_log)}"
    )
    remote_bash(ssh_user, ssh_host, container, inner, identity=identity, check=True)


def _orch_ts_line() -> str:
    """与推理日志常见格式接近的时间前缀行。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def format_marker_new_service(
    section_headline: str,
    replicas: int,
    concurrency: int,
) -> str:
    ts = _orch_ts_line()
    lines = [
        f"{ts} INFO ORCH_MARKER 新启动服务 replicas={replicas} concurrency={concurrency} section={section_headline.rstrip('.')}",
        "########################################",
        "########################################",
        "########################################",
        "",
    ]
    return "\n".join(lines)


def format_marker_locust(
    section_label: str,
    users: int,
    ramp: int,
    t_str: str,
    locust_host: str,
) -> str:
    ts = _orch_ts_line()
    lines = [
        f"{ts} INFO ORCH_MARKER 新启动locust并发测试 u={users} r={ramp} t={t_str} host={locust_host} section={section_label}",
        "----------------------------------------",
        "----------------------------------------",
        "----------------------------------------",
        "",
    ]
    return "\n".join(lines)


def kill_server_on_port(
    ssh_user: str,
    ssh_host: str,
    container: str,
    port: int = 8000,
    *,
    identity: Optional[str] = None,
) -> None:
    inner = f"""
set -e
if command -v fuser >/dev/null 2>&1; then
  fuser -k {port}/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  pids=$(lsof -t -iTCP:{port} -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; fi
else
  pkill -f 'openai_server_v4.py' 2>/dev/null || true
fi
sleep 2
""".strip()
    remote_bash(ssh_user, ssh_host, container, inner, identity=identity, check=False)


def start_server_background(
    ssh_user: str,
    ssh_host: str,
    container: str,
    server_bash: str,
    *,
    identity: Optional[str] = None,
) -> None:
    """在容器 /data/tts/faster-qwen3-tts 下后台启动 server_bash（md 中代码块原文，含续行）。"""
    cmd_body = server_bash.strip()
    inner = (
        "set -e\n"
        "cd /data/tts/faster-qwen3-tts\n"
        f"nohup bash -c {shlex.quote(cmd_body)} >> /tmp/openai_server_v4.log 2>&1 &\n"
        "sleep 2\n"
    )
    remote_bash(ssh_user, ssh_host, container, inner, identity=identity, check=True)


def wait_for_health(
    health_url: str,
    timeout_sec: float = DEFAULT_HEALTH_TIMEOUT_SEC,
    poll_sec: float = DEFAULT_HEALTH_POLL_SEC,
) -> None:
    deadline = time.monotonic() + timeout_sec
    last_err: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            if data.get("status") == "ok":
                log.info("health ok: %s", health_url)
                return
            last_err = f"status field={data.get('status')!r}"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            last_err = str(e)
        log.debug("health wait: %s", last_err)
        time.sleep(poll_sec)
    raise TimeoutError(f"健康检查超时 ({timeout_sec}s): {health_url} 最后错误: {last_err}")


def run_locust(
    workdir: Path,
    locust_file: str,
    users: int,
    ramp: int,
    t_str: str,
    locust_host: str,
) -> int:
    argv = [
        "locust",
        "-f",
        locust_file,
        "--headless",
        "-u",
        str(users),
        "-r",
        str(ramp),
        "-t",
        t_str,
        "--host",
        locust_host,
    ]
    log.info("locust: cwd=%s %s", workdir, " ".join(shlex.quote(a) for a in argv))
    p = subprocess.run(argv, cwd=str(workdir))
    return int(p.returncode)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="远程 TTS + Locust 串联编排（见 remote_benchmark_orchestrator_plan.md）")
    p.add_argument(
        "--markdown",
        default=os.environ.get("ORCH_BENCHMARK_MD") or _default_markdown_path(),
        help="《faster-qwen3-tts修改后4090上并发测试.md》路径（也可用环境变量 ORCH_BENCHMARK_MD）",
    )
    p.add_argument("--ssh-user", default=DEFAULT_SSH_USER, help="SSH 登录用户名")
    p.add_argument("--ssh-host", default=DEFAULT_SSH_HOST, help="SSH 目标主机")
    p.add_argument("--ssh-identity", default=os.environ.get("ORCH_SSH_IDENTITY", ""), help="SSH 私钥路径（可选）")
    p.add_argument("--docker-container", default=DEFAULT_CONTAINER, help="Docker 容器名")
    p.add_argument(
        "--health-url",
        default=os.environ.get("ORCH_HEALTH_URL", DEFAULT_HEALTH_URL),
        help="就绪判定用的 HTTP health 完整 URL",
    )
    p.add_argument(
        "--locust-host",
        default=os.environ.get("ORCH_LOCUST_HOST", ""),
        help="传给 locust --host（默认从 --health-url 推导：去掉 /health 等路径）",
    )
    p.add_argument(
        "--inference-log",
        default=os.environ.get("ORCH_INFERENCE_LOG", DEFAULT_INFERENCE_LOG),
        help="容器内 inference.log 绝对路径",
    )
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="Locust 工作目录")
    p.add_argument("--locust-file", default=DEFAULT_LOCUST_FILE, help="Locust 脚本文件名")
    p.add_argument("--post-service-sleep", type=int, default=DEFAULT_POST_SERVICE_SLEEP)
    p.add_argument("--post-locust-sleep", type=int, default=DEFAULT_POST_LOCUST_SLEEP)
    p.add_argument("--health-timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT_SEC)
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不执行 SSH/Locust")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Locust 非零退出时仍继续后续用例（默认遇错中止）",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    md_path = Path(args.markdown)
    if not md_path.is_file():
        log.error("Markdown 不存在: %s", md_path)
        return 2

    locust_host = args.locust_host.strip()
    if not locust_host:
        hu = args.health_url.rstrip("/")
        if hu.endswith("/health"):
            locust_host = hu[: -len("/health")]
        else:
            locust_host = re.sub(r"/health/?$", "", hu) or hu
        if not locust_host.startswith("http"):
            locust_host = "http://" + locust_host

    identity = args.ssh_identity.strip() or None

    text = md_path.read_text(encoding="utf-8")
    try:
        sections = parse_benchmark_md(text)
    except ValueError as e:
        log.error("解析 Markdown 失败: %s", e)
        return 2

    log.info("解析到 %d 个章节", len(sections))
    if args.dry_run:
        for s in sections:
            log.info(
                "DRY %s replicas=%s concurrency=%s locust_cases=%d",
                s.headline,
                s.replicas,
                s.concurrency,
                len(s.locust_cases),
            )
            for c in s.locust_cases:
                log.info(
                    "  DRY locust %s u=%d r=%d -t %dm",
                    c.section_label,
                    c.users,
                    c.ramp,
                    c.duration_min,
                )
        return 0

    if not os.environ.get("TTS_VOICE"):
        log.warning("环境变量 TTS_VOICE 未设置，Locust 可能使用占位 voice 导致合成失败")

    for si, sec in enumerate(sections):
        log.info("======== 章节 %s (%d/%d) ========", sec.headline, si + 1, len(sections))
        kill_server_on_port(
            args.ssh_user, args.ssh_host, args.docker_container, identity=identity
        )
        try:
            start_server_background(
                args.ssh_user,
                args.ssh_host,
                args.docker_container,
                sec.server_bash,
                identity=identity,
            )
            wait_for_health(args.health_url, timeout_sec=args.health_timeout)
        except Exception as e:
            log.exception("启动服务或健康检查失败: %s", e)
            return 1

        time.sleep(args.post_service_sleep)

        append_inference_log_marker(
            args.ssh_user,
            args.ssh_host,
            args.docker_container,
            args.inference_log,
            format_marker_new_service(sec.headline, sec.replicas, sec.concurrency),
            identity=identity,
        )

        for c in sec.locust_cases:
            t_str = c.locust_time
            append_inference_log_marker(
                args.ssh_user,
                args.ssh_host,
                args.docker_container,
                args.inference_log,
                format_marker_locust(
                    c.section_label, c.users, c.ramp, t_str, locust_host
                ),
                identity=identity,
            )
            rc = run_locust(
                args.workdir,
                args.locust_file,
                c.users,
                c.ramp,
                t_str,
                locust_host,
            )
            if rc != 0:
                log.error("Locust 退出码 %d: section=%s case=%s", rc, sec.headline, c.section_label)
                if not args.continue_on_error:
                    return rc
            time.sleep(args.post_locust_sleep)

    log.info("全部章节执行完毕")
    return 0


if __name__ == "__main__":
    sys.exit(main())
