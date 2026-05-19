#!/usr/bin/env python3
"""
§6.2.8 顺序 2：多路 ``ParityStreamSession`` 在 **parity** 路径下 **轮流 ``step()``**。

每轮对每个尚未 ``finished`` 的 session 各调用一次 ``step()``；若某次调用产出 codec 块则对外 yield。
"""
from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import torch

from .parity_stream_session import ParityStreamSession


def iter_round_robin_codec_chunks(
    sessions: Sequence[ParityStreamSession],
) -> Iterator[Tuple[int, torch.Tensor, dict]]:
    """
    严格轮询：循环多轮；每轮按 session 下标顺序，对 **未结束** 的 session 各执行一次 ``step()``。

    Yields:
        ``(session_index, codec_chunk, timing_dict)``：仅当本次 ``step()`` 返回非 ``None`` 时产出，
        与 ``parity_generate_streaming`` 单次 ``yield`` 形状一致。
    """
    sess_list: List[ParityStreamSession] = list(sessions)
    while True:
        stepped = False
        for sid, sess in enumerate(sess_list):
            if sess.finished:
                continue
            stepped = True
            item = sess.step()
            if item is not None:
                chunk, timing = item
                yield sid, chunk, timing
        if not stepped:
            break
