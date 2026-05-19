#!/usr/bin/env python3
"""
Parity（非 CUDA Graph）流式 decode的显式会话：通过 ``ParityStreamSession`` + ``step()`` 推进。

§6.2.8 顺序 1：单 session 下将 ``parity_generate_streaming`` 的内层循环收成可调用步进，
便于路线 B 多 session 调度原型；对外 codec 块与 ``parity_generate_streaming`` 逐块对齐。

详见 ``faster-qwen3-tts服务使用与优化.md`` §6.3。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

from .sampling import apply_repetition_penalty, sample_logits


class ParityStreamSession:
    """
    Parity（非 CUDA Graph）流式解码会话：构造时完成 prefill，之后反复调用 ``step()`` 推进 decode。

    语义与 ``parity_generate_streaming`` 中单次 ``for`` 循环迭代一致，便于路线 B 多 session 调度原型。
    """

    __slots__ = (
        "talker",
        "talker_input_embeds",
        "attention_mask",
        "trailing_text_hiddens",
        "tts_pad_embed",
        "config",
        "max_new_tokens",
        "min_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "do_sample",
        "repetition_penalty",
        "chunk_size",
        "eos_id",
        "suppress_mask",
        "talker_past_kv",
        "past_hidden",
        "gen_step",
        "token",
        "t_prefill",
        "chunk_buffer",
        "all_first_tokens",
        "total_steps",
        "chunk_count",
        "chunk_start",
        "_iter",
        "_post_loop",
        "_done",
    )

    @torch.inference_mode()
    def __init__(
        self,
        talker,
        talker_input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        config: Any,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> None:
        """
        构造会话：执行与 ``parity_generate_streaming`` 相同的 prefill，并采样首 token，初始化攒块状态。

        Args:
            talker: Talker 模型，用于 ``forward``（动态 KV 路径）。
            talker_input_embeds: prefill 用 ``inputs_embeds``（与 ``_prepare_generation`` 产出 ``tie`` 一致）。
            attention_mask: 初始注意力掩码；非 ``None`` 时会在构造内 ``clone`` 供 decode 扩展。
            trailing_text_hiddens: 解码步条件文本隐层 ``tth``。
            tts_pad_embed: 超出文本长度后的 pad 嵌入 ``tpe``。
            config: 含 ``vocab_size``、``codec_eos_token_id`` 等。
            max_new_tokens: decode 最多推进的循环次数上限（与原版 ``range(max_new_tokens)`` 一致）。
            min_new_tokens: 至少生成多少步后才允许采到 EOS。
            temperature: 采样温度。
            top_k: Top-k 采样。
            top_p: Top-p 采样。
            do_sample: 是否采样。
            repetition_penalty: 重复惩罚系数。
            chunk_size: 每多少步 codec 叠成一块再对外等价 ``yield``（与 ``stream_chunk_size`` 含义一致）。
        """
        self.talker = talker
        self.talker_input_embeds = talker_input_embeds
        self.attention_mask = attention_mask
        self.trailing_text_hiddens = trailing_text_hiddens
        self.tts_pad_embed = tts_pad_embed
        self.config = config
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.chunk_size = chunk_size

        device = talker_input_embeds.device
        vocab_size = config.vocab_size
        eos_id = config.codec_eos_token_id
        self.eos_id = eos_id
        suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        suppress_start = max(0, vocab_size - 1024)
        for i in range(suppress_start, vocab_size):
            if i != eos_id:
                suppress_mask[i] = True
        self.suppress_mask = suppress_mask

        t_start = time.time()
        out = talker.forward(
            inputs_embeds=talker_input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
        self.talker_past_kv = out.past_key_values
        self.past_hidden = out.past_hidden
        self.gen_step = out.generation_step
        logits = out.logits[:, -1, :]
        suppress_eos = min_new_tokens > 0
        self.token = sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )
        if attention_mask is not None:
            self.attention_mask = attention_mask.clone()
        torch.cuda.synchronize()
        self.t_prefill = time.time() - t_start

        self.chunk_buffer = []
        self.all_first_tokens = []
        self.total_steps = 0
        self.chunk_count = 0
        self.chunk_start = time.time()
        self._iter = 0
        self._post_loop = False
        self._done = False

    @property
    def finished(self) -> bool:
        """
        会话是否已结束（不应再调用 ``step()`` 期望新产出）。

        Returns:
            ``True`` 表示尾块已发出或确认无尾块；``False`` 表示仍可 ``step()``。
        """
        return self._done

    def _emit_chunk(self, *, is_final: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        将当前 ``chunk_buffer`` 栈化为一块 codec 张量，并生成与 METRICS 一致的 ``timing`` 字典。

        Args:
            is_final: 是否为流式最后一块（``is_final`` 字段）；为真时在方法末尾置 ``_done``。

        Returns:
            ``(stacked_codec, timing_dict)``，其中 ``timing_dict`` 含 ``chunk_index``、``chunk_steps``、
            ``prefill_ms``、``decode_ms``、``total_steps_so_far``、``is_final``。
        """
        torch.cuda.synchronize()
        chunk_decode_time = time.time() - self.chunk_start
        self.total_steps += len(self.chunk_buffer)
        stacked = torch.stack(self.chunk_buffer)
        timing: Dict[str, Any] = {
            "chunk_index": self.chunk_count,
            "chunk_steps": len(self.chunk_buffer),
            "prefill_ms": self.t_prefill * 1000 if self.chunk_count == 0 else 0,
            "decode_ms": chunk_decode_time * 1000,
            "total_steps_so_far": self.total_steps,
            "is_final": is_final,
        }
        self.chunk_buffer = []
        if not is_final:
            self.chunk_count += 1
            self.chunk_start = time.time()
        else:
            self._done = True
        return stacked, timing

    def _flush_tail_if_any(self) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        在 EOS / 步数耗尽 / ``hidden_states[1] is None`` 等结束路径上，尝试发出不足 ``chunk_size`` 的尾块。

        Returns:
            ``chunk_buffer`` 非空时返回 ``(codec_chunk, timing)`` 且 ``is_final=True``；否则返回 ``None``
            并置 ``finished``。
        """
        if not self.chunk_buffer:
            self._done = True
            return None
        return self._emit_chunk(is_final=True)

    @torch.inference_mode()
    def step(self) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        推进一次 decode，等价于 ``parity_generate_streaming`` 中 ``for`` 循环的一次迭代（含首部 EOS 判断与尾块 flush）。

        调用方在 ``not finished`` 时可反复调用；某次调用可能仅更新 KV/token/buffer 而不对外产出块。

        Returns:
            当本步凑满 ``chunk_size`` 或进入结束路径需发出尾块时，返回 ``(codec_chunk, timing)``，
            形状与字段同 ``parity_generate_streaming`` 的 ``yield``；仅内部前进时返回 ``None``。
            已 ``finished`` 后恒为 ``None``。
        """
        if self._done:
            return None
        if self._post_loop:
            self._post_loop = False
            return self._flush_tail_if_any()

        if self._iter >= self.max_new_tokens:
            self._post_loop = True
            return self.step()

        if self.token.item() == self.eos_id:
            self._post_loop = True
            return self.step()

        attn = self.attention_mask
        if attn is not None:
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=1)
            self.attention_mask = attn
            cache_position = torch.tensor([attn.shape[1] - 1], device=attn.device)
        else:
            cache_position = None

        out = self.talker.forward(
            input_ids=self.token.view(1, 1),
            attention_mask=self.attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=self.trailing_text_hiddens,
            tts_pad_embed=self.tts_pad_embed,
            generation_step=self.gen_step,
            past_hidden=self.past_hidden,
            past_key_values=self.talker_past_kv,
            subtalker_dosample=self.do_sample,
            subtalker_top_k=self.top_k,
            subtalker_top_p=self.top_p,
            subtalker_temperature=self.temperature,
            cache_position=cache_position,
        )
        codec_ids = out.hidden_states[1]
        if codec_ids is None:
            self._post_loop = True
            return self.step()

        self.chunk_buffer.append(codec_ids.squeeze(0).detach())
        self.all_first_tokens.append(self.token.detach())

        logits = out.logits[:, -1, :]
        if self.repetition_penalty != 1.0 and self.all_first_tokens:
            history = torch.stack(self.all_first_tokens)
            logits = apply_repetition_penalty(logits, history, self.repetition_penalty)

        suppress_eos = len(self.all_first_tokens) < self.min_new_tokens
        self.token = sample_logits(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            suppress_mask=self.suppress_mask,
            suppress_tokens=[self.eos_id] if suppress_eos else None,
        )
        self.talker_past_kv = out.past_key_values
        self.past_hidden = out.past_hidden
        self.gen_step = out.generation_step
        self._iter += 1

        if len(self.chunk_buffer) >= self.chunk_size:
            return self._emit_chunk(is_final=False)
        return None
