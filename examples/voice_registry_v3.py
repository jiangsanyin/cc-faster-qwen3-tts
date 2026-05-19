"""
v3 音色注册表：JSON 文件 + 文件锁 + 原子替换写入。

供 voice_manager_api_v3.py（读写）与 openai_server_concurrent_v3.py（热加载只读）使用。
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

try:
    import fcntl  # type: ignore
except ImportError:
    fcntl = None  # Windows 等环境无 flock，仅单进程时可用


def load_config_v3(config_path: str) -> dict:
    """
    加载 v3 配置文件。
    
    Args:
        config_path: 配置文件路径。
        
    Returns:
        解析后的配置字典。
    """
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def _flock_path(lock_path: str, exclusive: bool):
    """
    文件锁上下文管理器。
    
    用于在读写注册表时进行进程间同步。在 Windows 下若无 fcntl 则降级为无锁模式。
    
    Args:
        lock_path: 锁文件路径。
        exclusive: 是否使用排他锁（写锁）。True 为排他锁，False 为共享锁（读锁）。
    """
    d = os.path.dirname(os.path.abspath(lock_path))
    if d:
        os.makedirs(d, exist_ok=True)
    lf = open(lock_path, "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lf.fileno(), flag)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        lf.close()


def _entry_to_voice_cfg(voice_id: str, entry: dict) -> dict:
    """
    将注册表原始条目转换为 TTS 推理所需的配置格式。
    
    Args:
        voice_id: 音色唯一标识。
        entry: 注册表中的原始 JSON 对象。
        
    Returns:
        包含推理必要字段的字典。
    """
    rt = entry.get("ref_text", "") or ""
    return {
        "voice_id": voice_id,
        "ref_audio": entry["ref_audio"],
        "ref_text": rt,
        "language": entry.get("language", "Auto"),
        "status": entry.get("status", "active"),
        "version": int(entry.get("version", 1)),
    }


class VoiceRegistryV3:
    """注册表读写类（管理端使用）：支持独占锁 + 原子替换写入。"""

    def __init__(self, registry_path: str):
        """
        初始化注册表对象。
        
        Args:
            registry_path: 注册表 JSON 文件的磁盘路径。
        """
        self.registry_path = os.path.abspath(registry_path)
        self.lock_path = self.registry_path + ".lock"
        parent = os.path.dirname(self.registry_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def ensure_file_exists(self) -> None:
        """确保注册表文件存在，如果不存在则创建一个空的 JSON 对象。"""
        if os.path.isfile(self.registry_path):
            return
        with _flock_path(self.lock_path, True):
            if not os.path.isfile(self.registry_path):
                self._write_unlocked({})

    def _read_unlocked(self) -> Dict[str, Any]:
        """
        在不加锁的情况下读取注册表内容。仅供内部持锁方法调用。
        
        Returns:
            注册表完整字典。
        """
        if not os.path.isfile(self.registry_path):
            return {}
        with open(self.registry_path, encoding="utf-8") as rf:
            raw = rf.read()
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _write_unlocked(self, data: Dict[str, Any]) -> None:
        """
        在不加锁的情况下执行原子写操作。仅供内部持锁方法调用。
        
        通过写入临时文件并执行 os.replace 实现原子性，防止 JSON 损坏。
        
        Args:
            data: 待写入的完整数据字典。
        """
        dird = os.path.dirname(self.registry_path) or "."
        fd, tmp = tempfile.mkstemp(suffix=".tmp", prefix="voices_reg_", dir=dird)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as wf:
                json.dump(data, wf, ensure_ascii=False, indent=2)
                wf.write("\n")
                wf.flush()
                os.fsync(wf.fileno())
            os.replace(tmp, self.registry_path)
        except Exception:
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def read_all(self) -> Dict[str, Any]:
        """
        加共享锁读取完整注册表。
        
        Returns:
            注册表完整字典。
        """
        with _flock_path(self.lock_path, False):
            return self._read_unlocked()

    def mutate(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        """
        在独占锁保护下修改注册表。
        
        流程：持锁 -> 读取 -> 执行回调修改 -> 原子写回 -> 释放锁。
        
        Args:
            fn: 修改数据的回调函数，接收当前数据字典作为参数。
        """
        with _flock_path(self.lock_path, True):
            data = self._read_unlocked()
            fn(data)
            self._write_unlocked(data)

    def get_voice_active(self, voice_id: str) -> Optional[dict]:
        """
        获取指定 ID 的 active 音色配置。
        
        Args:
            voice_id: 音色唯一标识。
            
        Returns:
            转换后的配置字典，如果不存在或已禁用则返回 None。
        """
        data = self.read_all()
        entry = data.get(voice_id)
        if not entry:
            return None
        if entry.get("status", "active") == "disabled":
            return None
        return _entry_to_voice_cfg(voice_id, entry)

    def get_entry_raw(self, voice_id: str) -> Optional[dict]:
        """
        获取指定 ID 的原始注册表条目（不区分状态）。
        
        Args:
            voice_id: 音色唯一标识。
            
        Returns:
            原始条目字典拷贝，不存在则返回 None。
        """
        data = self.read_all()
        e = data.get(voice_id)
        return dict(e) if e else None

    def list_voices(self, status: str = "active") -> List[dict]:
        """
        根据状态列出音色。
        
        Args:
            status: 'active' 或 'disabled'。
            
        Returns:
            转换后的配置字典列表，按 ID 排序。
        """
        data = self.read_all()
        out: List[dict] = []
        for vid, entry in data.items():
            st = entry.get("status", "active")
            if status == "active" and st == "disabled":
                continue
            if status == "disabled" and st != "disabled":
                continue
            out.append(_entry_to_voice_cfg(vid, entry))
        out.sort(key=lambda x: x.get("voice_id", ""))
        return out


class HotVoiceRegistryV3:
    """TTS 侧热加载注册表：基于文件修改时间（mtime）实现内存缓存失效。"""

    def __init__(self, registry_path: str):
        """
        初始化热加载注册表。
        
        Args:
            registry_path: 注册表 JSON 路径。
        """
        self.registry_path = os.path.abspath(registry_path)
        self.lock_path = self.registry_path + ".lock"
        self._mtime: Optional[float] = None
        self._cache: Dict[str, Any] = {}

    def _load_from_disk(self) -> None:
        """从磁盘加载数据到内存缓存，并记录 mtime。"""
        if not os.path.isfile(self.registry_path):
            self._cache = {}
            self._mtime = -1.0
            return
        with _flock_path(self.lock_path, False):
            try:
                m = os.path.getmtime(self.registry_path)
            except OSError:
                self._cache = {}
                self._mtime = -1.0
                return
            with open(self.registry_path, encoding="utf-8") as rf:
                raw = rf.read()
            self._cache = json.loads(raw) if raw.strip() else {}
            self._mtime = m

    def _reload_if_needed(self) -> None:
        """检查磁盘文件 mtime，如果发生变化则重新加载。"""
        if not os.path.isfile(self.registry_path):
            if self._mtime is not None and self._mtime != -1.0:
                self._cache = {}
                self._mtime = -1.0
            elif self._mtime is None:
                self._load_from_disk()
            return
        try:
            m = os.path.getmtime(self.registry_path)
        except OSError:
            self._cache = {}
            self._mtime = -1.0
            return
        if self._mtime is None or m != self._mtime:
            self._load_from_disk()

    def get_registry_copy(self) -> Dict[str, Any]:
        """
        获取当前注册表快照的拷贝（触发热加载检查）。
        
        Returns:
            完整注册表字典。
        """
        self._reload_if_needed()
        return dict(self._cache)

    def count_active(self) -> int:
        """
        统计当前 active 状态的音色数量。
        
        Returns:
            active 音色总数。
        """
        self._reload_if_needed()
        n = 0
        for _vid, entry in self._cache.items():
            if entry.get("status", "active") != "disabled":
                n += 1
        return n

    def resolve_active_voice(self, voice_id: str) -> Optional[dict]:
        """
        解析指定 ID 的 active 音色配置。
        
        Args:
            voice_id: 音色唯一标识。
            
        Returns:
            转换后的配置字典，不可用则返回 None。
        """
        self._reload_if_needed()
        entry = self._cache.get(voice_id)
        if not entry:
            return None
        if entry.get("status", "active") == "disabled":
            return None
        return _entry_to_voice_cfg(voice_id, entry)

    def list_active_voice_cfgs(self) -> List[dict]:
        """
        列出所有 active 状态的音色配置。
        
        Returns:
            配置字典列表。
        """
        self._reload_if_needed()
        out: List[dict] = []
        for vid, entry in self._cache.items():
            if entry.get("status", "active") == "disabled":
                continue
            out.append(_entry_to_voice_cfg(vid, entry))
        return out


def valid_voice_prompt_cache_keys(registry: Dict[str, Any]) -> set:
    """
    根据注册表生成所有合法的模型缓存 Key 集合。
    
    规则与 faster_qwen3_tts.model._resolve_voice_clone_prompt_from_reference 保持一致。
    
    Args:
        registry: 注册表数据字典。
        
    Returns:
        包含合法 cache_key 元组的集合。
    """
    S = set()
    for _vid, entry in registry.items():
        if entry.get("status", "active") == "disabled":
            continue
        ra = entry.get("ref_audio")
        if not ra:
            continue
        rt = entry.get("ref_text", "") or ""
        S.add((str(ra), rt, False, True))
    return S


def maybe_cleanup_voice_prompt_cache(
    model: Any,
    registry: Dict[str, Any],
    *,
    enable: bool,
    threshold: int,
) -> int:
    """
    若满足启发式阈值，则清理模型内存缓存中的“孤儿”条目。
    
    Args:
        model: FasterQwen3TTS 模型实例。
        registry: 当前注册表数据快照。
        enable: 是否启用清理。
        threshold: 触发清理的差值阈值 (N_cache - N_reg)。
        
    Returns:
        实际删除的缓存条目数量。
    """
    if not enable:
        return 0
    cache = getattr(model, "_voice_prompt_cache", None)
    if not isinstance(cache, dict):
        return 0
    n_reg = sum(
        1
        for _v, e in registry.items()
        if e.get("status", "active") != "disabled"
    )
    n_cache = len(cache)
    if n_cache - n_reg < threshold:
        return 0
    S = valid_voice_prompt_cache_keys(registry)
    removed = 0
    for k in list(cache.keys()):
        if k not in S:
            del cache[k]
            removed += 1
    return removed


def utc_now_iso() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串（不含微秒）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def audio_path_for_version(tone_dir: str, voice_id: str, ext: str, version: int) -> str:
    """
    根据版本号生成音频存储路径。
    
    首版 v1 使用 {voice_id}.{ext}，后续版本使用 {voice_id}_v{N}.{ext}。
    
    Args:
        tone_dir: 音频根目录。
        voice_id: 音色唯一标识。
        ext: 文件扩展名。
        version: 版本号。
        
    Returns:
        生成的完整磁盘路径。
    """
    if version <= 1:
        return os.path.join(tone_dir, f"{voice_id}.{ext}")
    return os.path.join(tone_dir, f"{voice_id}_v{version}.{ext}")
