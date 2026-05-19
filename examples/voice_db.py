"""
音色数据访问层：MySQL（权威数据源）+ Redis（只读缓存）。

管理 API（voice_manager_api.py）和 TTS 推理服务（openai_server_concurrent_v2.py）
共用此模块，避免重复代码。
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pymysql
import redis

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """从 config.json 加载配置；支持传入路径或自动查找项目根目录。"""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.json"
        )
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


class VoiceDB:
    """封装 MySQL + Redis 的音色数据访问。"""

    def __init__(self, config: dict):
        mysql_cfg = config["mysql"]["faster_qwen3_tts_tone"]
        redis_cfg = config["redis"]["faster_qwen3_tts_tone"]

        self._mysql_host = mysql_cfg["host"]
        self._mysql_port = int(mysql_cfg["port"])
        self._mysql_user = mysql_cfg["user"]
        self._mysql_password = mysql_cfg["password"]
        self._mysql_db = mysql_cfg["database"]
        self._mysql_table = mysql_cfg["table_name"]
        self._mysql_charset = mysql_cfg.get("charset", "utf8mb4")
        self._mysql_timeout = int(mysql_cfg.get("connect_timeout", 10))
        self._mysql_retry = int(mysql_cfg.get("retry", 3))

        self._key_prefix = redis_cfg.get("key_prefix", "tts:voice:")
        self._cache_ttl = int(redis_cfg.get("cache_ttl_seconds", 86400))

        self._redis = redis.Redis(
            host=redis_cfg["redis_ip"],
            port=int(redis_cfg["redis_port"]),
            password=redis_cfg.get("password"),
            db=int(redis_cfg.get("db", 0)),
            decode_responses=True,
            socket_connect_timeout=5,
        )

        self._tone_wav_dir = config.get("tone_wav_file_dir", "")
        self._allowed_formats = config.get("allowed_audio_formats", ["wav"])
        self._max_file_mb = config.get("max_audio_file_size_mb", 20)

    # ------------------------------------------------------------------
    # MySQL 连接（每次操作短连接，线程安全）
    # ------------------------------------------------------------------

    def _get_conn(self) -> pymysql.Connection:
        last_exc = None
        for attempt in range(1, self._mysql_retry + 1):
            try:
                return pymysql.connect(
                    host=self._mysql_host,
                    port=self._mysql_port,
                    user=self._mysql_user,
                    password=self._mysql_password,
                    database=self._mysql_db,
                    charset=self._mysql_charset,
                    connect_timeout=self._mysql_timeout,
                    cursorclass=pymysql.cursors.DictCursor,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "MySQL connect attempt %d/%d failed: %s",
                    attempt, self._mysql_retry, exc,
                )
                if attempt < self._mysql_retry:
                    time.sleep(min(attempt, 3))
        raise ConnectionError(
            f"无法连接 MySQL ({self._mysql_host}:{self._mysql_port}): {last_exc}"
        )

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """如果表不存在则自动建表。"""
        ddl = f"""
        CREATE TABLE IF NOT EXISTS `{self._mysql_table}` (
            `voice_id`       VARCHAR(128)  NOT NULL PRIMARY KEY,
            `ref_text`       TEXT          NOT NULL,
            `language`       VARCHAR(32)   NOT NULL DEFAULT 'Auto',
            `ref_audio_path` VARCHAR(512)  NOT NULL,
            `status`         VARCHAR(16)   NOT NULL DEFAULT 'active',
            `version`        INT           NOT NULL DEFAULT 1,
            `created_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
            `updated_at`     DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info("Table `%s` ensured.", self._mysql_table)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Redis 缓存辅助
    # ------------------------------------------------------------------

    def _redis_key(self, voice_id: str) -> str:
        return f"{self._key_prefix}{voice_id}"

    def _get_from_redis(self, voice_id: str) -> Optional[dict]:
        try:
            raw = self._redis.get(self._redis_key(voice_id))
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis GET 失败 (voice_id=%s): %s", voice_id, exc)
        return None

    def _set_to_redis(self, voice_id: str, data: dict) -> None:
        try:
            self._redis.setex(
                self._redis_key(voice_id),
                self._cache_ttl,
                json.dumps(data, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("Redis SET 失败 (voice_id=%s): %s", voice_id, exc)

    def invalidate_cache(self, voice_id: str) -> None:
        """清除 Redis 中指定 voice 的缓存。"""
        try:
            self._redis.delete(self._redis_key(voice_id))
        except Exception as exc:
            logger.warning("Redis DEL 失败 (voice_id=%s): %s", voice_id, exc)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _row_to_voice_cfg(self, row: dict) -> dict:
        """将 MySQL 行转为 TTS 推理侧期望的 voice config 格式。

        TTS 代码中使用 voice_cfg["ref_audio"]，这里做字段映射。
        """
        return {
            "voice_id": row["voice_id"],
            "ref_audio": row["ref_audio_path"],
            "ref_text": row.get("ref_text", ""),
            "language": row.get("language", "Auto"),
            "status": row.get("status", "active"),
            "version": row.get("version", 1),
        }

    def get_voice(self, voice_id: str) -> Optional[dict]:
        """按 voice_id 获取音色配置：先 Redis → 未命中则 MySQL → 回写 Redis。"""
        cached = self._get_from_redis(voice_id)
        if cached:
            return cached

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM `{self._mysql_table}` "
                    f"WHERE `voice_id` = %s AND `status` = 'active'",
                    (voice_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return None

        cfg = self._row_to_voice_cfg(row)
        self._set_to_redis(voice_id, cfg)
        return cfg

    def list_voices(self, status: str = "active") -> List[dict]:
        """列出指定状态的所有音色。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM `{self._mysql_table}` WHERE `status` = %s "
                    f"ORDER BY `created_at`",
                    (status,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [self._row_to_voice_cfg(r) for r in rows]

    # ------------------------------------------------------------------
    # 增加
    # ------------------------------------------------------------------

    def add_voice(
        self,
        voice_id: str,
        ref_audio_path: str,
        ref_text: str,
        language: str = "Auto",
    ) -> dict:
        """新增音色，返回插入后的 voice config。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO `{self._mysql_table}` "
                    f"(`voice_id`, `ref_audio_path`, `ref_text`, `language`, "
                    f" `status`, `version`, `created_at`, `updated_at`) "
                    f"VALUES (%s, %s, %s, %s, 'active', 1, NOW(), NOW())",
                    (voice_id, ref_audio_path, ref_text, language),
                )
            conn.commit()
        finally:
            conn.close()

        cfg = {
            "voice_id": voice_id,
            "ref_audio": ref_audio_path,
            "ref_text": ref_text,
            "language": language,
            "status": "active",
            "version": 1,
        }
        return cfg

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------

    def update_voice(self, voice_id: str, **kwargs) -> Optional[dict]:
        """更新音色字段（ref_audio_path / ref_text / language / status）。

        写库成功后清除 Redis 缓存；返回更新后的 voice config。
        """
        allowed = {"ref_audio_path", "ref_text", "language", "status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return self.get_voice(voice_id)

        set_clauses = ", ".join(f"`{k}` = %s" for k in updates)
        values = list(updates.values())
        sql = (
            f"UPDATE `{self._mysql_table}` SET {set_clauses}, "
            f"`version` = `version` + 1, `updated_at` = NOW() "
            f"WHERE `voice_id` = %s"
        )
        values.append(voice_id)

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                affected = cur.execute(sql, values)
            conn.commit()
        finally:
            conn.close()

        if affected == 0:
            return None

        self.invalidate_cache(voice_id)
        return self.get_voice(voice_id)

    # ------------------------------------------------------------------
    # 恢复（软删除后）
    # ------------------------------------------------------------------

    def restore_voice(self, voice_id: str) -> Optional[dict]:
        """将软删除（status=disabled）的音色恢复为 active；成功则清 Redis 并返回配置。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                affected = cur.execute(
                    f"UPDATE `{self._mysql_table}` SET `status` = 'active', "
                    f"`version` = `version` + 1, `updated_at` = NOW() "
                    f"WHERE `voice_id` = %s AND `status` = 'disabled'",
                    (voice_id,),
                )
            conn.commit()
        finally:
            conn.close()

        if affected == 0:
            return None

        self.invalidate_cache(voice_id)
        return self.get_voice(voice_id)

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_voice(self, voice_id: str, hard: bool = False) -> bool:
        """删除音色。hard=True 硬删除行，否则软删除（status='disabled'）。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                if hard:
                    affected = cur.execute(
                        f"DELETE FROM `{self._mysql_table}` WHERE `voice_id` = %s",
                        (voice_id,),
                    )
                else:
                    affected = cur.execute(
                        f"UPDATE `{self._mysql_table}` SET `status` = 'disabled', "
                        f"`updated_at` = NOW() WHERE `voice_id` = %s",
                        (voice_id,),
                    )
            conn.commit()
        finally:
            conn.close()

        self.invalidate_cache(voice_id)
        return affected > 0

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def check_connections(self) -> dict:
        """检测 MySQL 和 Redis 连通性，返回状态。"""
        result = {"mysql": False, "redis": False}
        try:
            conn = self._get_conn()
            conn.ping()
            conn.close()
            result["mysql"] = True
        except Exception as exc:
            logger.error("MySQL health check failed: %s", exc)
        try:
            self._redis.ping()
            result["redis"] = True
        except Exception as exc:
            logger.error("Redis health check failed: %s", exc)
        return result

    @property
    def tone_wav_dir(self) -> str:
        return self._tone_wav_dir

    @property
    def allowed_formats(self) -> list:
        return self._allowed_formats

    @property
    def max_file_mb(self) -> int:
        return self._max_file_mb
