"""
Job state store: Redis when REDIS_URL/REDIS_HOST is set, in-memory otherwise.

Always use the singleton `job_store`. The API surface is tiny on purpose:
`set`, `get`. Callers own the shape of the dict.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("workers.job_store")


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memory: dict[str, dict] = {}
        self._redis: Any = None
        self._redis_disabled = False

    # ---- public API -------------------------------------------------------

    def set(self, job_id: str, **fields: Any) -> None:
        client = self._get_redis()
        if client is not None:
            key = self._redis_key(job_id)
            raw = client.get(key)
            prev = json.loads(raw) if raw else {}
            merged = {**prev, **fields}
            client.setex(key, max(60, settings.redis.job_ttl_sec), json.dumps(merged))
            return
        with self._lock:
            prev = self._memory.get(job_id, {})
            self._memory[job_id] = {**prev, **fields}

    def get(self, job_id: str) -> dict | None:
        client = self._get_redis()
        if client is not None:
            raw = client.get(self._redis_key(job_id))
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        with self._lock:
            v = self._memory.get(job_id)
            return dict(v) if v else None

    # ---- helpers ----------------------------------------------------------

    def _redis_key(self, job_id: str) -> str:
        return f"{settings.redis.job_store_prefix}:{job_id}"

    def _build_redis_url(self) -> str:
        r = settings.redis
        if r.url:
            return r.url
        if not r.host:
            return ""
        auth = ""
        if r.username and r.password:
            auth = f"{r.username}:{r.password}@"
        elif r.password:
            auth = f":{r.password}@"
        return f"redis://{auth}{r.host}:{r.port}"

    def _get_redis(self):
        if self._redis_disabled:
            return None
        if self._redis is not None:
            return self._redis
        url = self._build_redis_url()
        if not url:
            self._redis_disabled = True
            return None
        try:
            import redis as redis_lib

            c = redis_lib.Redis.from_url(url, decode_responses=True)
            c.ping()
            self._redis = c
            logger.info("redis connected (%s)", url.split("@")[-1])
            return self._redis
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis disabled: %s", exc)
            self._redis_disabled = True
            return None


job_store = JobStore()
