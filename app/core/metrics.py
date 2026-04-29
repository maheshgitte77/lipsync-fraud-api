"""Lightweight per-stage timing. Return stage timings with every job response."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from app.core.logger import get_logger

logger = get_logger("core.metrics")


class StageTimer:
    """Accumulate wall-clock time per named stage."""

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}
        self._started: float = time.perf_counter()

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self._stages[name] = round(self._stages.get(name, 0.0) + dur_ms, 2)
            logger.debug("stage %s took %.2f ms", name, dur_ms)

    def mark(self, name: str, duration_ms: float) -> None:
        self._stages[name] = round(self._stages.get(name, 0.0) + float(duration_ms), 2)

    def snapshot(self) -> dict:
        total_ms = round((time.perf_counter() - self._started) * 1000.0, 2)
        return {"totalMs": total_ms, "stages": dict(self._stages)}
