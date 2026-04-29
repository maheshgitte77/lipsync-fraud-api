"""Per-job temp directory lifecycle."""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("utils.file_manager")


def new_job_id(length: int = 12) -> str:
    return uuid.uuid4().hex[:length]


def job_dir(job_id: str) -> Path:
    d = settings.paths.temp_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def temporary_job_dir(job_id: str | None = None, *, keep: bool = False) -> Iterator[tuple[str, Path]]:
    """Yield (job_id, dir); remove dir on exit unless keep=True."""
    jid = job_id or new_job_id()
    d = job_dir(jid)
    try:
        yield jid, d
    finally:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)
            logger.debug("cleaned job dir %s", d)
