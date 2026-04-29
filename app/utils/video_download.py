"""HTTP video download with retries + extension inference."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("utils.video_download")


ALLOWED_VIDEO_EXT: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})


def guess_video_extension(url: str) -> str:
    """Infer extension from URL path or query string hints. Defaults to .mp4."""
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext in ALLOWED_VIDEO_EXT:
        return ext
    qs = parse_qs(parsed.query or "")
    for key in ("filename", "file", "name"):
        for v in qs.get(key) or []:
            qext = Path(v).suffix.lower()
            if qext in ALLOWED_VIDEO_EXT:
                return qext
    return ".mp4"


def download_to_file(url: str, destination: Path) -> None:
    """Download video at `url` to `destination` with retry + timeout."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("videoUrl must be http/https")

    destination.parent.mkdir(parents=True, exist_ok=True)
    timeout_sec = max(30, settings.syncnet.download_timeout_sec)
    retries = max(1, settings.syncnet.download_retries)

    req = Request(url, headers={"User-Agent": f"{settings.app_name}/{settings.version}", "Connection": "close"})
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=timeout_sec) as response, open(destination, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if destination.is_file() and destination.stat().st_size > 0:
                logger.info("downloaded %s → %s (%d bytes)", url, destination, destination.stat().st_size)
                return
            raise RuntimeError("download completed with empty file")
        except (TimeoutError, socket.timeout, OSError, RuntimeError) as exc:
            last_err = exc
            logger.warning("download attempt %d/%d failed: %s", attempt, retries, exc)
            if destination.exists():
                destination.unlink(missing_ok=True)
            if attempt >= retries:
                break
            time.sleep(min(2 * attempt, 5))

    raise RuntimeError(
        f"video download failed after {retries} attempts (timeout={timeout_sec}s): {last_err}"
    )
