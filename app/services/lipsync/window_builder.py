"""Select the single SyncNet analysis window (start / middle / end) from video duration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from app.utils.ffmpeg import trim_segment, video_duration_sec

logger = get_logger("services.lipsync.window")


@dataclass(frozen=True)
class SyncNetWindow:
    path: Path
    start_sec: float
    duration_sec: float


def build_window(duration_sec: float) -> dict | None:
    """Compute {startSec, durationSec} for the active window policy."""
    duration = max(0.0, float(duration_sec))
    if duration <= 0:
        return None
    only_dur = min(max(1.0, min(settings.syncnet.trim_max_seconds, 600.0)), duration)
    pos = settings.syncnet.window_position
    if pos == "middle":
        start = max(0.0, (duration - only_dur) / 2.0)
    elif pos == "end":
        start = max(0.0, duration - only_dur)
    else:
        start = 0.0
    return {"startSec": round(start, 3), "durationSec": round(only_dur, 3)}


def prepare_windows(video_path: Path, job_dir: Path) -> tuple[list[SyncNetWindow], dict]:
    """
    Build one SyncNet window (or use full source when trim disabled).

    Returns (windows, metadata) where metadata surfaces the policy in API responses.
    """
    src_dur = video_duration_sec(video_path)
    max_sec = max(1.0, min(settings.syncnet.trim_max_seconds, 600.0))
    meta: dict = {
        "singleWindowMode": True,
        "position": settings.syncnet.window_position,
        "trimEnabled": bool(settings.syncnet.trim_enabled),
        "trimMaxSeconds": max_sec,
        "sourceDurationSec": src_dur,
        "applied": False,
        "count": 1,
    }

    if not settings.syncnet.trim_enabled or src_dur is None:
        return [SyncNetWindow(path=video_path, start_sec=0.0, duration_sec=src_dur or 0.0)], meta

    win = build_window(src_dur)
    if not win:
        return [SyncNetWindow(path=video_path, start_sec=0.0, duration_sec=src_dur)], meta

    if float(win["startSec"]) <= 0.001 and src_dur <= max_sec + 0.05:
        return [SyncNetWindow(path=video_path, start_sec=0.0, duration_sec=src_dur)], meta

    seg = job_dir / "syncnet_window_1.mp4"
    trim_segment(
        video_path,
        seg,
        start_sec=float(win["startSec"]),
        seconds=float(win["durationSec"]),
    )
    meta["applied"] = True
    return [
        SyncNetWindow(
            path=seg,
            start_sec=float(win["startSec"]),
            duration_sec=float(win["durationSec"]),
        )
    ], meta
