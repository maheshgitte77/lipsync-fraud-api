"""
FFmpeg / FFprobe wrappers used for trimming, normalization and duration probing.

All callers go through here — never shell out to ffmpeg directly from services.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("utils.ffmpeg")


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe returns a non-zero exit code."""


def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def ffprobe_duration_sec(path: Path) -> float | None:
    """Container duration via ffprobe; returns None if unavailable."""
    try:
        r = _run(
            [
                settings.paths.ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
        return None
    if r.returncode != 0 or not (r.stdout or "").strip():
        return None
    try:
        doc = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    fmt = doc.get("format") or {}
    s = fmt.get("duration")
    if s not in (None, "", "N/A"):
        try:
            d = float(s)
            if d > 0:
                return d
        except ValueError:
            pass
    for st in doc.get("streams") or []:
        if st.get("codec_type") != "video":
            continue
        ds = st.get("duration")
        if ds in (None, "", "N/A"):
            continue
        try:
            d2 = float(ds)
            if d2 > 0:
                return d2
        except ValueError:
            continue
    return None


def opencv_duration_sec(path: Path) -> float | None:
    """Fallback duration probe for containers ffprobe cannot read."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps > 0.1 and frames > 0:
            return frames / fps
    except Exception as exc:  # noqa: BLE001
        logger.debug("opencv duration probe failed for %s: %s", path, exc)
    return None


def video_duration_sec(path: Path) -> float | None:
    d = ffprobe_duration_sec(path)
    if d is not None and d > 0:
        return d
    return opencv_duration_sec(path)


def trim_head(src: Path, dest: Path, seconds: float, *, timeout: int = 3600) -> None:
    """Write the first `seconds` of src to dest (stream copy, encode fallback)."""
    _trim_segment(src, dest, start_sec=0.0, seconds=seconds, timeout=timeout)


def trim_segment(src: Path, dest: Path, *, start_sec: float, seconds: float, timeout: int = 3600) -> None:
    """Write segment [start_sec, start_sec+seconds) of src to dest."""
    _trim_segment(src, dest, start_sec=start_sec, seconds=seconds, timeout=timeout)


def _trim_segment(src: Path, dest: Path, *, start_sec: float, seconds: float, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    start = str(max(0.0, float(start_sec)))
    dur = str(max(0.5, float(seconds)))

    copy_cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        start,
        "-i",
        str(src),
        "-t",
        dur,
        "-c",
        "copy",
        str(dest),
    ]
    r = _run(copy_cmd, timeout=min(timeout, 1200))
    if r.returncode == 0 and dest.is_file() and dest.stat().st_size > 256:
        return
    err_copy = (r.stderr or "") + (r.stdout or "")
    if dest.is_file():
        dest.unlink()

    enc_cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        start,
        "-i",
        str(src),
        "-t",
        dur,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    r2 = _run(enc_cmd, timeout=timeout)
    if r2.returncode != 0 or not dest.is_file() or dest.stat().st_size < 256:
        err2 = (r2.stderr or "") + (r2.stdout or "")
        raise FFmpegError(
            "ffmpeg trim failed (copy and re-encode). "
            f"copy_err={err_copy[-1200:]!r} encode_err={err2[-1200:]!r}"
        )


def normalize_container(src: Path, out_dir: Path, *, job_id: str = "") -> Path:
    """
    Normalize source into a duration-friendly container (remux → transcode fallback).

    Used before SyncNet which requires reliable duration metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    remux_out = out_dir / f"syncnet_source{('_' + job_id) if job_id else ''}.mkv"
    if remux_out.is_file():
        remux_out.unlink()
    remux_cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0",
        "-c",
        "copy",
        str(remux_out),
    ]
    r1 = _run(remux_cmd, timeout=1800)
    dur = video_duration_sec(remux_out) if r1.returncode == 0 and remux_out.is_file() else None
    if r1.returncode == 0 and remux_out.is_file() and dur and dur > 0:
        return remux_out

    trans_out = out_dir / f"syncnet_source{('_' + job_id) if job_id else ''}.mp4"
    if trans_out.is_file():
        trans_out.unlink()
    trans_cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-movflags",
        "+faststart",
        str(trans_out),
    ]
    r2 = _run(trans_cmd, timeout=3600)
    dur2 = video_duration_sec(trans_out) if r2.returncode == 0 and trans_out.is_file() else None
    if r2.returncode == 0 and trans_out.is_file() and dur2 and dur2 > 0:
        return trans_out

    err = (r2.stderr or "") + (r2.stdout or "") + "\n" + (r1.stderr or "") + (r1.stdout or "")
    raise FFmpegError(
        f"SyncNet normalization failed; no valid duration after remux/transcode. {err[-2000:]}"
    )


def extract_wav(src: Path, dest: Path, *, sample_rate: int = 16000, channels: int = 1, timeout: int = 300) -> None:
    """Extract mono WAV from any audio/video source."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(int(sample_rate)),
        "-ac",
        str(int(channels)),
        str(dest),
    ]
    r = _run(cmd, timeout=timeout)
    if r.returncode != 0 or not dest.is_file() or dest.stat().st_size < 64:
        raise FFmpegError(f"ffmpeg wav extraction failed: {(r.stderr or '')[-1000:]}")


def resize_video(src: Path, dest: Path, *, height: int = 720, timeout: int = 1800) -> None:
    """Re-encode source so that the shortest side <= `height` (keeps aspect)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    filt = f"scale=-2:'min({int(height)},ih)'"
    cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        filt,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    r = _run(cmd, timeout=timeout)
    if r.returncode != 0 or not dest.is_file() or dest.stat().st_size < 256:
        raise FFmpegError(f"ffmpeg resize failed: {(r.stderr or '')[-1000:]}")


def reencode_fps(src: Path, dest: Path, *, fps: float, timeout: int = 1200) -> None:
    """Re-encode video with a target output FPS, preserving audio."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    target_fps = max(1.0, float(fps))
    cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={target_fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    r = _run(cmd, timeout=timeout)
    if r.returncode != 0 or not dest.is_file() or dest.stat().st_size < 256:
        raise FFmpegError(f"ffmpeg fps re-encode failed: {(r.stderr or '')[-1000:]}")


def mux_video_audio(video: Path, audio: Path, dest: Path, *, timeout: int = 600) -> None:
    """Replace the audio track of `video` with `audio`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    cmd = [
        settings.paths.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    r = _run(cmd, timeout=timeout)
    if r.returncode != 0 or not dest.is_file() or dest.stat().st_size < 256:
        raise FFmpegError(f"ffmpeg mux failed: {(r.stderr or '')[-1000:]}")
