"""
Lip-sync authenticity API: SyncNet (subprocess) + MediaPipe/librosa correlation.

Setup: README.md and requirements.txt. SyncNet is not in git — run scripts/setup_syncnet.sh (or .ps1) after clone.

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000

Env:
  LIPSYNC_WINDOW_POSITION=start|middle|end
  MIN_DIST_PASS, CONFIDENCE_PASS — SyncNet thresholds
  LIPSYNC_VIDEO_TRIM=true|false — if true, SyncNet analyzes only first LIPSYNC_TRIM_MAX_SECONDS (default 15); MediaPipe/proctor still use original video.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import os
import re
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.proctor_signals import ProctorThresholds, analyze_eye_head_pose

# Default: sibling folder "syncnet_python" under project root (parent of app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SYNCNET = _PROJECT_ROOT / "syncnet_python"

SYNCNET_DIR = Path(os.environ.get("SYNCNET_DIR", str(_DEFAULT_SYNCNET))).resolve()
TEMP_DIR = Path(os.environ.get("LIPSYNC_TEMP_DIR", str(_PROJECT_ROOT / "temp_videos"))).resolve()
MIN_DIST_PASS = float(os.environ.get("MIN_DIST_PASS", "6.0"))
CONFIDENCE_PASS = float(os.environ.get("CONFIDENCE_PASS", "3.0"))
LIPSYNC_FFMPEG = (os.environ.get("LIPSYNC_FFMPEG") or "ffmpeg").strip() or "ffmpeg"
# SyncNet is the only lip-sync source in this service.
PROCTOR_SAMPLE_FPS = float(os.environ.get("PROCTOR_SAMPLE_FPS", "1.0"))
PROCTOR_OFFSCREEN_RATIO_THRESHOLD = float(
    os.environ.get("PROCTOR_OFFSCREEN_RATIO_THRESHOLD", "0.35")
)
PROCTOR_IMPROPER_HEAD_RATIO_THRESHOLD = float(
    os.environ.get("PROCTOR_IMPROPER_HEAD_RATIO_THRESHOLD", "0.35")
)
PROCTOR_REPETITIVE_PATTERN_THRESHOLD = int(
    os.environ.get("PROCTOR_REPETITIVE_PATTERN_THRESHOLD", "6")
)
PROCTOR_LIPSYNC_FLAG_SOURCE = (
    os.environ.get("PROCTOR_LIPSYNC_FLAG_SOURCE", "syncnet_only").strip().lower()
)
# Skip SyncNet on /analyze/proctor-signals (much faster; lip-sync = MediaPipe only via fusion mediapipe_only)
PROCTOR_SKIP_SYNCNET = os.environ.get("PROCTOR_SKIP_SYNCNET", "").lower() in (
    "1",
    "true",
    "yes",
)
# Trim long videos to the first N seconds (single clip) before SyncNet / MediaPipe / proctor — faster, less RAM.
LIPSYNC_VIDEO_TRIM = os.environ.get("LIPSYNC_VIDEO_TRIM", "").lower() in ("1", "true", "yes")
LIPSYNC_TRIM_MAX_SECONDS = float(os.environ.get("LIPSYNC_TRIM_MAX_SECONDS", "15"))
LIPSYNC_FFPROBE = (os.environ.get("LIPSYNC_FFPROBE") or "ffprobe").strip() or "ffprobe"
LIPSYNC_WINDOW_POSITION = os.environ.get("LIPSYNC_WINDOW_POSITION", "start").strip().lower()
if LIPSYNC_WINDOW_POSITION not in {"start", "middle", "end"}:
    LIPSYNC_WINDOW_POSITION = "start"


def _syncnet_pipeline_cli_extras() -> list[str]:
    """Optional run_pipeline.py flags from env (unset = upstream defaults)."""
    out: list[str] = []
    if v := os.environ.get("SYNCNET_FRAME_RATE", "").strip():
        out.extend(["--frame_rate", v])
    if v := os.environ.get("SYNCNET_MIN_TRACK", "").strip():
        out.extend(["--min_track", v])
    if v := os.environ.get("SYNCNET_FACEDET_SCALE", "").strip():
        out.extend(["--facedet_scale", v])
    if v := os.environ.get("SYNCNET_CROP_SCALE", "").strip():
        out.extend(["--crop_scale", v])
    if v := os.environ.get("SYNCNET_NUM_FAILED_DET", "").strip():
        out.extend(["--num_failed_det", v])
    if v := os.environ.get("SYNCNET_MIN_FACE_SIZE", "").strip():
        out.extend(["--min_face_size", v])
    return out


def _syncnet_run_syncnet_cli_extras() -> list[str]:
    """Optional run_syncnet.py flags from env."""
    out: list[str] = []
    if v := os.environ.get("SYNCNET_BATCH_SIZE", "").strip():
        out.extend(["--batch_size", v])
    if v := os.environ.get("SYNCNET_VSHIFT", "").strip():
        out.extend(["--vshift", v])
    return out

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_TRIM_OUT_NAME = "trimmed_for_analysis.mp4"


def _ffprobe_duration_sec(path: Path) -> float | None:
    try:
        r = subprocess.run(
            [
                LIPSYNC_FFPROBE,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        if r.returncode != 0:
            return None
        raw = (r.stdout or "").strip()
        if not raw:
            return None
        doc = json.loads(raw)
        fmt = doc.get("format") or {}
        s = fmt.get("duration")
        if s not in (None, "", "N/A"):
            d = float(s)
            if d > 0:
                return d
        for st in doc.get("streams") or []:
            if st.get("codec_type") != "video":
                continue
            ds = st.get("duration")
            if ds in (None, "", "N/A"):
                continue
            d2 = float(ds)
            if d2 > 0:
                return d2
        return None
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return None


def _opencv_duration_sec(path: Path) -> float | None:
    """Fallback duration probe when ffprobe cannot read container metadata."""
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
    except Exception:
        return None
    return None


def _video_duration_sec(path: Path) -> float | None:
    """Primary ffprobe duration with OpenCV fallback for problematic WebM metadata."""
    d = _ffprobe_duration_sec(path)
    if d is not None and d > 0:
        return d
    d2 = _opencv_duration_sec(path)
    return d2


def _guess_video_extension_from_url(video_url: str) -> str:
    """Infer extension from URL path or query string hints."""
    parsed = urlparse(video_url)
    ext = Path(parsed.path).suffix.lower()
    if ext in ALLOWED_EXT:
        return ext
    qs = parse_qs(parsed.query or "")
    for key in ("filename", "file", "name"):
        vals = qs.get(key) or []
        for v in vals:
            qext = Path(v).suffix.lower()
            if qext in ALLOWED_EXT:
                return qext
    return ".mp4"


def _ffmpeg_trim_video_head(src: Path, dest: Path, seconds: float) -> None:
    """Write the first `seconds` of `src` to `dest`. Prefer stream copy; fall back to H.264/AAC."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    t = str(max(0.5, seconds))
    copy_cmd = [
        LIPSYNC_FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-t",
        t,
        "-c",
        "copy",
        str(dest),
    ]
    r = subprocess.run(
        copy_cmd,
        capture_output=True,
        text=True,
        timeout=600,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode == 0 and dest.is_file() and dest.stat().st_size > 256:
        return
    err_copy = (r.stderr or "") + (r.stdout or "")
    if dest.is_file():
        dest.unlink()
    enc_cmd = [
        LIPSYNC_FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-t",
        t,
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
    r2 = subprocess.run(
        enc_cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        encoding="utf-8",
        errors="replace",
    )
    if r2.returncode != 0 or not dest.is_file() or dest.stat().st_size < 256:
        err2 = (r2.stderr or "") + (r2.stdout or "")
        raise RuntimeError(
            "ffmpeg trim failed (copy and re-encode). "
            f"copy_err={err_copy[-1500:]!r} encode_err={err2[-1500:]!r}"
        )


def _ffmpeg_trim_video_segment(src: Path, dest: Path, start_sec: float, seconds: float) -> None:
    """Write segment [start_sec, start_sec + seconds) from src to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        dest.unlink()
    start = str(max(0.0, start_sec))
    dur = str(max(0.5, seconds))
    copy_cmd = [
        LIPSYNC_FFMPEG,
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
    r = subprocess.run(
        copy_cmd,
        capture_output=True,
        text=True,
        timeout=1200,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode == 0 and dest.is_file() and dest.stat().st_size > 256:
        return
    err_copy = (r.stderr or "") + (r.stdout or "")
    if dest.is_file():
        dest.unlink()
    enc_cmd = [
        LIPSYNC_FFMPEG,
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
    r2 = subprocess.run(
        enc_cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        encoding="utf-8",
        errors="replace",
    )
    if r2.returncode != 0 or not dest.is_file() or dest.stat().st_size < 256:
        err2 = (r2.stderr or "") + (r2.stdout or "")
        raise RuntimeError(
            "ffmpeg segment trim failed (copy and re-encode). "
            f"copy_err={err_copy[-1500:]!r} encode_err={err2[-1500:]!r}"
        )


def _normalize_for_syncnet(src: Path, job_dir: Path, job_id: str) -> Path:
    """
    Normalize source into a duration-friendly container for probing/windowing.
    Try fast remux first, then transcode fallback.
    """
    remux_out = job_dir / "syncnet_source.mkv"
    if remux_out.is_file():
        remux_out.unlink()
    remux_cmd = [
        LIPSYNC_FFMPEG,
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
    r1 = subprocess.run(
        remux_cmd,
        capture_output=True,
        text=True,
        timeout=1800,
        encoding="utf-8",
        errors="replace",
    )
    remux_dur = _video_duration_sec(remux_out) if r1.returncode == 0 and remux_out.is_file() else None
    if r1.returncode == 0 and remux_out.is_file() and remux_dur and remux_dur > 0:
        return remux_out

    trans_out = job_dir / "syncnet_source.mp4"
    if trans_out.is_file():
        trans_out.unlink()
    trans_cmd = [
        LIPSYNC_FFMPEG,
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
    r2 = subprocess.run(
        trans_cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        encoding="utf-8",
        errors="replace",
    )
    trans_dur = _video_duration_sec(trans_out) if r2.returncode == 0 and trans_out.is_file() else None
    if r2.returncode == 0 and trans_out.is_file() and trans_dur and trans_dur > 0:
        return trans_out

    err = (r2.stderr or "") + (r2.stdout or "") + "\n" + (r1.stderr or "") + (r1.stdout or "")
    raise RuntimeError(f"SyncNet normalization failed; no valid duration after remux/transcode. {err[-2000:]}")


def _build_syncnet_window(duration_sec: float) -> dict | None:
    """
    Build exactly one SyncNet window based on:
      - LIPSYNC_TRIM_MAX_SECONDS
      - LIPSYNC_WINDOW_POSITION (start|middle|end)
    """
    duration = max(0.0, duration_sec)
    if duration <= 0.0:
        return None

    only_dur = min(max(1.0, min(LIPSYNC_TRIM_MAX_SECONDS, 600.0)), duration)
    if LIPSYNC_WINDOW_POSITION == "middle":
        start = max(0.0, (duration - only_dur) / 2.0)
    elif LIPSYNC_WINDOW_POSITION == "end":
        start = max(0.0, duration - only_dur)
    else:
        start = 0.0
    return {"startSec": round(start, 3), "durationSec": round(only_dur, 3)}


def _prepare_analysis_video(video_path: Path, job_dir: Path) -> tuple[Path, dict]:
    """
    Optionally replace `video_path` with a shorter MP4 (first N seconds only).
    Returns (path_to_use, metadata dict for API responses).
    """
    max_sec = max(1.0, min(LIPSYNC_TRIM_MAX_SECONDS, 600.0))
    meta: dict = {
        "trimEnabled": bool(LIPSYNC_VIDEO_TRIM),
        "trimMaxSeconds": max_sec,
        "trimApplied": False,
        "sourceDurationSec": None,
        "analyzedDurationSec": None,
    }
    src_dur = _ffprobe_duration_sec(video_path)
    meta["sourceDurationSec"] = src_dur

    if not LIPSYNC_VIDEO_TRIM:
        meta["analyzedDurationSec"] = src_dur
        return video_path, meta

    need_trim = src_dur is None or src_dur > max_sec + 0.05
    if not need_trim:
        meta["analyzedDurationSec"] = src_dur
        return video_path, meta

    out = job_dir / _TRIM_OUT_NAME
    _ffmpeg_trim_video_head(video_path, out, max_sec)
    meta["trimApplied"] = True
    meta["analyzedDurationSec"] = _ffprobe_duration_sec(out) or max_sec
    return out, meta


def _prepare_syncnet_window_paths(video_path: Path, job_dir: Path) -> tuple[list[dict], dict]:
    """
    Prepare one or more inputs for SyncNet.
    Returns:
      - list of {path, startSec, durationSec}
      - metadata about windowing behavior for API response
    """
    src_dur = _video_duration_sec(video_path)
    max_sec = max(1.0, min(LIPSYNC_TRIM_MAX_SECONDS, 600.0))
    meta: dict = {
        "singleWindowMode": True,
        "position": LIPSYNC_WINDOW_POSITION,
        "trimEnabled": bool(LIPSYNC_VIDEO_TRIM),
        "trimMaxSeconds": max_sec,
        "sourceDurationSec": src_dur,
        "applied": False,
        "count": 1,
    }
    if not LIPSYNC_VIDEO_TRIM or src_dur is None:
        return [{"path": video_path, "startSec": 0.0, "durationSec": src_dur}], meta

    win = _build_syncnet_window(src_dur)
    if not win:
        return [{"path": video_path, "startSec": 0.0, "durationSec": src_dur}], meta

    # If selected window is effectively full source, skip segment extraction.
    if float(win["startSec"]) <= 0.001 and src_dur <= max_sec + 0.05:
        return [{"path": video_path, "startSec": 0.0, "durationSec": src_dur}], meta

    seg = job_dir / "syncnet_window_1.mp4"
    _ffmpeg_trim_video_segment(
        video_path,
        seg,
        start_sec=float(win["startSec"]),
        seconds=float(win["durationSec"]),
    )
    meta["applied"] = True
    return [
        {
            "path": seg,
            "startSec": float(win["startSec"]),
            "durationSec": float(win["durationSec"]),
        }
    ], meta


_SYNCNET_BLOCK = re.compile(
    r"AV offset:\s*([-\d]+)\s*\n\s*Min dist:\s*([\d.]+)\s*\n\s*Confidence:\s*([\d.]+)",
    re.MULTILINE,
)

app = FastAPI(
    title="Lip-sync fraud detection (SyncNet + MediaPipe)",
    description="Upload interview video; SyncNet + lip–audio correlation; fused PASS/FAIL.",
    version="1.1.0",
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "usage": "POST /analyze with multipart field 'file' (video)",
        "fusion": "syncnet_only",
    }


@app.get("/health")
def health():
    ok = SYNCNET_DIR.is_dir() and (SYNCNET_DIR / "run_pipeline.py").is_file()
    model = SYNCNET_DIR / "data" / "syncnet_v2.model"
    return {
        "syncnet_dir": str(SYNCNET_DIR),
        "syncnet_present": ok,
        "model_present": model.is_file(),
        "lipsync_engine": "syncnet_only",
    }


def _model_path() -> Path:
    p = SYNCNET_DIR / "data" / "syncnet_v2.model"
    if not p.is_file():
        raise FileNotFoundError(
            f"SyncNet weights not found at {p}. Run download_model.sh inside syncnet_python."
        )
    return p


def parse_syncnet_stdout(stdout: str) -> list[dict]:
    tracks = []
    for m in _SYNCNET_BLOCK.finditer(stdout):
        tracks.append(
            {
                "av_offset_frames": int(m.group(1)),
                "min_dist": float(m.group(2)),
                "confidence": float(m.group(3)),
            }
        )
    return tracks


def aggregate_scores(tracks: list[dict]) -> dict:
    if not tracks:
        raise ValueError(
            "No SyncNet scores in subprocess output. "
            "Face may be missing, video too short, or pipeline failed silently."
        )
    worst = max(tracks, key=lambda t: t["min_dist"])
    min_conf = min(t["confidence"] for t in tracks)
    mean_offset = sum(abs(t["av_offset_frames"]) for t in tracks) / len(tracks)
    return {
        "av_offset_frames": worst["av_offset_frames"],
        "min_dist": worst["min_dist"],
        "confidence": min_conf,
        "tracks_evaluated": len(tracks),
        "all_tracks": tracks,
        "mean_abs_offset_frames": round(mean_offset, 3),
    }


def build_verdict(scores: dict) -> dict:
    min_dist = scores["min_dist"]
    confidence = scores["confidence"]
    sync_ok = min_dist <= MIN_DIST_PASS
    confident_ok = confidence >= CONFIDENCE_PASS
    passed = sync_ok and confident_ok

    if passed:
        reason = (
            "Lip motion and audio appear in sync; candidate likely speaking in the recording."
        )
    elif not sync_ok and not confident_ok:
        reason = (
            "Poor audio-visual sync and low model confidence; possible dubbed or mismatched audio."
        )
    elif not sync_ok:
        reason = "Audio-visual sync distance is high; possible dubbed or pre-recorded audio."
    else:
        reason = "Sync distance is acceptable but confidence is low; recommend manual review."

    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "reason": reason,
        "scores": {
            "av_offset_frames": scores["av_offset_frames"],
            "min_dist": round(min_dist, 4),
            "confidence": round(confidence, 4),
            "tracks_evaluated": scores["tracks_evaluated"],
            "mean_abs_offset_frames": scores["mean_abs_offset_frames"],
        },
        "thresholds": {
            "min_dist_max_pass": MIN_DIST_PASS,
            "confidence_min_pass": CONFIDENCE_PASS,
        },
    }


def run_syncnet(video_path: Path, reference: str, data_dir: Path) -> dict:
    video_path = video_path.resolve()
    data_dir = data_dir.resolve()
    model = _model_path()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    def run_script(name: str) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            str(SYNCNET_DIR / name),
            "--videofile",
            str(video_path),
            "--reference",
            reference,
            "--data_dir",
            str(data_dir),
        ]
        if name == "run_pipeline.py":
            cmd.extend(_syncnet_pipeline_cli_extras())
        if name == "run_syncnet.py":
            cmd.extend(["--initial_model", str(model)])
            cmd.extend(_syncnet_run_syncnet_cli_extras())
        return subprocess.run(
            cmd,
            cwd=str(SYNCNET_DIR),
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
            errors="replace",
        )

    p1 = run_script("run_pipeline.py")
    if p1.returncode != 0:
        err = (p1.stderr or "") + (p1.stdout or "")
        raise RuntimeError(f"run_pipeline.py failed (exit {p1.returncode}):\n{err[-8000:]}")

    p2 = run_script("run_syncnet.py")
    out = (p2.stdout or "") + "\n" + (p2.stderr or "")
    if p2.returncode != 0:
        raise RuntimeError(f"run_syncnet.py failed (exit {p2.returncode}):\n{out[-8000:]}")

    tracks = parse_syncnet_stdout(out)
    if not tracks:
        raise ValueError(
            "Could not parse SyncNet scores. Raw tail:\n" + out[-4000:]
        )
    agg = aggregate_scores(tracks)
    agg["raw_log_tail"] = out[-6000:].strip()
    return agg


def _safe_syncnet(video_path: Path, reference: str, job_dir: Path) -> dict:
    try:
        scores = run_syncnet(video_path, reference, job_dir)
        raw_tail = scores.pop("raw_log_tail", None)
        verdict = build_verdict(scores)
        out = {**verdict, "all_tracks": scores.get("all_tracks")}
        if os.environ.get("LIPSYNC_DEBUG_LOG") == "1" and raw_tail:
            out["debug_log_tail"] = raw_tail
        return out
    except FileNotFoundError as e:
        return {"passed": False, "verdict": "ERROR", "error": str(e), "scores": {}}
    except (ValueError, RuntimeError) as e:
        return {"passed": False, "verdict": "ERROR", "error": str(e)[:8000], "scores": {}}


def _safe_syncnet_multi_window(video_path: Path, reference: str, job_dir: Path) -> dict:
    """Runs SyncNet with single-position window mode (or full source if trim off)."""
    try:
        windows, windowing = _prepare_syncnet_window_paths(video_path, job_dir)
    except Exception as e:
        return {"passed": False, "verdict": "ERROR", "error": str(e)[:8000], "scores": {}}

    if len(windows) == 1:
        single = _safe_syncnet(Path(windows[0]["path"]), reference, job_dir / "w0")
        single["windowing"] = {
            **windowing,
            "windows": [
                {
                    "index": 1,
                    "startSec": windows[0]["startSec"],
                    "durationSec": windows[0]["durationSec"],
                }
            ],
        }
        return single

    results: list[dict] = []
    for idx, w in enumerate(windows):
        res = _safe_syncnet(
            Path(w["path"]),
            f"{reference}_w{idx + 1}",
            job_dir / f"w{idx + 1}",
        )
        results.append(
            {
                "index": idx + 1,
                "startSec": w["startSec"],
                "durationSec": w["durationSec"],
                "result": res,
            }
        )

    ok_results = [r for r in results if "error" not in r["result"]]
    if not ok_results:
        first_err = results[0]["result"].get("error", "No SyncNet window produced a score")
        return {
            "passed": False,
            "verdict": "ERROR",
            "error": first_err,
            "scores": {},
            "windowing": {
                **windowing,
                "windows": [
                    {
                        "index": r["index"],
                        "startSec": r["startSec"],
                        "durationSec": r["durationSec"],
                        "verdict": r["result"].get("verdict"),
                        "passed": r["result"].get("passed"),
                        "error": r["result"].get("error"),
                    }
                    for r in results
                ],
            },
        }

    agg_track = {
        "av_offset_frames": 0,
        "min_dist": max(float(r["result"]["scores"].get("min_dist", 999.0)) for r in ok_results),
        "confidence": min(float(r["result"]["scores"].get("confidence", 0.0)) for r in ok_results),
        "tracks_evaluated": sum(int(r["result"]["scores"].get("tracks_evaluated", 0)) for r in ok_results),
        "mean_abs_offset_frames": round(
            sum(float(r["result"]["scores"].get("mean_abs_offset_frames", 0.0)) for r in ok_results)
            / max(len(ok_results), 1),
            3,
        ),
    }
    agg_verdict = build_verdict(agg_track)
    if any(r["result"].get("passed") is False for r in ok_results):
        agg_verdict["reason"] += " Dynamic windowing observed at least one failing window."

    agg_verdict["windowing"] = {
        **windowing,
        "syncnetTrimEnabled": bool(LIPSYNC_VIDEO_TRIM),
        "syncnetTrimMaxSeconds": max(1.0, min(LIPSYNC_TRIM_MAX_SECONDS, 600.0)),
        "windows": [
            {
                "index": r["index"],
                "startSec": r["startSec"],
                "durationSec": r["durationSec"],
                "verdict": r["result"].get("verdict"),
                "passed": r["result"].get("passed"),
                "error": r["result"].get("error"),
                "scores": r["result"].get("scores", {}),
            }
            for r in results
        ],
    }
    return agg_verdict


def _syncnet_only_fusion(syncnet_out: dict, *, skipped: bool) -> dict:
    if skipped:
        return {
            "verdict": "SKIPPED",
            "passed": True,
            "reason": "SyncNet was skipped by request/configuration.",
            "fusion_mode": "syncnet_only",
            "positive_methods": [],
        }
    passed = bool(syncnet_out.get("passed", False))
    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "reason": syncnet_out.get("reason", "SyncNet-only lip-sync result"),
        "fusion_mode": "syncnet_only",
        "positive_methods": ["syncnet"] if passed else [],
    }


def _download_video_from_url(video_url: str, destination: Path) -> None:
    parsed = urlparse(video_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("videoUrl must be http/https")

    req = Request(video_url, headers={"User-Agent": "lipsync-fraud-api/1.2"})
    with urlopen(req, timeout=180) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)


class ProctorSignalsRequest(BaseModel):
    videoUrl: str = Field(..., min_length=5)
    questionId: str | None = None
    candidateId: str | None = None
    sampleFps: float | None = None
    offscreenRatioThreshold: float | None = None
    improperHeadRatioThreshold: float | None = None
    repetitivePatternThreshold: int | None = None
    skipSyncNet: bool | None = None


def _build_proctor_thresholds(payload: ProctorSignalsRequest) -> ProctorThresholds:
    return ProctorThresholds(
        sample_fps=payload.sampleFps or PROCTOR_SAMPLE_FPS,
        offscreen_ratio_threshold=(
            payload.offscreenRatioThreshold or PROCTOR_OFFSCREEN_RATIO_THRESHOLD
        ),
        improper_head_ratio_threshold=(
            payload.improperHeadRatioThreshold or PROCTOR_IMPROPER_HEAD_RATIO_THRESHOLD
        ),
        repetitive_pattern_threshold=(
            payload.repetitivePatternThreshold or PROCTOR_REPETITIVE_PATTERN_THRESHOLD
        ),
    )


@app.post("/analyze/proctor-signals")
async def analyze_proctor_signals(payload: ProctorSignalsRequest = Body(...)):
    job_id = uuid.uuid4().hex[:12]
    reference = f"cand_{job_id}"
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = _guess_video_extension_from_url(payload.videoUrl)
    video_path = job_dir / f"input{ext}"
    try:
        _download_video_from_url(payload.videoUrl, video_path)
        syncnet_path = _normalize_for_syncnet(video_path, job_dir, job_id)
        thresholds = _build_proctor_thresholds(payload)

        skip_sn = PROCTOR_SKIP_SYNCNET or (payload.skipSyncNet is True)

        if skip_sn:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut_pose = pool.submit(analyze_eye_head_pose, video_path, thresholds)
                pose_result = fut_pose.result()
            syncnet_out = {"skipped": True, "passed": False, "scores": {}}
            fused_lipsync = _syncnet_only_fusion(syncnet_out, skipped=True)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_sn = pool.submit(_safe_syncnet_multi_window, syncnet_path, reference, job_dir)
                fut_pose = pool.submit(analyze_eye_head_pose, video_path, thresholds)
                syncnet_result = fut_sn.result()
                pose_result = fut_pose.result()

            syncnet_out = {k: v for k, v in syncnet_result.items() if k != "all_tracks"}
            fused_lipsync = _syncnet_only_fusion(syncnet_out, skipped=False)

        # IMPORTANT: by default, flag lip-sync mismatch only from SyncNet result.
        # This avoids false flags when MediaPipe correlation fails but SyncNet passes.
        if PROCTOR_LIPSYNC_FLAG_SOURCE == "none":
            lip_sync_mismatch = False
        else:
            if skip_sn:
                lip_sync_mismatch = False
            else:
                lip_sync_mismatch = not bool(syncnet_out.get("passed", False))
        eye_tracking = pose_result["eyeMovement"].get("eyeTracking", {})
        eye_reliable = bool(eye_tracking.get("reliable", True))
        offscreen_ratio = pose_result["eyeMovement"]["offScreenRatio"]
        improper_head_ratio = pose_result["headPose"]["improperHeadRatio"]
        repetitive_pattern = pose_result["eyeMovement"]["repetitivePatternDetected"]
        pattern_verdict = pose_result["eyeMovement"].get("patternVerdict")
        reading_pattern_flag = eye_reliable and pattern_verdict == "READING_LIKE"
        eye_flag = eye_reliable and (
            offscreen_ratio >= thresholds.offscreen_ratio_threshold or repetitive_pattern
        )
        head_flag = improper_head_ratio >= thresholds.improper_head_ratio_threshold

        flags = []
        if lip_sync_mismatch:
            flags.append(
                {
                    "type": "LIP_SYNC_MISMATCH",
                    "severity": "HIGH",
                    "evidence": "Audio-video synchronization mismatch detected by fused model pipeline",
                    "keyTimestamps": [],
                }
            )
        if eye_flag:
            flags.append(
                {
                    "type": "OFF_SCREEN_GAZE",
                    "severity": "MEDIUM",
                    "evidence": (
                        f"Off-screen eye ratio {offscreen_ratio:.2f}"
                        + (" with repetitive glance pattern" if repetitive_pattern else "")
                    ),
                    "keyTimestamps": [s["range"] for s in pose_result["eyeMovement"]["segments"][:5]],
                }
            )
        if reading_pattern_flag:
            flags.append(
                {
                    "type": "SUBTLE_READING",
                    "severity": "MEDIUM",
                    "evidence": f"Eye reading pattern detected (score: {pose_result['eyeMovement'].get('readingPatternScore', 0)})",
                    "keyTimestamps": [s["range"] for s in pose_result["eyeMovement"]["segments"][:5]],
                }
            )
        if head_flag:
            flags.append(
                {
                    "type": "READING_FROM_EXTERNAL",
                    "severity": "MEDIUM",
                    "evidence": f"Improper head pose ratio {improper_head_ratio:.2f}",
                    "keyTimestamps": [s["range"] for s in pose_result["headPose"]["segments"][:5]],
                }
            )

        suspicious = len(flags) > 0
        verdict = "SUSPECT" if suspicious else "CLEAR"
        confidence = 0.88 if suspicious else 0.2
        return JSONResponse(
            content={
                "jobId": job_id,
                "questionId": payload.questionId,
                "candidateId": payload.candidateId,
                "videoUrl": payload.videoUrl,
                "lipSync": {
                    "passed": fused_lipsync.get("passed", False),
                    "verdict": fused_lipsync.get("verdict"),
                    "reason": fused_lipsync.get("reason"),
                    "fusion": {
                        "mode": "syncnet_only",
                        "positiveMethods": fused_lipsync.get("positive_methods", []),
                        "syncNetSkipped": skip_sn,
                        "flagSource": PROCTOR_LIPSYNC_FLAG_SOURCE,
                        "mediapipeLipSyncSkipped": True,
                    },
                    "syncnet": syncnet_out,
                },
                "eyeMovement": pose_result["eyeMovement"],
                "headPose": pose_result["headPose"],
                "videoMeta": pose_result["videoMeta"],
                "integrityAnalysis": {
                    "verdict": verdict,
                    "confidenceScore": confidence,
                    "flags": flags,
                },
                "summary": {
                    "suspicious": suspicious,
                    "signalCount": len(flags),
                    "rules": pose_result["summary"]["rules"],
                },
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)[:4000]) from exc
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {ext!r}. Allowed: {sorted(ALLOWED_EXT)}",
        )

    if not SYNCNET_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"SYNCNET_DIR not found: {SYNCNET_DIR}. Clone joonson/syncnet_python.",
        )

    job_id = uuid.uuid4().hex[:12]
    reference = f"cand_{job_id}"
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"input{ext}"

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        syncnet_path = _normalize_for_syncnet(video_path, job_dir, job_id)
        src_dur = _video_duration_sec(syncnet_path)
        win = _build_syncnet_window(src_dur or 0.0) if LIPSYNC_VIDEO_TRIM else None
        syncnet_trim_meta = {
            "trimEnabled": bool(LIPSYNC_VIDEO_TRIM),
            "trimMaxSeconds": max(1.0, min(LIPSYNC_TRIM_MAX_SECONDS, 600.0)),
            "trimApplied": bool(LIPSYNC_VIDEO_TRIM and win and (float(win["startSec"]) > 0.001 or (src_dur or 0.0) > float(win["durationSec"]) + 0.05)),
            "sourceDurationSec": src_dur,
            "analyzedDurationSec": (float(win["durationSec"]) if win else src_dur),
            "position": LIPSYNC_WINDOW_POSITION,
            "singleWindowMode": True,
        }

        syncnet_result = _safe_syncnet_multi_window(syncnet_path, reference, job_dir)

        # Drop bulky SyncNet track list from default payload
        syncnet_out = {k: v for k, v in syncnet_result.items() if k != "all_tracks"}
        if "all_tracks" in syncnet_result and os.environ.get("LIPSYNC_DEBUG_LOG") == "1":
            syncnet_out["all_tracks"] = syncnet_result["all_tracks"]

        final = _syncnet_only_fusion(syncnet_out, skipped=False)

        body = {
            "job_id": job_id,
            "file": file.filename,
            "video_trim": syncnet_trim_meta,
            "verdict": final["verdict"],
            "passed": final["passed"],
            "reason": final["reason"],
            "fusion": {
                "mode": "syncnet_only",
                "positive_methods": final.get("positive_methods", []),
                "mediapipe_lipsync_skipped": True,
            },
            "syncnet": syncnet_out,
        }
        body["scores"] = syncnet_out.get("scores")
        body["method_used"] = "syncnet" if final["passed"] else None

        if syncnet_out.get("error"):
            raise HTTPException(status_code=422, detail={"syncnet": syncnet_out.get("error")})

        return JSONResponse(content=body)

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

