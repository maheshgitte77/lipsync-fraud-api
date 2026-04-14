"""
Lip-sync authenticity API: SyncNet (subprocess) + MediaPipe/librosa correlation.

Setup: README.md and requirements.txt. SyncNet is not in git — run scripts/setup_syncnet.sh (or .ps1) after clone.

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000

Env:
  LIPSYNC_FUSION=any|all|syncnet_only|mediapipe_only|best  (default: any)
  MEDIAPIPE_CORR_THRESHOLD=0.4
  MIN_DIST_PASS, CONFIDENCE_PASS — SyncNet thresholds
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
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.mediapipe_lipsync import analyze_mediapipe_correlation
from app.proctor_signals import ProctorThresholds, analyze_eye_head_pose

# Default: sibling folder "syncnet_python" under project root (parent of app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SYNCNET = _PROJECT_ROOT / "syncnet_python"

SYNCNET_DIR = Path(os.environ.get("SYNCNET_DIR", str(_DEFAULT_SYNCNET))).resolve()
TEMP_DIR = Path(os.environ.get("LIPSYNC_TEMP_DIR", str(_PROJECT_ROOT / "temp_videos"))).resolve()
MIN_DIST_PASS = float(os.environ.get("MIN_DIST_PASS", "6.0"))
CONFIDENCE_PASS = float(os.environ.get("CONFIDENCE_PASS", "3.0"))
MEDIAPIPE_CORR_THRESHOLD = float(os.environ.get("MEDIAPIPE_CORR_THRESHOLD", "0.4"))
MEDIAPIPE_CORR_BORDERLINE = float(os.environ.get("MEDIAPIPE_CORR_BORDERLINE", "0.25"))
LIPSYNC_FFMPEG = (os.environ.get("LIPSYNC_FFMPEG") or "ffmpeg").strip() or "ffmpeg"
MEDIAPIPE_MAX_NUM_FACES = int(os.environ.get("MEDIAPIPE_MAX_NUM_FACES", "1"))
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = float(
    os.environ.get("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.5")
)
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = float(
    os.environ.get("MEDIAPIPE_MIN_TRACKING_CONFIDENCE", "0.5")
)
MEDIAPIPE_AUDIO_SAMPLE_RATE = int(os.environ.get("MEDIAPIPE_AUDIO_SAMPLE_RATE", "16000"))
MEDIAPIPE_FPS_FALLBACK = float(os.environ.get("MEDIAPIPE_FPS_FALLBACK", "25"))
MEDIAPIPE_MIN_FRAMES = int(os.environ.get("MEDIAPIPE_MIN_FRAMES", "5"))
# any: PASS if either method passes | all: both must pass | best: same as any + closer_to_pass hint
LIPSYNC_FUSION = os.environ.get("LIPSYNC_FUSION", "any").strip().lower()
LIPSYNC_DISABLE_MEDIAPIPE_LIPSYNC = os.environ.get(
    "LIPSYNC_DISABLE_MEDIAPIPE_LIPSYNC", ""
).lower() in ("1", "true", "yes")
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
        "fusion": LIPSYNC_FUSION,
    }


@app.get("/health")
def health():
    ok = SYNCNET_DIR.is_dir() and (SYNCNET_DIR / "run_pipeline.py").is_file()
    model = SYNCNET_DIR / "data" / "syncnet_v2.model"
    mp_ok = False
    try:
        import mediapipe  # noqa: F401

        mp_ok = True
    except ImportError:
        pass
    lib_ok = False
    try:
        import librosa  # noqa: F401

        lib_ok = True
    except ImportError:
        pass
    return {
        "syncnet_dir": str(SYNCNET_DIR),
        "syncnet_present": ok,
        "model_present": model.is_file(),
        "mediapipe_import_ok": mp_ok,
        "librosa_import_ok": lib_ok,
        "fusion_default": LIPSYNC_FUSION,
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


def _safe_mediapipe(video_path: Path) -> dict:
    try:
        return analyze_mediapipe_correlation(
            video_path,
            corr_threshold=MEDIAPIPE_CORR_THRESHOLD,
            corr_borderline=MEDIAPIPE_CORR_BORDERLINE,
            ffmpeg_binary=LIPSYNC_FFMPEG,
            audio_sample_rate=MEDIAPIPE_AUDIO_SAMPLE_RATE,
            fps_fallback=MEDIAPIPE_FPS_FALLBACK,
            max_num_faces=MEDIAPIPE_MAX_NUM_FACES,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            min_frames=MEDIAPIPE_MIN_FRAMES,
        )
    except Exception as e:
        return {
            "passed": False,
            "verdict": "ERROR",
            "error": str(e)[:2500],
            "scores": {},
        }


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


def _mediapipe_skipped_result(reason: str = "MediaPipe lip-sync skipped by configuration") -> dict:
    return {
        "skipped": True,
        "passed": None,
        "verdict": "SKIPPED",
        "reason": reason,
        "scores": {},
    }


def _should_skip_mediapipe_lipsync(fusion_mode: str) -> bool:
    return LIPSYNC_DISABLE_MEDIAPIPE_LIPSYNC or fusion_mode == "syncnet_only"


def _fuse(
    syncnet: dict,
    mediapipe: dict,
    mode: str,
) -> dict:
    sn_ok = "error" not in syncnet
    mp_ok = "error" not in mediapipe
    sn_pass = sn_ok and syncnet.get("passed") is True
    mp_pass = mp_ok and mediapipe.get("passed") is True

    mode = mode if mode in ("any", "all", "syncnet_only", "mediapipe_only", "best") else "any"

    if mode == "all":
        passed = sn_ok and mp_ok and sn_pass and mp_pass
    elif mode == "syncnet_only":
        passed = sn_pass
    elif mode == "mediapipe_only":
        passed = mp_pass
    else:
        # any | best — same boolean; best adds hint below
        passed = sn_pass or mp_pass

    parts = []
    if sn_pass:
        parts.append("syncnet")
    if mp_pass:
        parts.append("mediapipe")

    if passed:
        reason = (
            f"Fused PASS ({mode}). Positive signal from: {', '.join(parts) if parts else 'criteria'}."
        )
    else:
        sn_lbl = "PASS" if sn_pass else ("FAIL" if sn_ok else "ERROR")
        mp_lbl = "PASS" if mp_pass else ("FAIL" if mp_ok else "ERROR")
        reason = f"Fused FAIL ({mode}). SyncNet: {sn_lbl}; MediaPipe: {mp_lbl}."

    detail: dict = {"fusion_mode": mode, "positive_methods": parts}

    if mode == "best" and not passed and sn_ok and mp_ok:
        sn_s = syncnet.get("scores") or {}
        mp_s = mediapipe.get("scores") or {}
        md = float(sn_s.get("min_dist") or 99.0)
        cf = float(sn_s.get("confidence") or 0.0)
        corr = float(mp_s.get("correlation") or 0.0)
        # Margin: positive = closer to passing (heuristic for review only)
        sn_margin = (MIN_DIST_PASS - md) / max(MIN_DIST_PASS, 0.01)
        sn_margin += (cf - CONFIDENCE_PASS) / max(CONFIDENCE_PASS, 0.01)
        mp_margin = (corr - MEDIAPIPE_CORR_THRESHOLD) / max(MEDIAPIPE_CORR_THRESHOLD, 0.01)
        if sn_margin >= mp_margin:
            detail["closer_to_pass"] = "syncnet"
            detail["margins"] = {"syncnet": round(sn_margin, 4), "mediapipe": round(mp_margin, 4)}
        else:
            detail["closer_to_pass"] = "mediapipe"
            detail["margins"] = {"syncnet": round(sn_margin, 4), "mediapipe": round(mp_margin, 4)}

    return {
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "reason": reason,
        **detail,
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
    video_path = job_dir / "input.mp4"

    try:
        _download_video_from_url(payload.videoUrl, video_path)
        thresholds = _build_proctor_thresholds(payload)

        skip_sn = PROCTOR_SKIP_SYNCNET or (payload.skipSyncNet is True)

        if skip_sn:
            skip_mp_lipsync = _should_skip_mediapipe_lipsync("mediapipe_only")
            with ThreadPoolExecutor(max_workers=1 if skip_mp_lipsync else 2) as pool:
                fut_pose = pool.submit(analyze_eye_head_pose, video_path, thresholds)
                fut_mp = (
                    None
                    if skip_mp_lipsync
                    else pool.submit(_safe_mediapipe, video_path)
                )
                pose_result = fut_pose.result()
                mediapipe_result = (
                    _mediapipe_skipped_result()
                    if skip_mp_lipsync
                    else fut_mp.result()
                )
            # No "error" key — keeps _fuse sn_ok True under mediapipe_only
            syncnet_out = {"skipped": True, "passed": False, "scores": {}}
            fused_lipsync = _fuse(syncnet_out, mediapipe_result, "mediapipe_only")
        else:
            requested_fusion_mode = LIPSYNC_FUSION
            skip_mp_lipsync = _should_skip_mediapipe_lipsync(requested_fusion_mode)
            effective_fusion_mode = (
                "syncnet_only" if skip_mp_lipsync else requested_fusion_mode
            )

            with ThreadPoolExecutor(max_workers=2 if skip_mp_lipsync else 3) as pool:
                fut_mp = (
                    None
                    if skip_mp_lipsync
                    else pool.submit(_safe_mediapipe, video_path)
                )
                fut_sn = pool.submit(_safe_syncnet, video_path, reference, job_dir)
                fut_pose = pool.submit(analyze_eye_head_pose, video_path, thresholds)
                syncnet_result = fut_sn.result()
                pose_result = fut_pose.result()
                mediapipe_result = (
                    _mediapipe_skipped_result()
                    if skip_mp_lipsync
                    else fut_mp.result()
                )

            syncnet_out = {k: v for k, v in syncnet_result.items() if k != "all_tracks"}
            fused_lipsync = _fuse(syncnet_out, mediapipe_result, effective_fusion_mode)

        # IMPORTANT: by default, flag lip-sync mismatch only from SyncNet result.
        # This avoids false flags when MediaPipe correlation fails but SyncNet passes.
        if PROCTOR_LIPSYNC_FLAG_SOURCE == "none":
            lip_sync_mismatch = False
        elif PROCTOR_LIPSYNC_FLAG_SOURCE == "fused":
            lip_sync_mismatch = not fused_lipsync.get("passed", False)
        elif PROCTOR_LIPSYNC_FLAG_SOURCE == "mediapipe_only":
            lip_sync_mismatch = not bool(mediapipe_result.get("passed", False))
        else:
            # Default: syncnet_only
            if skip_sn:
                # SyncNet skipped -> don't auto-flag from MediaPipe alone
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
                        "mode": fused_lipsync.get("fusion_mode", LIPSYNC_FUSION),
                        "positiveMethods": fused_lipsync.get("positive_methods", []),
                        "syncNetSkipped": skip_sn,
                        "flagSource": PROCTOR_LIPSYNC_FLAG_SOURCE,
                        "mediapipeLipSyncSkipped": bool(mediapipe_result.get("skipped")),
                    },
                    "syncnet": syncnet_out,
                    "mediapipe": mediapipe_result,
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

        requested_fusion_mode = LIPSYNC_FUSION
        skip_mp_lipsync = _should_skip_mediapipe_lipsync(requested_fusion_mode)
        effective_fusion_mode = "syncnet_only" if skip_mp_lipsync else requested_fusion_mode

        with ThreadPoolExecutor(max_workers=1 if skip_mp_lipsync else 2) as pool:
            fut_mp = (
                None
                if skip_mp_lipsync
                else pool.submit(_safe_mediapipe, video_path)
            )
            fut_sn = pool.submit(_safe_syncnet, video_path, reference, job_dir)
            syncnet_result = fut_sn.result()
            mediapipe_result = (
                _mediapipe_skipped_result()
                if skip_mp_lipsync
                else fut_mp.result()
            )

        # Drop bulky SyncNet track list from default payload
        syncnet_out = {k: v for k, v in syncnet_result.items() if k != "all_tracks"}
        if "all_tracks" in syncnet_result and os.environ.get("LIPSYNC_DEBUG_LOG") == "1":
            syncnet_out["all_tracks"] = syncnet_result["all_tracks"]

        final = _fuse(syncnet_out, mediapipe_result, effective_fusion_mode)

        body = {
            "job_id": job_id,
            "file": file.filename,
            "verdict": final["verdict"],
            "passed": final["passed"],
            "reason": final["reason"],
            "fusion": {
                "mode": final["fusion_mode"],
                "positive_methods": final.get("positive_methods", []),
                "mediapipe_lipsync_skipped": bool(mediapipe_result.get("skipped")),
            },
            "syncnet": syncnet_out,
            "mediapipe": mediapipe_result,
        }
        if "closer_to_pass" in final:
            body["fusion"]["closer_to_pass"] = final["closer_to_pass"]
            body["fusion"]["margins"] = final.get("margins")

        # Top-level backward-compatible scores: primary method that passed, else syncnet then mediapipe
        if final["passed"]:
            if "syncnet" in final.get("positive_methods", []):
                body["scores"] = syncnet_out.get("scores")
                body["method_used"] = "syncnet"
            elif "mediapipe" in final.get("positive_methods", []):
                body["scores"] = mediapipe_result.get("scores")
                body["method_used"] = "mediapipe"
            else:
                body["scores"] = syncnet_out.get("scores") or mediapipe_result.get("scores")
                body["method_used"] = "fused"
        else:
            body["scores"] = {
                "syncnet": syncnet_out.get("scores"),
                "mediapipe": mediapipe_result.get("scores"),
            }
            body["method_used"] = None

        if syncnet_out.get("error") and mediapipe_result.get("error"):
            raise HTTPException(
                status_code=422,
                detail={
                    "syncnet": syncnet_out.get("error"),
                    "mediapipe": mediapipe_result.get("error"),
                },
            )

        return JSONResponse(content=body)

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

