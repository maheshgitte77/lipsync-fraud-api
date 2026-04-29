"""Video-analysis endpoints: synchronous upload, proctor signals, async jobs."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import get_logger
from app.core.metrics import StageTimer
from app.models.proctor import ProctorSignalsRequest
from app.services.lipsync.syncnet_service import syncnet_service
from app.services.lipsync.window_builder import build_window
from app.services.orchestration import proctor_orchestrator
from app.utils.ffmpeg import normalize_container, video_duration_sec
from app.utils.file_manager import job_dir, new_job_id
from app.utils.video_download import ALLOWED_VIDEO_EXT
from app.workers.job_store import job_store
from app.workers.kafka_worker import enqueue_proctor_job

logger = get_logger("api.analyze")
router = APIRouter(tags=["analyze"])


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Upload a video and get a SyncNet-based lip-sync verdict."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {ext!r}. Allowed: {sorted(ALLOWED_VIDEO_EXT)}",
        )

    if not syncnet_service.is_available():
        raise HTTPException(
            status_code=503,
            detail=f"SYNCNET_DIR not found: {settings.paths.syncnet_dir}. Clone joonson/syncnet_python.",
        )

    jid = new_job_id()
    work = job_dir(jid)
    video_path = work / f"input{ext}"
    timer = StageTimer()
    try:
        with timer.track("upload"):
            with open(video_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

        with timer.track("normalize"):
            syncnet_path = normalize_container(video_path, work, job_id=jid)
        src_dur = video_duration_sec(syncnet_path)
        win = build_window(src_dur or 0.0) if settings.syncnet.trim_enabled else None
        trim_meta = {
            "trimEnabled": bool(settings.syncnet.trim_enabled),
            "trimMaxSeconds": max(1.0, min(settings.syncnet.trim_max_seconds, 600.0)),
            "trimApplied": bool(
                settings.syncnet.trim_enabled
                and win
                and (
                    float(win["startSec"]) > 0.001
                    or (src_dur or 0.0) > float(win["durationSec"]) + 0.05
                )
            ),
            "sourceDurationSec": src_dur,
            "analyzedDurationSec": (float(win["durationSec"]) if win else src_dur),
            "position": settings.syncnet.window_position,
            "singleWindowMode": True,
        }

        with timer.track("syncnet"):
            syncnet_result = syncnet_service.analyze_windowed(syncnet_path, f"cand_{jid}", work)
        syncnet_out = {k: v for k, v in syncnet_result.items() if k != "all_tracks"}
        if "all_tracks" in syncnet_result and settings.syncnet.debug_log:
            syncnet_out["all_tracks"] = syncnet_result["all_tracks"]

        final = syncnet_service.syncnet_only_fusion(syncnet_out, skipped=False)

        body = {
            "job_id": jid,
            "file": file.filename,
            "video_trim": trim_meta,
            "verdict": final["verdict"],
            "passed": final["passed"],
            "reason": final["reason"],
            "fusion": {
                "mode": "syncnet_only",
                "positive_methods": final.get("positive_methods", []),
                "mediapipe_lipsync_skipped": True,
            },
            "syncnet": syncnet_out,
            "scores": syncnet_out.get("scores"),
            "method_used": "syncnet" if final["passed"] else None,
            "metrics": timer.snapshot(),
        }
        if syncnet_out.get("error"):
            raise HTTPException(status_code=422, detail={"syncnet": syncnet_out.get("error")})
        return JSONResponse(content=body)
    finally:
        shutil.rmtree(work, ignore_errors=True)


@router.post("/analyze/proctor-signals")
async def analyze_proctor_signals(payload: ProctorSignalsRequest = Body(...)):
    """Synchronous fraud-signal analysis (downloads the video from `videoUrl`)."""
    jid = new_job_id()
    try:
        body = proctor_orchestrator.execute(payload, job_id=jid)
        return JSONResponse(content=body)
    except Exception as exc:  # noqa: BLE001
        logger.exception("proctor analysis failed for %s", jid)
        raise HTTPException(status_code=422, detail=str(exc)[:4000]) from exc


@router.post("/analyze/proctor-signals/submit")
async def submit_proctor_signals(payload: ProctorSignalsRequest = Body(...)):
    """Publish an analysis job to Kafka and return immediately."""
    if not settings.kafka.enabled:
        raise HTTPException(status_code=503, detail="Kafka mode disabled. Set PROCTOR_KAFKA_ENABLED=true")

    jid = new_job_id()
    job_store.set(
        jid,
        status="QUEUED",
        createdAt=int(time.time()),
        updatedAt=int(time.time()),
        result=None,
        error=None,
    )
    try:
        enqueue_proctor_job(jid, payload)
    except Exception as exc:  # noqa: BLE001
        job_store.set(jid, status="FAILED", updatedAt=int(time.time()), error=str(exc)[:4000])
        raise HTTPException(status_code=500, detail=f"kafka enqueue failed: {str(exc)[:1000]}") from exc
    return {"jobId": jid, "status": "QUEUED"}


@router.get("/analyze/proctor-signals/jobs/{job_id}")
async def get_proctor_signals_job(job_id: str):
    state = job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return {"jobId": job_id, **state}
