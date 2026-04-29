"""Root + health + config inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.core.device import device_manager
from app.services.lipsync.syncnet_service import syncnet_service
from app.services.tts import tts_service

router = APIRouter(tags=["health"])


@router.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "endpoints": [
            "GET /health",
            "GET /config",
            "POST /analyze",
            "POST /analyze/proctor-signals",
            "POST /analyze/proctor-signals/submit",
            "GET /analyze/proctor-signals/jobs/{job_id}",
            "POST /tts/generate",
            "POST /heygen/generate",
        ],
    }


@router.get("/health")
def health() -> dict:
    syncnet_model = settings.paths.syncnet_dir / "data" / "syncnet_v2.model"
    return {
        "service": settings.app_name,
        "version": settings.version,
        "device": device_manager.describe(),
        "lipsync": {
            "syncnetDir": str(settings.paths.syncnet_dir),
            "syncnetPresent": syncnet_service.is_available(),
            "modelPresent": syncnet_model.is_file(),
            "engine": "syncnet_only",
        },
        "tts": {
            "default": settings.tts.default_provider,
            "fallbackOrder": list(settings.tts.fallback_order),
            "providers": tts_service.list_providers(),
        },
    }


@router.get("/config")
def config() -> dict:
    """Non-secret runtime config. Useful for debugging deploys."""
    return {
        "app": {"name": settings.app_name, "version": settings.version, "logLevel": settings.log_level},
        "device": device_manager.describe(),
        "paths": {
            "syncnetDir": str(settings.paths.syncnet_dir),
            "tempDir": str(settings.paths.temp_dir),
            "ffmpeg": settings.paths.ffmpeg,
            "ffprobe": settings.paths.ffprobe,
        },
        "syncnet": {
            "minDistPass": settings.syncnet.min_dist_pass,
            "confidencePass": settings.syncnet.confidence_pass,
            "trimEnabled": settings.syncnet.trim_enabled,
            "trimMaxSeconds": settings.syncnet.trim_max_seconds,
            "windowPosition": settings.syncnet.window_position,
        },
        "kafka": {
            "enabled": settings.kafka.enabled,
            "startWorker": settings.kafka.start_worker,
            "requestTopic": settings.kafka.request_topic,
            "resultTopic": settings.kafka.result_topic,
        },
        "storage": {
            "backend": settings.storage.backend,
            "localDir": str(settings.storage.local_dir),
            "s3Bucket": settings.storage.s3_bucket,
        },
    }
