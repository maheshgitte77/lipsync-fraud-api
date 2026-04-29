"""
FastAPI entry point for lipsync-fraud-api.

Thin by design: all logic lives in app/services/*. This file only wires up
the ASGI app, routes, and lifecycle hooks (Kafka threads).

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 [--reload]
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import api_router
from app.core.config import settings
from app.core.device import device_manager
from app.core.logger import get_logger
from app.workers.kafka_worker import start_background_threads, stop_background_threads

logger = get_logger(settings.app_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "starting %s v%s on %s", settings.app_name, settings.version, device_manager.torch_device()
    )
    start_background_threads()
    try:
        yield
    finally:
        stop_background_threads()
        logger.info("shutting down %s", settings.app_name)


app = FastAPI(
    title="Lipsync fraud service",
    description=(
        "Unified service for lip-sync authenticity (SyncNet + MediaPipe), "
        "proctor signals, pluggable TTS, and HeyGen avatar generation."
    ),
    version=settings.version,
    lifespan=lifespan,
)
app.include_router(api_router)
