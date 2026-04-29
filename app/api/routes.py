"""Aggregate all sub-routers into one `api_router` mounted by main.py."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.analyze import router as analyze_router
from app.api.health import router as health_router
from app.api.heygen import router as heygen_router
from app.api.tts import router as tts_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(analyze_router)
api_router.include_router(tts_router)
api_router.include_router(heygen_router)
