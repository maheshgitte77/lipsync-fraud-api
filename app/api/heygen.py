"""/heygen — cloud avatar video generation via HeyGen v2 API.

This router only uses HeyGen's avatar-video-creation endpoints; no other HeyGen features are
consumed. Every field is user-customizable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from app.core.logger import get_logger
from app.models.heygen import (
    HeyGenGenerateRawRequest,
    HeyGenGenerateResponse,
    HeyGenGenerateSimpleRequest,
    HeyGenListResponse,
)
from app.services.heygen import HeyGenError, HeyGenNotConfiguredError, heygen_service

logger = get_logger("api.heygen")
router = APIRouter(prefix="/heygen", tags=["heygen"])


_SIMPLE_EXAMPLES = {
    "text_script": {
        "summary": "Avatar speaks a text script (HeyGen TTS)",
        "value": {
            "question_text": "Hello! Tell me about object-oriented programming.",
            "avatar_id": "Daisy-inskirt-20220818",
            "voice_id": "1bd001e7e50f421d891986aad5158bc8",
            "speed": 1.0,
            "emotion": "Friendly",
            "width": 1280,
            "height": 720,
            "background_color": "#ffffff",
            "title": "oops-intro",
            "wait_for_completion": True,
        },
    },
    "custom_audio": {
        "summary": "Avatar lip-syncs to a provided audio URL (no HeyGen TTS)",
        "value": {
            "question_text": "ignored when audio_url is set",
            "avatar_id": "Daisy-inskirt-20220818",
            "audio_url": "https://example.com/my-speech.mp3",
            "width": 1080,
            "height": 1920,
            "background_color": "#000000",
            "wait_for_completion": True,
        },
    },
    "talking_photo_transparent": {
        "summary": "Talking-photo with transparent (matting) background",
        "value": {
            "question_text": "Welcome to the interview.",
            "talking_photo_id": "your_talking_photo_id",
            "voice_id": "1bd001e7e50f421d891986aad5158bc8",
            "matting": True,
            "scale": 1.2,
            "offset_x": 0,
            "offset_y": -0.1,
            "wait_for_completion": False,
        },
    },
}


_RAW_EXAMPLE = {
    "multi_scene": {
        "summary": "Full HeyGen v2 payload (multi-scene capable)",
        "value": {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": "Daisy-inskirt-20220818",
                        "avatar_style": "normal",
                    },
                    "voice": {
                        "type": "text",
                        "input_text": "Scene one.",
                        "voice_id": "1bd001e7e50f421d891986aad5158bc8",
                        "speed": 1.0,
                        "emotion": "Friendly",
                    },
                    "background": {"type": "color", "value": "#ffffff"},
                },
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": "Daisy-inskirt-20220818",
                    },
                    "voice": {
                        "type": "text",
                        "input_text": "Scene two.",
                        "voice_id": "1bd001e7e50f421d891986aad5158bc8",
                    },
                    "background": {"type": "color", "value": "#000000"},
                },
            ],
            "dimension": {"width": 1280, "height": 720},
            "title": "multi-scene",
            "caption": False,
        },
    }
}


def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, HeyGenNotConfiguredError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, HeyGenError):
        return HTTPException(status_code=502, detail=str(exc)[:2000])
    return HTTPException(status_code=500, detail=str(exc)[:2000])


@router.get("/health")
def heygen_health() -> dict:
    return heygen_service.describe()


# ---------- talking photo (upload your own photo -> talking_photo_id) ----------

_ALLOWED_UPLOAD_CT = {"image/jpeg", "image/png", "image/webp"}


class TalkingPhotoFromUrlRequest(BaseModel):
    image_url: str = Field(
        ..., examples=["https://example.com/my-photo.jpg"],
        description="HTTP(S) URL of a JPG/PNG/WEBP photo. We fetch it and upload to HeyGen.",
    )


class TalkingPhotoUploadResponse(BaseModel):
    talking_photo_id: str | None
    talking_photo_url: str | None = None
    raw: dict | None = None


@router.post("/talking-photo/upload", response_model=TalkingPhotoUploadResponse)
async def upload_talking_photo(
    file: UploadFile = File(..., description="JPG / PNG / WEBP image of a face."),
) -> TalkingPhotoUploadResponse:
    """
    Upload a local photo to HeyGen and get back a `talking_photo_id`.

    Use the returned id in `POST /heygen/avatar/generate` as `talking_photo_id`
    to make *your* photo speak.
    """
    ct = (file.content_type or "").lower()
    if ct not in _ALLOWED_UPLOAD_CT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content-type '{ct}'. Allowed: {sorted(_ALLOWED_UPLOAD_CT)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        result = heygen_service.upload_talking_photo_bytes(data, content_type=ct)
        return TalkingPhotoUploadResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen talking-photo upload failed")
        raise _handle(exc) from exc


@router.post("/talking-photo/from-url", response_model=TalkingPhotoUploadResponse)
def talking_photo_from_url(
    req: TalkingPhotoFromUrlRequest,
) -> TalkingPhotoUploadResponse:
    """Fetch an image from a URL and upload it to HeyGen as a talking photo."""
    try:
        result = heygen_service.upload_talking_photo_from_url(req.image_url)
        return TalkingPhotoUploadResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen talking-photo from-url failed")
        raise _handle(exc) from exc


# ---------- discovery ----------

@router.get("/avatars", response_model=HeyGenListResponse)
def list_avatars() -> HeyGenListResponse:
    try:
        raw = heygen_service.list_avatars()
        return HeyGenListResponse(data=raw.get("data", raw), raw=raw)
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen list avatars failed")
        raise _handle(exc) from exc


@router.get("/voices", response_model=HeyGenListResponse)
def list_voices() -> HeyGenListResponse:
    try:
        raw = heygen_service.list_voices()
        return HeyGenListResponse(data=raw.get("data", raw), raw=raw)
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen list voices failed")
        raise _handle(exc) from exc


# ---------- status ----------

@router.get("/video/status", response_model=HeyGenGenerateResponse)
def video_status(
    video_id: str = Query(..., description="HeyGen video_id returned by /video/generate."),
) -> HeyGenGenerateResponse:
    try:
        return heygen_service.get_status(video_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen status failed")
        raise _handle(exc) from exc


@router.get("/video/wait", response_model=HeyGenGenerateResponse)
def video_wait(
    video_id: str = Query(..., description="HeyGen video_id to wait for."),
    max_wait_seconds: int | None = Query(default=None, ge=30, le=3600),
    mirror_to_storage: bool | None = Query(default=None),
) -> HeyGenGenerateResponse:
    try:
        final = heygen_service.wait_for_completion(
            video_id, max_wait_seconds=max_wait_seconds
        )
        if mirror_to_storage and final.status == "completed":
            final = heygen_service.mirror_video_to_storage(final)
        return final
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen wait failed")
        raise _handle(exc) from exc


# ---------- generate: simple convenience ----------

@router.post("/avatar/generate", response_model=HeyGenGenerateResponse)
def generate_avatar(
    req: HeyGenGenerateSimpleRequest = Body(..., openapi_examples=_SIMPLE_EXAMPLES),
) -> HeyGenGenerateResponse:
    """
    One-shot avatar video creation via HeyGen.

    By default (`wait_for_completion=true`) this blocks until the video is
    ready and returns the final `video_url`. Set `wait_for_completion=false`
    to submit and return only `video_id` immediately; poll `/heygen/video/status`.
    """
    try:
        return heygen_service.generate_simple(req)
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen generate_simple failed")
        raise _handle(exc) from exc


# ---------- generate: full passthrough ----------

@router.post("/video/generate/raw", response_model=HeyGenGenerateResponse)
def generate_raw(
    req: HeyGenGenerateRawRequest = Body(..., openapi_examples=_RAW_EXAMPLE),
    wait: bool = Query(default=True, description="Block until ready."),
    max_wait_seconds: int | None = Query(default=None, ge=30, le=3600),
    mirror_to_storage: bool | None = Query(default=None),
) -> HeyGenGenerateResponse:
    """Submit the full HeyGen v2 `/video/generate` payload unchanged."""
    try:
        return heygen_service.generate_raw(
            req,
            wait=wait,
            max_wait_seconds=max_wait_seconds,
            mirror=mirror_to_storage,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen generate_raw failed")
        raise _handle(exc) from exc


# ---------- submit only (fire and forget) ----------

@router.post("/video/submit")
def submit_only(
    req: HeyGenGenerateSimpleRequest = Body(..., openapi_examples=_SIMPLE_EXAMPLES),
) -> dict:
    """Submit a job and return only `video_id` (no polling)."""
    try:
        req = req.model_copy(update={"wait_for_completion": False})
        video_id = heygen_service.submit_simple(req)
        return {"video_id": video_id, "status": "processing"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("heygen submit failed")
        raise _handle(exc) from exc
