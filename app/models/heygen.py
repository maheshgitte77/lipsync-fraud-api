"""Schemas for the HeyGen avatar video generation integration.

We expose the full HeyGen v2 `/video/generate` payload surface so callers can customize every
knob, plus a convenience "simple" request shape that assembles the payload
for the common "one avatar + one voice + one script" case.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------- Character ----------

class HeyGenCharacter(BaseModel):
    """HeyGen v2 `character` block. `type` defaults to `avatar`."""
    type: Literal["avatar", "talking_photo"] = "avatar"
    avatar_id: str | None = Field(
        default=None,
        description="HeyGen avatar_id (for type=avatar).",
        examples=["Daisy-inskirt-20220818"],
    )
    talking_photo_id: str | None = Field(
        default=None,
        description="HeyGen talking photo id (for type=talking_photo).",
    )
    avatar_style: Literal["normal", "circle", "closeUp"] | None = "normal"
    scale: float | None = Field(default=1.0, ge=0.1, le=5.0)
    offset: dict[str, float] | None = Field(
        default=None,
        description='Translation in normalized units, e.g. {"x": 0, "y": 0}.',
    )
    matting: bool | None = None
    circle_background_color: str | None = None


# ---------- Voice ----------

class HeyGenVoice(BaseModel):
    """HeyGen v2 `voice` block. Supports `text`, `audio`, and `silence`."""
    type: Literal["text", "audio", "silence"] = "text"

    # type=text
    input_text: str | None = Field(
        default=None,
        max_length=4000,
        description="Script to synthesize (type=text).",
    )
    voice_id: str | None = None
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    pitch: float | None = Field(default=None, ge=-50, le=50)
    emotion: Literal[
        "Excited", "Friendly", "Serious", "Soothing", "Broadcaster"
    ] | None = None
    locale: str | None = Field(default=None, examples=["en-IN"])

    # type=audio
    audio_url: str | None = Field(
        default=None,
        description="Publicly-accessible audio URL (type=audio).",
    )
    audio_asset_id: str | None = None

    # type=silence
    duration: float | None = Field(
        default=None, ge=0.1, le=60.0, description="Silence duration in seconds."
    )


# ---------- Background ----------

class HeyGenBackground(BaseModel):
    type: Literal["color", "image", "video"] = "color"
    value: str | None = Field(default=None, examples=["#008000"])
    image_asset_id: str | None = None
    video_asset_id: str | None = None
    url: str | None = None
    play_style: Literal["loop", "once"] | None = None
    fit: Literal["cover", "contain", "crop", "none"] | None = None


# ---------- Video input ----------

class HeyGenVideoInput(BaseModel):
    """One scene in HeyGen's `video_inputs` array."""
    character: HeyGenCharacter
    voice: HeyGenVoice
    background: HeyGenBackground | None = None


class HeyGenDimension(BaseModel):
    width: int = Field(default=1280, ge=360, le=3840)
    height: int = Field(default=720, ge=360, le=3840)


# ---------- Full passthrough request ----------

class HeyGenGenerateRawRequest(BaseModel):
    """
    Full-fidelity HeyGen v2 `/video/generate` payload. Every field is passed
    through to HeyGen as-is. Use this when you need maximum customization
    (multi-scene videos, captions, callback, etc.).
    """
    video_inputs: list[HeyGenVideoInput] = Field(..., min_length=1)
    dimension: HeyGenDimension | None = None
    title: str | None = None
    caption: bool | None = None
    callback_id: str | None = None
    callback_url: str | None = None
    folder_id: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_inputs": [
                    {
                        "character": {
                            "type": "avatar",
                            "avatar_id": "Daisy-inskirt-20220818",
                            "avatar_style": "normal",
                        },
                        "voice": {
                            "type": "text",
                            "input_text": "Hello, welcome to the interview.",
                            "voice_id": "1bd001e7e50f421d891986aad5158bc8",
                            "speed": 1.0,
                            "emotion": "Friendly",
                        },
                        "background": {"type": "color", "value": "#ffffff"},
                    }
                ],
                "dimension": {"width": 1280, "height": 720},
                "title": "intro-video",
                "caption": False,
            }
        }
    )


# ---------- Simple convenience request ----------

class HeyGenGenerateSimpleRequest(BaseModel):
    """
    Convenience shape for the common "one avatar speaks one script" case.
    We build the full v2 payload internally. Use `/heygen/video/generate/raw`
    if you need multi-scene or full control.
    """
    question_text: str = Field(
        ..., min_length=1, max_length=4000,
        examples=["Hello, welcome to the interview. Please introduce yourself."],
    )
    avatar_id: str | None = Field(
        default=None,
        description="HeyGen avatar_id. Falls back to HEYGEN_DEFAULT_AVATAR_ID.",
    )
    talking_photo_id: str | None = Field(
        default=None,
        description="Use a talking-photo instead of an avatar_id.",
    )
    voice_id: str | None = Field(
        default=None,
        description="HeyGen voice_id. Falls back to HEYGEN_DEFAULT_VOICE_ID.",
    )
    audio_url: str | None = Field(
        default=None,
        description="If provided, overrides TTS: HeyGen will lip-sync to this audio URL.",
    )
    audio_asset_id: str | None = None

    # voice tuning
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    pitch: float | None = Field(default=None, ge=-50, le=50)
    emotion: Literal[
        "Excited", "Friendly", "Serious", "Soothing", "Broadcaster"
    ] | None = None
    locale: str | None = None

    # character tuning
    avatar_style: Literal["normal", "circle", "closeUp"] | None = "normal"
    scale: float | None = Field(default=None, ge=0.1, le=5.0)
    offset_x: float | None = None
    offset_y: float | None = None
    matting: bool | None = Field(
        default=None,
        description="If true, use transparent background (green-screen removed).",
    )

    # output tuning
    width: int | None = Field(default=None, ge=360, le=3840)
    height: int | None = Field(default=None, ge=360, le=3840)
    background_color: str | None = Field(
        default=None, examples=["#ffffff"],
        description="Solid background color (hex). Mutually exclusive with background_image_url.",
    )
    background_image_url: str | None = None
    background_video_url: str | None = None
    title: str | None = None
    caption: bool | None = None
    callback_url: str | None = None
    folder_id: str | None = None

    # orchestration
    wait_for_completion: bool = Field(
        default=True,
        description="If true, the endpoint polls HeyGen until the video is ready "
                    "and returns the final URL. If false, returns only video_id.",
    )
    max_wait_seconds: int | None = Field(
        default=None, ge=30, le=3600,
        description="Max seconds to poll when wait_for_completion=true.",
    )
    mirror_to_storage: bool | None = Field(
        default=None,
        description="If true, also mirror the HeyGen MP4 into our S3/local storage. "
                    "Defaults to HEYGEN_MIRROR_TO_STORAGE.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
            }
        }
    )

    @field_validator(
        "avatar_id", "talking_photo_id", "voice_id", "audio_url",
        "audio_asset_id", "background_color", "background_image_url",
        "background_video_url", "title", "callback_url", "folder_id",
        "locale", mode="before",
    )
    @classmethod
    def _strip_sentinels(cls, v):
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() == "string":
                return None
            return s
        return v

    @model_validator(mode="after")
    def _one_character_source(self) -> "HeyGenGenerateSimpleRequest":
        if self.avatar_id and self.talking_photo_id:
            raise ValueError(
                "Provide either avatar_id or talking_photo_id, not both."
            )
        return self


# ---------- Responses ----------

class HeyGenGenerateResponse(BaseModel):
    video_id: str
    status: str = Field(
        description="HeyGen status: pending | processing | completed | failed."
    )
    video_url: str | None = None
    thumbnail_url: str | None = None
    duration_seconds: float | None = None
    storage_url: str | None = Field(
        default=None,
        description="Mirrored URL in our S3/local storage (if mirror_to_storage=true).",
    )
    error: str | None = None
    raw: dict[str, Any] | None = Field(
        default=None, description="Raw HeyGen response payload (for debugging)."
    )


class HeyGenListResponse(BaseModel):
    data: Any
    raw: dict[str, Any] | None = None
