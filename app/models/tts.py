"""Schemas for the TTS endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TTSProviderName = Literal[
    "elevenlabs",
    "google",
    "deepgram",
    "cartesia",
    "inworld",
    "xai",
    "sarvam",
]

_VALID_PROVIDERS = {"elevenlabs", "google", "deepgram", "cartesia", "inworld", "xai", "sarvam"}


class TTSVoiceConfig(BaseModel):
    """
    Provider-agnostic voice configuration. Each provider maps these fields to
    its own SDK/REST params (see `app/providers/tts/*.py`).

    - `voice_id` -> ElevenLabs voice_id, Cartesia voice UUID, Inworld voice name,
                    Sarvam `speaker`, xAI voice, Google voice name.
    - `model_id` -> ElevenLabs model_id, Cartesia model, Deepgram Aura family,
                    Inworld model, Sarvam model.
    - `language` -> Google language_code, Sarvam target_language_code,
                    xAI language, Deepgram language suffix.
    - `speed` / `stability` / `similarity_boost` -> only respected when the
      provider supports them (otherwise ignored).

    **Leave any field `null` or omit it** to use the .env default for the
    chosen provider. Never send the literal string `"string"` (that is the
    Swagger UI placeholder and providers will reject it).
    """

    voice_id: str | None = Field(default=None, examples=["Saanvi"])
    model_id: str | None = Field(default=None, examples=["inworld-tts-1.5-max"])
    language: str | None = Field(default=None, examples=["en-IN"])
    speed: float | None = Field(default=None, ge=0.25, le=4.0, examples=[1.0])
    stability: float | None = Field(default=None, ge=0.0, le=1.0, examples=[0.5])
    similarity_boost: float | None = Field(default=None, ge=0.0, le=1.0, examples=[0.75])
    sample_rate: int | None = Field(default=None, ge=8000, le=48000, examples=[24000])

    @field_validator("voice_id", "model_id", "language", mode="before")
    @classmethod
    def _strip_sentinels(cls, v):
        """Turn Swagger's literal `"string"` placeholder (and empty strings)
        into `None` so the provider falls back to its .env default."""
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() == "string":
                return None
            return s
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "voice_id": "Saanvi",
                "model_id": "inworld-tts-1.5-max",
                "language": "en-IN",
                "speed": 1.0,
                "stability": 0.5,
                "similarity_boost": 0.75,
                "sample_rate": 24000,
            }
        }
    )


class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["What is Java? Explain OOPS and its four important pillars."],
    )
    provider: str | None = Field(
        default=None,
        description=(
            "Which TTS provider to use (case-insensitive). One of: "
            "elevenlabs, google, deepgram, cartesia, inworld, xai, sarvam. "
            "Defaults to TTS_DEFAULT_PROVIDER."
        ),
        examples=["inworld"],
    )
    voice: TTSVoiceConfig | None = None
    allow_fallback: bool = Field(
        default=True,
        description="On provider failure, try the next provider from TTS_FALLBACK_ORDER.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "What is Java? Explain OOPS and its four important pillars.",
                "provider": "inworld",
                "voice": {
                    "voice_id": "Saanvi",
                    "model_id": "inworld-tts-1.5-max",
                    "language": "en-IN",
                    "speed": 1.0,
                    "sample_rate": 24000,
                },
                "allow_fallback": True,
            }
        }
    )

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, v: str | None) -> str | None:
        if v is None:
            return None
        name = v.strip().lower()
        if not name or name == "string":
            return None
        if name not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unknown TTS provider '{v}'. Valid: {sorted(_VALID_PROVIDERS)}"
            )
        return name


class TTSResponse(BaseModel):
    audioUrl: str
    provider: str
    durationSec: float | None = None
    sampleRate: int | None = None
    metrics: dict | None = None
