"""
Sarvam AI TTS provider.

Endpoint: https://api.sarvam.ai/text-to-speech
Speakers: shubh (v3), anushka (v2), aditya, ritu, priya, neha, rahul, ...
Models:   bulbul:v3 | bulbul:v2 | bulbul:v3-beta
Languages: hi-IN, en-IN, ta-IN, te-IN, ...

Maps from our generic TTSVoiceConfig:
    voice_id  -> speaker (name)
    model_id  -> model
    language  -> target_language_code (required, default en-IN)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import settings
from app.core.logger import get_logger
from app.models.tts import TTSVoiceConfig
from app.providers.tts.base import (
    BaseTTSProvider,
    TTSProviderError,
    TTSProviderUnavailableError,
    TTSResult,
)

logger = get_logger("providers.tts.sarvam")

_REST_URL = "https://api.sarvam.ai/text-to-speech"


class SarvamProvider(BaseTTSProvider):
    name = "sarvam"

    def is_available(self) -> bool:
        return bool(settings.tts.sarvam_api_key)

    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        if not self.is_available():
            raise TTSProviderUnavailableError("SARVAM_API_KEY not set")

        speaker = (voice and voice.voice_id) or settings.tts.sarvam_speaker
        model = (voice and voice.model_id) or settings.tts.sarvam_model
        language = (voice and voice.language) or settings.tts.sarvam_language
        sr = (voice and voice.sample_rate) or settings.tts.audio_sample_rate
        speed = (voice and voice.speed) or 1.0

        body = {
            "inputs": [text],
            "target_language_code": language,
            "speaker": speaker,
            "model": model,
            "speech_sample_rate": sr,
            "pitch": 0.0,
            "pace": speed,
            "loudness": 1.0,
            "enable_preprocessing": True,
        }
        req = Request(
            _REST_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "API-Subscription-Key": settings.tts.sarvam_api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=90) as resp:
                raw = resp.read()
        except HTTPError as exc:
            raise TTSProviderError(
                f"sarvam HTTP {exc.code} (speaker={speaker}, model={model}): "
                f"{exc.read()[:500].decode(errors='replace')}"
            ) from exc
        except URLError as exc:
            raise TTSProviderError(f"sarvam network error: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TTSProviderError(f"sarvam invalid JSON response: {exc}") from exc

        audios = payload.get("audios") or []
        if not audios:
            raise TTSProviderError(f"sarvam returned no audio: {str(payload)[:500]}")

        try:
            wav_bytes = base64.b64decode(audios[0])
        except (ValueError, TypeError) as exc:
            raise TTSProviderError(f"sarvam invalid base64 audio: {exc}") from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
        logger.info(
            "sarvam tts speaker=%s model=%s lang=%s wrote %d bytes -> %s",
            speaker,
            model,
            language,
            len(wav_bytes),
            out_path,
        )
        return TTSResult(audio_path=out_path, provider=self.name, sample_rate=sr)
