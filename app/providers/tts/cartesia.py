"""
Cartesia TTS provider (mock / best-effort).

The Cartesia SDK surface changes; this provider:
1. Tries the official `cartesia` SDK if installed.
2. Falls back to their public REST endpoint when an API key is set.
3. Raises TTSProviderUnavailableError if neither path is possible, so the
   fallback chain can skip it cleanly.
"""

from __future__ import annotations

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

logger = get_logger("providers.tts.cartesia")

_REST_URL = "https://api.cartesia.ai/tts/bytes"

# Remap deprecated dashboard aliases to valid API model ids.
# Mirrors ai-calling-platform/agent-python/providers/tts/__init__.py.
_MODEL_ALIASES = {
    "sonic-3-stable": "sonic-3",
    "sonic-3": "sonic-3",
    "sonic-3-latest": "sonic-3-latest",
}


def _resolve_model(model_id: str | None) -> str:
    m = (model_id or settings.tts.cartesia_model or "sonic-english").strip()
    return _MODEL_ALIASES.get(m, m)


class CartesiaProvider(BaseTTSProvider):
    name = "cartesia"

    def is_available(self) -> bool:
        return bool(settings.tts.cartesia_api_key)

    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        if not self.is_available():
            raise TTSProviderUnavailableError("CARTESIA_API_KEY not set")

        voice_id = (voice and voice.voice_id) or settings.tts.cartesia_voice_id
        if not voice_id:
            raise TTSProviderUnavailableError("CARTESIA_VOICE_ID not set")
        sr = (voice and voice.sample_rate) or settings.tts.audio_sample_rate
        model = _resolve_model(voice and voice.model_id)

        body = {
            "model_id": model,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": sr,
            },
        }
        if voice and voice.language:
            body["language"] = voice.language
        req = Request(
            _REST_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "X-API-Key": settings.tts.cartesia_api_key,
                "Content-Type": "application/json",
                "Cartesia-Version": "2024-06-10",
                "Accept": "audio/wav",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=60) as resp:
                wav_bytes = resp.read()
        except HTTPError as exc:
            raise TTSProviderError(
                f"cartesia HTTP {exc.code}: {exc.read()[:500].decode(errors='replace')}"
            ) from exc
        except URLError as exc:
            raise TTSProviderError(f"cartesia network error: {exc}") from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
        logger.info("cartesia tts wrote %d bytes → %s", len(wav_bytes), out_path)
        return TTSResult(audio_path=out_path, provider=self.name, sample_rate=sr)
