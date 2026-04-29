"""ElevenLabs TTS provider. Uses the public REST API (no SDK dependency)."""

from __future__ import annotations

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
from app.utils.ffmpeg import extract_wav

logger = get_logger("providers.tts.elevenlabs")

_API_BASE = "https://api.elevenlabs.io/v1"


class ElevenLabsProvider(BaseTTSProvider):
    name = "elevenlabs"

    def is_available(self) -> bool:
        return bool(settings.tts.elevenlabs_api_key)

    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        if not self.is_available():
            raise TTSProviderUnavailableError("ELEVENLABS_API_KEY not set")

        voice_id = (voice and voice.voice_id) or settings.tts.elevenlabs_voice_id
        model_id = (voice and voice.model_id) or settings.tts.elevenlabs_model_id
        stability = voice.stability if voice and voice.stability is not None else 0.5
        similarity = voice.similarity_boost if voice and voice.similarity_boost is not None else 0.75

        url = f"{_API_BASE}/text-to-speech/{voice_id}"
        body = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {"stability": stability, "similarity_boost": similarity},
        }
        req = Request(
            url,
            data=_json_bytes(body),
            headers={
                "xi-api-key": settings.tts.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=60) as resp:
                mp3_bytes = resp.read()
        except HTTPError as exc:
            raise TTSProviderError(f"elevenlabs HTTP {exc.code}: {exc.read()[:500].decode(errors='replace')}") from exc
        except URLError as exc:
            raise TTSProviderError(f"elevenlabs network error: {exc}") from exc

        mp3_path = out_path.with_suffix(".mp3")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path.write_bytes(mp3_bytes)
        sr = (voice and voice.sample_rate) or settings.tts.audio_sample_rate
        extract_wav(mp3_path, out_path, sample_rate=sr)
        mp3_path.unlink(missing_ok=True)

        logger.info("elevenlabs generated %d bytes → %s", len(mp3_bytes), out_path)
        return TTSResult(audio_path=out_path, provider=self.name, sample_rate=sr)


def _json_bytes(body: dict) -> bytes:
    import json

    return json.dumps(body).encode("utf-8")
