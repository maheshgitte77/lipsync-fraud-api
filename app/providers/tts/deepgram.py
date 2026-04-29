"""
Deepgram Aura TTS provider.

Deepgram's REST `/v1/speak` endpoint takes a single `model` query param which
combines model + voice + language, e.g.:

    aura-2-athena-en          (Aura 2, voice=athena, en)
    aura-asteria-en           (Aura 1 legacy, voice=asteria, en)

Callers can either:
1. pass the full compound id via `model_id` (e.g. "aura-asteria-en"), or
2. pass model family (`aura` / `aura-2` / `aura-asteria-en`) in `model_id` +
   voice in `voice_id` (e.g. "athena") and we compose it for them.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
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

logger = get_logger("providers.tts.deepgram")


def _compose_model(model: str, voice: str | None, language: str | None) -> str:
    """Turn (family, voice, lang) into the compound Aura model id."""
    m = (model or "").strip().lower()
    if not m:
        m = "aura-2"
    # If caller already gave a full compound id, leave it alone.
    if m.count("-") >= 2 and not voice:
        return m
    v = (voice or "").strip().lower()
    lang = (language or "en").strip().lower()
    # Legacy single-aura: "aura" → "aura-<voice>-<lang>".
    # Aura 2: "aura-2" → "aura-2-<voice>-<lang>".
    if not v:
        return m
    return f"{m}-{v}-{lang}"


class DeepgramProvider(BaseTTSProvider):
    name = "deepgram"

    def is_available(self) -> bool:
        return bool(settings.tts.deepgram_api_key)

    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        if not self.is_available():
            raise TTSProviderUnavailableError("DEEPGRAM_API_KEY not set")

        family = (voice and voice.model_id) or settings.tts.deepgram_model
        v_name = (voice and voice.voice_id) or settings.tts.deepgram_voice
        lang = (voice and voice.language) or settings.tts.deepgram_language
        model = _compose_model(family, v_name, lang)
        sr = (voice and voice.sample_rate) or settings.tts.audio_sample_rate

        params = urlencode({"model": model, "encoding": "linear16", "sample_rate": sr})
        url = f"https://api.deepgram.com/v1/speak?{params}"
        body = {"text": text}

        req = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Token {settings.tts.deepgram_api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=60) as resp:
                wav_bytes = resp.read()
        except HTTPError as exc:
            raise TTSProviderError(
                f"deepgram HTTP {exc.code} (model={model}): "
                f"{exc.read()[:500].decode(errors='replace')}"
            ) from exc
        except URLError as exc:
            raise TTSProviderError(f"deepgram network error: {exc}") from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
        logger.info("deepgram tts model=%s wrote %d bytes -> %s", model, len(wav_bytes), out_path)
        return TTSResult(audio_path=out_path, provider=self.name, sample_rate=sr)
