"""
xAI (Grok) TTS provider.

Endpoint: https://api.x.ai/v1/audio/speech  (OpenAI-compatible)
Voices: ara | eve | rex | sal | leo
Model:  tts-1

Maps from our generic TTSVoiceConfig:
    voice_id  -> voice
    model_id  -> model
    language  -> language (default "auto")
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
from app.providers.tts.inworld import _sniff_audio_ext
from app.utils.ffmpeg import FFmpegError, extract_wav

logger = get_logger("providers.tts.xai")

_REST_URL = "https://api.x.ai/v1/audio/speech"


class XAIProvider(BaseTTSProvider):
    name = "xai"

    def is_available(self) -> bool:
        return bool(settings.tts.xai_api_key)

    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        if not self.is_available():
            raise TTSProviderUnavailableError("XAI_API_KEY not set")

        voice_name = (voice and voice.voice_id) or settings.tts.xai_voice
        model = (voice and voice.model_id) or settings.tts.xai_model
        language = (voice and voice.language) or settings.tts.xai_language
        sr = (voice and voice.sample_rate) or settings.tts.audio_sample_rate

        body = {
            "model": model,
            "input": text,
            "voice": voice_name,
            "response_format": "mp3",
        }
        if language and language.lower() != "auto":
            body["language"] = language

        req = Request(
            _REST_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.tts.xai_api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=60) as resp:
                audio_bytes = resp.read()
        except HTTPError as exc:
            raise TTSProviderError(
                f"xai HTTP {exc.code} (voice={voice_name}, model={model}): "
                f"{exc.read()[:500].decode(errors='replace')}"
            ) from exc
        except URLError as exc:
            raise TTSProviderError(f"xai network error: {exc}") from exc

        if len(audio_bytes) < 128:
            raise TTSProviderError(
                f"xai: audio too small ({len(audio_bytes)}B) — likely an error response: "
                f"{audio_bytes[:500]!r}"
            )

        ext = _sniff_audio_ext(audio_bytes)
        if ext == ".bin":
            raise TTSProviderError(
                f"xai: unrecognized audio format, first bytes={audio_bytes[:16]!r}"
            )
        src_path = out_path.with_suffix(ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_bytes(audio_bytes)

        try:
            extract_wav(src_path, out_path, sample_rate=sr)
        except FFmpegError as exc:
            raise TTSProviderError(f"xai: ffmpeg conversion failed from {ext}: {exc}") from exc
        finally:
            src_path.unlink(missing_ok=True)

        logger.info(
            "xai tts voice=%s model=%s lang=%s src=%s wrote %d bytes -> %s",
            voice_name,
            model,
            language,
            ext,
            len(audio_bytes),
            out_path,
        )
        return TTSResult(audio_path=out_path, provider=self.name, sample_rate=sr)
