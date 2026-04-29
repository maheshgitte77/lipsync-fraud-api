"""
Inworld AI TTS provider.

Inworld's REST endpoint accepts the voice by name (e.g. "Arjun", "Saanvi") and
a model id (e.g. "inworld-tts-1.5-mini", "inworld-tts-1.5-max").

Our TTSVoiceConfig maps:
    voice_id  -> voice (name, case-sensitive)
    model_id  -> model
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
from app.utils.ffmpeg import FFmpegError, extract_wav

logger = get_logger("providers.tts.inworld")

_REST_URL = "https://api.inworld.ai/tts/v1/voice"


def _sniff_audio_ext(data: bytes) -> str:
    """Return a filename suffix based on magic bytes."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:3] == b"ID3":
        return ".mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:4] == b"fLaC":
        return ".flac"
    return ".bin"


def _decode_inworld_response(payload: bytes, content_type: str) -> bytes:
    """
    Inworld's REST endpoint returns JSON `{"audioContent": "<base64>"}` by default
    even when you ask for `Accept: audio/mpeg`. Fall back to raw bytes if the
    response is clearly not JSON.
    """
    looks_json = "json" in content_type.lower() or payload[:1] in (b"{", b"[")
    if not looks_json:
        return payload
    try:
        doc = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload
    audio_b64 = (
        doc.get("audioContent")
        or doc.get("audio_content")
        or doc.get("audio")
        or doc.get("data")
    )
    if not audio_b64:
        # Surface the actual error body — usually has an error message
        raise TTSProviderError(
            f"inworld: unexpected JSON response (no audioContent): {str(doc)[:500]}"
        )
    try:
        return base64.b64decode(audio_b64)
    except (ValueError, TypeError) as exc:
        raise TTSProviderError(f"inworld: invalid base64 audio: {exc}") from exc


class InworldProvider(BaseTTSProvider):
    name = "inworld"

    def is_available(self) -> bool:
        return bool(settings.tts.inworld_api_key)

    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        if not self.is_available():
            raise TTSProviderUnavailableError("INWORLD_API_KEY not set")

        voice_name = (voice and voice.voice_id) or settings.tts.inworld_voice_id
        if not voice_name:
            raise TTSProviderUnavailableError("INWORLD_VOICE_ID not set")
        model = (voice and voice.model_id) or settings.tts.inworld_model
        sr = (voice and voice.sample_rate) or settings.tts.audio_sample_rate

        body = {
            "text": text,
            "voiceId": voice_name,
            "modelId": model,
        }
        req = Request(
            _REST_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Basic {settings.tts.inworld_api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=60) as resp:
                raw = resp.read()
                content_type = resp.headers.get("Content-Type", "")
        except HTTPError as exc:
            raise TTSProviderError(
                f"inworld HTTP {exc.code} (voice={voice_name}, model={model}): "
                f"{exc.read()[:500].decode(errors='replace')}"
            ) from exc
        except URLError as exc:
            raise TTSProviderError(f"inworld network error: {exc}") from exc

        audio_bytes = _decode_inworld_response(raw, content_type)
        if len(audio_bytes) < 128:
            raise TTSProviderError(
                f"inworld: audio too small ({len(audio_bytes)}B) — likely an error response"
            )

        ext = _sniff_audio_ext(audio_bytes)
        if ext == ".bin":
            # Unknown format — dump first bytes to the error so we can debug quickly
            raise TTSProviderError(
                f"inworld: unrecognized audio format, first bytes={audio_bytes[:16]!r}"
            )
        src_path = out_path.with_suffix(ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_bytes(audio_bytes)

        try:
            extract_wav(src_path, out_path, sample_rate=sr)
        except FFmpegError as exc:
            raise TTSProviderError(
                f"inworld: ffmpeg conversion failed from {ext}: {exc}"
            ) from exc
        finally:
            src_path.unlink(missing_ok=True)

        logger.info(
            "inworld tts voice=%s model=%s src=%s wrote %d bytes -> %s",
            voice_name,
            model,
            ext,
            len(audio_bytes),
            out_path,
        )
        return TTSResult(audio_path=out_path, provider=self.name, sample_rate=sr)
