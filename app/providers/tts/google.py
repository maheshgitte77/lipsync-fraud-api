"""Google Cloud Text-to-Speech provider."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from app.models.tts import TTSVoiceConfig
from app.providers.tts.base import (
    BaseTTSProvider,
    TTSProviderError,
    TTSProviderUnavailableError,
    TTSResult,
)

logger = get_logger("providers.tts.google")


class GoogleProvider(BaseTTSProvider):
    name = "google"

    def is_available(self) -> bool:
        if importlib.util.find_spec("google.cloud.texttospeech") is None:
            return False
        cred = settings.tts.google_credentials_json
        if not cred:
            return False
        # Verify the service-account JSON actually exists — otherwise the
        # provider would claim to be available and then blow up on generate().
        return Path(cred).is_file()

    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        try:
            from google.cloud import texttospeech  # type: ignore
        except ImportError as exc:
            raise TTSProviderUnavailableError(
                "google-cloud-texttospeech not installed. pip install google-cloud-texttospeech"
            ) from exc
        if not settings.tts.google_credentials_json:
            raise TTSProviderUnavailableError("GOOGLE_APPLICATION_CREDENTIALS not set")

        language = (voice and voice.language) or settings.tts.google_language
        voice_name = (voice and voice.voice_id) or settings.tts.google_voice
        sr = (voice and voice.sample_rate) or settings.tts.audio_sample_rate

        try:
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            v = texttospeech.VoiceSelectionParams(language_code=language, name=voice_name)
            audio_cfg = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=sr,
                speaking_rate=(voice and voice.speed) or 1.0,
            )
            resp = client.synthesize_speech(input=synthesis_input, voice=v, audio_config=audio_cfg)
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(f"google tts error: {exc}") from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.audio_content)
        logger.info("google tts wrote %d bytes → %s", len(resp.audio_content), out_path)
        return TTSResult(audio_path=out_path, provider=self.name, sample_rate=sr)
