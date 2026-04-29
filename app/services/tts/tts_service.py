"""
TTS service: pick a provider, optionally fall back to the next on failure.

Providers are DI-registered at construction time so tests can swap in fakes.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from app.models.tts import TTSVoiceConfig
from app.providers.tts import (
    BaseTTSProvider,
    CartesiaProvider,
    DeepgramProvider,
    ElevenLabsProvider,
    GoogleProvider,
    InworldProvider,
    SarvamProvider,
    TTSProviderError,
    TTSProviderUnavailableError,
    TTSResult,
    XAIProvider,
)

logger = get_logger("services.tts")


class TTSService:
    """Strategy dispatcher with optional fallback chain."""

    def __init__(self, providers: dict[str, BaseTTSProvider] | None = None) -> None:
        self._providers: dict[str, BaseTTSProvider] = providers or self._default_providers()

    # ---- public API -------------------------------------------------------

    def list_providers(self) -> list[dict]:
        return [
            {"name": name, "available": provider.is_available()}
            for name, provider in self._providers.items()
        ]

    def generate(
        self,
        text: str,
        *,
        out_path: Path,
        provider: str | None = None,
        voice: TTSVoiceConfig | None = None,
        allow_fallback: bool = True,
    ) -> TTSResult:
        text = self._validate_text(text)
        candidates = self._resolve_chain(provider, allow_fallback)
        last_err: Exception | None = None
        for name in candidates:
            p = self._providers.get(name)
            if p is None:
                logger.warning("unknown TTS provider %s, skipping", name)
                continue
            if not p.is_available():
                logger.info("tts provider %s not available, skipping", name)
                continue
            try:
                logger.info("tts provider=%s text_len=%d", name, len(text))
                return p.generate(text, out_path=out_path, voice=voice)
            except TTSProviderUnavailableError as exc:
                logger.info("tts provider %s unavailable: %s", name, exc)
                last_err = exc
                continue
            except TTSProviderError as exc:
                logger.warning("tts provider %s failed: %s", name, exc)
                last_err = exc
                if not allow_fallback:
                    break
                continue
        raise TTSProviderError(
            f"All TTS providers failed or unavailable. Tried: {candidates}. Last error: {last_err}"
        )

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _validate_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise TTSProviderError("empty text")
        if len(text) > 5000:
            raise TTSProviderError("text exceeds 5000 characters")
        return text

    def _resolve_chain(self, provider: str | None, allow_fallback: bool) -> list[str]:
        primary = ((provider or settings.tts.default_provider) or "").strip().lower()
        if not primary:
            primary = "elevenlabs"
        if not allow_fallback:
            return [primary]
        chain: list[str] = [primary]
        for name in settings.tts.fallback_order:
            name = (name or "").strip().lower()
            if name and name not in chain:
                chain.append(name)
        return chain

    @staticmethod
    def _default_providers() -> dict[str, BaseTTSProvider]:
        return {
            "elevenlabs": ElevenLabsProvider(),
            "cartesia": CartesiaProvider(),
            "deepgram": DeepgramProvider(),
            "inworld": InworldProvider(),
            "xai": XAIProvider(),
            "sarvam": SarvamProvider(),
            "google": GoogleProvider(),
        }


tts_service = TTSService()
