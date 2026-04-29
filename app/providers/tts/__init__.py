"""TTS provider implementations (strategy pattern, all implement BaseTTSProvider)."""

from app.providers.tts.base import (
    BaseTTSProvider,
    TTSProviderError,
    TTSProviderUnavailableError,
    TTSResult,
)
from app.providers.tts.cartesia import CartesiaProvider
from app.providers.tts.catalog import TTS_CATALOG
from app.providers.tts.deepgram import DeepgramProvider
from app.providers.tts.elevenlabs import ElevenLabsProvider
from app.providers.tts.google import GoogleProvider
from app.providers.tts.inworld import InworldProvider
from app.providers.tts.sarvam import SarvamProvider
from app.providers.tts.xai import XAIProvider

__all__ = [
    "BaseTTSProvider",
    "TTSProviderError",
    "TTSProviderUnavailableError",
    "TTSResult",
    "TTS_CATALOG",
    "CartesiaProvider",
    "DeepgramProvider",
    "ElevenLabsProvider",
    "GoogleProvider",
    "InworldProvider",
    "SarvamProvider",
    "XAIProvider",
]
