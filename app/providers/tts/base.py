"""
Common TTS provider contract.

Every provider returns a :class:`TTSResult` with a path to a WAV file.
The service layer decides which provider to call and handles fallbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.models.tts import TTSVoiceConfig


class TTSProviderError(RuntimeError):
    """Generic provider failure (bad input, API error, etc.)."""


class TTSProviderUnavailableError(TTSProviderError):
    """Raised when a provider is missing credentials / dependency. Caller should skip."""


@dataclass(frozen=True)
class TTSResult:
    audio_path: Path
    provider: str
    sample_rate: int
    duration_sec: float | None = None


class BaseTTSProvider(ABC):
    """Strategy interface. Implement generate() to add a new provider."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check — credentials present, SDK importable."""

    @abstractmethod
    def generate(self, text: str, *, out_path: Path, voice: TTSVoiceConfig | None) -> TTSResult:
        """Generate speech for `text`, write a WAV file to `out_path`, return metadata."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{self.__class__.__name__} name={self.name} available={self.is_available()}>"
