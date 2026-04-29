"""Public request/response schemas."""

from app.models.proctor import ProctorSignalsRequest
from app.models.tts import TTSRequest

__all__ = ["ProctorSignalsRequest", "TTSRequest"]
