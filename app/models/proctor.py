"""Request schema for /analyze/proctor-signals and its async variants."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProctorSignalsRequest(BaseModel):
    videoUrl: str = Field(..., min_length=5, description="HTTPS URL to the candidate video")
    questionId: str | None = None
    candidateId: str | None = None
    sampleFps: float | None = None
    offscreenRatioThreshold: float | None = None
    improperHeadRatioThreshold: float | None = None
    repetitivePatternThreshold: int | None = None
    skipSyncNet: bool | None = None
