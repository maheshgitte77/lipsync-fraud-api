"""/tts endpoints — provider list, catalog, generate."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from app.core.logger import get_logger
from app.core.metrics import StageTimer
from app.models.tts import TTSRequest, TTSResponse
from app.providers.tts import TTS_CATALOG
from app.providers.tts.base import TTSProviderError
from app.services.tts import tts_service
from app.utils.file_manager import job_dir, new_job_id
from app.utils.storage import storage_client

logger = get_logger("api.tts")
router = APIRouter(prefix="/tts", tags=["tts"])


_TTS_EXAMPLES = {
    "inworld": {
        "summary": "Inworld (Saanvi voice, en-IN)",
        "value": {
            "text": "What is Java? Explain OOPS and its four important pillars.",
            "provider": "inworld",
            "voice": {
                "voice_id": "Saanvi",
                "model_id": "inworld-tts-1.5-max",
                "language": "en-IN",
                "speed": 1.0,
                "sample_rate": 24000,
            },
            "allow_fallback": True,
        },
    },
    "elevenlabs": {
        "summary": "ElevenLabs (Rachel)",
        "value": {
            "text": "Hello, tell me about yourself.",
            "provider": "elevenlabs",
            "voice": {
                "voice_id": "Rachel",
                "model_id": "eleven_turbo_v2_5",
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
            "allow_fallback": True,
        },
    },
    "deepgram": {
        "summary": "Deepgram Aura (athena/en)",
        "value": {
            "text": "Hello, welcome to the interview.",
            "provider": "deepgram",
            "voice": {
                "voice_id": "athena",
                "model_id": "aura-2",
                "language": "en",
                "sample_rate": 24000,
            },
            "allow_fallback": True,
        },
    },
    "cartesia": {
        "summary": "Cartesia Sonic",
        "value": {
            "text": "Hello from Cartesia.",
            "provider": "cartesia",
            "voice": {
                "voice_id": "f786b574-daa5-4673-aa0c-cbe3e8534c02",
                "model_id": "sonic-english",
                "language": "en",
                "speed": 1.0,
                "sample_rate": 24000,
            },
            "allow_fallback": True,
        },
    },
    "sarvam": {
        "summary": "Sarvam Bulbul v3 (Hindi)",
        "value": {
            "text": "Namaste, aap kaise hain?",
            "provider": "sarvam",
            "voice": {
                "voice_id": "shubh",
                "model_id": "bulbul:v3",
                "language": "hi-IN",
                "speed": 1.0,
                "sample_rate": 24000,
            },
            "allow_fallback": True,
        },
    },
    "xai": {
        "summary": "xAI Grok TTS",
        "value": {
            "text": "Hello from Grok.",
            "provider": "xai",
            "voice": {
                "voice_id": "ara",
                "model_id": "tts-1",
                "language": "auto",
            },
            "allow_fallback": True,
        },
    },
    "google": {
        "summary": "Google Cloud TTS (Neural2)",
        "value": {
            "text": "Hello from Google Cloud.",
            "provider": "google",
            "voice": {
                "voice_id": "en-US-Neural2-F",
                "language": "en-US",
            },
            "allow_fallback": True,
        },
    },
}


@router.get("/providers")
def list_tts_providers() -> dict:
    """Which providers are currently live (credentials present)."""
    return {"providers": tts_service.list_providers()}


@router.get("/catalog")
def tts_catalog() -> dict:
    """
    Full catalog of supported providers, their models, and voice ids.

    Merges the static catalog (valid model/voice ids per provider) with the
    live availability from `/tts/providers` so the frontend can disable
    providers that don't have API keys configured.
    """
    live = {p["name"]: p["available"] for p in tts_service.list_providers()}
    providers = []
    for entry in TTS_CATALOG:
        item = {**entry, "available": bool(live.get(entry["id"], False))}
        providers.append(item)
    return {"providers": providers}


@router.post("/generate", response_model=TTSResponse)
def generate_tts(
    req: TTSRequest = Body(..., openapi_examples=_TTS_EXAMPLES),
) -> TTSResponse:
    jid = new_job_id()
    work = job_dir(jid)
    audio = work / "speech.wav"
    timer = StageTimer()
    try:
        with timer.track("tts"):
            result = tts_service.generate(
                req.text,
                out_path=audio,
                provider=req.provider,
                voice=req.voice,
                allow_fallback=req.allow_fallback,
            )
        with timer.track("upload"):
            url = storage_client.upload(
                result.audio_path, f"{jid}/speech.wav", content_type="audio/wav"
            )
        return TTSResponse(
            audioUrl=url,
            provider=result.provider,
            sampleRate=result.sample_rate,
            durationSec=result.duration_sec,
            metrics=timer.snapshot(),
        )
    except TTSProviderError as exc:
        logger.warning("tts generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)[:2000]) from exc
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
