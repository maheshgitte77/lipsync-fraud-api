"""
TTS provider catalog — voices + models for each provider.

Mirrors `backend-node/src/providers.js::TTS_PROVIDERS` in the AI-calling-platform
project so the frontend can fetch a single source of truth for dropdowns.

Exposed via:  GET /tts/catalog

Voice / model ids here MUST match the ids each provider's REST/SDK accepts
(these ids are passed through as `voice_id` / `model_id` in TTSVoiceConfig).
"""

from __future__ import annotations

TTS_CATALOG: list[dict] = [
    {
        "id": "elevenlabs",
        "name": "ElevenLabs",
        "models": [
            {"id": "eleven_turbo_v2_5", "name": "Turbo v2.5"},
            {"id": "eleven_multilingual_v2", "name": "Multilingual v2"},
            {"id": "eleven_turbo_v2", "name": "Turbo v2"},
        ],
        "voices": [
            {"id": "Rachel", "name": "Rachel"},
            {"id": "Domi", "name": "Domi"},
            {"id": "Bella", "name": "Bella"},
            {"id": "Antoni", "name": "Antoni"},
            {"id": "Elli", "name": "Elli"},
            {"id": "Josh", "name": "Josh"},
        ],
        "voiceHint": "Use voice name or paste voice_id (UUID) from ElevenLabs dashboard",
        "fields": {
            "voice_id": "required (name or UUID)",
            "model_id": "optional",
            "stability": "optional 0–1",
            "similarity_boost": "optional 0–1",
        },
    },
    {
        "id": "cartesia",
        "name": "Cartesia AI",
        "models": [
            {"id": "sonic-english", "name": "Sonic English"},
            {"id": "sonic", "name": "Sonic"},
            {"id": "sonic-turbo", "name": "Sonic Turbo"},
            {"id": "sonic-2", "name": "Sonic 2.0"},
            {"id": "sonic-3", "name": "Sonic 3.0"},
            {"id": "sonic-3-2026-01-12", "name": "Sonic 3.0 (2026-01-12)"},
            {"id": "sonic-3-latest", "name": "Sonic 3.0 Latest (beta)"},
        ],
        "voices": [],
        "voiceHint": (
            "Paste voice_id (UUID) from Cartesia — e.g. "
            "791d5162-d5eb-40f0-8189-f19db44611d8 for Ayush"
        ),
        "fields": {
            "voice_id": "required (UUID)",
            "model_id": "optional (default sonic-english)",
        },
    },
    {
        "id": "deepgram",
        "name": "Deepgram TTS (Aura)",
        "models": [
            {"id": "aura-asteria-en", "name": "Aura Asteria (legacy)"},
            {"id": "aura", "name": "Aura"},
            {"id": "aura-2", "name": "Aura 2"},
        ],
        "voices": [
            {"id": "apollo", "name": "Apollo (male, en-US)"},
            {"id": "athena", "name": "Athena (female, en-US)"},
            {"id": "odysseus", "name": "Odysseus (male, en-US)"},
            {"id": "theia", "name": "Theia (female, en-AU)"},
            {"id": "asteria", "name": "Asteria (female)"},
            {"id": "hera", "name": "Hera (female)"},
            {"id": "zeus", "name": "Zeus (male)"},
        ],
        "voiceHint": "Aura 2 voices: apollo, athena, odysseus, theia",
        "fields": {
            "voice_id": "voice name (e.g. athena) — composed with model & language",
            "model_id": "aura | aura-2 | aura-asteria-en (legacy full id)",
            "language": "optional (default en)",
        },
    },
    {
        "id": "inworld",
        "name": "Inworld",
        "models": [
            {"id": "inworld-tts-1.5-mini", "name": "Inworld TTS 1.5 Mini"},
            {"id": "inworld-tts-1.5-max", "name": "Inworld TTS 1.5 Max"},
            {"id": "inworld-tts-1", "name": "Inworld TTS 1"},
            {"id": "inworld-tts-1-max", "name": "Inworld TTS 1 Max"},
        ],
        "voices": [
            {"id": "Arjun", "name": "Arjun"},
            {"id": "Ashley", "name": "Ashley"},
            {"id": "Diego", "name": "Diego"},
            {"id": "Edward", "name": "Edward"},
            {"id": "Hades", "name": "Hades"},
            {"id": "Liam", "name": "Liam"},
            {"id": "Saanvi", "name": "Saanvi"},
        ],
        "voiceHint": "Voice name from Inworld — Arjun, Ashley, Diego, Saanvi, ...",
        "fields": {
            "voice_id": "voice name (e.g. Saanvi)",
            "model_id": "optional (default inworld-tts-1.5-mini)",
        },
    },
    {
        "id": "xai",
        "name": "xAI (Grok TTS)",
        "models": [{"id": "tts-1", "name": "TTS 1"}],
        "voices": [
            {"id": "ara", "name": "Ara (warm, friendly)"},
            {"id": "eve", "name": "Eve (energetic, upbeat)"},
            {"id": "rex", "name": "Rex (confident, clear)"},
            {"id": "sal", "name": "Sal (smooth, balanced)"},
            {"id": "leo", "name": "Leo (authoritative)"},
        ],
        "voiceHint": "Voice: ara, eve, rex, sal, leo. XAI_API_KEY from console.x.ai",
        "fields": {
            "voice_id": "voice name (ara/eve/rex/sal/leo)",
            "language": "optional (default auto)",
        },
    },
    {
        "id": "sarvam",
        "name": "Sarvam AI",
        "models": [
            {"id": "bulbul:v3", "name": "Bulbul v3"},
            {"id": "bulbul:v3-beta", "name": "Bulbul v3 Beta"},
            {"id": "bulbul:v2", "name": "Bulbul v2"},
        ],
        "voices": [
            {"id": "shubh", "name": "Shubh (default v3)"},
            {"id": "anushka", "name": "Anushka (v2)"},
            {"id": "aditya", "name": "Aditya"},
            {"id": "ritu", "name": "Ritu"},
            {"id": "priya", "name": "Priya"},
            {"id": "neha", "name": "Neha"},
            {"id": "rahul", "name": "Rahul"},
            {"id": "abhilash", "name": "Abhilash (v2)"},
            {"id": "manisha", "name": "Manisha (v2)"},
            {"id": "vidya", "name": "Vidya (v2)"},
        ],
        "voiceHint": "Speaker: shubh (v3), anushka (v2). See docs.sarvam.ai",
        "languageHint": "Target language: hi-IN, en-IN, ta-IN, etc.",
        "fields": {
            "voice_id": "speaker name (shubh/anushka/...)",
            "model_id": "bulbul:v3 | bulbul:v2 | bulbul:v3-beta",
            "language": "required target_language_code (e.g. hi-IN)",
        },
    },
    {
        "id": "google",
        "name": "Google Cloud TTS",
        "models": [],
        "voices": [
            {"id": "en-US-Neural2-F", "name": "en-US-Neural2-F (female)"},
            {"id": "en-US-Neural2-J", "name": "en-US-Neural2-J (male)"},
            {"id": "en-IN-Neural2-A", "name": "en-IN-Neural2-A"},
            {"id": "hi-IN-Neural2-A", "name": "hi-IN-Neural2-A"},
        ],
        "voiceHint": "Paste any Google Cloud TTS voice name",
        "fields": {
            "voice_id": "Google voice name (e.g. en-US-Neural2-F)",
            "language": "language_code (e.g. en-US)",
            "speed": "optional speaking_rate",
        },
    },
]
