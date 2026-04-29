"""
Centralized configuration for the lipsync-fraud / avatar service.

All runtime knobs live here. Never read os.environ elsewhere — always
`from app.core.config import settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SYNCNET = _PROJECT_ROOT / "syncnet_python"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


DeviceMode = Literal["cpu", "gpu", "auto"]


@dataclass(frozen=True)
class DeviceSettings:
    """GPU / CPU runtime selection."""
    mode: DeviceMode = field(default_factory=lambda: _env_str("DEVICE", "auto").lower() or "auto")  # type: ignore
    gpu_id: int = field(default_factory=lambda: _env_int("GPU_ID", 0))
    torch_allow_tf32: bool = field(default_factory=lambda: _env_bool("TORCH_ALLOW_TF32", True))


@dataclass(frozen=True)
class PathSettings:
    project_root: Path = _PROJECT_ROOT
    syncnet_dir: Path = field(
        default_factory=lambda: Path(_env_str("SYNCNET_DIR", str(_DEFAULT_SYNCNET))).resolve()
    )
    temp_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("LIPSYNC_TEMP_DIR", str(_PROJECT_ROOT / "temp_videos"))
        ).resolve()
    )
    ffmpeg: str = field(default_factory=lambda: _env_str("LIPSYNC_FFMPEG", "ffmpeg") or "ffmpeg")
    ffprobe: str = field(default_factory=lambda: _env_str("LIPSYNC_FFPROBE", "ffprobe") or "ffprobe")


@dataclass(frozen=True)
class SyncNetSettings:
    min_dist_pass: float = field(default_factory=lambda: _env_float("MIN_DIST_PASS", 6.0))
    confidence_pass: float = field(default_factory=lambda: _env_float("CONFIDENCE_PASS", 3.0))
    trim_enabled: bool = field(default_factory=lambda: _env_bool("LIPSYNC_VIDEO_TRIM", False))
    trim_max_seconds: float = field(
        default_factory=lambda: _env_float("LIPSYNC_TRIM_MAX_SECONDS", 15.0)
    )
    window_position: Literal["start", "middle", "end"] = field(
        default_factory=lambda: (_env_str("LIPSYNC_WINDOW_POSITION", "start").lower() or "start")  # type: ignore
    )
    download_timeout_sec: int = field(
        default_factory=lambda: _env_int("LIPSYNC_DOWNLOAD_TIMEOUT_SEC", 600)
    )
    download_retries: int = field(
        default_factory=lambda: _env_int("LIPSYNC_DOWNLOAD_RETRIES", 3)
    )
    debug_log: bool = field(default_factory=lambda: _env_bool("LIPSYNC_DEBUG_LOG", False))

    # run_pipeline.py extras
    frame_rate: str = field(default_factory=lambda: _env_str("SYNCNET_FRAME_RATE", ""))
    min_track: str = field(default_factory=lambda: _env_str("SYNCNET_MIN_TRACK", ""))
    facedet_scale: str = field(default_factory=lambda: _env_str("SYNCNET_FACEDET_SCALE", ""))
    crop_scale: str = field(default_factory=lambda: _env_str("SYNCNET_CROP_SCALE", ""))
    num_failed_det: str = field(default_factory=lambda: _env_str("SYNCNET_NUM_FAILED_DET", ""))
    min_face_size: str = field(default_factory=lambda: _env_str("SYNCNET_MIN_FACE_SIZE", ""))

    # run_syncnet.py extras
    batch_size: str = field(default_factory=lambda: _env_str("SYNCNET_BATCH_SIZE", ""))
    vshift: str = field(default_factory=lambda: _env_str("SYNCNET_VSHIFT", ""))


@dataclass(frozen=True)
class ProctorSettings:
    sample_fps: float = field(default_factory=lambda: _env_float("PROCTOR_SAMPLE_FPS", 1.0))
    offscreen_ratio_threshold: float = field(
        default_factory=lambda: _env_float("PROCTOR_OFFSCREEN_RATIO_THRESHOLD", 0.35)
    )
    improper_head_ratio_threshold: float = field(
        default_factory=lambda: _env_float("PROCTOR_IMPROPER_HEAD_RATIO_THRESHOLD", 0.35)
    )
    repetitive_pattern_threshold: int = field(
        default_factory=lambda: _env_int("PROCTOR_REPETITIVE_PATTERN_THRESHOLD", 6)
    )
    lipsync_flag_source: str = field(
        default_factory=lambda: (
            _env_str("PROCTOR_LIPSYNC_FLAG_SOURCE", "syncnet_only").lower() or "syncnet_only"
        )
    )
    skip_syncnet: bool = field(default_factory=lambda: _env_bool("PROCTOR_SKIP_SYNCNET", False))
    max_frame_width: int = field(
        default_factory=lambda: _env_int("PROCTOR_MAX_FRAME_WIDTH", 480)
    )


@dataclass(frozen=True)
class KafkaSettings:
    enabled: bool = field(default_factory=lambda: _env_bool("PROCTOR_KAFKA_ENABLED", False))
    start_worker: bool = field(
        default_factory=lambda: _env_bool("PROCTOR_KAFKA_START_WORKER", False)
    )
    brokers: str = field(
        default_factory=lambda: (
            _env_str("PROCTOR_KAFKA_BROKERS")
            or _env_str("KAFKA_BROKERS")
            or _env_str("KAFKA_BROKER")
            or "localhost:9092"
        )
    )
    request_topic: str = field(
        default_factory=lambda: _env_str("PROCTOR_KAFKA_REQUEST_TOPIC", "proctor.signals.requests")
    )
    result_topic: str = field(
        default_factory=lambda: _env_str("PROCTOR_KAFKA_RESULT_TOPIC", "proctor.signals.results")
    )
    group: str = field(
        default_factory=lambda: _env_str("PROCTOR_KAFKA_GROUP", "lipsync-fraud-proctor-workers")
    )


@dataclass(frozen=True)
class RedisSettings:
    url: str = field(default_factory=lambda: _env_str("REDIS_URL", ""))
    host: str = field(default_factory=lambda: _env_str("REDIS_HOST", ""))
    port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    username: str = field(default_factory=lambda: _env_str("REDIS_USERNAME", ""))
    password: str = field(default_factory=lambda: _env_str("REDIS_PASSWORD", ""))
    job_store_prefix: str = field(
        default_factory=lambda: _env_str("PROCTOR_JOB_STORE_PREFIX", "proctor:job")
    )
    job_ttl_sec: int = field(default_factory=lambda: _env_int("PROCTOR_JOB_TTL_SEC", 86400))


@dataclass(frozen=True)
class TTSSettings:
    default_provider: str = field(
        default_factory=lambda: _env_str("TTS_DEFAULT_PROVIDER", "elevenlabs").lower()
    )
    fallback_order: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip().lower()
            for p in (
                _env_str(
                    "TTS_FALLBACK_ORDER",
                    "elevenlabs,cartesia,deepgram,inworld,xai,sarvam,google",
                ).split(",")
            )
            if p.strip()
        )
    )
    audio_sample_rate: int = field(
        default_factory=lambda: _env_int("TTS_AUDIO_SAMPLE_RATE", 24000)
    )
    # ElevenLabs
    elevenlabs_api_key: str = field(default_factory=lambda: _env_str("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(
        default_factory=lambda: _env_str("ELEVENLABS_VOICE_ID", "Rachel")
    )
    elevenlabs_model_id: str = field(
        default_factory=lambda: _env_str("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
    )
    # Google Cloud TTS
    google_credentials_json: str = field(
        default_factory=lambda: _env_str("GOOGLE_APPLICATION_CREDENTIALS", "")
    )
    google_voice: str = field(default_factory=lambda: _env_str("GOOGLE_TTS_VOICE", "en-US-Neural2-F"))
    google_language: str = field(default_factory=lambda: _env_str("GOOGLE_TTS_LANGUAGE", "en-US"))
    # Deepgram Aura (model carries voice, e.g. aura-2-athena-en; we also accept
    # model + voice separately and compose).
    deepgram_api_key: str = field(default_factory=lambda: _env_str("DEEPGRAM_API_KEY", ""))
    deepgram_model: str = field(
        default_factory=lambda: _env_str("DEEPGRAM_TTS_MODEL", "aura-2")
    )
    deepgram_voice: str = field(
        default_factory=lambda: _env_str("DEEPGRAM_TTS_VOICE", "athena")
    )
    deepgram_language: str = field(
        default_factory=lambda: _env_str("DEEPGRAM_TTS_LANGUAGE", "en")
    )
    # Cartesia
    cartesia_api_key: str = field(default_factory=lambda: _env_str("CARTESIA_API_KEY", ""))
    cartesia_voice_id: str = field(
        default_factory=lambda: _env_str(
            "CARTESIA_VOICE_ID", "f786b574-daa5-4673-aa0c-cbe3e8534c02"
        )
    )
    cartesia_model: str = field(
        default_factory=lambda: _env_str("CARTESIA_MODEL", "sonic-english")
    )
    # Inworld
    inworld_api_key: str = field(default_factory=lambda: _env_str("INWORLD_API_KEY", ""))
    inworld_voice_id: str = field(
        default_factory=lambda: _env_str("INWORLD_VOICE_ID", "Arjun")
    )
    inworld_model: str = field(
        default_factory=lambda: _env_str("INWORLD_MODEL", "inworld-tts-1.5-mini")
    )
    # xAI (Grok TTS)
    xai_api_key: str = field(default_factory=lambda: _env_str("XAI_API_KEY", ""))
    xai_voice: str = field(default_factory=lambda: _env_str("XAI_TTS_VOICE", "ara"))
    xai_language: str = field(default_factory=lambda: _env_str("XAI_TTS_LANGUAGE", "auto"))
    xai_model: str = field(default_factory=lambda: _env_str("XAI_TTS_MODEL", "tts-1"))
    # Sarvam
    sarvam_api_key: str = field(default_factory=lambda: _env_str("SARVAM_API_KEY", ""))
    sarvam_speaker: str = field(default_factory=lambda: _env_str("SARVAM_TTS_SPEAKER", "shubh"))
    sarvam_model: str = field(default_factory=lambda: _env_str("SARVAM_TTS_MODEL", "bulbul:v3"))
    sarvam_language: str = field(
        default_factory=lambda: _env_str("SARVAM_TTS_LANGUAGE", "en-IN")
    )


@dataclass(frozen=True)
class HeyGenSettings:
    """HeyGen v2 avatar video generation API."""
    api_key: str = field(default_factory=lambda: _env_str("HEYGEN_API_KEY", ""))
    base_url: str = field(default_factory=lambda: _env_str("HEYGEN_BASE_URL", "https://api.heygen.com"))
    default_avatar_id: str = field(default_factory=lambda: _env_str("HEYGEN_DEFAULT_AVATAR_ID", ""))
    default_voice_id: str = field(default_factory=lambda: _env_str("HEYGEN_DEFAULT_VOICE_ID", ""))
    default_width: int = field(default_factory=lambda: _env_int("HEYGEN_DEFAULT_WIDTH", 1280))
    default_height: int = field(default_factory=lambda: _env_int("HEYGEN_DEFAULT_HEIGHT", 720))
    request_timeout: int = field(default_factory=lambda: _env_int("HEYGEN_REQUEST_TIMEOUT", 60))
    poll_interval_sec: float = field(default_factory=lambda: _env_float("HEYGEN_POLL_INTERVAL_SEC", 5.0))
    poll_max_wait_sec: int = field(default_factory=lambda: _env_int("HEYGEN_POLL_MAX_WAIT_SEC", 900))
    mirror_to_storage: bool = field(default_factory=lambda: _env_bool("HEYGEN_MIRROR_TO_STORAGE", False))


@dataclass(frozen=True)
class StorageSettings:
    """Where final artifacts go (S3 / local)."""
    backend: Literal["local", "s3"] = field(
        default_factory=lambda: (_env_str("STORAGE_BACKEND", "local").lower() or "local")  # type: ignore
    )
    local_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("STORAGE_LOCAL_DIR", str(_PROJECT_ROOT / "output"))
        ).resolve()
    )
    s3_bucket: str = field(default_factory=lambda: _env_str("S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: _env_str("S3_PREFIX", "avatars/"))
    s3_region: str = field(default_factory=lambda: _env_str("AWS_REGION", "ap-south-1"))
    s3_signed_url_ttl: int = field(default_factory=lambda: _env_int("S3_SIGNED_URL_TTL", 3600))


@dataclass(frozen=True)
class Settings:
    """Root configuration container."""
    app_name: str = "lipsync-fraud-api"
    version: str = "2.0.0"
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())
    device: DeviceSettings = field(default_factory=DeviceSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    syncnet: SyncNetSettings = field(default_factory=SyncNetSettings)
    proctor: ProctorSettings = field(default_factory=ProctorSettings)
    kafka: KafkaSettings = field(default_factory=KafkaSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    heygen: HeyGenSettings = field(default_factory=HeyGenSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)

    def ensure_directories(self) -> None:
        """Create runtime directories that services may write to."""
        self.paths.temp_dir.mkdir(parents=True, exist_ok=True)
        if self.storage.backend == "local":
            self.storage.local_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
