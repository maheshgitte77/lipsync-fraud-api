# lipsync-fraud-api

Production-grade FastAPI service for:

- **Lip-sync authenticity** — SyncNet (+ optional MediaPipe correlation).
- **Proctor signals** — eye gaze + head pose + reading-pattern detection.
- **TTS** — pluggable providers (ElevenLabs, Google, Deepgram, Cartesia, Inworld) with fallback.
- **Avatar generation** — HeyGen cloud avatar video generation.

Async-ready: Kafka job queue + Redis job store. GPU/CPU toggle via env. S3 / local artifact storage.

---

## Architecture

```
            ┌──────────────────────────────┐
            │         FastAPI (app/)       │
            │   api/ routes delegate to    │
            │   services/ (no ML here)     │
            └──────────────┬───────────────┘
                           │
     ┌─────────────────────┼─────────────────────────┐
     ▼                     ▼                         ▼
services/lipsync     services/fraud           services/heygen
  ├─ syncnet_service   └─ proctor_service       └─ heygen_service
  ├─ mediapipe_service
  └─ window_builder                                         │
     │                                                       ▼
     │                                               services/tts
     │                                                 └─ tts_service
     │                                                        │
     │                                                        ▼
     │                                               providers/tts/
     │                                                 ├─ elevenlabs
     │                                                 ├─ google
     │                                                 ├─ deepgram
     │                                                 ├─ cartesia
     │                                                 └─ inworld
     ▼
workers/ (kafka_worker, job_store) ──── utils/ (ffmpeg, storage, video_download)
core/   (config, logger, device, metrics)
```

### Folder layout

```
lipsync-fraud-api/
├── app/
│   ├── main.py                      # thin FastAPI entry (lifespan + router)
│   ├── api/                         # HTTP routes
│   │   ├── routes.py                # aggregates sub-routers
│   │   ├── health.py                # /, /health, /config
│   │   ├── analyze.py               # /analyze, /analyze/proctor-signals*
│   │   ├── tts.py                   # /tts/generate, /tts/providers
│   │   └── heygen.py                # /heygen/generate, /heygen/providers
│   ├── core/
│   │   ├── config.py                # centralized Settings (dataclasses)
│   │   ├── logger.py                # get_logger(name)
│   │   ├── device.py                # DeviceManager (CPU/GPU toggle)
│   │   └── metrics.py               # StageTimer (per-stage ms)
│   ├── models/                      # pydantic request/response schemas
│   │   ├── proctor.py
│   │   ├── tts.py
│   │   └── heygen.py
│   ├── services/
│   │   ├── lipsync/                 # SyncNet + MediaPipe correlation
│   │   ├── fraud/                   # eye + head proctor analysis
│   │   ├── tts/                     # strategy dispatcher + fallback chain
│   │   ├── heygen/                  # HeyGen generation + polling
│   │   └── orchestration/           # proctor_orchestrator (download→run→fuse)
│   ├── providers/tts/               # one file per TTS provider
│   ├── workers/                     # kafka_worker, job_store
│   └── utils/                       # ffmpeg, video_download, file_manager, storage
├── syncnet_python/                  # legacy location, still supported via SYNCNET_DIR
├── scripts/                         # setup helpers, CI/dev ops
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

```bash
# 1. Install python deps
pip install -r requirements.txt
# Install torch matching your platform:
#   CPU:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Bring SyncNet into place (clone + weights)
bash scripts/setup_syncnet.sh    # or scripts/setup_syncnet.ps1 on Windows

# 3. Copy env & edit
cp .env.example .env

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## GPU / CPU toggle

Switch runtime by setting `DEVICE` in `.env`:

| DEVICE | Behaviour                                           |
| ------ | --------------------------------------------------- |
| `auto` | Use CUDA if available, else CPU (default).          |
| `gpu`  | Force CUDA. Logs a warning + uses CPU if missing.    |
| `cpu`  | Force CPU. Also passes `CUDA_VISIBLE_DEVICES=` to subprocesses. |

`GPU_ID` picks the CUDA device when multiple GPUs are present. Runtime info is
exposed at `GET /health` and `GET /config`.

---

## Adding a TTS provider

1. Add a file under `app/providers/tts/your_provider.py` implementing
   `BaseTTSProvider` (`is_available()`, `generate(text, out_path, voice) -> TTSResult`).
2. Register it in `app/providers/tts/__init__.py` and in
   `TTSService._default_providers()`.
3. Add config fields to `app.core.config.TTSSettings` and `.env.example`.

Selection is controlled via `TTS_DEFAULT_PROVIDER` and `TTS_FALLBACK_ORDER`.
The service tries providers in order and skips unavailable ones.

---

## HTTP endpoints

| Method | Path                                           | Purpose                               |
| ------ | ---------------------------------------------- | ------------------------------------- |
| GET    | `/`                                            | Service summary                        |
| GET    | `/health`                                      | Readiness + device + models check      |
| GET    | `/config`                                      | Non-secret runtime config              |
| POST   | `/analyze` (multipart)                         | SyncNet verdict from an uploaded video |
| POST   | `/analyze/proctor-signals`                     | Sync proctor analysis (downloads URL)  |
| POST   | `/analyze/proctor-signals/submit`              | Enqueue to Kafka, returns jobId        |
| GET    | `/analyze/proctor-signals/jobs/{job_id}`       | Poll job state                         |
| GET    | `/tts/providers`                               | Which TTS providers are configured     |
| POST   | `/tts/generate`                                | Text → audio file URL                  |
| GET    | `/heygen/providers`                            | HeyGen availability + defaults          |
| POST   | `/heygen/generate`                             | Text/script → cloud avatar video URL    |

---

## Scaling the service

- **Stateless API**: run multiple replicas behind a load balancer.
- **Kafka worker pool**: `PROCTOR_KAFKA_START_WORKER=true` on worker pods, `false` on API pods.
- **Redis**: shared job state across replicas.
- **GPU pool**: not required for avatar generation now (HeyGen is cloud-based).
- **S3 storage**: set `STORAGE_BACKEND=s3` for signed avatar URLs returned to clients.

---

## Migration notes (v1 → v2)

- All runtime config now comes from `app.core.config.settings`. **Do not read `os.environ` in service code.**
- `app/main.py` no longer contains business logic — only ASGI wiring.
- `app/proctor_signals.py` → `app/services/fraud/proctor_service.py`.
- `app/mediapipe_lipsync.py` → `app/services/lipsync/mediapipe_service.py`.
- Kafka / Redis code → `app/workers/*`.
- FFmpeg helpers → `app/utils/ffmpeg.py`.
- SyncNet script interface unchanged — still driven by `syncnet_python/`.
