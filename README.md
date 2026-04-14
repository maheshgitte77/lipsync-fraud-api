# lipsync-fraud-api

FastAPI service: **SyncNet** (audio–video lip sync), **MediaPipe** lip–audio correlation, and optional **proctor signals** (eye / head) for interview videos.

SyncNet upstream code and weight files are **not** stored in this repository. They are installed on the machine (or EC2 instance) at deploy time.

## What goes to GitHub

- Application code under `app/`, `scripts/`, `requirements.txt`, `.env.example`, and this README.
- The folder `syncnet_python/` is listed in `.gitignore`. After `git clone` of this repo, run **`scripts/setup_syncnet.sh`** (Linux) or **`scripts/setup_syncnet.ps1`** (Windows) once to create `syncnet_python/` and download weights.

If you previously committed `syncnet_python` by mistake, stop tracking it (files stay on disk) and commit the updated `.gitignore`:

```bash
git rm -r --cached syncnet_python
git add .gitignore
git commit -m "Stop tracking syncnet_python; use setup script at deploy"
```

## Local setup

**Prerequisites:** Git, Python 3.10+, `ffmpeg` on `PATH`.

1. Clone this repository.
2. Create a virtual environment and install Python dependencies (install **CPU PyTorch** before or with SyncNet deps; see `requirements.txt` comments).

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install --upgrade pip wheel
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

3. Install SyncNet + weights + CPU patches:

   ```bash
   bash scripts/setup_syncnet.sh
   pip install -r syncnet_python/requirements.txt
   ```

   On Windows:

   ```powershell
   .\scripts\setup_syncnet.ps1
   pip install -r syncnet_python\requirements.txt
   ```

4. Copy `.env.example` to `.env` and adjust.

5. Run the API from the repo root:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

Check **`GET /health`**: `model_present` should be `true` when `data/syncnet_v2.model` exists under `syncnet_python`.

### Environment overrides

| Variable | Purpose |
|----------|---------|
| `SYNCNET_DIR` | Path to the SyncNet tree (default: `./syncnet_python`). |
| `SYNCNET_REPO_URL` | Git URL for `setup_syncnet.sh` (default: `https://github.com/joonson/syncnet_python.git`). |
| `SKIP_SYNCNET_PATCH` | Set to `1` to skip `patch_syncnet_cpu.py` (not recommended on CPU servers). |

MediaPipe **Face Landmarker** `.task` files are downloaded on first use into `.cache/` (already ignored).

## AWS EC2 (Ubuntu 22.04)

Typical flow:

1. Launch an instance (x86_64), open security group port **8000** (or put nginx/ALB in front).
2. SSH in, install Git, clone **this** repo (your GitHub URL).
3. From the repo root:

   ```bash
   chmod +x scripts/bootstrap_ec2_ubuntu.sh scripts/setup_syncnet.sh
   bash scripts/bootstrap_ec2_ubuntu.sh
   ```

4. Edit `.env`, then run under a process manager or `tmux`:

   ```bash
   source .venv/bin/activate
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

For **Amazon Linux 2023**, use the same Python venv steps manually: install `git`, `ffmpeg`, `python3`, `python3-pip`, `wget` with `dnf`, then run the `python3 -m venv` / `pip` / `scripts/setup_syncnet.sh` commands from `bootstrap_ec2_ubuntu.sh` (that script is Ubuntu-oriented because of `apt-get`).

## API

- `POST /analyze` — multipart video upload, lip-sync analysis.
- `POST /analyze/proctor-signals` — combined lip sync + eye/head signals (see `PROCTOR_SIGNALS_FIELD_GUIDE.md`).
