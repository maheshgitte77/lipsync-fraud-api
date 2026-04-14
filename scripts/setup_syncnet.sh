#!/usr/bin/env bash
# Clone Oxford SyncNet demo (joonson/syncnet_python), download weights, apply CPU patch.
# Run from repo root: bash scripts/setup_syncnet.sh
# Env: SYNCNET_DIR (default: <repo>/syncnet_python), SYNCNET_REPO_URL, SKIP_SYNCNET_PATCH=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNCNET_DIR="${SYNCNET_DIR:-$ROOT/syncnet_python}"
REPO="${SYNCNET_REPO_URL:-https://github.com/joonson/syncnet_python.git}"

if [[ -d "$SYNCNET_DIR" && ! -f "$SYNCNET_DIR/run_pipeline.py" ]]; then
  echo "ERROR: $SYNCNET_DIR exists but is not a valid syncnet_python checkout (missing run_pipeline.py)."
  echo "Remove that directory and re-run, or set SYNCNET_DIR to an empty path."
  exit 1
fi

if [[ ! -d "$SYNCNET_DIR" ]]; then
  echo "Cloning SyncNet -> $SYNCNET_DIR"
  git clone --depth 1 "$REPO" "$SYNCNET_DIR"
fi

_dl() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]]; then
    local sz
    sz="$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)"
    if [[ "${sz:-0}" -gt 100000 ]]; then
      echo "Skip (exists): $dest"
      return 0
    fi
  fi
  echo "Download: $url"
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$dest" "$url"
  else
    echo "ERROR: need wget or curl to download models."
    exit 1
  fi
}

DATA="$SYNCNET_DIR/data"
W="$SYNCNET_DIR/detectors/s3fd/weights"
_dl "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model" "$DATA/syncnet_v2.model"
_dl "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi" "$DATA/example.avi"
_dl "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth" "$W/sfd_face.pth"

if [[ "${SKIP_SYNCNET_PATCH:-}" != "1" ]]; then
  echo "Applying CPU / NumPy compatibility patches..."
  python3 "$ROOT/scripts/patch_syncnet_cpu.py" "$SYNCNET_DIR"
else
  echo "SKIP_SYNCNET_PATCH=1 — not running patch_syncnet_cpu.py"
fi

echo "SyncNet ready at $SYNCNET_DIR (set SYNCNET_DIR in .env if you use another path)."
