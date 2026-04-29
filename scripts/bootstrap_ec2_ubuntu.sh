#!/usr/bin/env bash
# First-time server setup on Ubuntu 22.04+ (EC2). Run AFTER cloning this repo into the instance.
# Usage (from repo root, as a user with sudo):
#   chmod +x scripts/bootstrap_ec2_ubuntu.sh
#   bash scripts/bootstrap_ec2_ubuntu.sh
#
# Installs: git, ffmpeg, Python venv deps; creates .venv; pip installs; clones SyncNet + weights + patch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run as a normal user with sudo (not root), or adjust apt lines."
fi

sudo apt-get update -y
sudo apt-get install -y \
  git ffmpeg python3 python3-venv python3-pip wget curl \
  libgl1 libglib2.0-0 libegl1 libgles2 || \
  sudo apt-get install -y libgl1 libglib2.0-0 libegl1 libgles2-mesa

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

bash "$REPO_ROOT/scripts/setup_syncnet.sh"
python -m pip install -r "$REPO_ROOT/syncnet_python/requirements.txt"
# syncnet_python upstream pins scenedetect==0.5.1, but this service's run_pipeline
# uses open_video (0.6+ API). Re-upgrade to compatible version.
python -m pip install "scenedetect>=0.6.4"

echo ""
echo "Done. Activate and run API:"
echo "  cd $REPO_ROOT && source .venv/bin/activate"
echo "  cp -n .env.example .env   # then edit .env"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
