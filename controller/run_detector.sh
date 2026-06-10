#!/usr/bin/env bash
# Remote YOLO detector — run this on laptop 2 (the GPU machine).
# It connects to the controller server on laptop 1, pulls JPEG frames, runs YOLO,
# and sends bounding boxes back.
#
# Usage:
#   bash run_detector.sh ws://<laptop1-ip>:8080
#
# Find laptop 1's IP by running `hostname -I` on laptop 1.
set -e

SERVER="${1:?Usage: bash run_detector.sh ws://<laptop1-ip>:8080}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda python that has ultralytics (GPU torch) + opencv + websockets.
# Override if your env lives elsewhere:  CONDA_PY=/path/to/python bash run_detector.sh ...
CONDA_PY="${CONDA_PY:-/home/$USER/anaconda3/envs/drone_detect/bin/python3}"

echo "[detector] using python: $CONDA_PY"
# Make sure the lighter deps are present (torch/torchvision should already be the
# CUDA build — see controller/requirements-detector.txt for the one-time install).
"$CONDA_PY" -m pip install -q websockets opencv-python ultralytics
echo "[detector] connecting to $SERVER ..."
cd "$DIR"
"$CONDA_PY" detect_client.py --server "$SERVER"
