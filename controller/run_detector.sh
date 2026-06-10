#!/usr/bin/env bash
# Remote YOLO detector — run this on laptop 2 (the GPU machine).
# It connects to the controller server on laptop 1, pulls JPEG frames, runs YOLO,
# and sends bounding boxes back.
#
# Usage:
#   bash run_detector.sh ws://<laptop1-ip>:8080
#
# One-time setup on laptop 2:
#   conda env create -f controller/environment-detector.yml
#
# Find laptop 1's IP by running `hostname -I` on laptop 1.
set -e

# Laptop 1 IP — edit this once, then just run: bash run_detector.sh
LAPTOP1_IP="192.168.1.5"

SERVER="${1:-ws://$LAPTOP1_IP:8080}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda python — override with CONDA_PY=/path/to/python if your env lives elsewhere.
CONDA_PY="${CONDA_PY:-/home/$USER/anaconda3/envs/drone_detect/bin/python3}"

if [ ! -x "$CONDA_PY" ]; then
    echo "[detector] ERROR: python not found at: $CONDA_PY"
    echo "[detector] Create the env first:"
    echo "      conda env create -f controller/environment-detector.yml"
    echo "[detector] Then re-run this script."
    exit 1
fi

echo "[detector] using python: $CONDA_PY"
echo "[detector] connecting to $SERVER ..."
cd "$DIR"
"$CONDA_PY" detect_client.py --server "$SERVER"
