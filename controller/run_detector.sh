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
CONDA_PY="${CONDA_PY:-}"

if [ -z "$CONDA_PY" ]; then
    CONDA_BIN="$(conda info --base 2>/dev/null)/bin/conda"
    if [ ! -x "$CONDA_BIN" ]; then
        echo "[detector] ERROR: conda not found. Install Miniconda/Anaconda first."
        exit 1
    fi

    ENV_YML="$DIR/environment-detector.yml"
    if ! conda env list | grep -q '^drone_detect '; then
        echo "[detector] 'drone_detect' env not found — creating from $ENV_YML ..."
        echo "[detector] (this installs PyTorch + CUDA + YOLO and may take a few minutes)"
        "$CONDA_BIN" env create -f "$ENV_YML"
        echo "[detector] env created."
    else
        echo "[detector] 'drone_detect' env already exists — skipping create."
    fi

    CONDA_PY="$(conda run -n drone_detect which python3)"
fi

echo "[detector] using python: $CONDA_PY"
echo "[detector] connecting to $SERVER ..."
cd "$DIR"
"$CONDA_PY" detect_client.py --server "$SERVER"
