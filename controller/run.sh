#!/usr/bin/env bash
# Launch the drone + rover web controller.
# The sim stack (launch_baylands.sh or launch_default.sh) must already be running.
#
# Usage:
#   bash ~/github_desktop/circe/controller/run.sh
#
# Then open:  http://localhost:8080
# Over SSH:   ssh -L 8080:localhost:8080 user@host  → http://localhost:8080

CTRL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
    echo ""
    echo "[controller] Cleaning up..."
    pkill -f "python3.*server.py" 2>/dev/null || true
    fuser -k 8080/tcp 2>/dev/null || true
    echo "[controller] Done."
}
trap cleanup EXIT INT TERM

# ── Kill anything already running on port 8080 or a stale server.py ────────────
echo "[controller] Killing any stale controller processes..."
pkill -f "python3.*server.py" 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
sleep 1

source /opt/ros/jazzy/setup.bash
source "$HOME/ws_px4/install/local_setup.bash"

echo "[controller] Installing Python dependencies..."
pip3 install -q fastapi "uvicorn[standard]" opencv-python --break-system-packages

echo "[controller] Starting server on http://0.0.0.0:8080 ..."
cd "$CTRL_DIR"
python3 server.py
