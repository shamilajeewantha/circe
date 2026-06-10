#!/usr/bin/env bash
# Two-laptop mode — YOLO runs on laptop 2, this machine is server only.
# Usage: bash controller/run_remote.sh
#
# Then open: http://localhost:8080

CTRL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECT_MODE=remote

cleanup() {
    echo ""
    echo "[controller] Cleaning up..."
    pkill -f "python3.*server.py" 2>/dev/null || true
    fuser -k 8080/tcp 2>/dev/null || true
    echo "[controller] Log saved (see logs/ directory)."
    echo "[controller] Done."
}
trap cleanup EXIT INT TERM

echo "[controller] Killing any stale controller processes..."
pkill -f "python3.*server.py" 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
sleep 1

source /opt/ros/jazzy/setup.bash
source "$HOME/ws_px4/install/local_setup.bash"

echo "[controller] Installing Python dependencies..."
pip3 install -q fastapi "uvicorn[standard]" opencv-python --break-system-packages

IP=$(hostname -I | awk '{print $1}')
echo "[controller] DETECT_MODE=remote"
echo "[controller] ──────────────────────────────────────────────────"
echo "[controller] On laptop 2, run:"
echo "      bash controller/run_detector.sh ws://$IP:8080"
echo "[controller] ──────────────────────────────────────────────────"
echo "[controller] (firewall: sudo ufw allow 8080/tcp  — remove: sudo ufw delete allow 8080/tcp)"
echo "[controller] Starting server on http://0.0.0.0:8080 ..."
cd "$CTRL_DIR"
mkdir -p "$CTRL_DIR/logs"
LOG_FILE="$CTRL_DIR/logs/controller_$(date +%Y%m%d_%H%M%S).log"
echo "[controller] Full debug log: $LOG_FILE"
python3 -u server.py 2>&1 | tee "$LOG_FILE"
