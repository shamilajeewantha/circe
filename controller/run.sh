#!/usr/bin/env bash
# Launch the drone + rover web controller.
# The sim stack (launch_baylands.sh or launch_default.sh) must already be running.
#
# Usage:
#   bash ~/github_desktop/circe/controller/run.sh            # single laptop (local YOLO)
#   bash ~/github_desktop/circe/controller/run.sh --remote   # offload YOLO to laptop 2
#
# Then open:  http://localhost:8080
# Over SSH:   ssh -L 8080:localhost:8080 user@host  → http://localhost:8080

CTRL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detection mode: local (on-board YOLO) or remote (a detector on another laptop).
DETECT_MODE="${DETECT_MODE:-local}"
[ "$1" = "--remote" ] && DETECT_MODE=remote
export DETECT_MODE

cleanup() {
    echo ""
    echo "[controller] Cleaning up..."
    pkill -f "python3.*server.py" 2>/dev/null || true
    fuser -k 8080/tcp 2>/dev/null || true
    echo "[controller] Log saved (see logs/ directory)."
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

echo "[controller] DETECT_MODE=$DETECT_MODE"
if [ "$DETECT_MODE" = "remote" ]; then
    IP=$(hostname -I | awk '{print $1}')
    echo "[controller] REMOTE detection — on the OTHER laptop run:"
    echo "      bash run_detector.sh ws://$IP:8080"
    echo "[controller] (if you can't connect, open the firewall: sudo ufw allow 8080/tcp)"
fi

echo "[controller] Starting server on http://0.0.0.0:8080 ..."
cd "$CTRL_DIR"
mkdir -p "$CTRL_DIR/logs"
LOG_FILE="$CTRL_DIR/logs/controller_$(date +%Y%m%d_%H%M%S).log"
echo "[controller] Full debug log: $LOG_FILE"
python3 -u server.py 2>&1 | tee "$LOG_FILE"
