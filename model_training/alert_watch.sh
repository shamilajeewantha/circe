#!/usr/bin/env bash
# ACTIVE alarm for the training run - the companion to monitor_training.py.
#
# monitor_training.py only WRITES rows to monitor.log; nothing reads them, so a thermal or disk
# event at 3am would be faithfully recorded and acted on by nobody. This script instead BLOCKS
# until something is actually wrong and then EXITS - which fires a task-completion notification,
# i.e. it actively interrupts rather than passively logging.
#
# Exits (= raises the alarm) on any of:
#   - GPU temp above TEMP_MAX
#   - D: free space below DISK_MIN_GB
#   - the training log going stale for STALE_MIN minutes (process died/hung)
#   - training finishing normally (log says it completed)
#
# Usage: bash alert_watch.sh [TEMP_MAX] [DISK_MIN_GB] [STALE_MIN]

TEMP_MAX=${1:-85}
DISK_MIN_GB=${2:-5}
STALE_MIN=${3:-20}
LOG=/d/my_github/circe/model_training/reports/_v2_train.log

echo "alert_watch armed: temp>${TEMP_MAX}C, D:<${DISK_MIN_GB}GB, log stale>${STALE_MIN}min"

while true; do
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
    free_gb=$(df -BG /d 2>/dev/null | tail -1 | awk '{print $4}' | tr -dc '0-9')

    if [ -n "$temp" ] && [ "$temp" -gt "$TEMP_MAX" ]; then
        echo "ALERT THERMAL: GPU at ${temp}C (limit ${TEMP_MAX}C) at $(date)"
        exit 1
    fi
    if [ -n "$free_gb" ] && [ "$free_gb" -lt "$DISK_MIN_GB" ]; then
        echo "ALERT DISK: D: has ${free_gb}GB free (floor ${DISK_MIN_GB}GB) at $(date)"
        exit 1
    fi
    if [ -f "$LOG" ]; then
        age_min=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
        if [ "$age_min" -gt "$STALE_MIN" ]; then
            echo "ALERT STALLED: training log untouched for ${age_min}min at $(date)"
            exit 1
        fi
        if grep -qa "epochs completed" "$LOG" 2>/dev/null; then
            echo "TRAINING COMPLETE at $(date)"
            exit 0
        fi
    fi
    sleep 60
done
