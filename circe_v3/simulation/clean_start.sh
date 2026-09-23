#!/usr/bin/env bash
# Kill every stale sim/brain process and reset the ROS 2 daemon.
#
# RUN THIS BEFORE EVERY LAUNCH. Not optional.
#
# Why: leftover processes from a previous run do NOT announce themselves, they just
# quietly corrupt the next run. Real cases from this project:
#   * three stale ros_gz_bridge + robot_state_publisher instances (pids 5820, 41554,
#     69598) accumulated across runs, all publishing the SAME /rover/* topics. Sensor
#     rates collapsed and ROS looked "crashed".
#   * an orphaned brain kept streaming frames into the SLAM server, so a "fresh" run's
#     session counters were polluted by a process nobody knew was running.
#   * two `gz sim server` instances competed for the GPU; sensors silently stopped
#     publishing while every node looked healthy.
# Each of these cost a debugging session chasing the wrong thing.
#
# Usage:  ./clean_start.sh            # clean only
#         ./clean_start.sh --verify   # clean, then assert nothing is left
set -uo pipefail

PATTERNS='gz sim|lib/circe_|ros2 launch|parameter_bridge|robot_state_publisher|ros_gz|client_node|driver_node|mapping_node|coverage_node|explore_node|localization_node|circe_viz'

echo "[clean_start] surveying..."
ps aux | grep -E "$PATTERNS" | grep -v grep | awk '{print "  " $2, $11, $12}' || true

PIDS=$(ps aux | grep -E "$PATTERNS" | grep -v grep | awk '{print $2}')
if [ -n "${PIDS:-}" ]; then
    echo "[clean_start] killing: $(echo "$PIDS" | tr '\n' ' ')"
    # TERM first so Gazebo can release the GPU/ports, then hard kill the stragglers
    echo "$PIDS" | xargs -r kill 2>/dev/null
    sleep 3
    PIDS2=$(ps aux | grep -E "$PATTERNS" | grep -v grep | awk '{print $2}')
    [ -n "${PIDS2:-}" ] && { echo "$PIDS2" | xargs -r kill -9 2>/dev/null; sleep 2; }
else
    echo "[clean_start] nothing running"
fi

# The daemon caches the topic graph; after killing publishers it will happily serve a
# stale graph and `ros2 topic hz` then reports phantom or missing topics.
if command -v ros2 >/dev/null 2>&1; then
    echo "[clean_start] resetting ROS 2 daemon"
    ros2 daemon stop  >/dev/null 2>&1 || true
    sleep 2
    ros2 daemon start >/dev/null 2>&1 || true
fi

LEFT=$(ps aux | grep -E "$PATTERNS" | grep -v grep | wc -l)
echo "[clean_start] remaining processes: $LEFT"

if [ "${1:-}" = "--verify" ] && [ "$LEFT" -ne 0 ]; then
    echo "[clean_start] FAILED - processes survived:"
    ps aux | grep -E "$PATTERNS" | grep -v grep
    exit 1
fi
echo "[clean_start] OK - safe to launch"
