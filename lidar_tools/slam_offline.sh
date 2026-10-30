#!/usr/bin/env bash
# Run a recorded motion bag through slam_toolbox (scan matching + loop closure) and
# write a corrected copy with the same topics, so readers of the original work unchanged.
#   ./slam_offline.sh ~/lidar_bags/run4_fixed ~/lidar_bags/run4_slam [play_rate]
# Output: <out>/ (corrected bag), <out>_map.pgm / .yaml / .npy (final SLAM map)
set -e
set -m  # job control: background jobs keep normal SIGINT handling, so they stop cleanly
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$(realpath "$1")"; OUT="$(realpath -m "$2")"; RATE="${3:-1.0}"
TMP="$(mktemp -d)"
[ -e "$OUT" ] && { echo "$OUT already exists"; exit 1; }
source /opt/ros/jazzy/setup.bash

echo "=== Starting slam_toolbox (sync mode, every scan processed)"
# Run the binary directly (not via `ros2 run`) so signals reach the node itself.
/opt/ros/jazzy/lib/slam_toolbox/sync_slam_toolbox_node --ros-args \
    --params-file "$DIR/slam_params.yaml" > "$TMP/slam.log" 2>&1 &
SLAM=$!
stop() {  # stop <pid>: SIGINT, then SIGKILL if it hasn't exited after 5 s
    kill -INT "$1" 2>/dev/null || return 0
    for _ in $(seq 10); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.5; done
    kill -KILL "$1" 2>/dev/null || true
}
trap 'stop $SLAM' EXIT
sleep 3
# Jazzy's slam_toolbox is a lifecycle node: it does nothing until configured and activated.
ros2 lifecycle set /slam_toolbox configure > /dev/null
ros2 lifecycle set /slam_toolbox activate > /dev/null
echo "    slam_toolbox state: $(ros2 lifecycle get /slam_toolbox)"

echo "=== Recording SLAM output"
ros2 bag record -s mcap -o "$TMP/slam_out" /tf /map --use-sim-time > "$TMP/record.log" 2>&1 &
REC=$!
sleep 3

echo "=== Replaying $IN at ${RATE}x (only /scan and /tf)"
ros2 bag play "$IN" --clock --rate "$RATE" --topics /scan /tf > "$TMP/play.log" 2>&1
sleep 5  # let the last scans and map update go out

stop $REC
stop $SLAM
grep -iE "loop|error" "$TMP/slam.log" | tail -n 5 || true

echo "=== Building corrected bag"
python3 "$DIR/slam_apply.py" "$IN" "$TMP/slam_out" "$OUT"
rm -rf "$TMP"
