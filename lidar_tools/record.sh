#!/usr/bin/env bash
# Record lidar motion in the background; keeps running after SSH disconnects.
#   ./record.sh [bag_dir]   start (reuses the driver if run_on_screen.sh already started it)
#   ./record.sh stop        stop and close the bag cleanly
#   ./record.sh status      show the latest progress line
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/last_record.log"
DRIVER_FLAG="$DIR/.driver_started_by_record"

running() { pgrep -f 'lidar_motion_recorde[r]|lidar_motion[.]launch[.]py' >/dev/null; }

wait_gone() {  # wait_gone <pgrep pattern>
    for _ in $(seq 30); do pgrep -f "$1" >/dev/null || return 0; sleep 0.5; done
    return 1
}

case "${1:-start}" in
stop)
    if ! running; then echo "Not recording."; exit 0; fi
    # SIGINT straight to the recorder: that's the path that closes the bag cleanly.
    # (ros2 launch, started detached, dies on SIGTERM without passing it on.)
    pkill -INT -f 'lidar_motion_recorde[r]'
    wait_gone 'lidar_motion_recorde[r]' || echo "Recorder did not exit; check $LOG"
    if [ -f "$DRIVER_FLAG" ]; then
        pkill -INT -f 'sllidar_nod[e]'
        wait_gone 'sllidar_nod[e]'
        rm -f "$DRIVER_FLAG"
    fi
    pkill -TERM -f 'lidar_motion[.]launch[.]py' 2>/dev/null
    grep -E "Saved|ERROR|Traceback" "$LOG" | tail -n 3
    ;;
status)
    if running; then echo "Recording."; else echo "Not recording."; fi
    grep -E "Recording to|Hz \||Saved" "$LOG" 2>/dev/null | tail -n 2
    ;;
*)
    if running; then echo "Already recording. Log: $LOG"; exit 0; fi
    [ "$1" = "start" ] && shift
    source /opt/ros/jazzy/setup.bash
    source "$HOME/ros2_ws/install/setup.bash"  # sllidar_ros2 driver
    if pgrep -f 'sllidar_nod[e]' >/dev/null; then
        DRIVER=false; rm -f "$DRIVER_FLAG"
    else
        DRIVER=true; touch "$DRIVER_FLAG"
    fi
    setsid nohup ros2 launch "$DIR/lidar_motion.launch.py" driver:=$DRIVER ${1:+out:="$1"} \
        > "$LOG" 2>&1 < /dev/null &
    sleep 4
    grep -E "Recording to|ERROR" "$LOG" || echo "Starting... log: $LOG"
    echo "Driver started here: $DRIVER. Stop with: $DIR/record.sh stop"
    ;;
esac
