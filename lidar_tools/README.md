# lidar_tools — recording RPLIDAR C1 runs on the Pi

Code that runs on the Raspberry Pi 4 (Ubuntu 24.04, ROS 2 Jazzy) to record lidar runs
for the floor-plan pipeline in `PythonApplication1/`. Copy this folder to `~/lidar_motion`
on the Pi. The lidar driver is Slamtec's
[sllidar_ros2](https://github.com/Slamtec/sllidar_ros2), built in `~/ros2_ws`; the apt
`rplidar_ros` package can't start scans on the C1.

## Recording a run
```bash
~/lidar_motion/record.sh ~/lidar_bags/run5      # start (keeps going if SSH drops)
~/lidar_motion/record.sh status                 # pose, match %, lost scans
~/lidar_motion/record.sh stop                   # stop and close the bag cleanly
```
Start with the cart still for 2–3 s; the first scan sets the origin (0, 0, 0°).

Each bag (`metadata.yaml` + `.mcap`) holds, for every scan, all with the scan's timestamp:

| Topic | Type | Content |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | raw reading, 720 beams, 10 Hz |
| `/lidar_odom` | `nav_msgs/Odometry` | lidar pose (x, y, yaw) from scan matching |
| `/lidar_delta` | `geometry_msgs/TransformStamped` | motion since the previous scan |
| `/tf` | `tf2_msgs/TFMessage` | `odom -> laser` (`map -> laser` after SLAM) |

`ros_bag_reader.py` reads these directly (`/scan` + `/lidar_odom`).

## Corridors and long runs: correct with SLAM
Scan matching alone drifts in long corridors. After recording, run the bag through
slam_toolbox (loop closure) to get a corrected copy with the same topics:
```bash
~/lidar_motion/slam_offline.sh ~/lidar_bags/run5 ~/lidar_bags/run5_slam
```
Also writes `run5_slam_map.pgm/.yaml/.npy`. Takes about 1.2× the run's length on the Pi.

## Other tools
- `reprocess_bag.py IN OUT` — recompute poses from a bag's raw scans with the current tracker.
- `export_topics.py BAG` — one CSV per topic (`scan`, `pose`, `delta`, `tf`) in `BAG_csv/`.
- `read_motion_bag.py BAG [--json out.jsonl]` — summary, or one JSON record per scan.
- `stop_and_fetch.cmd RUN` — run on the Windows laptop: stops recording, exports CSVs,
  copies bag + CSVs into `data\` next to the script.

## Data in `../data/`
| Folder | What |
|---|---|
| `run2`, `run3` | room runs (two rooms with a partition) |
| `run4` | 3-minute corridor run, raw recording (tracking slipped in the corridor) |
| `run4_slam` (+ `run4_slam_map.*`) | run4 corrected with the current tracker + slam_toolbox; use this one |
