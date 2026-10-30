import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Topic names per bag source; the first pair present in the bag is used.
#   ROS1 dataset bag (.bag):          /base_scan + /robot_pose_ekf/odom_combined
#   RPLIDAR C1 recorder (ROS2 .mcap): /scan      + /lidar_odom
SCAN_TOPICS = ["/base_scan", "/scan"]
POSE_TOPICS = ["/robot_pose_ekf/odom_combined", "/lidar_odom"]
RESOLUTION = 0.05
SCAN_SKIP = 1          # Use every scan: the C1's 720-point scans are sparse, so skipping leaves gaps
RAY_SKIP = 1           # Set to 1 for maximum angular resolution along walls
MAX_RANGE = 12.0       # RPLIDAR C1 rated range; farther returns are noisy
MAX_RAY_LENGTH = 8.0   # Long grazing rays mostly add noise indoors
MIN_RANGE = 0.15       # Drop returns off the cable / mount right next to the lidar
OCCUPIED_RATIO = 0.25  # share of rays reaching a cell that must end there for it to count as wall
# Laser mount (x, y, yaw) relative to the pose frame, per pose topic.
# /lidar_odom is already the laser's own pose, so no offset.
LASER_MOUNTS = {
    "/robot_pose_ekf/odom_combined": (0.275, 0.0, 0.0),
    "/lidar_odom": (0.0, 0.0, 0.0),
}
MARGIN = 2.0

MAX_ROBOT_SPEED = 1.5              # m/s
MAX_POSE_GAP_NS = int(0.2 * 1e9)  # 0.2 seconds in ns


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    cells = []
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def filter_pose_outliers(pose_times, pose_x, pose_y, pose_yaw, max_speed=MAX_ROBOT_SPEED):
    valid = np.ones(len(pose_times), dtype=bool)
    dt = np.diff(pose_times) / 1e9
    dx = np.diff(pose_x)
    dy = np.diff(pose_y)
    dist = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, dist / dt, 0.0)

    bad = speed > max_speed
    valid[:-1][bad] = False
    valid[1:][bad] = False
    return valid


def pick_topic(available, candidates, kind):
    for topic in candidates:
        if topic in available:
            return topic
    raise ValueError(f"No {kind} topic found; looked for {candidates}, bag has {sorted(available)}")


def load_slam_map(bag_path):
    """If slam_toolbox's map was saved next to the bag (<bag>_map.npy + <bag>_map.yaml,
    as slam_offline.sh does), return (grid, min_x, min_y) in this module's -1/0/1 format.

    slam_toolbox builds that map from its final, loop-closed poses, so it is cleaner than
    a grid rebuilt from the per-scan poses stored in the bag. Returns None if there is no map.
    """
    prefix = str(bag_path).rstrip("/\\")
    if prefix.endswith(".mcap"):
        prefix = str(Path(prefix).parent)
    npy, yml = Path(prefix + "_map.npy"), Path(prefix + "_map.yaml")
    if not (npy.exists() and yml.exists()):
        return None

    meta = {}
    for line in yml.read_text().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    resolution = float(meta["resolution"])
    if abs(resolution - RESOLUTION) > 1e-6:
        print(f"SLAM map resolution {resolution} != RESOLUTION {RESOLUTION}; rebuilding from scans")
        return None
    origin_x, origin_y = (float(v) for v in meta["origin"].strip("[]").split(",")[:2])

    probability = np.load(npy)          # slam_toolbox: -1 unknown, 0..100 occupied %
    grid = np.full(probability.shape, -1, dtype=np.int8)
    grid[(probability >= 0) & (probability < 50)] = 0
    grid[probability >= 50] = 1
    return grid, origin_x, origin_y


def save_grid(grid, pose_x, pose_y, min_x, min_y):
    np.savez(
        "occupancy_grid_data.npz",
        grid=grid,
        pose_x=pose_x,
        pose_y=pose_y,
        min_x=min_x,
        min_y=min_y,
        resolution=RESOLUTION,
        max_range=MAX_RANGE
    )
    np.save("occupancy_grid.npy", grid)


def generate_raw_grid(bag_path):
    """bag_path: a ROS1 .bag, a ROS2 bag folder, or a ROS2 .mcap file."""
    print("Opening bag...")
    scans, poses = [], []

    # AnyReader handles ROS1 and ROS2 bags; the default typestore is only
    # used for ROS2 bags that don't embed their message definitions.
    with AnyReader([Path(bag_path)], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        available = {c.topic for c in reader.connections}
        scan_topic = pick_topic(available, SCAN_TOPICS, "scan")
        pose_topic = pick_topic(available, POSE_TOPICS, "pose")
        print(f"Using scan topic {scan_topic} and pose topic {pose_topic}")
        connections = [c for c in reader.connections if c.topic in (scan_topic, pose_topic)]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == scan_topic:
                scans.append((timestamp, msg))
            elif connection.topic == pose_topic:
                pose = msg.pose.pose
                yaw = quaternion_to_yaw(pose.orientation)
                poses.append((timestamp, pose.position.x, pose.position.y, yaw))

    print(f"Scans loaded: {len(scans)} | Poses loaded: {len(poses)}")
    LASER_X, LASER_Y, LASER_YAW = LASER_MOUNTS[pose_topic]

    pose_times = np.array([p[0] for p in poses], dtype=np.int64)
    pose_x = np.array([p[1] for p in poses], dtype=float)
    pose_y = np.array([p[2] for p in poses], dtype=float)
    pose_yaw = np.array([p[3] for p in poses], dtype=float)

    if len(pose_times) > 1:
        valid_mask = filter_pose_outliers(pose_times, pose_x, pose_y, pose_yaw)
        print(f"Removed {np.sum(~valid_mask)} outlier poses out of {len(valid_mask)}")
        pose_times = pose_times[valid_mask]
        pose_x = pose_x[valid_mask]
        pose_y = pose_y[valid_mask]
        pose_yaw = pose_yaw[valid_mask]

    slam_map = load_slam_map(bag_path)
    if slam_map is not None:
        grid, min_x, min_y = slam_map
        print(f"Using SLAM map next to the bag ({grid.shape[1]}x{grid.shape[0]} cells) "
              f"instead of rebuilding the grid from scans")
        save_grid(grid, pose_x, pose_y, min_x, min_y)
        return grid, pose_x, pose_y, min_x, min_y, RESOLUTION, MAX_RANGE

    # --------------------------------------------------------
    # FIRST PASS: COLLECT WORLD POINTS FOR MAP BOUNDS
    # --------------------------------------------------------
    world_points = []
    for scan_index, (scan_time, scan) in enumerate(scans):
        if scan_index % SCAN_SKIP != 0 or len(pose_times) == 0:
            continue

        pose_index = np.searchsorted(pose_times, scan_time)
        if pose_index == 0:
            nearest = 0
        elif pose_index >= len(pose_times):
            nearest = len(pose_times) - 1
        else:
            before, after = pose_index - 1, pose_index
            nearest = before if abs(scan_time - pose_times[before]) < abs(scan_time - pose_times[after]) else after

        if abs(scan_time - pose_times[nearest]) > MAX_POSE_GAP_NS:
            continue

        robot_x, robot_y, robot_yaw = pose_x[nearest], pose_y[nearest], pose_yaw[nearest]
        ranges = np.asarray(scan.ranges, dtype=float)[::RAY_SKIP]
        angles = (scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment)[::RAY_SKIP]

        valid = np.isfinite(ranges) & (ranges >= MIN_RANGE) & (ranges <= MAX_RANGE)
        ranges, angles = ranges[valid], angles[valid]
        if len(ranges) == 0:
            continue

        laser_x = ranges * np.cos(angles)
        laser_y = ranges * np.sin(angles)

        c_laser, s_laser = np.cos(LASER_YAW), np.sin(LASER_YAW)
        rotated_x = c_laser * laser_x - s_laser * laser_y + LASER_X
        rotated_y = s_laser * laser_x + c_laser * laser_y + LASER_Y

        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        world_x = robot_x + c * rotated_x - s * rotated_y
        world_y = robot_y + s * rotated_x + c * rotated_y
        world_points.append(np.column_stack((world_x, world_y)))

    if not world_points:
        raise ValueError("No valid LiDAR points were extracted.")

    world_points = np.vstack(world_points)
    min_x = np.min(world_points[:, 0]) - MARGIN
    max_x = np.max(world_points[:, 0]) + MARGIN
    min_y = np.min(world_points[:, 1]) - MARGIN
    max_y = np.max(world_points[:, 1]) + MARGIN

    width = int(np.ceil((max_x - min_x) / RESOLUTION))
    height = int(np.ceil((max_y - min_y) / RESOLUTION))
    grid = np.full((height, width), -1, dtype=np.int8)
    hits = np.zeros((height, width), dtype=np.int32)
    misses = np.zeros((height, width), dtype=np.int32)

    def world_to_grid(x, y):
        return int((x - min_x) / RESOLUTION), int((y - min_y) / RESOLUTION)

    # --------------------------------------------------------
    # SECOND PASS: BUILD OCCUPANCY GRID
    # --------------------------------------------------------
    print("Building occupancy grid with optimized ray bounds...")
    for scan_index, (scan_time, scan) in enumerate(scans):
        if scan_index % SCAN_SKIP != 0 or len(pose_times) == 0:
            continue

        pose_index = np.searchsorted(pose_times, scan_time)
        if pose_index == 0:
            nearest = 0
        elif pose_index >= len(pose_times):
            nearest = len(pose_times) - 1
        else:
            before, after = pose_index - 1, pose_index
            nearest = before if abs(scan_time - pose_times[before]) < abs(scan_time - pose_times[after]) else after

        if abs(scan_time - pose_times[nearest]) > MAX_POSE_GAP_NS:
            continue

        robot_x, robot_y, robot_yaw = pose_x[nearest], pose_y[nearest], pose_yaw[nearest]
        ranges = np.asarray(scan.ranges, dtype=float)[::RAY_SKIP]
        angles = (scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment)[::RAY_SKIP]

        valid = np.isfinite(ranges) & (ranges >= MIN_RANGE)
        ranges, angles = ranges[valid], angles[valid]
        if len(ranges) == 0:
            continue

        is_hit = ranges <= MAX_RANGE
        effective_ranges = np.minimum(ranges, MAX_RAY_LENGTH)

        lx = effective_ranges * np.cos(angles)
        ly = effective_ranges * np.sin(angles)
        c_laser, s_laser = np.cos(LASER_YAW), np.sin(LASER_YAW)
        x_laser = c_laser * lx - s_laser * ly + LASER_X
        y_laser = s_laser * lx + c_laser * ly + LASER_Y

        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        wx = robot_x + c * x_laser - s * y_laser
        wy = robot_y + s * x_laser + c * y_laser

        robot_gx, robot_gy = world_to_grid(robot_x, robot_y)

        for i, (x, y) in enumerate(zip(wx, wy)):
            end_gx, end_gy = world_to_grid(x, y)
            if not (0 <= end_gx < width and 0 <= end_gy < height):
                continue

            cells = bresenham(robot_gx, robot_gy, end_gx, end_gy)
            for cx, cy in cells[:-1]:
                if 0 <= cx < width and 0 <= cy < height:
                    misses[cy, cx] += 1

            if is_hit[i] and ranges[i] <= MAX_RAY_LENGTH:
                hits[end_gy, end_gx] += 1
            else:
                misses[end_gy, end_gx] += 1

    # A cell is a wall if enough of the rays reaching it ended there. Counting
    # (instead of last ray wins) stops one noisy ray from erasing a wall.
    seen = (hits + misses) > 0
    grid[seen] = 0
    grid[seen & (hits >= OCCUPIED_RATIO * (hits + misses))] = 1

    # --------------------------------------------------------
    # POST-PROCESSING: FLOOD-FILL UNREACHABLE SPACE
    # --------------------------------------------------------
    print("Cleaning up unreached laser fans...")
    free_mask = (grid == 0).astype(np.uint8)
    h, w = free_mask.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for px, py in zip(pose_x, pose_y):
        gx, gy = world_to_grid(px, py)
        if 0 <= gx < w and 0 <= gy < h and free_mask[gy, gx] == 1:
            cv2.floodFill(free_mask, flood_mask, (gx, gy), 2)

    grid[(grid == 0) & (free_mask != 2)] = -1

    # Save finalized output files
    save_grid(grid, pose_x, pose_y, min_x, min_y)
    return grid, pose_x, pose_y, min_x, min_y, RESOLUTION, MAX_RANGE