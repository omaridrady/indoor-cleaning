import cv2
import numpy as np
import matplotlib.pyplot as plt
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

SCAN_TOPIC = "/base_scan"
POSE_TOPIC = "/robot_pose_ekf/odom_combined"
RESOLUTION = 0.05
SCAN_SKIP = 5          # Reduced from 10 to capture denser scan data and prevent gaps
RAY_SKIP = 1           # Set to 1 for maximum angular resolution along walls
MAX_RANGE = 30.0
MAX_RAY_LENGTH = 15.0  # Increased from 6.0 to prevent cutting off long hallway walls!
MIN_RANGE = 0.1
LASER_X = 0.275
LASER_Y = 0.0
LASER_YAW = 0.0
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


def generate_raw_grid(bag_path):
    print("Opening bag...")
    typestore = get_typestore(Stores.ROS1_NOETIC)
    scans, poses = [], []

    with Reader(bag_path) as reader:
        connections = {
            c.topic: c for c in reader.connections 
            if c.topic in [SCAN_TOPIC, POSE_TOPIC]
        }

        for connection, timestamp, rawdata in reader.messages(connections=connections.values()):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            if connection.topic == SCAN_TOPIC:
                scans.append((timestamp, msg))
            elif connection.topic == POSE_TOPIC:
                pose = msg.pose.pose
                yaw = quaternion_to_yaw(pose.orientation)
                poses.append((timestamp, pose.position.x, pose.position.y, yaw))

    print(f"Scans loaded: {len(scans)} | Poses loaded: {len(poses)}")

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
                    grid[cy, cx] = 0

            if is_hit[i] and ranges[i] <= MAX_RAY_LENGTH:
                grid[end_gy, end_gx] = 1
            else:
                grid[end_gy, end_gx] = 0

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
    return grid, pose_x, pose_y, min_x, min_y, RESOLUTION, MAX_RANGE