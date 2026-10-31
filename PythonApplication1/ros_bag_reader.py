import cv2
import numpy as np
import yaml
from pathlib import Path


def _read_gray_image(path):
    # cv2.imread fails on non-ASCII Windows paths; decoding from bytes avoids that.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode map image: {path}")
    return img


def generate_raw_grid(yaml_file_path, max_range=12.0):
    """
    Reads a SLAM map (.yaml + .pgm) saved by map_saver / slam_toolbox.
    Returns the same tuple the rest of the pipeline expects.
    Grid values: -1 unknown, 0 free, 1 occupied. Row 0 = lowest Y (origin at bottom-left).
    """
    yaml_path = Path(yaml_file_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Map YAML not found: {yaml_path}")
    print(f"Loading SLAM map metadata from: {yaml_path}")

    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    for key in ("image", "resolution", "origin"):
        if key not in meta:
            raise KeyError(f"'{key}' missing from {yaml_path.name}")

    resolution = float(meta["resolution"])
    origin = meta["origin"]                      # [x, y, yaw]
    min_x, min_y = float(origin[0]), float(origin[1])
    if len(origin) > 2 and abs(float(origin[2])) > 1e-6:
        print(f"WARNING: map origin yaw is {origin[2]} rad; this pipeline ignores it.")
    negate = int(meta.get("negate", 0))

    # The image path may be relative, or absolute from the robot's filesystem.
    pgm_path = yaml_path.parent / meta["image"]
    if not pgm_path.exists():
        fallback = yaml_path.parent / Path(meta["image"]).name
        if not fallback.exists():
            raise FileNotFoundError(f"Map image not found at {pgm_path} or {fallback}")
        pgm_path = fallback
    print(f"Loading SLAM map image from: {pgm_path}")

    img = _read_gray_image(pgm_path)
    if negate:
        img = 255 - img

    # map_saver trinary: 254 = free, 205 = unknown, 0 = occupied
    grid = np.full(img.shape, -1, dtype=np.int8)
    grid[img >= 250] = 0
    grid[img <= 50] = 1

    # Image row 0 is the top (max Y); ROS grids start at the bottom-left.
    grid = np.flipud(grid)

    total = grid.size
    print(f"Grid: {grid.shape[1]} x {grid.shape[0]} cells "
          f"({grid.shape[1]*resolution:.1f} m x {grid.shape[0]*resolution:.1f} m), "
          f"res={resolution} m/cell, origin=({min_x:.2f}, {min_y:.2f})")
    print(f"Unknown {100*np.mean(grid == -1):.1f}% | "
          f"Free {100*np.mean(grid == 0):.1f}% | "
          f"Occupied {100*np.mean(grid == 1):.1f}%")

    # No raw trajectory anymore, so pose_x / pose_y are None.
    return grid, None, None, min_x, min_y, resolution, max_range