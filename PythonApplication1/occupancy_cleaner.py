import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.spatial import cKDTree

# ----------------------------------------------------------------
# Thresholds are in physical units (meters / m^2), so they keep their meaning
# if the map resolution changes. This cleaner is tuned for SLAM maps (slam_toolbox
# / map_saver), whose walls are already thin and continuous: it removes specks,
# bridges tiny gaps and drops short isolated pieces, but does NOT thicken walls.
# ----------------------------------------------------------------

# A wall pixel must touch at least this many other wall pixels (8-neighbourhood).
# 1 only removes completely isolated pixels; 2 would erase 1-cell-thin walls.
MIN_WALL_NEIGHBORS = 1

# Components with less area than this are specks (0.02 m^2 = ~8 px at 0.05 m/cell).
# Removed BEFORE closing, so specks cannot get welded onto real walls.
SPECK_AREA_M2 = 0.02

# Closing bridges small breaks in walls. 0.15 m = 3 px at 0.05 m/cell, which
# fixes 1-2 px gaps but stays well below a doorway (~0.8 m), so doors survive.
CLOSING_SIZE_M = 0.15

# Isolated wall pieces whose bounding-box diagonal is shorter than this are removed.
# Extent is used instead of pixel count because a thin 1.5 m wall and a compact
# 1.5 m^2 blob of clutter have very different shapes but could have the same count.
# Keep this at or below the minimum wall length used in wall_line_processing.py.
MIN_COMPONENT_EXTENT_M = 0.5

# Optional wall thickening. 0.0 = off. If line extraction struggles with 1-px
# walls, try 0.10 (becomes a symmetric 3-px kernel at 0.05 m/cell).
THICKEN_WALLS_M = 0.0

_EIGHT = np.ones((3, 3), dtype=int)   # 8-connectivity (diagonal pixels belong together)


def _odd_kernel_size(meters, resolution):
    """Pixel size for a square kernel, forced odd so it does not shift walls by half a cell."""
    k = max(1, int(round(meters / resolution)))
    if k % 2 == 0:
        k += 1
    return k


def _filter_components(mask, min_area_px=0, min_extent_px=0.0):
    """
    Keeps only connected components (8-connectivity) that have at least
    min_area_px pixels AND a bounding-box diagonal of at least min_extent_px.
    8-connectivity matters: a thin diagonal wall is a staircase of pixels that
    only touch at their corners, and 4-connectivity would shred it into dots.
    """
    labeled, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return mask.copy()

    areas = np.bincount(labeled.ravel(), minlength=n + 1)
    keep = areas >= min_area_px

    if min_extent_px > 0:
        for i, sl in enumerate(ndimage.find_objects(labeled), start=1):
            if sl is None:
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            if np.hypot(h, w) < min_extent_px:
                keep[i] = False

    keep[0] = False
    return keep[labeled]


def remove_unreachable_walls(cleaned_grid, pose_x, pose_y, min_x, min_y, resolution, max_range):
    """
    Safety net for bag-built grids: removes any wall pixel farther than max_range
    from all robot poses. Not used for SLAM maps (no pose data), where it is skipped.
    """
    tree = cKDTree(np.column_stack((pose_x, pose_y)))
    ys, xs = np.where(cleaned_grid == 1)

    if len(xs) == 0:
        return cleaned_grid

    world_x = xs * resolution + min_x
    world_y = ys * resolution + min_y
    dist, _ = tree.query(np.column_stack((world_x, world_y)))

    unreachable = dist > (max_range * 1.05)
    out = cleaned_grid.copy()
    out[ys[unreachable], xs[unreachable]] = 0
    return out


def clean_occupancy_grid(grid, resolution=0.05, pose_x=None, pose_y=None,
                         min_x=None, min_y=None, max_range=30.0):

    cell_area = resolution ** 2
    speck_px = max(1, int(round(SPECK_AREA_M2 / cell_area)))
    closing_px = _odd_kernel_size(CLOSING_SIZE_M, resolution)
    thicken_px = _odd_kernel_size(THICKEN_WALLS_M, resolution)
    min_extent_px = MIN_COMPONENT_EXTENT_M / resolution

    print(f"Resolution: {resolution} m/cell -> speck<{speck_px}px, closing={closing_px}px, "
          f"thicken={thicken_px}px, min component extent={min_extent_px:.1f}px")

    occupied = (grid == 1)
    raw_wall_px = int(occupied.sum())

    print("1. Removing isolated wall pixels...")
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.int8)
    neighbor_count = ndimage.convolve(occupied.astype(np.int8), kernel, mode="constant", cval=0)
    walls = occupied & (neighbor_count >= MIN_WALL_NEIGHBORS)

    print("2. Removing specks before they can fuse to walls...")
    walls = _filter_components(walls, min_area_px=speck_px)

    print("3. Bridging small gaps (closing)...")
    # Pad so closing cannot erode walls that touch the edge of the map.
    pad = closing_px
    padded = np.pad(walls, pad, mode="constant", constant_values=False)
    closed = ndimage.binary_closing(padded, structure=np.ones((closing_px, closing_px), dtype=bool))
    walls = closed[pad:-pad, pad:-pad]

    if thicken_px > 1:
        print("   Thickening walls...")
        walls = ndimage.binary_dilation(walls, structure=np.ones((thicken_px, thicken_px), dtype=bool))

    print("4. Discarding short isolated pieces...")
    walls = _filter_components(walls, min_extent_px=min_extent_px)

    # Rebuild the grid: every original wall pixel becomes free, then only the
    # confirmed walls are drawn back in.
    cleaned_grid = grid.copy()
    cleaned_grid[grid == 1] = 0
    cleaned_grid[walls] = 1

    if pose_x is not None and pose_y is not None and min_x is not None and min_y is not None:
        print("5. Removing unreachable walls (spatial safety net)...")
        cleaned_grid = remove_unreachable_walls(
            cleaned_grid, pose_x, pose_y, min_x, min_y, resolution, max_range
        )

    clean_wall_px = int((cleaned_grid == 1).sum())
    print(f"Wall pixels: {raw_wall_px} raw -> {clean_wall_px} cleaned")

    np.save("occupancy_grid_cleaned.npy", cleaned_grid)
    return cleaned_grid


def plot_comparison(raw_grid, cleaned_grid):
    cmap = ListedColormap(['#C8C8C8', '#FFFFFF', '#000000'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(raw_grid, cmap=cmap, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    axes[0].set_title("Raw Occupancy Grid")
    axes[0].set_xlabel("X (cells)")
    axes[0].set_ylabel("Y (cells)")

    axes[1].imshow(cleaned_grid, cmap=cmap, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    axes[1].set_title("Cleaned Grid")
    axes[1].set_xlabel("X (cells)")
    axes[1].set_ylabel("Y (cells)")

    plt.tight_layout()
    plt.show()