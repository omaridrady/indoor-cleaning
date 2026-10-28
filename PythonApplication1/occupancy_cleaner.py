import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.spatial import cKDTree

# ----------------------------------------------------------------
# Thresholds defined in physical units (meters / m²), not pixels.
# These stay meaningful across any RESOLUTION or dataset.
# ----------------------------------------------------------------
MIN_WALL_NEIGHBORS = 2          # count-based, not a distance -- stays as-is

PRE_FILTER_AREA_M2 = 0.1  # area equivalent of old PRE_FILTER_SIZE=15 px @ 0.05 m/cell
POST_FILTER_AREA_M2 = 100 * (0.05 ** 2)  # area equivalent of old POST_FILTER_SIZE=100 px @ 0.05 m/cell
DILATION_SIZE_M = 2 * 0.05               # 0.10 m
CLOSING_SIZE_M = 5 * 0.05                # 0.25 m


def remove_unreachable_walls(cleaned_grid, pose_x, pose_y, min_x, min_y, resolution, max_range):
    """Safety net: Removes any wall pixel situated farther than max_range from all valid robot poses."""
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

    # Convert physical thresholds into pixel-space, scaled for THIS grid's resolution.
    cell_area = resolution ** 2
    pre_filter_px = max(1, round(PRE_FILTER_AREA_M2 / cell_area))
    post_filter_px = max(1, round(POST_FILTER_AREA_M2 / cell_area))
    dilation_px = max(1, round(DILATION_SIZE_M / resolution))
    closing_px = max(1, round(CLOSING_SIZE_M / resolution))

    print(f"Resolution: {resolution} m/cell -> "
          f"pre_filter={pre_filter_px}px, post_filter={post_filter_px}px, "
          f"dilation={dilation_px}px, closing={closing_px}px")

    print("1. Filtering single-pixel ray endpoints...")
    occupied = (grid == 1)

    neighbor_kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    neighbor_count = ndimage.convolve(occupied.astype(int), neighbor_kernel, mode='constant', cval=0)
    dense_walls = occupied & (neighbor_count >= MIN_WALL_NEIGHBORS)

    print("2. Erasing internal clutter and furniture...")
    labeled_pre, _ = ndimage.label(dense_walls)
    pre_sizes = np.bincount(labeled_pre.ravel())
    keep_pre = pre_sizes >= pre_filter_px
    keep_pre[0] = False
    structural_base = keep_pre[labeled_pre]

    print("3. Morphological processing (Dilation + Closing)...")
    dilate_kernel = np.ones((dilation_px, dilation_px), dtype=bool)
    thickened = ndimage.binary_dilation(structural_base, structure=dilate_kernel)

    close_kernel = np.ones((closing_px, closing_px), dtype=bool)
    closed = ndimage.binary_closing(thickened, structure=close_kernel)

    print("4. Discarding isolated floating components...")
    labeled_post, _ = ndimage.label(closed)
    post_sizes = np.bincount(labeled_post.ravel())

    keep_post = post_sizes >= post_filter_px
    keep_post[0] = False
    certain_walls = keep_post[labeled_post]

    cleaned_grid = grid.copy()
    cleaned_grid[grid == 1] = 0
    cleaned_grid[certain_walls] = 1

    if pose_x is not None and pose_y is not None and min_x is not None and min_y is not None:
        print("5. Removing unreachable walls (spatial safety net)...")
        cleaned_grid = remove_unreachable_walls(
            cleaned_grid, pose_x, pose_y, min_x, min_y, resolution, max_range
        )

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
    axes[1].set_title("Cleaned Grid (Reachability & Outliers Filtered)")
    axes[1].set_xlabel("X (cells)")
    axes[1].set_ylabel("Y (cells)")

    plt.tight_layout()
    plt.show()