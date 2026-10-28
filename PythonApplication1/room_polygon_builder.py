import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# Maximum distance for an endpoint to snap to an intersecting perpendicular wall
# 0.8 meters = 16 pixels at 0.05 m/cell resolution
SNAP_TOLERANCE_METERS = 0.8

# Minimum final wall length to keep after adjustment
MIN_SNAPPED_LENGTH_METERS = 0.8


# ============================================================
# ORTHOGONAL INTERSECTION SNAPPING
# ============================================================
def plot_clean_vector_map(aligned_grid, snapped_lines, save_path="vector_floor_plan.png"):
    """
    Renders a CAD-style vector map:
    - Wall lines in solid black
    - Indoor walkable space in white
    - Unexplored surrounding space in neutral gray
    - Hides grid axes and raw cell pixelation
    """
    fig, ax = plt.subplots(figsize=(18, 10))

    # 1. Background mapping: -1 (unexplored) -> Gray, 0 (free) -> White, 1 (wall) -> White
    # We render free space as clean white and unexplored space as light gray
    bg_map = np.full(aligned_grid.shape, 255, dtype=np.uint8)  # White default
    bg_map[aligned_grid == -1] = 200                           # Light gray for unexplored area

    ax.imshow(
        bg_map,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="lower",
        interpolation="bilinear"
    )

    # 2. Draw clean black wall vector lines
    for line in snapped_lines:
        ax.plot(
            [line["x1"], line["x2"]],
            [line["y1"], line["y2"]],
            color="black",
            linewidth=3.0,
            solid_capstyle="round"
        )

    # 3. Clean up display aesthetics
    ax.set_title("Architectural Vector Floor Plan", fontsize=14, weight="bold")
    ax.axis("off")  # Hide grid axis numbers/ticks

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved vector map image: {save_path}")

def snap_wall_endpoints(wall_lines, resolution=0.05):
    """
    Snaps wall endpoints strictly to true orthogonal intersections (L-corners and T-junctions)
    rather than blind point averaging, preserving structural accuracy.
    """
    if not wall_lines:
        return []

    # Create mutable copies of the lines
    lines = [dict(l) for l in wall_lines]

    # Separate into horizontal and vertical families based on angle
    snap_pixels = SNAP_TOLERANCE_METERS / resolution
    horiz_lines = []
    vert_lines = []
    for l in lines:
        angle = l["angle_degrees"]
        if abs(angle - 90.0) < 25.0 or abs(angle + 90.0) < 25.0:
            vert_lines.append(l)
        else:
            horiz_lines.append(l)

    # Check every horizontal and vertical line pair for potential corner intersections
    for h in horiz_lines:
        hx1, hx2 = min(h["x1"], h["x2"]), max(h["x1"], h["x2"])
        hy = (h["y1"] + h["y2"]) / 2.0

        for v in vert_lines:
            vx = (v["x1"] + v["x2"]) / 2.0
            vy1, vy2 = min(v["y1"], v["y2"]), max(v["y1"], v["y2"])

            # Theoretical intersection of this horizontal and vertical line is at (vx, hy)
            # Check if this intersection falls on or near both segments
            if (hx1 - snap_pixels <= vx <= hx2 + snap_pixels) and (vy1 - snap_pixels <= hy <= vy2 + snap_pixels):
                
                # Snap horizontal line endpoints to intersection X coordinate if close
                if abs(h["x1"] - vx) <= snap_pixels:
                    h["x1"] = vx
                elif abs(h["x2"] - vx) <= snap_pixels:
                    h["x2"] = vx

                # Snap vertical line endpoints to intersection Y coordinate if close
                if abs(v["y1"] - hy) <= snap_pixels:
                    v["y1"] = hy
                elif abs(v["y2"] - hy) <= snap_pixels:
                    v["y2"] = hy

    # Recombine and recalculate properties for all lines
    all_updated = horiz_lines + vert_lines
    final_snapped_lines = []

    for l in all_updated:
        x1, y1 = l["x1"], l["y1"]
        x2, y2 = l["x2"], l["y2"]

        dx = x2 - x1
        dy = y2 - y1
        length_pixels = np.hypot(dx, dy)
        length_meters = length_pixels * resolution

        # Drop lines that collapsed or became too short
        if length_meters < MIN_SNAPPED_LENGTH_METERS:
            continue

        angle_degrees = np.degrees(np.arctan2(dy, dx)) % 180.0

        final_snapped_lines.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "length_pixels": float(length_pixels),
            "length_meters": float(length_meters),
            "angle_degrees": float(angle_degrees),
            "family": l.get("family", "horizontal")
        })

    return final_snapped_lines


# ============================================================
# MAIN ROOM CONSTRUCTION PROCESS
# ============================================================

def build_room_polygons(
    aligned_grid,
    wall_lines,
    resolution=0.05,
    save_result=True
):
    print()
    print("==============================================")
    print("ROOM CONSTRUCTION & CORNER SNAPPING")
    print("==============================================")

    print()
    print("1. Snapping endpoints via orthogonal intersections...")
    snapped_lines = snap_wall_endpoints(wall_lines, resolution)

    print(
        "Wall segments after geometric snapping:",
        len(snapped_lines)
    )

    # ========================================================
    # Save
    # ========================================================

    if save_result:

        data = []

        for line in snapped_lines:

            data.append([
                line["x1"],
                line["y1"],
                line["x2"],
                line["y2"],
                line["length_pixels"],
                line["length_meters"],
                line["angle_degrees"]
            ])

        np.save(
            "snapped_wall_lines.npy",
            np.asarray(
                data,
                dtype=float
            )
        )

        print()
        print("Saved: snapped_wall_lines.npy")

    print()
    print("Room construction processing finished.")

    return snapped_lines


# ============================================================
# PLOT SNAPPED WALL LINES
# ============================================================

def plot_snapped_walls(
    aligned_grid,
    snapped_lines
):

    fig, ax = plt.subplots(
        figsize=(18, 10)
    )

    ax.imshow(
        aligned_grid,
        cmap="gray",
        vmin=-1,
        vmax=1,
        origin="lower",
        interpolation="nearest"
    )

    for i, line in enumerate(
        snapped_lines
    ):

        ax.plot(
            [
                line["x1"],
                line["x2"]
            ],
            [
                line["y1"],
                line["y2"]
            ],
            linewidth=2.5,
            color="#FF5733"
        )

        mid_x = (
            line["x1"]
            +
            line["x2"]
        ) / 2.0

        mid_y = (
            line["y1"]
            +
            line["y2"]
        ) / 2.0

        ax.text(
            mid_x,
            mid_y,
            str(i + 1),
            fontsize=9,
            color="yellow",
            weight="bold"
        )

    ax.set_title(
        "Constructed Blueprint: Orthogonal Intersections"
    )

    ax.set_xlabel(
        "X (cells)"
    )

    ax.set_ylabel(
        "Y (cells)"
    )

    plt.tight_layout()
    plt.show()