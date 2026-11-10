import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# Maximum distance for an endpoint to snap to a perpendicular wall.
# Kept well below a doorway width (~0.8-0.9 m): a larger value would extend
# walls across door openings and hide them from the later door-detection stage.
# If real corners are left open, raise this slowly (try 0.4).
SNAP_TOLERANCE_METERS = 0.3

# Minimum final wall length to keep after snapping.
# Matches MIN_WALL_LENGTH_METERS in wall_line_processing.py, so this stage does
# not silently drop walls the previous stage accepted.
MIN_SNAPPED_LENGTH_METERS = 0.5


# ============================================================
# HELPERS
# ============================================================

def _line_family(line):
    """Returns 'horizontal' or 'vertical' (in the aligned grid frame)."""
    family = line.get("family")
    if family in ("horizontal", "vertical"):
        return family
    angle = line["angle_degrees"] % 180.0
    return "horizontal" if min(angle, 180.0 - angle) <= 45.0 else "vertical"


def _make_line(x1, y1, x2, y2, family, resolution):
    """Builds a line dict, or returns None if it is too short to keep."""
    length_pixels = float(np.hypot(x2 - x1, y2 - y1))
    length_meters = length_pixels * resolution
    if length_meters < MIN_SNAPPED_LENGTH_METERS:
        return None

    angle_degrees = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "length_pixels": length_pixels,
        "length_meters": float(length_meters),
        "angle_degrees": angle_degrees,
        "family": family
    }


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
    Snaps wall endpoints to true orthogonal intersections (L-corners and
    T-junctions).

    Each endpoint moves to the NEAREST perpendicular wall within
    SNAP_TOLERANCE_METERS, provided that wall actually spans the endpoint's
    position. All decisions are made from the original geometry, so the result
    does not depend on the order lines are processed in, and an endpoint can
    never be dragged along a chain of walls.

    Input lines must already be axis-aligned (horizontal/vertical) in the
    aligned grid frame, as produced by wall_line_processing.py.
    """
    if not wall_lines:
        return []

    snap = SNAP_TOLERANCE_METERS / resolution

    # Normalize every line to: fixed position, start/end along its own axis
    horiz = []   # y position, x extent
    vert = []    # x position, y extent
    for l in wall_lines:
        if _line_family(l) == "horizontal":
            horiz.append({
                "pos": (l["y1"] + l["y2"]) / 2.0,
                "lo": min(l["x1"], l["x2"]),
                "hi": max(l["x1"], l["x2"]),
            })
        else:
            vert.append({
                "pos": (l["x1"] + l["x2"]) / 2.0,
                "lo": min(l["y1"], l["y2"]),
                "hi": max(l["y1"], l["y2"]),
            })

    def snap_endpoint(value, own_pos, partners):
        """
        value    : this line's endpoint coordinate along its own axis
        own_pos  : this line's fixed (perpendicular) coordinate
        partners : perpendicular walls it may snap to
        """
        best_pos = None
        best_dist = snap
        for p in partners:
            d = abs(p["pos"] - value)
            if d > best_dist:
                continue
            # The partner must actually span this line's position
            if p["lo"] - snap <= own_pos <= p["hi"] + snap:
                best_pos = p["pos"]
                best_dist = d
        return value if best_pos is None else best_pos

    snapped = []

    for h in horiz:
        x1 = snap_endpoint(h["lo"], h["pos"], vert)
        x2 = snap_endpoint(h["hi"], h["pos"], vert)
        line = _make_line(min(x1, x2), h["pos"], max(x1, x2), h["pos"], "horizontal", resolution)
        if line is not None:
            snapped.append(line)

    for v in vert:
        y1 = snap_endpoint(v["lo"], v["pos"], horiz)
        y2 = snap_endpoint(v["hi"], v["pos"], horiz)
        line = _make_line(v["pos"], min(y1, y2), v["pos"], max(y1, y2), "vertical", resolution)
        if line is not None:
            snapped.append(line)

    return snapped


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