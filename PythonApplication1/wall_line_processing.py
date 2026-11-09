import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Douglas-Peucker polygon simplification tolerance, in METERS.
# (Previously a fraction of each contour's perimeter, which made the
# tolerance depend on how big the connected wall region was.)
# Lower = more detailed vertices, higher = smoother/stricter lines.
DP_EPSILON_METERS = 0.08

# Wall filtering
MIN_WALL_LENGTH_METERS = 0.5

# Parallel line clustering & merging
MAX_PARALLEL_DISTANCE_METERS = 0.2
MAX_MERGE_GAP_METERS = 0.3
MAX_PARALLEL_ANGLE_DEGREES = 12.0

# Duplicate suppression: a shorter parallel line is removed only if at least
# this fraction of it overlaps a longer, already-kept line.
DUPLICATE_OVERLAP_RATIO = 0.7

# Alignment: skip the rotation if the map is already this close to axis-aligned.
ALIGN_SKIP_BELOW_DEG = 1.0

# Alignment has its own constants so tuning the line extraction above can never
# change the rotation. Do not tune these: the alignment was checked against the
# real floor.
ALIGN_DP_EPSILON_METERS = 0.15
ALIGN_MIN_WALL_LENGTH_METERS = 0.6


# ============================================================
# ANGLE FUNCTIONS
# ============================================================

def normalize_angle(angle):
    return angle % 180.0


def angle_difference(angle1, angle2):
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    difference = abs(angle1 - angle2)
    if difference > 90.0:
        difference = 180.0 - difference
    return difference


# ============================================================
# DOMINANT ORIENTATION + ALIGNMENT
# ============================================================

def estimate_dominant_angle(grid, resolution):
    """
    Length-weighted dominant wall direction, in degrees, in the range (-45, 45].
    Perpendicular walls reinforce the same answer.
    """
    wall_mask = (grid == 1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    sx = 0.0
    sy = 0.0
    epsilon = ALIGN_DP_EPSILON_METERS / resolution

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        n = len(approx)
        if n < 2:
            continue
        for i in range(n):
            p1 = approx[i][0].astype(float)
            p2 = approx[(i + 1) % n][0].astype(float)
            d = p2 - p1
            length = float(np.hypot(d[0], d[1]))
            if length * resolution < ALIGN_MIN_WALL_LENGTH_METERS:
                continue
            a = np.arctan2(d[1], d[0])
            # Multiplying the angle by 4 makes 0, 90, 180 and 270 degrees all
            # point the same way, so perpendicular walls vote together.
            sx += length * np.cos(4.0 * a)
            sy += length * np.sin(4.0 * a)

    if sx == 0.0 and sy == 0.0:
        return 0.0
    return float(np.degrees(np.arctan2(sy, sx) / 4.0))


def align_grid_to_walls(grid, resolution, skip_below_deg=ALIGN_SKIP_BELOW_DEG, angle_override=None):
    """
    Rotates the grid so the dominant walls become horizontal/vertical.

    Returns (aligned_grid, angle_deg, M)
      aligned_grid : the rotated grid (the original if it was already aligned)
      angle_deg    : the estimated dominant wall angle
      M            : 2x3 affine transform, original cell coords -> aligned cell coords.
                     Use cv2.invertAffineTransform(M) to map results back to the
                     SLAM map frame when exporting meter coordinates.
    """
    # angle_override lets two different grids (e.g. classic-cleaned and
    # FPUNet-cleaned) be rotated by exactly the same angle, so their results
    # live in the same frame and can be compared directly.
    if angle_override is None:
        angle = estimate_dominant_angle(grid, resolution)
        print(f"Dominant wall angle: {angle:.2f} degrees")
    else:
        angle = float(angle_override)
        print(f"Using given wall angle: {angle:.2f} degrees")

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if abs(angle) < skip_below_deg:
        print("Map already axis-aligned; skipping rotation.")
        return grid, 0.0, identity

    h, w = grid.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

    # Grow the canvas so no corner of the map is cut off by the rotation.
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=float)
    moved = corners @ M.T
    min_xy = moved.min(axis=0)
    max_xy = moved.max(axis=0)
    M[:, 2] -= min_xy
    new_w = int(np.ceil(max_xy[0] - min_xy[0]))
    new_h = int(np.ceil(max_xy[1] - min_xy[1]))

    # warpAffine does not support int8, so shift {-1,0,1} to {0,1,2}, warp, shift back.
    shifted = (grid.astype(np.int16) + 1).astype(np.uint8)
    rotated = cv2.warpAffine(
        shifted,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0            # new border area becomes unknown (-1)
    )
    aligned = rotated.astype(np.int8) - 1

    print(f"Rotated grid from {w}x{h} to {new_w}x{new_h} cells")
    return aligned, angle, M


# ============================================================
# LINE PROPERTIES
# ============================================================

def calculate_line_properties(x1, y1, x2, y2, resolution):
    dx = x2 - x1
    dy = y2 - y1
    length_pixels = np.hypot(dx, dy)
    length_meters = length_pixels * resolution
    angle_degrees = np.degrees(np.arctan2(dy, dx))
    angle_degrees = normalize_angle(angle_degrees)

    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "length_pixels": float(length_pixels),
        "length_meters": float(length_meters),
        "angle_degrees": float(angle_degrees)
    }


# ============================================================
# DETERMINE LINE FAMILY
# ============================================================

def determine_line_family(angle):
    horizontal_difference = min(abs(angle), abs(angle - 180.0))
    vertical_difference = abs(angle - 90.0)

    if horizontal_difference <= vertical_difference:
        return "horizontal"
    return "vertical"


def line_center(line):
    return (
        (line["x1"] + line["x2"]) / 2.0,
        (line["y1"] + line["y2"]) / 2.0
    )


def perpendicular_position(line, family):
    center_x, center_y = line_center(line)
    if family == "horizontal":
        return center_y
    return center_x


def parallel_distance(line1, line2, family):
    pos1 = perpendicular_position(line1, family)
    pos2 = perpendicular_position(line2, family)
    return abs(pos1 - pos2)


def projection_interval(line, family):
    if family == "horizontal":
        p1, p2 = line["x1"], line["x2"]
    else:
        p1, p2 = line["y1"], line["y2"]
    return min(p1, p2), max(p1, p2)


def segment_gap(line1, line2, family):
    min1, max1 = projection_interval(line1, family)
    min2, max2 = projection_interval(line2, family)

    if max1 >= min2 and max2 >= min1:
        return 0.0
    if max1 < min2:
        return min2 - max1
    return min1 - max2


def segment_overlap_ratio(line1, line2, family):
    """Fraction of the SHORTER line that overlaps the other along the primary axis."""
    min1, max1 = projection_interval(line1, family)
    min2, max2 = projection_interval(line2, family)
    overlap = min(max1, max2) - max(min1, min2)
    shorter = min(max1 - min1, max2 - min2)
    if shorter <= 0:
        return 0.0
    return max(0.0, overlap) / shorter


def can_merge_segments(line1, line2, family, resolution):
    if determine_line_family(line1["angle_degrees"]) != family:
        return False
    if determine_line_family(line2["angle_degrees"]) != family:
        return False

    if angle_difference(line1["angle_degrees"], line2["angle_degrees"]) > MAX_PARALLEL_ANGLE_DEGREES:
        return False

    if parallel_distance(line1, line2, family) > (MAX_PARALLEL_DISTANCE_METERS / resolution):
        return False

    if segment_gap(line1, line2, family) > (MAX_MERGE_GAP_METERS / resolution):
        return False

    return True


def fit_merged_line(lines, family, resolution):
    # Endpoints are weighted by their segment's length, so a short stub cannot
    # pull a long wall away from where the scans put it.
    points = []
    weights = []
    for line in lines:
        length = max(line["length_pixels"], 1e-6)
        points.append([line["x1"], line["y1"]])
        points.append([line["x2"], line["y2"]])
        weights.extend([length, length])
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(points) < 2:
        return None

    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, np.argmax(eigenvalues)]

    projections = centered @ direction
    minimum, maximum = np.min(projections), np.max(projections)

    point1 = center + minimum * direction
    point2 = center + maximum * direction

    return calculate_line_properties(
        point1[0], point1[1],
        point2[0], point2[1],
        resolution
    )


def orthogonalize_line(line, family, resolution):
    x1, y1 = line["x1"], line["y1"]
    x2, y2 = line["x2"], line["y2"]

    if family == "horizontal":
        y = (y1 + y2) / 2.0
        new_x1, new_x2 = min(x1, x2), max(x1, x2)
        new_y1 = new_y2 = y
        final_angle = 0.0
    else:
        x = (x1 + x2) / 2.0
        new_y1, new_y2 = min(y1, y2), max(y1, y2)
        new_x1 = new_x2 = x
        final_angle = 90.0

    length_pixels = np.hypot(new_x2 - new_x1, new_y2 - new_y1)
    length_meters = length_pixels * resolution

    return {
        "x1": float(new_x1),
        "y1": float(new_y1),
        "x2": float(new_x2),
        "y2": float(new_y2),
        "length_pixels": float(length_pixels),
        "length_meters": float(length_meters),
        "angle_degrees": final_angle,
        "family": family
    }


def suppress_duplicate_lines(lines, family, resolution):
    """
    Removes shorter parallel lines that substantially overlap a longer,
    already-kept wall. A line is only removed when at least
    DUPLICATE_OVERLAP_RATIO of its length overlaps the kept line, so a short
    wall that merely touches the end of a longer one's span is preserved.
    """
    if not lines:
        return []

    # Longest first, so we always keep the most dominant walls
    lines.sort(key=lambda l: l["length_pixels"], reverse=True)

    keep = []
    for current in lines:
        is_duplicate = False

        for kept in keep:
            p_dist = parallel_distance(current, kept, family)

            if p_dist <= (MAX_PARALLEL_DISTANCE_METERS / resolution):
                if segment_overlap_ratio(current, kept, family) >= DUPLICATE_OVERLAP_RATIO:
                    is_duplicate = True
                    break

        if not is_duplicate:
            keep.append(current)

    return keep


def cluster_and_merge_family(lines, family, resolution):
    if len(lines) == 0:
        return []

    working = [dict(line) for line in lines]
    changed = True

    while changed:
        changed = False
        working.sort(key=lambda line: perpendicular_position(line, family))
        used = [False for _ in working]
        result = []

        for i in range(len(working)):
            if used[i]:
                continue
            current = working[i]
            used[i] = True

            merged = True
            while merged:
                merged = False
                for j in range(len(working)):
                    if used[j]:
                        continue
                    if can_merge_segments(current, working[j], family, resolution):
                        current = fit_merged_line([current, working[j]], family, resolution)
                        used[j] = True
                        merged = True
                        changed = True
            result.append(current)

        working = result

    final_lines = []
    for line in working:
        ortho = orthogonalize_line(line, family, resolution)
        if ortho["length_meters"] >= MIN_WALL_LENGTH_METERS:
            final_lines.append(ortho)

    # Remove stacked/overlapping duplicates
    final_lines = suppress_duplicate_lines(final_lines, family, resolution)

    return final_lines


# ============================================================
# DETECT WALL CANDIDATES VIA CONTOURS & DOUGLAS-PEUCKER
# ============================================================

def detect_wall_lines(aligned_grid, resolution=0.05):
    """
    NOTE: aligned_grid must already have its dominant walls horizontal/vertical
    (use align_grid_to_walls first). This stage snaps every line to exactly
    0 or 90 degrees.
    """
    print()
    print("==============================================")
    print("WALL LINE PROCESSING (CONTOUR APPROXIMATION)")
    print("==============================================")

    # 1. Wall mask
    print("1. Extracting wall mask...")
    wall_mask = np.zeros(aligned_grid.shape, dtype=np.uint8)
    wall_mask[aligned_grid == 1] = 255

    # 2. Find contours
    print("2. Finding wall contours...")
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw_segments = []

    # 3. Douglas-Peucker polygon approximation (tolerance in meters)
    print("3. Running Douglas-Peucker Polygon Approximation...")
    epsilon = DP_EPSILON_METERS / resolution

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        n_pts = len(approx)
        if n_pts < 2:
            continue

        for i in range(n_pts):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % n_pts][0]

            line = calculate_line_properties(
                float(pt1[0]), float(pt1[1]),
                float(pt2[0]), float(pt2[1]),
                resolution
            )

            if line["length_meters"] >= MIN_WALL_LENGTH_METERS:
                raw_segments.append(line)

    print(f"Raw polygon segments extracted: {len(raw_segments)}")

    # 4. Separate into families
    horizontal_lines = []
    vertical_lines = []

    for line in raw_segments:
        family = determine_line_family(line["angle_degrees"])
        if family == "horizontal":
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    print(f"Horizontal-family candidates: {len(horizontal_lines)}")
    print(f"Vertical-family candidates: {len(vertical_lines)}")

    # 5. Cluster and merge
    print("Clustering and merging horizontal walls...")
    final_horizontal = cluster_and_merge_family(horizontal_lines, "horizontal", resolution)

    print("Clustering and merging vertical walls...")
    final_vertical = cluster_and_merge_family(vertical_lines, "vertical", resolution)

    final_lines = final_horizontal + final_vertical
    final_lines.sort(key=lambda line: line["length_pixels"], reverse=True)

    return final_lines, wall_mask


# ============================================================
# COMPLETE PROCESS
# ============================================================

def process_wall_lines(aligned_grid, resolution=0.05, save_result=True):
    final_lines, edges = detect_wall_lines(aligned_grid, resolution)

    print()
    print("==============================================")
    print("FINAL WALL LINE RESULT")
    print("==============================================")
    print(f"Final wall lines: {len(final_lines)}")

    for i, line in enumerate(final_lines[:50]):
        print(
            f"{i + 1:2d}. Length = {line['length_meters']:.2f} m, "
            f"Angle = {line['angle_degrees']:.2f}°, "
            f"Family = {line['family']}"
        )

    if save_result:
        data = []
        for line in final_lines:
            data.append([
                line["x1"], line["y1"],
                line["x2"], line["y2"],
                line["length_pixels"], line["length_meters"],
                line["angle_degrees"]
            ])
        np.save("wall_lines.npy", np.asarray(data, dtype=float))
        print("\nSaved: wall_lines.npy")

    return final_lines, edges


# ============================================================
# PLOTTERS
# ============================================================

def plot_wall_edges(edges):
    plt.figure(figsize=(12, 8))
    plt.imshow(edges, cmap="gray", origin="lower", interpolation="nearest")
    plt.title("Wall Mask Used for Contour Detection")
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    plt.tight_layout()
    plt.show()


def plot_wall_lines(aligned_grid, wall_lines, title="Processed Wall Lines (Contours)"):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(aligned_grid, cmap="gray", vmin=-1, vmax=1, origin="lower", interpolation="nearest")

    for i, line in enumerate(wall_lines):
        ax.plot([line["x1"], line["x2"]], [line["y1"], line["y2"]], linewidth=2.5)
        mid_x = (line["x1"] + line["x2"]) / 2.0
        mid_y = (line["y1"] + line["y2"]) / 2.0
        ax.text(mid_x, mid_y, str(i + 1), fontsize=9, color="yellow", weight="bold")

    ax.set_title(title)
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    plt.tight_layout()
    plt.show()