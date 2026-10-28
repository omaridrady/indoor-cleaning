import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Douglas-Peucker Polygon Approximation Sensitivity
# Lower = more detailed vertices, Higher = smoother/stricter lines
DP_EPSILON_FACTOR = 0.035

# Wall filtering
MIN_WALL_LENGTH_METERS = 1.2

# Parallel line clustering & merging
MAX_PARALLEL_DISTANCE_METERS = 0.5
MAX_MERGE_GAP_METERS = 1.2 
MAX_PARALLEL_ANGLE_DEGREES = 12.0


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
    points = []
    for line in lines:
        points.append([line["x1"], line["y1"]])
        points.append([line["x2"], line["y2"]])
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return None

    center = np.mean(points, axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
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
    Removes shorter parallel lines that overlap larger dominant walls.
    Works dynamically for both horizontal and vertical families.
    """
    if not lines:
        return []
    
    # Sort by length descending, so we always keep the longest, most dominant walls
    lines.sort(key=lambda l: l["length_pixels"], reverse=True)
    
    keep = []
    for current in lines:
        is_duplicate = False
        
        for kept in keep:
            # Check if they are side-by-side within the tolerance
            p_dist = parallel_distance(current, kept, family)
            
            if p_dist <= (MAX_PARALLEL_DISTANCE_METERS / resolution):
                # Check if they physically overlap along their primary axis (gap == 0.0)
                if segment_gap(current, kept, family) == 0.0:
                    is_duplicate = True
                    break
        
        # Only add the line if it didn't overlap a larger, already-kept line
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

   
    # ---> NEW: Run the suppressor to wipe out the stacked/overlapping duplicates
    final_lines = suppress_duplicate_lines(final_lines, family, resolution)

    return final_lines


# ============================================================
# DETECT WALL CANDIDATES VIA CONTOURS & DOUGLAS-PEUCKER
# ============================================================

def detect_wall_lines(aligned_grid, resolution=0.05):
    print()
    print("==============================================")
    print("WALL LINE PROCESSING (CONTOUR APPROXIMATION)")
    print("==============================================")

    # 1. Wall mask
    print("1. Extracting wall mask...")
    wall_mask = np.zeros(aligned_grid.shape, dtype=np.uint8)
    wall_mask[aligned_grid == 1] = 255

    # 2. Find Contours
    print("2. Finding wall contours...")
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw_segments = []

    # 3. Douglas-Peucker Polygon Approximation
    print("3. Running Douglas-Peucker Polygon Approximation...")
    for cnt in contours:
        epsilon = DP_EPSILON_FACTOR * cv2.arcLength(cnt, True)
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