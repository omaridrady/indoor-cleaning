import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# PARAMETERS
# ============================================================

MIN_LINE_LENGTH_METERS = 3.0

ANGLE_BIN_SIZE = 1.0

OUTPUT_GRID_FILE = "occupancy_grid_aligned.npy"
OUTPUT_INFO_FILE = "orientation_alignment_data.npz"


# ============================================================
# ANGLE NORMALIZATION
# ============================================================
def normalize_angle(angle):
    """
    Normalize a line orientation to [-90, 90).
    Supports both individual scalar values and NumPy arrays.
    """
    # Check if the input is a single scalar number
    is_scalar = np.isscalar(angle)
    
    # Convert to a numpy array for uniform processing
    arr = np.asarray(angle, dtype=float)
    
    arr = arr % 180.0
    
    # Vectorized condition check using np.where instead of a Python 'if'
    arr = np.where(arr >= 90.0, arr - 180.0, arr)
    
    # Return a standard Python float if a scalar was passed, otherwise return the array
    return arr.item() if is_scalar else arr


# ============================================================
# EXTRACT LINE DATA
# ============================================================

def extract_line_data(
    detected_lines,
    min_line_length=MIN_LINE_LENGTH_METERS
):
    """
    Extract lengths and orientations from the dictionaries
    produced by hough_detection.py.

    Each Hough line has:

        x1
        y1
        x2
        y2
        length_pixels
        length_meters
        angle_degrees
    """

    if detected_lines is None:
        raise ValueError(
            "detected_lines is None."
        )

    if len(detected_lines) == 0:
        raise ValueError(
            "No Hough lines were detected."
        )

    lengths = []
    angles = []

    for line in detected_lines:

        # ----------------------------------------------------
        # The Hough detector returns dictionaries.
        # ----------------------------------------------------

        length = float(
            line["length_meters"]
        )

        angle = float(
            line["angle_degrees"]
        )

        # ----------------------------------------------------
        # Ignore short lines.
        # ----------------------------------------------------

        if length < min_line_length:
            continue

        lengths.append(
            length
        )

        angles.append(
            normalize_angle(angle)
        )

    lengths = np.asarray(
        lengths,
        dtype=float
    )

    angles = np.asarray(
        angles,
        dtype=float
    )

    if len(angles) == 0:
        raise ValueError(
            "No Hough lines remain after the "
            "minimum-length filter."
        )

    print(
        f"Valid lines used for orientation analysis: "
        f"{len(angles)}"
    )

    return lengths, angles


# ============================================================
# FIND DOMINANT ORIENTATION
# ============================================================

def find_dominant_orientation(lengths, angles):
    """
    Finds the optimal angle by searching for a pair of orthogonal peaks 
    (~90° apart) that maximize combined wall length.
    """
    best_angle = 0.0
    max_score = -1.0
    
    # Search every possible base angle from -45 to 45 degrees
    for candidate_angle in np.linspace(-45.0, 45.0, 180):
        # Score lines matching candidate or candidate + 90
        diffs1 = np.abs(normalize_angle(angles - candidate_angle))
        diffs2 = np.abs(normalize_angle(angles - (candidate_angle + 90.0)))
        
        match1 = diffs1 <= 5.0
        match2 = diffs2 <= 5.0
        
        # Combined weight of both orthogonal families
        score = np.sum(lengths[match1]) + np.sum(lengths[match2])
        
        if score > max_score:
            max_score = score
            best_angle = candidate_angle

    # Many candidates inside the ±5° window tie on score, and the loop keeps the
    # first (an edge of the window). Refine to the length-weighted mean angle
    # of the lines that matched.
    offsets1 = normalize_angle(angles - best_angle)
    offsets2 = normalize_angle(angles - (best_angle + 90.0))
    offsets = np.where(np.abs(offsets1) <= np.abs(offsets2), offsets1, offsets2)
    matched = np.abs(offsets) <= 5.0
    if np.any(matched):
        best_angle += np.average(offsets[matched], weights=lengths[matched])

    return normalize_angle(best_angle)


# ============================================================
# ROTATE OCCUPANCY GRID
# ============================================================

def rotate_occupancy_grid(
    grid,
    rotation_angle
):
    """
    Rotate the occupancy grid without cropping the map.

    Occupancy values remain:

        -1 = unknown
         0 = free
         1 = occupied

    Nearest-neighbor interpolation is used so that we do not
    create intermediate occupancy values.
    """

    height, width = grid.shape

    center = (
        width / 2.0,
        height / 2.0
    )

    # --------------------------------------------------------
    # Create rotation matrix.
    # --------------------------------------------------------

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        rotation_angle,
        1.0
    )

    # --------------------------------------------------------
    # Calculate new dimensions.
    # --------------------------------------------------------

    cos_value = abs(
        rotation_matrix[0, 0]
    )

    sin_value = abs(
        rotation_matrix[0, 1]
    )

    new_width = int(
        (height * sin_value) +
        (width * cos_value)
    )

    new_height = int(
        (height * cos_value) +
        (width * sin_value)
    )

    # --------------------------------------------------------
    # Move rotated image to the center of the new canvas.
    # --------------------------------------------------------

    rotation_matrix[0, 2] += (
        new_width / 2.0 -
        center[0]
    )

    rotation_matrix[1, 2] += (
        new_height / 2.0 -
        center[1]
    )

    # --------------------------------------------------------
    # Rotate.
    # --------------------------------------------------------

    aligned_grid = cv2.warpAffine(
        grid,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1
    )

    return aligned_grid


# ============================================================
# MAIN ALIGNMENT FUNCTION
# ============================================================

def align_occupancy_grid(
    cleaned_grid,
    detected_lines,
    resolution=0.05,
    min_line_length=MIN_LINE_LENGTH_METERS,
    save_result=True
):
    """
    Align the cleaned occupancy grid using Hough line data.

    Parameters
    ----------
    cleaned_grid:
        Cleaned occupancy grid produced by occupancy_cleaner.py.

    detected_lines:
        List of dictionaries returned by hough_detection.py.

    resolution:
        Occupancy grid resolution in meters/cell.

    min_line_length:
        Minimum Hough line length used for orientation analysis.

    save_result:
        Save the aligned grid if True.

    Returns
    -------
    aligned_grid
    dominant_angle
    rotation_angle
    """

    print()
    print("==============================================")
    print("ORIENTATION ALIGNMENT")
    print("==============================================")

    # ========================================================
    # Validate occupancy grid
    # ========================================================

    if cleaned_grid is None:

        raise ValueError(
            "cleaned_grid is None."
        )

    if cleaned_grid.ndim != 2:

        raise ValueError(
            "cleaned_grid must be a 2D numpy array."
        )

    print(
        "Input grid shape:",
        cleaned_grid.shape
    )

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print(
        "1. Analyzing Hough line orientations..."
    )

    lengths, angles = extract_line_data(
        detected_lines,
        min_line_length=min_line_length
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print(
        "2. Finding dominant wall orientation..."
    )

    dominant_angle = find_dominant_orientation(
        lengths,
        angles
    )

    print(
        f"Dominant wall orientation: "
        f"{dominant_angle:.2f}°"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print(
        "3. Calculating required rotation..."
    )

    # To make the dominant wall horizontal,
    # rotate the map in the opposite direction.

    rotation_angle = dominant_angle

    print(
        f"Rotation applied to map: "
        f"{rotation_angle:.2f}°"
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print(
        "4. Rotating occupancy grid..."
    )

    aligned_grid = rotate_occupancy_grid(
        cleaned_grid,
        rotation_angle
    )

    print(
        "Output grid shape:",
        aligned_grid.shape
    )

    # ========================================================
    # STEP 5
    # ========================================================

    if save_result:

        print()
        print(
            "5. Saving aligned occupancy grid..."
        )

        np.save(
            OUTPUT_GRID_FILE,
            aligned_grid
        )

        np.savez(
            OUTPUT_INFO_FILE,
            dominant_angle=dominant_angle,
            rotation_angle=rotation_angle,
            resolution=resolution
        )

        print(
            "Saved:",
            OUTPUT_GRID_FILE
        )

        print(
            "Saved:",
            OUTPUT_INFO_FILE
        )

    print()
    print(
        "Orientation alignment finished."
    )

    return (
        aligned_grid,
        dominant_angle,
        rotation_angle
    )


# ============================================================
# VISUALIZATION
# ============================================================

def plot_alignment(
    cleaned_grid,
    aligned_grid,
    dominant_angle,
    rotation_angle
):
    """
    Display the occupancy grid before and after alignment.
    """

    cmap = ListedColormap([
        "#C8C8C8",   # unknown
        "#FFFFFF",   # free
        "#000000"    # occupied
    ])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 9)
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    axes[0].imshow(
        cleaned_grid,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin="lower",
        interpolation="nearest"
    )

    axes[0].set_title(
        "Before Alignment\n"
        f"Dominant Orientation = "
        f"{dominant_angle:.2f}°"
    )

    axes[0].set_xlabel(
        "X (cells)"
    )

    axes[0].set_ylabel(
        "Y (cells)"
    )

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    axes[1].imshow(
        aligned_grid,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin="lower",
        interpolation="nearest"
    )

    axes[1].set_title(
        "After Alignment\n"
        f"Rotation = "
        f"{rotation_angle:.2f}°"
    )

    axes[1].set_xlabel(
        "X (cells)"
    )

    axes[1].set_ylabel(
        "Y (cells)"
    )

    plt.tight_layout()
    plt.show()