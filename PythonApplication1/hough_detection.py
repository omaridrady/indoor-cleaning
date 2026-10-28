import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# PARAMETERS
# ============================================================

# Canny edge detection
CANNY_LOW = 50
CANNY_HIGH = 150

# Hough Transform
HOUGH_THRESHOLD = 40

# Minimum line length in pixels
# At 0.05 m/cell:
# 100 pixels = 5 meters
MIN_LINE_LENGTH = 60

# Maximum gap allowed between points belonging to the
# same detected line
MAX_LINE_GAP = 15

# Thickness used when drawing detected lines
LINE_DRAW_THICKNESS = 2


# ============================================================
# DETECT HOUGH LINES
# ============================================================

def detect_hough_lines(cleaned_grid, resolution=0.05):

    print()
    print("==============================================")
    print("HOUGH LINE DETECTION")
    print("==============================================")

    # --------------------------------------------------------
    # 1. Extract wall pixels
    # --------------------------------------------------------

    print("1. Extracting wall pixels...")

    wall_mask = np.zeros(
        cleaned_grid.shape,
        dtype=np.uint8
    )

    wall_mask[cleaned_grid == 1] = 255


    # --------------------------------------------------------
    # 2. Edge detection
    # --------------------------------------------------------

    print("2. Detecting wall edges with Canny...")

    edges = cv2.Canny(
        wall_mask,
        CANNY_LOW,
        CANNY_HIGH
    )


    # --------------------------------------------------------
    # 3. Hough Line Transform
    # --------------------------------------------------------

    print("3. Running Probabilistic Hough Transform...")

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )


    # --------------------------------------------------------
    # No lines found
    # --------------------------------------------------------

    if lines is None:

        print("No lines were detected.")

        return [], edges


    # --------------------------------------------------------
    # 4. Convert Hough result
    # --------------------------------------------------------

    detected_lines = []

    # Flatten to a list of [x1, y1, x2, y2]
    lines = np.asarray(lines).reshape(-1, 4)

    for x1, y1, x2, y2 in lines:

        dx = x2 - x1
        dy = y2 - y1

        length_pixels = np.hypot(
            dx,
            dy
        )

        length_meters = (
            length_pixels * resolution
        )

        angle_degrees = np.degrees(
            np.arctan2(dy, dx)
        )

        # Normalize angle to [-90, 90)
        if angle_degrees >= 90:
            angle_degrees -= 180

        if angle_degrees < -90:
            angle_degrees += 180


        detected_lines.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "length_pixels": float(length_pixels),
            "length_meters": float(length_meters),
            "angle_degrees": float(angle_degrees)
        })


    # --------------------------------------------------------
    # 5. Sort longest lines first
    # --------------------------------------------------------

    detected_lines.sort(
        key=lambda line: line["length_pixels"],
        reverse=True
    )


    # --------------------------------------------------------
    # 6. Print results
    # --------------------------------------------------------

    print()
    print("Detected lines:", len(detected_lines))

    print()
    print("Longest detected lines:")

    for i, line in enumerate(
        detected_lines[:20]
    ):

        print(
            f"{i + 1:2d}. "
            f"Length = "
            f"{line['length_meters']:.2f} m, "
            f"Angle = "
            f"{line['angle_degrees']:.2f}°"
        )


    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    np.save(
        "hough_lines.npy",
        np.array(
            [
                [
                    l["x1"],
                    l["y1"],
                    l["x2"],
                    l["y2"],
                    l["length_pixels"],
                    l["length_meters"],
                    l["angle_degrees"]
                ]
                for l in detected_lines
            ],
            dtype=float
        )
    )

    print()
    print("Saved: hough_lines.npy")


    return detected_lines, edges


# ============================================================
# PLOT EDGES
# ============================================================

def plot_edges(edges):

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        edges,
        cmap="gray",
        origin="lower",
        interpolation="nearest"
    )

    plt.title(
        "Canny Edges of Cleaned Occupancy Grid"
    )

    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT HOUGH LINES
# ============================================================

def plot_hough_lines(
    cleaned_grid,
    detected_lines,
    resolution=0.05
):

    # --------------------------------------------------------
    # Floor-plan background
    #
    # Gray  = unknown
    # White = free
    # Black = wall
    # --------------------------------------------------------

    cmap = ListedColormap([
        "#C8C8C8",
        "#FFFFFF",
        "#000000"
    ])


    fig, ax = plt.subplots(
        figsize=(18, 10)
    )


    ax.imshow(
        cleaned_grid,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin="lower",
        interpolation="nearest"
    )


    # --------------------------------------------------------
    # Draw detected lines
    # --------------------------------------------------------

    for i, line in enumerate(
        detected_lines
    ):

        x1 = line["x1"]
        y1 = line["y1"]

        x2 = line["x2"]
        y2 = line["y2"]


        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.5
        )


        # Label only the first 20 lines
        # to avoid making the map unreadable.

        if i < 20:

            mid_x = (
                x1 + x2
            ) / 2

            mid_y = (
                y1 + y2
            ) / 2

            ax.text(
                mid_x,
                mid_y,
                str(i + 1),
                fontsize=8
            )


    ax.set_title(
        "Hough Line Detection"
    )

    ax.set_xlabel(
        "X (cells)"
    )

    ax.set_ylabel(
        "Y (cells)"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT ONLY LONGEST LINES
# ============================================================

def plot_longest_lines(
    cleaned_grid,
    detected_lines,
    number_of_lines=30
):

    cmap = ListedColormap([
        "#C8C8C8",
        "#FFFFFF",
        "#000000"
    ])


    fig, ax = plt.subplots(
        figsize=(18, 10)
    )


    ax.imshow(
        cleaned_grid,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin="lower",
        interpolation="nearest"
    )


    lines_to_draw = detected_lines[
        :number_of_lines
    ]


    for i, line in enumerate(
        lines_to_draw
    ):

        ax.plot(
            [line["x1"], line["x2"]],
            [line["y1"], line["y2"]],
            linewidth=2
        )


        mid_x = (
            line["x1"] +
            line["x2"]
        ) / 2

        mid_y = (
            line["y1"] +
            line["y2"]
        ) / 2


        ax.text(
            mid_x,
            mid_y,
            str(i + 1),
            fontsize=8
        )


    ax.set_title(
        f"Longest {len(lines_to_draw)} Hough Lines"
    )

    ax.set_xlabel(
        "X (cells)"
    )

    ax.set_ylabel(
        "Y (cells)"
    )

    plt.tight_layout()
    plt.show()