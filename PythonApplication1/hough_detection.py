import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.morphology import skeletonize

# ============================================================
# PARAMETERS (distances in meters, so they survive resolution changes)
# ============================================================
USE_SKELETON = True        # one centerline per wall; False = Canny on the raw mask
CANNY_LOW = 50             # only used if USE_SKELETON is False
CANNY_HIGH = 150

HOUGH_THRESHOLD_M = 0.6    # minimum evidence (in meters of line) for a candidate
MIN_LINE_LENGTH = 1.0      # meters
MAX_LINE_GAP = 0.25        # meters; keep below door width (~0.8 m)


def detect_hough_lines(cleaned_grid, resolution, min_x=None, min_y=None):
    print()
    print("==============================================")
    print("HOUGH LINE DETECTION")
    print("==============================================")

    min_len_px = max(1, int(round(MIN_LINE_LENGTH / resolution)))
    max_gap_px = max(1, int(round(MAX_LINE_GAP / resolution)))
    threshold = max(1, int(round(HOUGH_THRESHOLD_M / resolution)))
    print(f"min length={min_len_px}px, max gap={max_gap_px}px, threshold={threshold} votes")

    print("1. Extracting wall pixels...")
    wall_mask = (cleaned_grid == 1)

    if USE_SKELETON:
        print("2. Skeletonizing walls to centerlines...")
        edges = skeletonize(wall_mask).astype(np.uint8) * 255
    else:
        print("2. Detecting wall edges with Canny...")
        edges = cv2.Canny(wall_mask.astype(np.uint8) * 255, CANNY_LOW, CANNY_HIGH)

    print("3. Running Probabilistic Hough Transform...")
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=threshold,
        minLineLength=min_len_px, maxLineGap=max_gap_px
    )

    lines = []
    if raw is not None:
       for x1, y1, x2, y2 in raw.reshape(-1, 4):
            length_px = float(np.hypot(x2 - x1, y2 - y1))
            line = {
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),  # cells
                "length_m": length_px * resolution,
                "angle_deg": float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0),
            }
            if min_x is not None and min_y is not None:
                line["x1_m"] = min_x + x1 * resolution
                line["y1_m"] = min_y + y1 * resolution
                line["x2_m"] = min_x + x2 * resolution
                line["y2_m"] = min_y + y2 * resolution
            lines.append(line)

    lines.sort(key=lambda l: l["length_m"], reverse=True)
    print(f"Detected {len(lines)} line segments")
    return lines, edges
def plot_edges(edges):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(edges, cmap="gray", origin="lower", interpolation="nearest")
    ax.set_title("Wall centerlines used for Hough detection")
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    plt.tight_layout()
    plt.show()


def plot_hough_lines(
    cleaned_grid,
    detected_lines,
    resolution
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

