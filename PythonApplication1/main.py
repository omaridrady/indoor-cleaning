import os
import sys
import numpy as np

from ros_bag_reader import generate_raw_grid
from occupancy_cleaner import clean_occupancy_grid
from FPUNet_cleaner import (
    load_fpunet,
    clean_with_fpunet,
    plot_three_way,
    plot_vector_comparison,
    compare_line_sets,
)
from wall_line_processing import (
    estimate_dominant_angle,
    align_grid_to_walls,
    process_wall_lines,
    plot_wall_lines,
    plot_wall_edges,
)
from room_polygon_builder import (
    build_room_polygons,
    plot_clean_vector_map,
)

# ============================================================
# SETTINGS
# ============================================================

# The SLAM map (.yaml; the .pgm it points to must be next to it)
BAG_PATH = r"D:\rosbag\PythonApplication1\data\run5_slam_map.yaml"

# Trained FPUNet weights (.pth). FPUNet.py must be in this project folder.
# If this file is missing, the pipeline still runs with the classic cleaner only.
FPUNET_WEIGHTS_PATH = r"D:\rosbag\best_fpunet_128_10.pth"

# True  = stop with a clear error if FPUNet cannot run (so you never get a
#         classic-only result by accident).
# False = print a warning and continue with the classic cleaner only.
REQUIRE_FPUNET = True

SHOW_DEBUG_PLOTS = False        # per-approach wall mask + wall line plots
SHOW_INDIVIDUAL_MAPS = False    # extra single-map windows (the side-by-side figure already shows both)


# ============================================================
# HELPERS
# ============================================================

def save_lines(path, lines):
    data = [[l["x1"], l["y1"], l["x2"], l["y2"],
             l["length_pixels"], l["length_meters"], l["angle_degrees"]]
            for l in lines]
    np.save(path, np.asarray(data, dtype=float))


def run_vector_pipeline(label, grid, resolution, angle):
    """
    Align -> wall lines -> corner snapping, for one cleaned grid.
    `angle` is shared by both approaches so their results are in the same frame.
    """
    print()
    print("##############################################")
    print(f"VECTORIZING: {label}")
    print("##############################################")

    aligned_grid, rotation_deg, align_M = align_grid_to_walls(
        grid, resolution, angle_override=angle
    )

    wall_lines, wall_edges = process_wall_lines(
        aligned_grid, resolution=resolution, save_result=False
    )
    print(f"{label}: final wall candidates = {len(wall_lines)}")

    if SHOW_DEBUG_PLOTS:
        plot_wall_edges(wall_edges)
        plot_wall_lines(aligned_grid, wall_lines, title=f"Wall lines ({label})")

    snapped = build_room_polygons(
        aligned_grid, wall_lines, resolution=resolution, save_result=False
    )
    save_lines(f"snapped_wall_lines_{label}.npy", snapped)
    print(f"{label}: snapped wall segments = {len(snapped)}")

    return {
        "aligned_grid": aligned_grid,
        "rotation_deg": rotation_deg,
        "align_M": align_M,
        "wall_lines": wall_lines,
        "snapped": snapped,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================================")
    print("INDOOR FLOOR PLAN PROCESSING PIPELINE")
    print("==============================================")

    # ========================================================
    # STEP 1: load the SLAM map
    # ========================================================
    print()
    print("STEP 1: Loading SLAM map...")

    (
        raw_grid,
        pose_x,
        pose_y,
        min_x,
        min_y,
        resolution,
        max_range
    ) = generate_raw_grid(BAG_PATH)

    print("Grid shape:", raw_grid.shape)
    print("Resolution:", resolution, "m/cell")

    # ========================================================
    # STEP 2: classic cleaning
    # ========================================================
    print()
    print("STEP 2: Cleaning with the classic filters...")

    classic_grid = clean_occupancy_grid(
        raw_grid,
        resolution=resolution,
        pose_x=pose_x,
        pose_y=pose_y,
        min_x=min_x,
        min_y=min_y,
        max_range=max_range
    )

    # ========================================================
    # STEP 3: FPUNet cleaning (skipped with a message if not available)
    # ========================================================
    print()
    print("STEP 3: Cleaning with FPUNet...")

    fpunet_grid = None
    problem = None
    if not os.path.exists(FPUNET_WEIGHTS_PATH):
        problem = f"FPUNet weights not found at: {FPUNET_WEIGHTS_PATH}"
    else:
        try:
            net, device = load_fpunet(FPUNET_WEIGHTS_PATH)
            print(f"Loaded FPUNet on {device}.")
            fpunet_grid, _ = clean_with_fpunet(raw_grid, resolution, net, device)
            np.save("occupancy_grid_fpunet.npy", fpunet_grid)
        except ImportError as e:
            problem = (f"FPUNet could not be imported: {e}\n"
                       f"This script runs with: {sys.executable}\n"
                       f"torch / torchvision must be installed for THAT Python, and "
                       f"FPUNet.py must be in the project folder.")

    if problem:
        if REQUIRE_FPUNET:
            raise RuntimeError(
                "\n" + problem + "\nFix this, or set REQUIRE_FPUNET = False to run "
                "the classic cleaner only."
            )
        print()
        print("!" * 60)
        print("WARNING: FPUNet was SKIPPED. You will only see the classic result.")
        print(problem)
        print("!" * 60)

    # ========================================================
    # STEP 4: compare the cleaned grids
    # ========================================================
    print()
    print("STEP 4: Showing the cleaning results...")

    if fpunet_grid is not None:
        plot_three_way(raw_grid, classic_grid, fpunet_grid)
    else:
        from occupancy_cleaner import plot_comparison
        plot_comparison(raw_grid, classic_grid)

    # ========================================================
    # STEP 5: vectorize each approach with the SAME rotation
    # ========================================================
    # The rotation is estimated once, from the classic-cleaned grid, and used for
    # both approaches, so both blueprints share one coordinate frame.
    angle = estimate_dominant_angle(classic_grid, resolution)

    results = {"classic": run_vector_pipeline("classic", classic_grid, resolution, angle)}
    if fpunet_grid is not None:
        results["fpunet"] = run_vector_pipeline("fpunet", fpunet_grid, resolution, angle)

    # ========================================================
    # STEP 6: compare the blueprints
    # ========================================================
    print()
    print("STEP 6: Comparing the final blueprints...")

    if "fpunet" in results:
        c, f = results["classic"], results["fpunet"]
        compare_line_sets(c["snapped"], "classic", f["snapped"], "fpunet", resolution)
        plot_vector_comparison(
            c["aligned_grid"], c["snapped"], "Classic cleaning",
            f["aligned_grid"], f["snapped"], "FPUNet cleaning"
        )

    if SHOW_INDIVIDUAL_MAPS:
        for label, r in results.items():
            plot_clean_vector_map(
                r["aligned_grid"], r["snapped"],
                save_path=f"vector_floor_plan_{label}.png"
            )

    # ========================================================
    # FINISHED
    # ========================================================
    print()
    print("==============================================")
    print("PROCESSING FINISHED")
    print("==============================================")
    print("Generated files:")
    print("  occupancy_grid_cleaned.npy")
    for label in results:
        print(f"  snapped_wall_lines_{label}.npy")
        print(f"  vector_floor_plan_{label}.png")
    if fpunet_grid is not None:
        print("  occupancy_grid_fpunet.npy")


if __name__ == "__main__":
    main()