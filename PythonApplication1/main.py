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
from vector_export import merge_walls, export_floorplan
from door_outline_prototype import (
    outline as build_outline,
    export as export_outline,
    plot_vector_panels,
    doors_closeup,
    count_stretches,
)

# ============================================================
# SETTINGS
# ============================================================

# The SLAM map (.yaml; the .pgm it points to must be next to it)
BAG_PATH = r"D:\rosbag\PythonApplication1\data\KASIT_1st_floor\KASIT_1st_floor_slam_map.yaml"

# Every result file goes to OUTPUT_ROOT\<map name>, never into this code folder.
# The map name is the .yaml file name without "_slam_map", e.g.
#   data\run5\run5_slam_map.yaml                       -> outputs\run5
#   data\KASIT_3rd_floor\KASIT_3rd_floor_slam_map.yaml -> outputs\KASIT_3rd_floor
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

# Trained FPUNet weights (.pth). FPUNet.py must be in this project folder.
# If this file is missing, the pipeline still runs with the classic cleaner only.
FPUNET_WEIGHTS_PATH = r"D:\rosbag\best_fpunet_128_10.pth"

# True  = stop with a clear error if FPUNet cannot run (so you never get a
#         classic-only result by accident).
# False = print a warning and continue with the classic cleaner only.
REQUIRE_FPUNET = True

MERGE_WALLS = True              # join broken walls + drop isolated stubs (see vector_export.py)
SHOW_DEBUG_PLOTS = False       # per-approach wall mask + wall line plots
SHOW_INDIVIDUAL_MAPS = False    # extra single-map windows (the side-by-side figure already shows both)

# Third result: outline of the explored free space as ONE closed border, with
# doors / wide openings left open (beam-fan rule, see door_outline_prototype.py).
# It is built from the FPUNet grid (the classic grid if FPUNet was skipped) and
# appears as the third panel of the vector figure. False = previous behaviour.
MAKE_OUTLINE = True


# ============================================================
# HELPERS
# ============================================================

def output_dir_for(map_yaml):
    name = os.path.splitext(os.path.basename(map_yaml))[0]
    if name.endswith("_slam_map"):
        name = name[: -len("_slam_map")]
    path = os.path.abspath(os.path.join(OUTPUT_ROOT, name))
    os.makedirs(path, exist_ok=True)
    return path


def save_lines(path, lines):
    data = [[l["x1"], l["y1"], l["x2"], l["y2"],
             l["length_pixels"], l["length_meters"], l["angle_degrees"]]
            for l in lines]
    np.save(path, np.asarray(data, dtype=float))


def run_vector_pipeline(label, grid, resolution, angle, map_origin):
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
    print(f"{label}: snapped wall segments = {len(snapped)}")

    # Join broken / duplicate walls into long ones, drop tiny isolated stubs.
    if MERGE_WALLS:
        snapped = merge_walls(snapped, resolution)
    save_lines(f"snapped_wall_lines_{label}.npy", snapped)

    # Vector file for the app (aligned frame, meters)
    export_floorplan(
        snapped, resolution, align_M, map_origin, rotation_deg,
        base_name=f"floorplan_{label}", source_name=os.path.basename(BAG_PATH)
    )

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

    origin = (min_x, min_y)
    results = {"classic": run_vector_pipeline("classic", classic_grid, resolution, angle, origin)}
    if fpunet_grid is not None:
        results["fpunet"] = run_vector_pipeline("fpunet", fpunet_grid, resolution, angle, origin)

    # ========================================================
    # STEP 5b: third result - free-space outline with doors
    # ========================================================
    # Uses the aligned grid and rotation STEP 5 already made; classic and
    # FPUNet results above are not touched.
    outline_lines = None
    outline_src = "fpunet" if "fpunet" in results else "classic"
    if MAKE_OUTLINE:
        print()
        print("##############################################")
        print(f"OUTLINE WITH DOORS: {outline_src}")
        print("##############################################")
        r = results[outline_src]
        raw_aligned, _, _ = align_grid_to_walls(raw_grid, resolution, angle_override=angle)
        outline_lines, _, _, _ = build_outline(r["aligned_grid"], raw_aligned, resolution)
        export_outline(
            outline_lines, resolution, r["align_M"], origin, r["rotation_deg"],
            "floorplan_outline", source_name=os.path.basename(BAG_PATH), cleaner=outline_src
        )
        doors_closeup(raw_aligned, outline_lines, resolution)
        lengths = {k: sum(l["length_meters"] for l in outline_lines if l["kind"] == k)
                   for k in ("wall", "door", "opening", "unseen")}
        print(f"outline: {len(outline_lines)} segments, closed border, "
              f"{count_stretches(outline_lines, 'door')} doors, "
              f"{count_stretches(outline_lines, 'opening')} wide openings")
        print("outline lengths (m): " + ", ".join(f"{k} {v:.1f}" for k, v in lengths.items()))

    # ========================================================
    # STEP 6: compare the blueprints
    # ========================================================
    print()
    print("STEP 6: Comparing the final blueprints...")

    if "fpunet" in results:
        c, f = results["classic"], results["fpunet"]
        compare_line_sets(c["snapped"], "classic", f["snapped"], "fpunet", resolution)
        if outline_lines is None:
            plot_vector_comparison(
                c["aligned_grid"], c["snapped"], "Classic cleaning",
                f["aligned_grid"], f["snapped"], "FPUNet cleaning"
            )

    if outline_lines is not None:
        # Same two panels as before, plus the outline as the third one.
        panels = [(results["classic"]["aligned_grid"], results["classic"]["snapped"], "Classic cleaning", "cad")]
        if "fpunet" in results:
            panels.append((results["fpunet"]["aligned_grid"], results["fpunet"]["snapped"], "FPUNet cleaning", "cad"))
        panels.append((results[outline_src]["aligned_grid"], outline_lines,
                       f"Outline with doors ({outline_src})", "outline"))
        plot_vector_panels(panels)

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
    if outline_lines is not None:
        print("  floorplan_outline.json / .geojson / .svg")
        print("  doors_closeup.png")
        print("  vector_comparison.png (3 panels)")


if __name__ == "__main__":
    # The pipeline modules save their files to the current folder, so run
    # inside the map's output folder and come back afterwards.
    out_dir = output_dir_for(BAG_PATH)
    start_dir = os.getcwd()
    os.chdir(out_dir)
    try:
        print(f"Output folder: {out_dir}")
        main()
    finally:
        os.chdir(start_dir)