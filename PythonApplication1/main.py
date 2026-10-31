import numpy as np
from ros_bag_reader import generate_raw_grid
from occupancy_cleaner import (
    clean_occupancy_grid,
    plot_comparison
)

from wall_line_processing import (
    process_wall_lines,
    plot_wall_lines,
    plot_wall_edges,
    align_grid_to_walls   
)
from room_polygon_builder import (
    build_room_polygons,
    plot_clean_vector_map,
    plot_snapped_walls
)

BAG_PATH = r"D:\rosbag\PythonApplication1\data\run4_slam_map.yaml"
def main():

    print("==============================================")
    print("INDOOR FLOOR PLAN PROCESSING PIPELINE")
    print("==============================================")

    # ========================================================
    # STEP 1
    # ========================================================
    print()
    print("STEP 1: Generating occupancy grid from ROS bag...")

    (
        raw_grid,
        pose_x,
        pose_y,
        min_x,
        min_y,
        resolution,
        max_range
    ) = generate_raw_grid(BAG_PATH)

    print()
    print("Raw grid generated.")
    print("Grid shape:", raw_grid.shape)
    print("Resolution:", resolution, "m/cell")

    # ========================================================
    # STEP 2
    # ========================================================
    print()
    print("STEP 2: Cleaning occupancy grid...")

    cleaned_grid = clean_occupancy_grid(
        raw_grid,
        pose_x=pose_x,
        pose_y=pose_y,
        min_x=min_x,
        min_y=min_y,
        resolution=resolution,
        max_range=max_range
    )

    print()
    print("Cleaning finished.")
    plot_comparison(raw_grid, cleaned_grid)

    aligned_grid, rotation_deg, align_M = align_grid_to_walls(cleaned_grid, resolution)

    # ========================================================
    # STEP 7
    # ========================================================
    print()
    print("STEP 7: Detecting final wall lines on cleaned map...")

    (
        wall_lines,
        wall_edges
    ) = process_wall_lines(
        aligned_grid,
        resolution=resolution
    )

    print()
    print("Wall line processing finished.")
    print("Final wall candidates:", len(wall_lines))
    
    # ========================================================
    # STEP 8
    # ========================================================
    print()
    print("STEP 8: Displaying wall detection edges...")
    plot_wall_edges(wall_edges) 
    
    # ========================================================
    # STEP 9
    # ========================================================
    print()
    print("STEP 9: Displaying final wall lines...")

    plot_wall_lines(
        aligned_grid,
        wall_lines
    )

    # ========================================================
    # STEP 10
    # ========================================================
    print()
    print("STEP 10: Building rooms and snapping corners...")

    snapped_wall_lines = build_room_polygons(
        aligned_grid,
        wall_lines,
        resolution=resolution
    )

    # ========================================================
    # STEP 11
    # ========================================================
    print()
    print("STEP 11: Displaying snapped blueprint walls...")

    plot_snapped_walls(
        aligned_grid,
        snapped_wall_lines
    )

    # ========================================================
    # STEP 12
    # ========================================================
    print()
    print("STEP 12: Displaying final CAD-style vector map...")
   
    plot_clean_vector_map(
        aligned_grid,
        snapped_wall_lines
    )
    
    # ========================================================
    # FINISHED
    # ========================================================
    print()
    print("==============================================")
    print("PROCESSING FINISHED")
    print("==============================================")

    print()
    print("Generated files:")
    print("  occupancy_grid.npy")
    print("  occupancy_grid_cleaned.npy")
    print("  occupancy_grid_data.npz")
    print("  hough_lines.npy")
    print("  wall_lines.npy")
    print("  snapped_wall_lines.npy")

    print()
    print("Final wall candidates:", len(wall_lines))
    print("Snapped wall segments:", len(snapped_wall_lines))

if __name__ == "__main__":
    main()