import numpy as np
from ros_bag_reader import generate_raw_grid

from occupancy_cleaner import (
    clean_occupancy_grid,
    plot_comparison
)

from hough_detection import (
    detect_hough_lines,
    plot_edges,
    plot_hough_lines,
    plot_longest_lines
)

from orientation_alignment import (
    align_occupancy_grid,
    plot_alignment
)

from wall_line_processing import (
    process_wall_lines,
    plot_wall_lines,
    plot_wall_edges,
)

from room_polygon_builder import (
    build_room_polygons,
    plot_clean_vector_map,
    plot_snapped_walls,
    plot_clean_vector_map
)

BAG_PATH = r"D:\indoor mapping\2011-08-09-11-17-42.bag"  


def main():

    print("==============================================")
    print("INDOOR FLOOR PLAN PROCESSING PIPELINE")
    print("==============================================")


    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print(
        "STEP 1: Generating occupancy grid from ROS bag..."
    )

    (
        raw_grid,
        pose_x,
        pose_y,
        min_x,
        min_y,
        resolution,
        max_range
    ) = generate_raw_grid(
        BAG_PATH
    )

    print()
    print("Raw grid generated.")

    print(
        "Grid shape:",
        raw_grid.shape
    )

    print(
        "Resolution:",
        resolution,
        "m/cell"
    )


    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print(
        "STEP 2: Cleaning occupancy grid..."
    )

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
    print(
        "Cleaning finished."
    )

    plot_comparison(
        raw_grid,
        cleaned_grid
    )


    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print(
        "STEP 3: Detecting structural lines..."
    )

    detected_lines, edges = detect_hough_lines(
        cleaned_grid,
        resolution=resolution
    )

    print()
    print(
        "Hough detection finished."
    )


    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print(
        "STEP 4: Displaying Canny edges..."
    )

    plot_edges(
        edges
    )


    # ========================================================
    # STEP 5
    # ========================================================

    print()
    print(
        "STEP 5: Displaying Hough lines..."
    )

    plot_hough_lines(
        cleaned_grid,
        detected_lines,
        resolution=resolution
    )


    # ========================================================
    # STEP 6
    # ========================================================

    print()
    print(
        "STEP 6: Displaying longest candidate lines..."
    )

    plot_longest_lines(
        cleaned_grid,
        detected_lines,
        number_of_lines=30
    )


    # ========================================================
    # STEP 7
    # ========================================================

    print()
    print(
        "STEP 7: Aligning occupancy grid..."
    )

    (
        aligned_grid,
        dominant_angle,
        rotation_angle
    ) = align_occupancy_grid(
        cleaned_grid,
        detected_lines,
        resolution=resolution
    )

    print()
    print(
        "Alignment finished."
    )

    print(
        f"Dominant orientation: "
        f"{dominant_angle:.2f}°"
    )

    print(
        f"Applied rotation: "
        f"{rotation_angle:.2f}°"
    )


    # ========================================================
    # STEP 8
    # ========================================================

    print()
    print(
        "STEP 8: Displaying aligned occupancy grid..."
    )

    plot_alignment(
        cleaned_grid,
        aligned_grid,
        dominant_angle,
        rotation_angle
    )


    # ========================================================
    # STEP 9
    # ========================================================

    print()
    print("STEP 9: Detecting final wall lines on aligned map...")

    (
        wall_lines,
        wall_edges
    ) = process_wall_lines(
        aligned_grid,
        resolution=resolution
    )

    print()
    print("Wall line processing finished.")

    print(
        "Final wall candidates:",
        len(wall_lines)
    )
    

    # ========================================================
    # STEP 10
    # ========================================================
    
    print()
    print("STEP 10: Displaying wall detection edges...")

    plot_wall_edges(
        wall_edges
    ) 
    
    
    # ========================================================
    # STEP 11
    # ========================================================
    
    print()
    print("STEP 11: Displaying final wall lines...")

    plot_wall_lines(
        aligned_grid,
        wall_lines
    )


    # ========================================================
    # STEP 12
    # ========================================================

    print()
    print("STEP 12: Building rooms and snapping corners...")

    snapped_wall_lines = build_room_polygons(
        aligned_grid,
        wall_lines,
        resolution=resolution
    )


    # ========================================================
    # STEP 13
    # ========================================================

    print()
    print("STEP 13: Displaying snapped blueprint walls...")

    plot_snapped_walls(
        aligned_grid,
        snapped_wall_lines
    )

    # ========================================================
 # STEP 14
 # ========================================================

    print()
    print("STEP 14: Displaying final CAD-style vector map...")
   
    plot_clean_vector_map(
        aligned_grid,
        snapped_wall_lines
    )
    
    
    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("==============================================")
    print(
        "PROCESSING FINISHED"
    )
    print("==============================================")

    print()
    print("Generated files:")
    print("  occupancy_grid.npy")
    print("  occupancy_grid_cleaned.npy")
    print("  occupancy_grid_data.npz")
    print("  hough_lines.npy")
    print("  occupancy_grid_aligned.npy")
    print("  orientation_alignment_data.npz")
    print("  wall_lines.npy")
    print("  snapped_wall_lines.npy")

    print()
    print(
        "Final wall candidates:",
        len(wall_lines)
    )
    print(
        "Snapped wall segments:",
        len(snapped_wall_lines)
    )


if __name__ == "__main__":
    main()