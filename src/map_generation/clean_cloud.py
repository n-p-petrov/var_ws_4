import open3d as o3d
import numpy as np

def clean_point_cloud(input_path, output_path):
    # --- 1. Load the original PLY ---
    print(f"Loading {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"Original point count: {len(pcd.points)}")

    # Visualize Original (Paint it red for contrast)
    print("Visualizing ORIGINAL cloud... (Close window to proceed)")
    pcd.paint_uniform_color([1, 0, 0])  # Red
    o3d.visualization.draw_geometries([pcd], window_name="Original Noisy Cloud")

    # --- 2. Radius Outlier Removal ---
    # PARAMETERS TO TUNE:
    # nb_points: The minimum number of neighbors a point must have to be kept.
    # radius: The size of the sphere (in meters) to search for neighbors.
    # High 'nb_points' + Low 'radius' = Very aggressive cleaning
    
    nb_points = 64   # A point needs 16 neighbors...
    radius = 0.05    # ...within a 5cm radius to survive.
    
    print(f"Applying Radius Outlier Removal (Min {nb_points} neighbors in {radius}m radius)...")
    
    # cl is the cleaned cloud, ind is the list of indices kept
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    
    print(f"Cleaned point count: {len(cl.points)}")
    removed_count = len(pcd.points) - len(cl.points)
    print(f"Removed {removed_count} points.")

    # --- 3. Save and Visualize Result ---
    # Paint the clean cloud Green
    cl.paint_uniform_color([0, 1, 0])
    
    print("Visualizing CLEANED cloud... (Close window to finish)")
    o3d.visualization.draw_geometries([cl], window_name="Cleaned Cloud")
    
    o3d.io.write_point_cloud(output_path, cl)
    print(f"Saved cleaned cloud to {output_path}")

if __name__ == "__main__":
    # Edit these filenames
    clean_point_cloud("3cloud4.ply", "my_maze_cleaned.ply")