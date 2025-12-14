import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

def process_point_cloud(ply_path, output_image, resolution=0.05, height_min=0.2, height_max=2.0):
    """
    Converts a 3D PLY point cloud to a 2D occupancy grid map.
    
    Args:
        ply_path: Path to input .ply file.
        output_image: Path to save the result .png map.
        resolution: Map resolution in meters per pixel (default 5cm).
        height_min: Min height to consider for obstacles (removes floor).
        height_max: Max height to consider (removes ceiling).
    """
    
    # --- 1. Load and Visualize Raw Cloud ---
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Original points: {len(pcd.points)}")

    # --- 2. Downsampling (Voxel Grid Filter) ---
    # Reduces data density while preserving structure. Speeds up subsequent steps.
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    
    # --- 3. Outlier Removal (Statistical) ---
    # Removes points that are further away from their neighbors compared to the average.
    # Good for removing sensor noise/sparkles.
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    print(f"Points after cleanup: {len(pcd.points)}")

    # --- 4. Pass-Through Filter (Height Slicing) ---
    # We convert points to numpy array to filter by Z (height)
    points = np.asarray(pcd.points)
    
    # Filter: Keep points between min and max height
    # This automatically removes the ground plane and ceiling
    mask = (points[:, 2] > height_min) & (points[:, 2] < height_max)
    sliced_points = points[mask]

    if len(sliced_points) == 0:
        print("Error: No points found in the specified height range.")
        return

    # --- 5. Project to 2D Plane ---
    # We only care about X and Y coordinates now
    points_2d = sliced_points[:, :2]

    # --- 6. Rasterization (Grid Generation) ---
    # Determine the bounds of the map
    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)
    
    # Calculate grid dimensions
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    
    print(f"Map Size: {width} x {height} pixels")

    # Create an empty grid (0 = free space, 1 = obstacle)
    occupancy_grid = np.zeros((height, width), dtype=np.uint8)

    # Convert continuous coordinates to grid indices
    # We shift points by min_x/min_y so the map starts at (0,0)
    idxs_x = ((points_2d[:, 0] - min_x) / resolution).astype(int)
    idxs_y = ((points_2d[:, 1] - min_y) / resolution).astype(int)

    # Clip indices to ensure they are within bounds (handling edge cases)
    idxs_x = np.clip(idxs_x, 0, width - 1)
    idxs_y = np.clip(idxs_y, 0, height - 1)

    # Mark obstacles on the grid
    # Note: Images are typically accessed as [row, col] -> [y, x]
    occupancy_grid[idxs_y, idxs_x] = 1

    # --- 7. Post-Processing (Optional) ---
    # Dilate the walls slightly to make them more distinct and add safety margin
    # This connects sparse points into solid walls
    occupancy_grid = binary_dilation(occupancy_grid, iterations=2).astype(np.uint8)

    # --- 8. Save and Visualize ---
    # Invert colors: usually map means White=Free, Black=Obstacle
    plt.imsave(output_image, 1 - occupancy_grid, cmap='gray')
    print(f"Map saved to {output_image}")
    
    # Optional: Show the map
    plt.imshow(1 - occupancy_grid, cmap='gray')
    plt.title("Generated 2D Map")
    plt.show()

# Run the function
if __name__ == "__main__":
    # Replace with your specific file path
    process_point_cloud("3cloud4.ply", "maze_map.png")