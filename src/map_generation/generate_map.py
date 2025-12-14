import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

def generate_navigation_map(input_path, output_img_path, resolution=0.05, robot_radius=0.20):
    """
    Projects a wall-only point cloud to a 2D grid and dilates obstacles 
    to account for robot size (Configuration Space).
    
    Args:
        resolution: Meters per pixel (e.g., 0.05 = 5cm per pixel).
        robot_radius: The safety margin to add around walls (in meters).
    """
    # --- 1. Load Walls ---
    print(f"Loading {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)

    if len(points) == 0:
        print("Error: No points in cloud.")
        return

    # --- 2. Define Map Bounds ---
    # We only care about X and Y now
    vals_x = points[:, 0]
    vals_y = points[:, 1]
    
    min_x, max_x = np.min(vals_x), np.max(vals_x)
    min_y, max_y = np.min(vals_y), np.max(vals_y)
    
    # Calculate grid size
    width_meters = max_x - min_x
    height_meters = max_y - min_y
    
    # +1 ensures we don't go out of bounds due to rounding
    cols = int(np.ceil(width_meters / resolution)) + 1
    rows = int(np.ceil(height_meters / resolution)) + 1
    
    print(f"Map Grid Size: {rows} x {cols}")
    print(f"Real World Dimensions: {width_meters:.2f}m x {height_meters:.2f}m")

    # --- 3. Rasterization (Quantization) ---
    # Create the empty grid (0 = Free, 1 = Obstacle)
    grid = np.zeros((rows, cols), dtype=np.uint8)
    
    # Convert continuous coordinates to integer grid indices
    # (row = y, col = x)
    idx_cols = ((vals_x - min_x) / resolution).astype(int)
    idx_rows = ((vals_y - min_y) / resolution).astype(int)
    
    # Mark the walls on the grid
    grid[idx_rows, idx_cols] = 1

    # --- 4. Inflation (Configuration Space) ---
    # We calculate how many pixels correspond to the robot's radius
    # e.g., if radius is 0.2m and res is 0.05, we need to dilate by 4 pixels.
    dilation_pixels = int(np.ceil(robot_radius / resolution))
    
    print(f"Inflating obstacles by {dilation_pixels} pixels ({robot_radius}m)...")
    
    # Use scipy to dilate the walls
    # This creates the "Keep Out" zones for the path planner
    inflated_grid = binary_dilation(grid, iterations=dilation_pixels).astype(np.uint8)

    # --- 5. Save and Visualize ---
    
    # Save as an image (White = Free, Black = Obstacle)
    # Most planners expect: 255 (white) = free, 0 (black) = occupied
    plt.imsave(output_img_path, 1 - inflated_grid, cmap='gray')
    
    # ALSO save as a raw numpy array. 
    # This is what you should load into your A* script to avoid image compression artifacts.
    np.save(output_img_path.replace('.png', '.npy'), grid)
    
    print(f"Saved map image to {output_img_path}")
    print(f"Saved raw grid data to {output_img_path.replace('.png', '.npy')}")

    # Plotting for verification
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(1 - grid, cmap='gray', origin='lower')
    ax1.set_title("Extracted Walls")
    
    ax2.imshow(1 - inflated_grid, cmap='gray', origin='lower')
    ax2.set_title(f"Inflated Map (Safety Radius {robot_radius}m)")
    
    plt.show()

if __name__ == "__main__":
    # Use the walls-only file
    generate_navigation_map("my_maze_walls_clean.ply", "final_map.png")