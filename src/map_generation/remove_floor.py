import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def segment_and_remove_floor(input_path, output_path, floor_buffer=0.10):
    """
    Detects the floor by finding the most common Z-coordinate and removes it.
    
    Args:
        input_path: Path to cleaned .ply file.
        output_path: Path to save the wall-only .ply file.
        floor_buffer: How many meters ABOVE the detected floor to cut.
                      (e.g., 0.10 means cut 10cm above the calculated floor height)
    """
    # --- 1. Load Cloud ---
    print(f"Loading {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print("Error: Point cloud is empty.")
        return

    # --- 2. Histogram Analysis of Z-Axis ---
    # We create a histogram of just the Z (height) values.
    # The 'bins' determine the resolution of our height check (e.g., 2cm slices).
    z_values = points[:, 2]
    density, bin_edges = np.histogram(z_values, bins=100)
    
    # Find the bin with the most points (The Peak)
    peak_index = np.argmax(density)
    
    # The floor height is the center of that bin
    floor_height = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2
    print(f"Detected Floor Height: {floor_height:.3f} meters")

    # --- Optional: Visualize the Histogram ---
    # This helps you verify if the peak actually corresponds to the floor.
    plt.figure(figsize=(10, 4))
    plt.plot(bin_edges[:-1], density)
    plt.axvline(x=floor_height, color='r', linestyle='--', label='Detected Floor')
    plt.title("Point Cloud Height Distribution (Z-Axis)")
    plt.xlabel("Height (m)")
    plt.ylabel("Point Count")
    plt.legend()
    plt.show()

    # --- 3. Filter the Points ---
    # We keep everything ABOVE (floor + buffer).
    # We also (optionally) cut the ceiling if it's too high to be relevant.
    cutoff_height = floor_height + floor_buffer
    
    print(f"Cutting everything below: {cutoff_height:.3f} meters")
    
    # Create a boolean mask: True if point is above cutoff
    mask = points[:, 2] > cutoff_height
    
    # Apply the mask
    points_filtered = points[mask]
    
    # --- 4. Create New Point Cloud & Save ---
    walls_pcd = o3d.geometry.PointCloud()
    walls_pcd.points = o3d.utility.Vector3dVector(points_filtered)
    
    # Preserve colors if they exist (optional, helps visualization)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        walls_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    print(f"Original points: {len(points)}")
    print(f"Remaining points (Walls): {len(points_filtered)}")
    
    o3d.visualization.draw_geometries([walls_pcd], window_name="Walls Only")
    
    o3d.io.write_point_cloud(output_path, walls_pcd)
    print(f"Saved wall segments to {output_path}")

if __name__ == "__main__":
    # Use the cleaned file from the previous step
    segment_and_remove_floor("my_maze_cleaned.ply", "my_maze_walls.ply")