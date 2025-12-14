import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def clean_and_bound_map(input_path, output_img_path, min_cluster_size=70):
    """
    Removes small noise blobs and adds a border to the map.
    """
    # 1. Load the raw grid (0=Free, 1=Obstacle)
    try:
        grid = np.load(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return

    print(f"Loaded map: {grid.shape}")
    
    # --- Step 1: Remove Small Noise Blobs ---
    # We label every distinct 'island' of black pixels
    # structure=[[1,1,1],[1,1,1],[1,1,1]] defines connectivity (8-connected)
    structure = np.ones((3, 3), dtype=np.int8)
    labeled_array, num_features = label(grid, structure=structure)
    
    print(f"Found {num_features} distinct obstacle clusters.")
    
    # Iterate through all clusters and check their size
    new_grid = grid.copy()
    cleaned_count = 0
    
    # 'bincount' counts the number of pixels in each label index
    sizes = np.bincount(labeled_array.ravel())
    
    # Note: Label 0 is the background (free space), so we skip index 0
    mask_sizes = sizes < min_cluster_size
    mask_sizes[0] = 0 # Ensure we don't remove the background!
    
    # Identify labels to remove
    labels_to_remove = np.where(mask_sizes)[0]
    
    # Set pixels belonging to those labels to 0 (Free)
    # isin creates a boolean mask where the labeled_array matches labels_to_remove
    noise_mask = np.isin(labeled_array, labels_to_remove)
    new_grid[noise_mask] = 0
    
    print(f"Removed {len(labels_to_remove)} small noise clusters (size < {min_cluster_size}).")

    # --- Step 2: Bound the Map (Add Border) ---
    # This forces the path planner to stay inside the known area.
    # We set the outer edges of the matrix to 1 (Obstacle).
    border_thickness = 2
    
    rows = np.any(new_grid == 1, axis=1)
    cols = np.any(new_grid == 1, axis=0)

    new_grid = new_grid[rows][:, cols]

    # Top and Bottom
    new_grid[:border_thickness, :] = 1
    new_grid[-border_thickness:, :] = 1
    
    # Left and Right
    new_grid[:, :border_thickness] = 1
    new_grid[:, -border_thickness:] = 1
    
    print("Added bounding walls to map edges.")

    # --- Step 3: Save and Visualize ---
    
    # Save as .npy (for A*) and .png (for viewing)
    np.save(output_img_path.replace('.png', '.npy'), new_grid)
    plt.imsave(output_img_path, 1 - new_grid, cmap='gray')
    
    print(f"Saved cleaned map to {output_img_path}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(1 - grid, cmap='gray', origin='lower')
    ax1.set_title("Before (Noisy & Open)")
    
    ax2.imshow(1 - new_grid, cmap='gray', origin='lower')
    ax2.set_title("After (Cleaned & Bounded)")
    
    plt.show()

if __name__ == "__main__":
    # Make sure to point to the .npy file generated in the previous step
    clean_and_bound_map("final_map.npy", "clean_map.png")