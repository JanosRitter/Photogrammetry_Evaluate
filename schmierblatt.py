import numpy as np
import os
from file_io.plotting import plot_2d_points

def generate_xy_pattern(
    size_x, size_y=None, 
    scale=1.0, scale_factors=None, 
    save_dir="output/"
):
    """
    Generates an x, y pattern of coordinates and saves each pattern as an .npy file.

    Parameters:
    - size_x (int): Number of points in the x-direction.
    - size_y (int, optional): Number of points in the y-direction. Defaults to size_x if None.
    - scale (float, optional): General scaling factor for the grid. Default is 1.0.
    - scale_factors (list or np.ndarray, optional): Additional scaling factors to loop over.
    - save_dir (str, optional): Directory to save the generated .npy files. Default is "output/".

    Returns:
    - None
    """
    if size_y is None:
        size_y = size_x

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Loop through scale_factors and save each pattern
    if scale_factors is None:
        scale_factors = [1.0]

    for sf in scale_factors:
        # Generate a grid of points for the current scale factor
        x_coords = np.linspace(scale * sf, scale * sf * size_x, size_x)
        y_coords = np.linspace(scale * sf, scale * sf * size_y, size_y)
        x, y = np.meshgrid(x_coords, y_coords)

        pattern = np.column_stack((x.ravel(), y.ravel()))
        
        # Debug output for verification
        print(f"Scale Factor {sf}:")
        print(pattern)
        plot_2d_points(pattern)
        
        # Save pattern with the scale factor as a suffix in the file name
        file_name = f"xy_pattern_scale_{int(sf)}.npy"
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, pattern)
        print(f"Pattern with scale factor {sf:.2f} saved to {file_path}")

# Example usage
save_dir = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\sim_dat_large_scale"

scale_factors = [1]
generate_xy_pattern(
    size_x=10, 
    scale=50.0, 
    scale_factors=scale_factors, 
    save_dir=save_dir
)





























