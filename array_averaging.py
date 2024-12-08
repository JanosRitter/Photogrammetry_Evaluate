import numpy as np


def downsample_and_average(array, factor):
    """
    Downsamples a 2D array by block averaging and rounds values to ensure they stay within 0-255.

    Parameters:
        - array (np.ndarray): The input 2D array to be downsampled.
        - factor (int): The downsampling factor (e.g., 2 for (n, m) -> (n/2, m/2)).

    Returns:
        - np.ndarray: The downsampled array with values rounded and clipped to 0-255.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if factor <= 0:
        raise ValueError("Factor must be a positive integer.")

    new_rows = array.shape[0] // factor
    new_cols = array.shape[1] // factor

    trimmed_array = array[:new_rows * factor, :new_cols * factor]

    reshaped = trimmed_array.reshape(new_rows, factor, new_cols, factor)
    averaged = reshaped.mean(axis=(1, 3))

    averaged = np.clip(np.round(averaged), 0, 255).astype(np.uint8)

    return averaged

filepath = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\noise_spots_scaled\noise_spots"

savepath = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\noise_spots_scaled\noise_spots\averaged_noise_spots"

filename = r"Noise_projection_cam2_scale_2.npy"

array = load_npy_file(filepath, filename)

new_array = downsample_and_average(array, 2)

save_array_as_npy(new_array, savepath, filename)
