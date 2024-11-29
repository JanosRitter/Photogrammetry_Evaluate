import numpy as np

def compute_center_of_mass_with_uncertainty(data_3d):
    """
    Computes the center of mass and its uncertainty for each 2D slice of a 3D array.

    Parameters:
    - data_3d (np.ndarray): Input 3D array of shape (num, height, width), 
                            where each slice represents an intensity distribution.

    Returns:
    - centers_of_mass (np.ndarray): Array of shape (num, 2) containing x and y coordinates of 
                                     the center of mass for each slice.
    - uncertainties (np.ndarray): Array of shape (num, 2) containing uncertainties in the 
                                   x and y coordinates for each slice.
    """
    if not isinstance(data_3d, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if data_3d.ndim != 3:
        raise ValueError("Input array must be 3D.")

    num_slices, height, width = data_3d.shape
    centers_of_mass = np.zeros((num_slices, 2))
    uncertainties = np.zeros((num_slices, 2))

    for i in range(num_slices):
        slice_2d = data_3d[i]
        total_mass = np.sum(slice_2d)
        if total_mass == 0:
            centers_of_mass[i] = [np.nan, np.nan]
            uncertainties[i] = [np.nan, np.nan]
            continue

        x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
        x_center = np.sum(x_indices * slice_2d) / total_mass
        y_center = np.sum(y_indices * slice_2d) / total_mass
        centers_of_mass[i] = [x_center, y_center]

        x_uncertainty = np.sqrt(np.sum((x_indices - x_center) ** 2 * slice_2d) / total_mass)
        y_uncertainty = np.sqrt(np.sum((y_indices - y_center) ** 2 * slice_2d) / total_mass)
        uncertainties[i] = [x_uncertainty, y_uncertainty]

    return centers_of_mass, uncertainties
