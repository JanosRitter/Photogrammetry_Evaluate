"""
Module: non_linear_center_of_mass

This module provides functionality to compute the center of mass and its associated uncertainties
for intensity distributions represented in 3D numpy arrays. Each slice of the input 3D array
is treated as a 2D intensity distribution, allowing for per-slice analysis of center-of-mass
properties. To increase the contribution of high values an exponent > 1 is applied.

Key Functionality:
- `non_linear_center_of_mass`: Calculates the center of mass and uncertainties
  for each 2D slice of a 3D array. This is particularly useful for analyzing intensity
  distributions in applications such as imaging, material analysis, or scientific experiments.
"""
import numpy as np

def non_linear_center_of_mass(data_3d):
    """
    Computes the center of mass and its uncertainty for each 2D slice of a 3D array,
    with non-linear weighting (square of values).

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

        weighted_slice = slice_2d ** 3

        total_mass = np.sum(weighted_slice)
        if total_mass == 0:
            centers_of_mass[i] = [np.nan, np.nan]
            uncertainties[i] = [np.nan, np.nan]
            continue

        x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
        x_center = np.sum(x_indices * weighted_slice) / total_mass
        y_center = np.sum(y_indices * weighted_slice) / total_mass
        centers_of_mass[i] = [x_center, y_center]

        x_uncertainty = np.sqrt(np.sum((x_indices - x_center) ** 2 * weighted_slice) / total_mass)
        y_uncertainty = np.sqrt(np.sum((y_indices - y_center) ** 2 * weighted_slice) / total_mass)
        uncertainties[i] = [x_uncertainty, y_uncertainty]

    return centers_of_mass, uncertainties
