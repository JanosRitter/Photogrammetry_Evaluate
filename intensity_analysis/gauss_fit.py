"""
Module: gaussian_fitting

This module provides tools for fitting 2D Gaussian distributions to slices of 3D intensity data.
The primary goal is to estimate the mean positions, deviations, and amplitude of Gaussian
intensity distributions for each slice in the 3D dataset.

Functions:
----------
1. `gaussian_2d_model`: Generates a 2D Gaussian distribution based on given parameters.
2. `gaussian_2d_residuals`: Computes the weighted residuals between the Gaussian model
and a 2D slice.
3. `prepare_meshgrid`: Prepares a meshgrid for the given dimensions, used in Gaussian computations.
4. `fit_single_slice`: Fits a 2D Gaussian to a single 2D intensity distribution slice.
5. `fit_gaussian_3d`: Applies Gaussian fitting to each 2D slice in a 3D intensity dataset.

This module is particularly useful for analyzing intensity distributions in applications like
image processing, experimental physics, or material science.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

def gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude):
    """Generates a 2D Gaussian distribution based on given parameters.
    Parameters:
        - xy (tuple): Meshgrid coordinates (x, y).
        - mu_x (float): Mean position in the x-direction.
        - mu_y (float): Mean position in the y-direction.
        - sigma_x (float): Standard deviation in the x-direction.
        - sigma_y (float): Standard deviation in the y-direction.
        - amplitude (float): Peak amplitude of the Gaussian.

    Returns:
        - np.ndarray: Flattened 2D Gaussian intensity values.
    """
    x, y = xy
    return amplitude * np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +
                                ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))

def gaussian_2d_residuals(params, xy, slice_2d):
    mu_x, mu_y, sigma_x, sigma_y, amplitude = params
    model = gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude)
    residuals = (model - slice_2d.ravel()) ** 2
    weights = slice_2d.ravel()  # Höhere Intensitäten stärker gewichten
    return np.sum(residuals * weights)

def prepare_meshgrid(width, height):
    """Creates a meshgrid for given width and height dimensions.
    Parameters:
        - width (int): Width of the grid.
        - height (int): Height of the grid.

    Returns:
        - tuple: Meshgrid arrays (x, y).
    """
    x = np.arange(width)
    y = np.arange(height)
    return np.meshgrid(x, y)

def fit_single_slice(slice_2d, width, height):
    """Fits a 2D Gaussian to a single slice and returns mean, deviations, and amplitude.
    Parameters:
        - slice_2d (np.ndarray): 2D intensity data.
        - width (int): Width of the slice.
        - height (int): Height of the slice.

    Returns:
        - tuple: ([mu_x, mu_y, sigma_x, sigma_y], amplitude) or
        ([nan, nan, nan, nan], nan) on failure.
    """
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    slice_2d = np.asarray(slice_2d, dtype=np.float64)
    smoothed_slice = gaussian_filter(slice_2d, sigma=2)
    y_max, x_max = np.unravel_index(np.argmax(smoothed_slice), smoothed_slice.shape)
    initial_guess = [x_max, y_max, width / 8, height / 8, smoothed_slice.max()]
    bounds = [(0, width), (0, height), (1, width), (1, height), (0.1, None)]
    x_value, y_value = prepare_meshgrid(width, height)
    xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])
    result = minimize(gaussian_2d_residuals, initial_guess, args=(xy_grid, slice_2d), bounds=bounds, method='L-BFGS-B')
    return result.x[:4], result.x[4] if result.success else ([np.nan] * 4, np.nan)

def fit_gaussian_3d(intensity_array):
    """Performs Gaussian fitting on each 2D slice of a 3D intensity array and returns fitted data.
    Parameters:
        - intensity_array (np.ndarray): 3D intensity array with shape (num_slices, height, width).

    Returns:
        - mean_values (np.ndarray): Array of shape (num_slices, 2) containing [mu_x, mu_y]
        for each slice.
        - deviations (np.ndarray): Array of shape (num_slices, 2) containing [sigma_x, sigma_y]
        for each slice.
        - fitted_data (np.ndarray): 3D array containing the Gaussian-fitted data for each slice.
    """
    num_slices, height, width = intensity_array.shape
    mean_values = np.zeros((num_slices, 2))
    deviations = np.zeros((num_slices, 2))
    fitted_data = np.zeros((num_slices, height, width))
    for i in range(num_slices):
        slice_2d = intensity_array[i]
        (mu_x, mu_y, sigma_x, sigma_y), amplitude = fit_single_slice(slice_2d, width, height)
        mean_values[i] = [mu_x, mu_y]
        deviations[i] = [sigma_x, sigma_y]
        if not np.isnan(mu_x):
            x_value, y_value = prepare_meshgrid(width, height)
            xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])
            fitted_data[i] = gaussian_2d_model(xy_grid, mu_x, mu_y, sigma_x, sigma_y, amplitude).reshape(height, width)
        else:
            fitted_data[i] = np.full((height, width), np.nan)
    return mean_values, deviations, fitted_data
