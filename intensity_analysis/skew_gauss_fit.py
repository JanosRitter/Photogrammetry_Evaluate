"""
Skewed Gaussian Fitting Module
================================

This module provides functions for fitting a skewed 2D Gaussian model
to 2D slices within a 3D dataset. The primary goal is to compute the
parameters of the skewed Gaussian, including means, standard deviations,
amplitudes, and skewness, as well as to generate the fitted models.

Functions
---------
1. skew_gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y):
    Generates a skewed 2D Gaussian distribution based on the given parameters.

2. skew_gaussian_2d_residuals(params, xy, slice_2d):
    Calculates residuals for the skewed 2D Gaussian fitting process.

3. prepare_meshgrid(width, height):
    Creates a meshgrid for the specified width and height dimensions.

4. fit_skewed_single_slice(slice_2d, width, height):
    Fits a skewed 2D Gaussian to a single 2D slice and returns the fitted parameters.

5. fit_skewed_gaussian_3d(intensity_array):
    Performs skewed Gaussian fitting on each 2D slice of a 3D intensity array
    and returns the parameters and fitted models for all slices.

Usage
-----
This module is designed to analyze intensity distributions in 3D datasets
by fitting skewed Gaussian models to individual 2D slices. The results include
the mean, deviations, skewness, and the reconstructed fitted data.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

def skew_gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y):
    """
    Generates a skewed 2D Gaussian distribution.

    Parameters:
        - xy (tuple of np.ndarray): Meshgrid arrays for x and y coordinates.
        - mu_x (float): Mean (center) of the Gaussian along the x-axis.
        - mu_y (float): Mean (center) of the Gaussian along the y-axis.
        - sigma_x (float): Standard deviation of the Gaussian along the x-axis.
        - sigma_y (float): Standard deviation of the Gaussian along the y-axis.
        - amplitude (float): Amplitude of the Gaussian.
        - alpha_x (float): Skewness parameter along the x-axis.
        - alpha_y (float): Skewness parameter along the y-axis.

    Returns:
        - np.ndarray: The skewed 2D Gaussian distribution as a flattened array.
    """
    x, y = xy
    gauss = amplitude * np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +
                                 ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))
    skew_x = 2 * norm.cdf(alpha_x * (x - mu_x) / sigma_x) - 1
    skew_y = 2 * norm.cdf(alpha_y * (y - mu_y) / sigma_y) - 1
    return gauss * skew_x * skew_y

def skew_gaussian_2d_residuals(params, xy, slice_2d):
    """
    Calculates residuals for the skewed 2D Gaussian fitting.

    Parameters:
        - params (list of float): Parameters of the skewed Gaussian model:
          [mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y].
        - xy (tuple of np.ndarray): Meshgrid arrays for x and y coordinates.
        - slice_2d (np.ndarray): 2D intensity data to fit.

    Returns:
        - float: Sum of squared weighted residuals between the model and the data.
    """
    mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y = params
    model = skew_gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y)
    residuals = (model - slice_2d.ravel()) ** 2
    weights = slice_2d.ravel()
    return np.sum(residuals * weights)

def prepare_meshgrid(width, height):
    """
    Creates a meshgrid for given width and height dimensions.

    Parameters:
        - width (int): Width of the grid.
        - height (int): Height of the grid.

    Returns:
        - tuple of np.ndarray: Meshgrid arrays for x and y coordinates.
    """
    x = np.arange(width)
    y = np.arange(height)
    return np.meshgrid(x, y)

def fit_skewed_single_slice(slice_2d, width, height):
    """
    Fits a skewed 2D Gaussian to a single slice and returns parameters.

    Parameters:
        - slice_2d (np.ndarray): 2D intensity data to fit.
        - width (int): Width of the slice.
        - height (int): Height of the slice.

    Returns:
        - list of float: Fitted parameters
        [mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y].
        - If fitting fails, returns [nan, nan, nan, nan, nan, nan, nan].
    """
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    slice_2d = np.asarray(slice_2d, dtype=np.float64)
    smoothed_slice = gaussian_filter(slice_2d, sigma=2)
    y_max, x_max = np.unravel_index(np.argmax(smoothed_slice), smoothed_slice.shape)
    initial_guess = [x_max, y_max, width / 8, height / 8, smoothed_slice.max(), 0, 0]
    bounds = [(0, width), (0, height), (1, width), (1, height), (0.1, None), (-10, 10), (-10, 10)]
    x_value, y_value = prepare_meshgrid(width, height)
    xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])
    result = minimize(skew_gaussian_2d_residuals, initial_guess, args=(xy_grid, slice_2d),
                      bounds=bounds, method='L-BFGS-B')
    return result.x if result.success else [np.nan] * 7

def fit_skewed_gaussian_3d(intensity_array):
    """
    Performs skewed Gaussian fitting on each 2D slice of a 3D intensity array.

    Parameters:
        - intensity_array (np.ndarray): 3D intensity data of shape (num_slices, height, width).

    Returns:
        - mean_values (np.ndarray): Array of shape (num_slices, 2)
        containing [mu_x, mu_y] for each slice.
        - deviations (np.ndarray): Array of shape (num_slices, 2)
        containing [sigma_x, sigma_y] for each slice.
        - skewness (np.ndarray): Array of shape (num_slices, 2)
        containing [alpha_x, alpha_y] for each slice.
        - fitted_data (np.ndarray): 3D array of the same shape as the input,
        containing the fitted skewed Gaussian model.
    """
    num_slices, height, width = intensity_array.shape
    mean_values = np.zeros((num_slices, 2))
    deviations = np.zeros((num_slices, 2))
    skewness = np.zeros((num_slices, 2))
    fitted_data = np.zeros((num_slices, height, width))
    for i in range(num_slices):
        slice_2d = intensity_array[i]
        params = fit_skewed_single_slice(slice_2d, width, height)
        mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y = params
        mean_values[i] = [mu_x, mu_y]
        deviations[i] = [sigma_x, sigma_y]
        skewness[i] = [alpha_x, alpha_y]
        if not np.isnan(mu_x):
            x_value, y_value = prepare_meshgrid(width, height)
            xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])
            fitted_data[i] = skew_gaussian_2d_model(xy_grid, mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y).reshape(height, width)
        else:
            fitted_data[i] = np.full((height, width), np.nan)
    return mean_values, deviations, skewness, fitted_data
