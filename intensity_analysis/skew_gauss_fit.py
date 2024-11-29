import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

def skew_gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y):
    """
    Generates a skewed 2D Gaussian distribution based on given parameters.
    """
    x, y = xy
    # Base Gaussian
    gauss = amplitude * np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +
                                 ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))
    # Skewness factors
    skew_x = 2 * norm.cdf(alpha_x * (x - mu_x) / sigma_x) - 1
    skew_y = 2 * norm.cdf(alpha_y * (y - mu_y) / sigma_y) - 1
    return gauss * skew_x * skew_y

def skew_gaussian_2d_residuals(params, xy, slice_2d):
    """
    Calculates residuals for the skewed 2D Gaussian fitting.
    """
    mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y = params
    model = skew_gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude, alpha_x, alpha_y)
    residuals = (model - slice_2d.ravel()) ** 2
    weights = slice_2d.ravel()  # Höhere Intensitäten stärker gewichten
    return np.sum(residuals * weights)

def prepare_meshgrid(width, height):
    """Creates a meshgrid for given width and height dimensions."""
    x = np.arange(width)
    y = np.arange(height)
    return np.meshgrid(x, y)

def fit_skewed_single_slice(slice_2d, width, height):
    """
    Fits a skewed 2D Gaussian to a single slice and returns parameters.
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

