"""
This module contains all functions to find maximum points in an array. A 2D
array that contains the intensity of every pixel is analysed. Due to a lot of
noise the maximum point is not easy to find. The array contains a set number of
maximums up to 256+1 that are rughly alligned in a cross pattern. The maximums
are first estimated by averaging the array and than fitted using different
methods to find the laser point centers.
Current methods available:
    - fit_gaussian_2D function that uses as gaussian function to fit the distribution
"""

import numpy as np
from scipy.optimize import curve_fit
from file_io import save_array_as_dat




def closest_divisor(shape, factor):
    """
    Find the divisor of n that is closest to the given factor.

    Parameters:
        - n (int): The number to find divisors for.
        - factor (int): The reference factor to find the closest divisor to.

    Returns:
        - int: The divisor of n closest to the input factor.
    """
    divisors = [i for i in range(1, shape + 1) if shape % i == 0]
    closest = min(divisors, key=lambda x: abs(x - factor))
    return closest

def block_average(brightness_array, factor=15):
    """
    Averages the array over blocks of a given factor. If the factor is not a divisor
    of the array dimensions, the nearest possible divisor is used, and a message is printed.

    Parameters:
        - brightness_array (np.ndarray): The 2D array to be averaged.
        - factor (int): The block size for averaging. If the factor doesn't divide both dimensions,
                  the closest divisor will be used instead.

    Returns:
        - np.ndarray: The reduced array after block averaging.
        - int: The factor used for block averaging.
    """
    
    shape_x, shape_y = brightness_array.shape

    factor_x = closest_divisor(shape_x, factor)
    factor_y = closest_divisor(shape_y, factor)

    new_factor = min(factor_x, factor_y)

    if factor_x != factor or factor_y != factor:
        print(f"Closest divisor used for averaging: factor_x = {factor_x}, factor_y = {factor_y}")

    new_shape = (brightness_array.shape[0] // new_factor, brightness_array.shape[1] // new_factor)

    truncated_array = brightness_array[:new_shape[0] * new_factor, :new_shape[1] * new_factor]

    reshaped = truncated_array.reshape(new_shape[0], new_factor, new_shape[1], new_factor)
    reduced_array = reshaped.mean(axis=(1, 3))

    return reduced_array, new_factor




def find_peaks(brightness_array, factor=15, threshold=40):
    """
    Finds the maximum points in a 7x7 Subarray and applies a threshold to filter noise.

    Parameters:
        - brightness_array (np.ndarray): A 2D array with the intensity data.
        - factor (int): The block averaging factor.
        - threshold (int): Minimum intensity value for a point to be considered a peak.

    Returns:
        - np.ndarray: An (n, 2) array with the x-y coordinates of the peaks.
    """

    data, factor = block_average(brightness_array, factor)

    max_peaks = (data.shape[0] // 7) * (data.shape[1] // 7)
    peaks = np.zeros((max_peaks, 2), dtype=int)
    peak_count = 0

    rows, cols = data.shape

    for j in range(3, cols - 3):
        for i in range(3, rows - 3):
            subarray = data[i-3:i+4, j-3:j+4]

            if data[i, j] == np.max(subarray) and data[i, j] > threshold:
                if peak_count < max_peaks:
                    peaks[peak_count] = [j * factor, i * factor]
                    peak_count += 1

    file_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner"
    file_name = r"estimated_peak_array.dat"

    save_array_as_dat(peaks, file_path, file_name)

    return peaks[:peak_count]




def find_limit(brightness_array_shape, peaks, max_limits=(60, 60)):
    """
    Determines the smallest x and y limits that ensure all subarrays around each
    peak stay within array boundaries, while respecting maximum limits.

    Parameters:
    - brightness_array_shape: Shape of the 2D brightness array (height, width).
    - peaks: Array of points [(x1, y1), (x2, y2), ...] for which to calculate the minimum limit.
    - max_limits: (1,2) array specifying maximum limits for (x_limit, y_limit). Default is (60, 60).

    Returns:
    - min_limit: The minimum distance (x_limit, y_limit) that can be used
    for all peaks without exceeding array boundaries or max_limits.
    """

    min_x_limit = float('inf')
    min_y_limit = float('inf')

    max_x_limit, max_y_limit = max_limits

    for peak in peaks:
        x_center, y_center = peak

        # Calculate the maximum limits based on array boundaries
        x_max_limit = min(x_center, brightness_array_shape[1] - x_center)
        y_max_limit = min(y_center, brightness_array_shape[0] - y_center)

        # Ensure that the limits don't exceed the specified max_limits
        min_x_limit = min(min_x_limit, x_max_limit, max_x_limit)
        min_y_limit = min(min_y_limit, y_max_limit, max_y_limit)

    return min_x_limit, min_y_limit




def brightness_subarray_creator(brightness_array, peaks):
    """
    Creates subarrays around each point in `peaks` with a single, globally
    calculated limit based on array boundaries.
    Returns a 3D array where each slice is a subarray for one of the points.

    Parameters:
    - brightness_array: 2D array of brightness values.
    - peaks: List of points [(x1, y1), (x2, y2), ...].

    Returns:
    - A 3D array where each slice corresponds to a subarray for one of the points.
    """

    x_limit, y_limit = find_limit(brightness_array.shape, peaks)


    subarrays = []

    for peak in peaks:
        x_center, y_center = peak


        x_min = max(x_center - x_limit, 0)
        x_max = min(x_center + x_limit, brightness_array.shape[1])
        y_min = max(y_center - y_limit, 0)
        y_max = min(y_center + y_limit, brightness_array.shape[0])

        brightness_subarray = brightness_array[y_min:y_max, x_min:x_max]

        subarrays.append(brightness_subarray)

    subarrays_3d = np.array(subarrays, dtype=object)

    return subarrays_3d



def lpc_calc(brightness_array, mean_values, peaks):
    """
    Calculates the laser point centers from the peaks and mean values
    calculated before hand by the choosen method

    Parameters:
        - brightness_array (np.ndarray): only needed for the find_limit function
        - mean_values (n,2) array: contains the calculated mean values of each subarray
        - peaks (n,2) array: contains the position of each subarray

    Returns:
        - laser_point_centers (n,2) array: contains the position of the laser
        point centers in the original array
    """


    x_limit, y_limit = find_limit(brightness_array.shape, peaks)

    real_mean_values = np.zeros(mean_values.shape)

    real_mean_values[:, 0] = peaks[:, 0] + mean_values[:, 0] - x_limit
    real_mean_values[:, 1] = peaks[:, 1] + mean_values[:, 1] - y_limit

    laser_point_center = np.round(real_mean_values, decimals=1)

    return laser_point_center




def gaussian_2d(xy_grid, mu_x, mu_y, sigma_x, sigma_y, amplitude):
    """
    Defines a 2D Gaussian function with separate mean and standard deviation parameters.

    Parameters:
    - xy_grid (np.ndarray): A tuple of arrays (x, y) representing the grid of coordinates.
    - mu_x, mu_y (float): Mean values along the x and y axes.
    - sigma_x, sigma_y (float): Standard deviations along the x and y axes.
    - amplitude (float): Amplitude of the Gaussian function.

    Returns:
    - np.ndarray: The Gaussian function evaluated at the given (x, y) coordinates.
    """
    x_value, y_value = xy_grid
    exp_term_x = ((x_value - mu_x) ** 2) / (2 * sigma_x ** 2)
    exp_term_y = ((y_value - mu_y) ** 2) / (2 * sigma_y ** 2)

    return np.asarray(amplitude * np.exp(-(exp_term_x + exp_term_y)).ravel(), dtype=np.float64)




def prepare_meshgrid(width, height):
    """
    Creates a meshgrid of x and y coordinates based on the width and height of the array.

    Parameters:
    - width (int): The number of columns (x-axis) in the 2D array.
    - height (int): The number of rows (y-axis) in the 2D array.

    Returns:
    - np.ndarray, np.ndarray: Two arrays representing the x and y coordinates of the grid.
    """
    x_value = np.arange(0, width)
    y_value = np.arange(0, height)
    return np.meshgrid(x_value, y_value)




def fit_single_slice(slice_2d, width, height):
    """
    Fits a 2D Gaussian to a single slice (2D array) of intensity data.

    Parameters:
    - slice_2d (np.ndarray): A 2D array of intensity data representing one slice.
    - width (int): The width of the slice (number of columns).
    - height (int): The height of the slice (number of rows).

    Returns:
    - tuple: A tuple containing the fitted parameters ((mu_x, mu_y, sigma_x, sigma_y), amplitude).
    """
    x_value, y_value = prepare_meshgrid(width, height)
    xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])

    xy_grid = np.asarray(xy_grid, dtype=np.float64)
    slice_2d = np.asarray(slice_2d, dtype=np.float64)

    est_mu_x = float(width / 2)
    est_mu_y = float(height / 2)
    amp_est = float(slice_2d.max())
    initial_guess = (float(est_mu_x), float(est_mu_y), float(width / 4), float(height / 4))

    try:
        p0 = initial_guess + (amp_est,)
        popt, _ = curve_fit(gaussian_2d, xy_grid, slice_2d.ravel(), p0=p0)
        return popt[:4], popt[4]
    except RuntimeError:
        return [np.nan] * 4, np.nan




def fit_gaussian_3d(intensity_array):
    """
    Applies 2D Gaussian fitting to each slice of a 3D intensity array.

    Parameters:
    - intensity_array (np.ndarray): A 3D array where each slice (2D) represents intensity data.

    Returns:
    - np.ndarray: Array of mean values [mu_x, mu_y] for each slice.
    - np.ndarray: Array of standard deviations [sigma_x, sigma_y] for each slice.
    - np.ndarray: 3D array of fitted data for each slice, with the same shape as the input array.
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
            fitted_slice = gaussian_2d(xy_grid, mu_x, mu_y, sigma_x, sigma_y, amplitude).reshape(height, width)
            fitted_data[i] = fitted_slice
        else:
            fitted_data[i] = np.full((height, width), np.nan)

    return mean_values, deviations, fitted_data




def compute_centroids_with_uncertainty_limited(intensity_array, estimated_positions, max_limits=(60, 60)):
    """
    Computes the centroids of laser points and their uncertainties in a given intensity array,
    with limits on window size around each point.

    Parameters:
    - intensity_array (np.ndarray): 2D array of intensity values (0-255).
    - estimated_positions (np.ndarray): (n, 2) array of estimated (x, y) positions of laser points.
    - max_limits (tuple): Maximum window size for (x_limit, y_limit).

    Returns:
    - centroids (np.ndarray): (n, 2) array of centroid positions (x, y) for each laser point.
    - uncertainties (np.ndarray): (n, 2) array of uncertainties (sigma_x, sigma_y) for each centroid.
    """

    num_points = estimated_positions.shape[0]
    centroids = np.zeros((num_points, 2))
    uncertainties = np.zeros((num_points, 2))

    # Find the window size for each point using the find_limit function
    window_x_limit, window_y_limit = find_limit(intensity_array.shape, estimated_positions, max_limits)

    for i, (x_est, y_est) in enumerate(estimated_positions):
        # Define the window around the estimated position
        x_min = max(int(x_est - window_x_limit), 0)
        x_max = min(int(x_est + window_x_limit), intensity_array.shape[1] - 1)
        y_min = max(int(y_est - window_y_limit), 0)
        y_max = min(int(y_est + window_y_limit), intensity_array.shape[0] - 1)

        # Extract the window around the estimated point
        sub_array = intensity_array[y_min:y_max+1, x_min:x_max+1]

        # Create a grid of x and y positions
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))

        # Calculate the weighted sum of the positions using the intensity as weights
        total_intensity = np.sum(sub_array)

        if total_intensity > 0:
            x_centroid = np.sum(x_grid * sub_array) / total_intensity
            y_centroid = np.sum(y_grid * sub_array) / total_intensity

            # Calculate uncertainty (standard deviation) based on weighted variance
            x_var = np.sum(((x_grid - x_centroid) ** 2) * sub_array) / total_intensity
            y_var = np.sum(((y_grid - y_centroid) ** 2) * sub_array) / total_intensity

            centroids[i] = [x_centroid, y_centroid]
            uncertainties[i] = [np.sqrt(x_var), np.sqrt(y_var)]
        else:
            # If no intensity is found, return NaN for the centroid and uncertainty
            centroids[i] = [np.nan, np.nan]
            uncertainties[i] = [np.nan, np.nan]

    return centroids, uncertainties
