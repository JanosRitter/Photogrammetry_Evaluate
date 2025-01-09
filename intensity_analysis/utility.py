"""
This module contains various utility functions that are needed for intensity
evaluation in a dataset. These functions help with background subtraction,
calculating distances between peaks, extracting subarrays around peaks, and
determining laser point centers based on peak positions and their associated
mean values.

Functions:
- subtract_mean_background: Removes the mean background intensity from an array
  and sets negative values to zero.
- find_average_distance: Calculates the average distance between peaks and
  scales it by a given factor to determine limits for subarray extraction.
- brightness_subarray_creator: Extracts subarrays centered around peaks in the
  intensity array, based on the calculated limits from average distances.
- lpc_calc: Calculates laser point centers by combining peak positions with
  the computed mean values.

Each function is designed to aid in the preprocessing, analysis, and extraction of
features from intensity data, particularly for scenarios like peak detection,
laser point localization, and background correction.

Usage:
    - The `subtract_mean_background` function can be used to remove baseline
      intensity values.
    - The `find_average_distance` function helps determine appropriate subarray
      limits around each peak.
    - The `brightness_subarray_creator` function is used to generate subarrays
      around each peak, which can then be analyzed separately.
    - The `lpc_calc` function computes the laser point centers from detected peaks
      and their corresponding mean values, useful for localizing the laser spot.

Note:
    These utilities assume the input arrays are numpy arrays and may raise errors
    if the input types do not match the expected format.
"""

import numpy as np
from scipy.spatial import KDTree

def subtract_mean_background(array):
    """
    Subtracts the mean value of an array from each element
    and sets negative values to zero.

    This function is useful for background removal in intensity data, where the
    background is considered to be the mean intensity across the entire array.
    After subtracting this background, any resulting negative values are set to zero
    to ensure that the array only contains non-negative values.

    Parameters:
    - array (np.ndarray): Input array with intensity values.

    Returns:
    - np.ndarray: Array after mean background subtraction with non-negative values.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    mean_value = np.mean(array)
    adjusted_array = array - mean_value
    adjusted_array[adjusted_array < 0] = 0
    return adjusted_array

def find_average_distance(peaks, factor=0.35):
    """
    Computes the average distance between peaks and scales it by a factor.

    This function calculates the average distance between all pairs of peaks, and
    then scales this distance by a factor to create limits used for further analysis.
    The result is used for setting the radius around each peak for subarray extraction.

    Parameters:
    - peaks (np.ndarray): Array of coordinates of the peaks (e.g., [(x1, y1), (x2, y2), ...]).
    - factor (float): Scaling factor to adjust the distance for subarray limits.

    Returns:
    - tuple: A pair of values (x_limit, y_limit), representing the limits for each peak.
    """
    kdtree = KDTree(peaks)
    distances, _ = kdtree.query(peaks, k=2)
    avg_distance = np.mean(distances[:, 1])
    limits = (int(avg_distance * factor), int(avg_distance * factor))
    print(f"Calculated limits (x_limit, y_limit): {limits}")
    return limits

def brightness_subarray_creator(brightness_array, peaks):
    """
    Creates subarrays around each peak in `peaks` based on a fixed limit.

    This function extracts subarrays around each peak. The size of the subarray is determined
    by the average distance between peaks, and it is consistent across all peaks. Each subarray
    corresponds to one of the peaks and represents a region of interest in the intensity data.

    Parameters:
    - brightness_array (np.ndarray): 2D array containing intensity values.
    - peaks (np.ndarray): List of peak coordinates, each in the form (x, y).

    Returns:
    - np.ndarray: A 3D array, where each slice is a subarray around a peak.
    """
    x_limit, y_limit = find_average_distance(peaks)

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

def lpc_calc(mean_values, peaks):
    """
    Calculates the laser point centers from the peaks and mean values.

    This function computes the actual laser point centers by combining the
    peak positions with the mean values calculated earlier. The result is the
    adjusted coordinates that reflect the laser point positions in the original data.

    Parameters:
    - mean_values (np.ndarray): Array of calculated mean values for each subarray (shape: n, 2).
    - peaks (np.ndarray): Array of peak positions (shape: n, 2), representing the centers
                           of each subarray in the original array.

    Returns:
    - np.ndarray: Array of laser point centers (shape: n, 2), containing the position of
                  the laser points in the original array.
    """
    x_limit, y_limit = find_average_distance(peaks)

    real_mean_values = np.zeros(mean_values.shape)

    real_mean_values[:, 0] = peaks[:, 0] + mean_values[:, 0] - x_limit
    real_mean_values[:, 1] = peaks[:, 1] + mean_values[:, 1] - y_limit

    laser_point_center = np.round(real_mean_values, decimals=1)

    return laser_point_center
