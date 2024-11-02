"""
This module contains all functions to find maximum points in an array. A 2D
array that contains the intensity of every pixel is analysed. Due to a lot of
noise the maximum point is not easy to find. The array contains a set number of
maximums up to 256+1 that are roughly alligned in a cross pattern. The maximums
are first estimated by averaging the array and than fitted using different
methods to find the laser point centers.
Current methods available:
    - fit_gaussian_2D function that uses as gaussian function to fit the distribution
"""

import numpy as np
from scipy.spatial import KDTree
from file_io import save_array_as_npy





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




def find_peaks(brightness_array, factor=15, threshold=40, filename='estimated_peak_array.dat'):
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
    file_name = filename

    save_array_as_npy(peaks, file_path, file_name)

    return peaks[:peak_count]




def peak_filter(points, tolerance=1.5, boundary_factor=2.5):
    """
    Filters out points that lie outside an automatically detected boundary or have
    an unusually large distance to their nearest neighbor.

    Parameters:
        points (np.ndarray): An (n, 2) array of x and y coordinates.
        tolerance (float): Multiplier for the standard deviation of distances,
                           defining outliers based on neighbor distance.
        boundary_factor (float): Multiplier for the range around the average location,
                                 defining the allowable boundary for points.

    Returns:
        np.ndarray: A filtered (m, 2) array excluding outliers and points outside the
        detected boundary.
    """

    center, std_dev = np.mean(points, axis=0), np.std(points, axis=0)
    bounds = [(center[i] - boundary_factor * std_dev[i], center[i] + boundary_factor * std_dev[i]) for i in range(2)]

    in_bounds = (bounds[0][0] <= points[:, 0]) & (points[:, 0] <= bounds[0][1]) & \
                (bounds[1][0] <= points[:, 1]) & (points[:, 1] <= bounds[1][1])
    points_in_bounds = points[in_bounds]

    distances = KDTree(points_in_bounds).query(points_in_bounds, k=2)[0][:, 1]

    threshold = np.mean(distances) + tolerance * np.std(distances)
    return points_in_bounds[distances <= threshold]




def find_limit(brightness_array_shape, peaks, max_limits=(60, 60), distance_factor=0.4):
    """
    Determines the smallest x and y limits to ensure subarrays around each peak
    stay within array boundaries, respecting max limits and distances to neighboring peaks.

    Parameters:
    - brightness_array_shape: Shape of the 2D brightness array (height, width).
    - peaks: Array of points [(x1, y1), (x2, y2), ...] for which to calculate the minimum limit.
    - max_limits: Tuple specifying maximum limits for (x_limit, y_limit). Default is (60, 60).
    - distance_factor: Factor of the distance to the nearest peak to determine limits.

    Returns:
    - min_limit: The minimum distance (x_limit, y_limit) within boundaries, max_limits,
    and distance-based limits, rounded to integers.
    """
    min_limits = [float('inf'), float('inf')]
    kdtree = KDTree(peaks)

    for peak in peaks:
        nearest_distance = kdtree.query([peak], k=2)[0][0, 1] * distance_factor
        boundary_limits = [min(peak[i], brightness_array_shape[i] - peak[i]) for i in range(2)]

        limits = [min(nearest_distance, boundary_limits[i], max_limits[i]) for i in range(2)]

        min_limits = [min(min_limits[i], limits[i]) for i in range(2)]

    return tuple(map(lambda x: int(round(x)), min_limits))




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
