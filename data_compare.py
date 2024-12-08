"""
This module compares two arrays to each other and returns the difference
"""

import numpy as np
#from file_io import plot_differences_as_bar_chart, load_brightness_arrays, create_image_plot
#from peak_find import find_peaks, brightness_subarray_creator, lpc_calc
#import intensity_analysis

def sort_experimental_to_theoretical(theoretical_points, experimental_points):
    """
    Sorts experimental points to minimize the distance to corresponding theoretical points.

    Parameters:
    - theoretical_points (np.ndarray): Array of shape (n, 2) with theoretical (x, y) coordinates.
    - experimental_points (np.ndarray): Array of shape (n, 2) with experimental (x, y) coordinates.

    Returns:
    - np.ndarray: Sorted experimental points matching the theoretical order.
    """
    print(f"Initial theoretical_points shape: {theoretical_points.shape}")
    print(f"Initial experimental_points shape: {experimental_points.shape}")

    sorted_experimental = np.zeros_like(theoretical_points)

    for i, theoretical_point in enumerate(theoretical_points):
        #print(f"\nIteration {i}:")
        #print(f"  Current theoretical point: {theoretical_point}")
        #print(f"  Remaining experimental points shape: {experimental_points.shape}")
        
        if experimental_points.shape[0] == 0:
            raise ValueError("Ran out of experimental points during sorting. "
                             "Check input data for inconsistencies.")

        # Calculate distances
        distances = np.linalg.norm(experimental_points - theoretical_point, axis=1)
        closest_index = np.argmin(distances)
        #print(f"  Closest index: {closest_index}, Distance: {distances[closest_index]}")

        # Assign the closest experimental point
        sorted_experimental[i] = experimental_points[closest_index]
        #print(f"  Assigned experimental point: {experimental_points[closest_index]}")

        # Remove the selected point from experimental_points
        experimental_points = np.delete(experimental_points, closest_index, axis=0)
        #print(f"  Experimental points after deletion: {experimental_points.shape}")

    return sorted_experimental

def calculate_differences(theoretical_points, experimental_points):
    """
    Calculates the difference between theoretical and experimental points.

    Parameters:
    - theoretical_points (np.ndarray): Array of shape (n, 2) with theoretical (x, y) coordinates.
    - experimental_points (np.ndarray): Array of shape (n, 2) with experimental (x, y) coordinates.

    Returns:
    - np.ndarray: Array of shape (n, 2) representing the (x, y) differences.
    """
    sorted_experimental = sort_experimental_to_theoretical(theoretical_points, experimental_points)
    return theoretical_points - sorted_experimental









