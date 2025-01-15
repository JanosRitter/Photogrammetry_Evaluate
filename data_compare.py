"""
This module compares two arrays to each other and returns the difference
"""

import numpy as np

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

    sorted_experimental = np.zeros(theoretical_points.shape, dtype=theoretical_points.dtype)

    for i, theoretical_point in enumerate(theoretical_points):

        if experimental_points.shape[0] == 0:
            raise ValueError("Ran out of experimental points during sorting. "
                             "Check input data for inconsistencies.")

        distances = np.linalg.norm(experimental_points - theoretical_point, axis=1)
        closest_index = np.argmin(distances)

        sorted_experimental[i] = experimental_points[closest_index]

        experimental_points = np.delete(experimental_points, closest_index, axis=0)

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
