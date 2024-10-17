"""
This module uses the before hand calculated laser point centers finds
the center point as assigns an index in the form of ij to each laser point
center.
"""


from scipy.spatial.distance import pdist, squareform
import numpy as np




def find_outlier_point(coords):
    """
    Finds the outlier point from an array of nearly square coordinates

    Parameters:
        - coords (2,n) array: nearly square coordinates with an outlier point

    Returns:
        - coords[outlier_index] (2,1) array: coordinates of the outlier point
        - outlier_index integer: index of the outlier point
    """
    distances = squareform(pdist(coords))

    np.fill_diagonal(distances, np.inf)

    min_distances = np.min(distances, axis=1)

    median_distance = np.median(min_distances)

    outlier_index = np.argmax(np.abs(min_distances - median_distance))

    return coords[outlier_index], outlier_index



def unique_with_tolerance(laser_point_centers, tolerance=40):
    """
    Extracts unique values with a tolerance to merge nearby values.

    Parameters:
        - values (np.ndarray): Input array of values.
        - tolerance (float): Tolerance for merging nearby values.

    Returns:
        - np.ndarray: Array of unique values.
    """
    unique_vals = []
    for v in np.sort(laser_point_centers):
        if not unique_vals or abs(v - unique_vals[-1]) > tolerance:
            unique_vals.append(v)
    return np.array(unique_vals)

def assign_indices(laser_point_centers, tolerance=40):
    """
    Assigns indices to points based on their proximity to unique values.

    Parameters:
        - points (np.ndarray): Input array of points.
        - unique_vals (np.ndarray): Unique values array for comparison.
        - tolerance (float): Tolerance for index assignment.

    Returns:
        - np.ndarray: Array of assigned indices.
    """
    unique_vals = unique_with_tolerance(laser_point_centers, tolerance=40)
    indices = np.zeros(len(laser_point_centers), dtype=int)
    for i, val in enumerate(laser_point_centers):
        for j, u_val in enumerate(unique_vals):
            if abs(val - u_val) <= tolerance:
                indices[i] = j
                break
    return indices

def analyze_coordinates(laser_point_centers, tolerance=40.0):
    """
    Analyzes the input coordinates to assign indices based on a nearly square grid.

    Parameters:
        - data (np.ndarray): Input array of coordinates, shape (n, 2).
        - tolerance (float): Tolerance for merging nearby values into a grid.

    Returns:
        - np.ndarray: Result array with original coordinates and assigned x, y indices, shape (n, 4).
    """
    # Find outlier point using the existing function
    outlier, outlier_index = find_outlier_point(laser_point_centers)


    # Assign indices based on the unique values
    x_indices = assign_indices(laser_point_centers[:, 0], tolerance)
    y_indices = assign_indices(laser_point_centers[:, 1], tolerance)

    # Adjust the indices relative to the outlier point
    x_indices -= x_indices[outlier_index]
    y_indices -= y_indices[outlier_index]

    # Combine original coordinates with calculated indices
    result = np.hstack((laser_point_centers, x_indices[:, np.newaxis], y_indices[:, np.newaxis]))

    return result
