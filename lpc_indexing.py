"""
This module uses the before hand calculated laser point centers finds
the center point as assigns an index in the form of ij to each laser point
center.
"""


from scipy.spatial.distance import pdist, squareform
import numpy as np





def find_outlier_point(coords, k=3):
    """
    Findet den Ausreißerpunkt in den Koordinaten, der am weitesten von
    den k nächsten Nachbarn abweicht.

    Parameters:
        - coords (np.ndarray): 2D-Array der Punktkoordinaten.
        - k (int): Anzahl der kleinsten Abstände zur Berechnung des durchschnittlichen Abstandes.

    Returns:
        - outlier_coord (np.ndarray): Koordinaten des Ausreißerpunkts.
        - outlier_index (int): Index des Ausreißerpunkts im Originalarray.
    """
    distances = squareform(pdist(coords))
    np.fill_diagonal(distances, np.inf)
    k_smallest_distances = np.sort(distances, axis=1)[:, :k]
    avg_min_distances = np.mean(k_smallest_distances, axis=1)
    median_distance = np.median(avg_min_distances)
    deviations = np.abs(avg_min_distances - median_distance)
    outlier_index = np.argmax(deviations)
    return coords[outlier_index], outlier_index




def calculate_rotation_angle(coords):
    """
    Bestimmt den Drehwinkel basierend auf den 4 Punkten, die dem Ausreißerpunkt am nächsten sind.

    Parameters:
        - coords (np.ndarray): 2D-Array der Punktkoordinaten.

    Returns:
        - angle (float): Berechneter Drehwinkel in Radiant.
    """
    outlier_coord, _ = find_outlier_point(coords)
    distances = np.linalg.norm(coords - outlier_coord, axis=1)
    nearest_indices = np.argsort(distances)[:5]
    nearest_points = coords[nearest_indices]

    nearest_points = nearest_points[~np.all(nearest_points == outlier_coord, axis=1)]

    upper_points = nearest_points[nearest_points[:, 1] > outlier_coord[1]]
    lower_points = nearest_points[nearest_points[:, 1] < outlier_coord[1]]

    if len(upper_points) >= 2 and len(lower_points) >= 2:
        upper_slope = (upper_points[1][1] - upper_points[0][1]) / (upper_points[1][0] - upper_points[0][0])  # pylint: disable=line-too-long
        lower_slope = (lower_points[1][1] - lower_points[0][1]) / (lower_points[1][0] - lower_points[0][0])  # pylint: disable=line-too-long
    else:
        raise ValueError("Nicht genug Punkte zur Bestimmung der Steigungen.")

    average_slope = (upper_slope + lower_slope) / 2
    angle = np.arctan(average_slope)

    return angle




def rotate_coordinates(coords):
    """
    Dreht die Koordinaten um den berechneten Winkel um den Ausreißerpunkt.

    Parameters:
        - coords (np.ndarray): 2D-Array der Punktkoordinaten.

    Returns:
        - rotated_coords (np.ndarray): Array der gedrehten Koordinaten.
    """
    outlier_coord, _ = find_outlier_point(coords)
    angle = calculate_rotation_angle(coords)

    angle = -angle

    shifted_coords = coords - outlier_coord

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    rotated_coords = np.dot(shifted_coords, rotation_matrix.T)

    return rotated_coords




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
    for value in np.sort(laser_point_centers):
        if not unique_vals or abs(value - unique_vals[-1]) > tolerance:
            unique_vals.append(value)
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




def sort_result_by_indices(result):
    """
    Sortiert das Array basierend auf den Indices in den Spalten 3 und 4.

    Parameters:
        - result (np.ndarray): Array mit den Koordinaten und den zugewiesenen Indizes.

    Returns:
        - np.ndarray: Sortiertes Array nach Spalte 3 und, falls gleich, nach Spalte 4.
    """

    sorted_result = result[np.lexsort((result[:, 3], result[:, 2]))]
    return sorted_result




def analyze_coordinates(laser_point_centers, tolerance=40.0):
    """
    Analyzes the input coordinates to assign indices based on a nearly square grid.

    Parameters:
        - data (np.ndarray): Input array of coordinates, shape (n, 2).
        - tolerance (float): Tolerance for merging nearby values into a grid.

    Returns:
        - np.ndarray: Result array with original coordinates and assigned x, y indices, shape (n, 4)
    """
    outlier_index = find_outlier_point(laser_point_centers)[1]

    x_indices = assign_indices(laser_point_centers[:, 0], tolerance)
    y_indices = assign_indices(laser_point_centers[:, 1], tolerance)

    x_indices -= x_indices[outlier_index]
    y_indices -= y_indices[outlier_index]

    result = np.hstack((laser_point_centers, x_indices[:, np.newaxis], y_indices[:, np.newaxis]))

    sorted_result = sort_result_by_indices(result)

    return sorted_result
