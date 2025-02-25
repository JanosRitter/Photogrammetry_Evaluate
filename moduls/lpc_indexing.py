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




def compute_grid_size(coords, k=3):
    """
    Computes the typical grid size for a nearly square grid by analyzing the median
    distance to the k nearest neighbors, excluding the identified outlier.

    Parameters:
        coords (np.ndarray): Array of shape (n, 2) representing 2D coordinates.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: The computed grid size, representing the typical distance between points.
    """
    _, outlier_idx = find_outlier_point(coords, k)
    coords_wo_outlier = np.delete(coords, outlier_idx, axis=0)
    distances = squareform(pdist(coords_wo_outlier))
    np.fill_diagonal(distances, np.inf)
    k_smallest_distances = np.sort(distances, axis=1)[:, :k]
    grid_size = np.median(np.mean(k_smallest_distances, axis=1))

    print(f"Computed grid size: {grid_size}")

    return grid_size


def assign_grid_indices(coords, k=3):
    """
    Assigns grid indices to points based on a grid where the outlier is positioned
    at the corner of four central cells, forming a chessboard-like grid.

    Parameters:
        coords (np.ndarray): Array of shape (n, 2) representing 2D coordinates.
        k (int): Number of nearest neighbors to consider for outlier detection
                 and grid size computation.

    Returns:
        np.ndarray: Array of shape (n, 4) where each row contains:
            - Original x-coordinate
            - Original y-coordinate
            - x-index in the grid
            - y-index in the grid
    """
    _, outlier_idx = find_outlier_point(coords, k)
    grid_size = compute_grid_size(coords, k)
    origin = coords[outlier_idx]

    relative_positions = coords - origin

    grid_indices = custom_round(relative_positions / grid_size)

    grid_indices[outlier_idx] = [0, 0]

    return np.hstack((coords, grid_indices))



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



def check_unique_indices(indices):
    """
    Checks whether all assigned indices are unique.

    Parameters:
        indices (np.ndarray): Array of shape (n, 2) containing grid indices.

    Returns:
        bool: True if all indices are unique, False otherwise.
    """
    unique_indices = {tuple(idx) for idx in indices}
    return len(unique_indices) == len(indices)



def custom_round(values):
    """
    Custom rounding logic:
    - Negative values are rounded to the next smaller integer (further from zero).
    - Positive values are rounded to the next larger integer (further from zero).

    Parameters:
        values (np.ndarray): Array of floating-point values.

    Returns:
        np.ndarray: Array of integers rounded according to the specified logic.
    """
    return np.where(values < 0, np.floor(values), np.ceil(values)).astype(int)


def analyze_coordinates(coords, k=3):
    """
    Analyzes a set of 2D coordinates and assigns grid-based indices, identifying
    an outlier point as the grid origin. Ensures proper alignment and validates index uniqueness.

    Parameters:
        coords (np.ndarray): Array of shape (n, 2) representing 2D coordinates.
        k (int): Number of nearest neighbors to consider for outlier detection
                 and grid size computation.

    Returns:
        np.ndarray: Sorted array of shape (n, 4), where each row contains:
            - Original x-coordinate
            - Original y-coordinate
            - x-index in the grid
            - y-index in the grid
    """
    result = assign_grid_indices(coords, k)
    if not check_unique_indices(result[:, 2:].astype(int)):
        print("Warning: Assigned indices are not unique!")
    return sort_result_by_indices(result)
