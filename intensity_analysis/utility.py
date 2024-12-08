"""
Contains different utility functions needed for the intensity
evaluation.
"""

import numpy as np
from scipy.spatial import KDTree

def subtract_mean_background(array):
    """
    Subtracts the mean value of an array from each element 
    and sets negative values to zero.

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




def find_average_distance(peaks, factor=0.4):
    """
    Berechnet den durchschnittlichen Abstand zwischen den Peaks und multipliziert
    diesen mit einem Faktor.

    Parameters:
    - peaks: Array von Koordinaten [(x1, y1), (x2, y2), ...].
    - factor: Skalierungsfaktor, der den Bereich um jeden Punkt steuert.

    Returns:
    - limits: Ein festgelegtes (x_limit, y_limit) basierend auf dem durchschnittlichen
    Abstand der Peaks.
    """
    kdtree = KDTree(peaks)
    distances, _ = kdtree.query(peaks, k=2)
    avg_distance = np.mean(distances[:, 1])
    limits = (int(avg_distance * factor), int(avg_distance * factor))
    print(f"Berechnete Limits (x_limit, y_limit): {limits}")
    return limits

def brightness_subarray_creator(brightness_array, peaks):
    """
    Erstellt Subarrays um jeden Punkt in `peaks` basierend auf einem festen Limit.
    Gibt ein 3D-Array zurück, wobei jeder Slice ein Subarray für einen der Punkte ist.

    Parameters:
    - brightness_array: 2D-Array mit Helligkeitswerten.
    - peaks: Liste von Punkten [(x1, y1), (x2, y2), ...].

    Returns:
    - Ein 3D-Array, wobei jeder Slice ein Subarray für einen der Punkte darstellt.
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


    x_limit, y_limit = find_average_distance(peaks)

    real_mean_values = np.zeros(mean_values.shape)

    real_mean_values[:, 0] = peaks[:, 0] + mean_values[:, 0] - x_limit
    real_mean_values[:, 1] = peaks[:, 1] + mean_values[:, 1] - y_limit

    laser_point_center = np.round(real_mean_values, decimals=1)

    return laser_point_center

