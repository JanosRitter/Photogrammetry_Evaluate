"""
This module calculates a 3D structure from 2D pixel coordinates using the camera positions
and resolution to convert pixel coordinates to real-world ones and then compute the intersection.
"""

import numpy as np


def calculate_angles(indices):
    """
    Calculates the X and Y angles based on grid indices.
    The angle for the index (0, 0) is 0, and for positive indices follows the pattern:
    Index * (1/3°) - (1/6°), while for negative indices:
    Index * (1/3°) + (1/6°).

    Parameters:
        - indices (np.ndarray): (n, 2) array of indices (x_index, y_index).

    Returns:
        - angles_x (np.ndarray): Array of X-axis angles in radians.
        - angles_y (np.ndarray): Array of Y-axis angles in radians.
    """
    angles_x = np.zeros(indices.shape[0])
    angles_y = np.zeros(indices.shape[0])

    for i, (x_idx, y_idx) in enumerate(indices):
        if (x_idx, y_idx) == (0, 0):
            angles_x[i] = 0
        else:
            angles_x[i] = (x_idx * (1/3) - (1/6)) if x_idx > 0 else (x_idx * (1/3) + (1/6))

        if (y_idx, y_idx) == (0, 0):
            angles_y[i] = 0
        else:
            angles_y[i] = (y_idx * (1/3) - (1/6)) if y_idx > 0 else (y_idx * (1/3) + (1/6))

    angles_x = np.deg2rad(angles_x)
    angles_y = np.deg2rad(angles_y)

    return angles_x, angles_y


def triangulate_3d(camera1_data, camera2_data, camera_stats, calibration_file=None):
    """
    Calculates the 3D coordinates of points based on image data from two cameras.
    This function uses the pixel coordinates from two different cameras and the camera parameters
    to compute the 3D positions of the points in real-world space using triangulation.
    
    Parameters:
        - camera1_data (np.ndarray): Array of image points from camera 1 in pixel coordinates (n, 2).
        - camera2_data (np.ndarray): Array of image points from camera 2 in pixel coordinates (n, 2).
        - camera_stats (dict): Dictionary containing camera parameters:
            - 'f' (float): Focal length of the cameras.
            - 'pixel_size' (float): Size of a pixel in meters.
            - 'resolution' (tuple): Camera resolution in pixels (width, height).
        - calibration_file (str or None): Path to a file containing the rotation matrix and
          translation vector for the second camera. If None, default values are used.

    Returns:
        - np.ndarray: Array of reconstructed 3D coordinates (n, 3)
          where each row represents a point in 3D space.
    """
    focal_length = camera_stats['f']
    pixel_size = camera_stats['pixel_size']
    resolution = camera_stats['resolution']

    n_points = camera1_data.shape[0]
    points_3d = np.zeros((n_points, 3))

    # Define default translation vector and rotation matrix for the second camera
    default_translation = np.array([0.4, 0, 0])  # Default translation in meters
    default_rotation = np.eye(3)  # Default rotation matrix (identity matrix)

    if calibration_file is not None:
        # Load translation vector and rotation matrix from calibration file
        calibration_data = np.load(calibration_file)
        translation_vector = calibration_data['translation']
        rotation_matrix = calibration_data['rotation']
    else:
        translation_vector = default_translation
        rotation_matrix = default_rotation

    # Define camera positions
    cam1_pos = np.array([0, 0, 0])  # First camera at the origin
    cam2_pos = translation_vector  # Second camera position from calibration

    # Apply rotation to second camera's direction vectors
    offset_x = resolution[0] / 2
    offset_y = resolution[1] / 2

    for i in range(n_points):
        x1_img = (camera1_data[i, 0] - offset_x) * pixel_size
        y1_img = (camera1_data[i, 1] - offset_y) * pixel_size
        x2_img = (camera2_data[i, 0] - offset_x) * pixel_size
        y2_img = (camera2_data[i, 1] - offset_y) * pixel_size

        dir_cam1 = np.array([x1_img, y1_img, focal_length])
        dir_cam2 = np.array([x2_img, y2_img, focal_length])
        dir_cam2 = rotation_matrix @ dir_cam2  # Rotate direction vector of the second camera

        points_3d[i] = find_closest_point(cam1_pos, dir_cam1, cam2_pos, dir_cam2)

    return points_3d


def find_closest_point(point1, direction1, point2, direction2):
    """
    Finds the closest point between two lines that pass through `point1` and `point2`
    and have directions `direction1` and `direction2`, respectively.

    This function calculates the intersection point between two lines defined by the points
    and direction vectors, and returns the closest point between them.

    Parameters:
        - point1 (np.ndarray): Origin point of the first line.
        - direction1 (np.ndarray): Direction vector of the first line.
        - point2 (np.ndarray): Origin point of the second line.
        - direction2 (np.ndarray): Direction vector of the second line.

    Returns:
        - np.ndarray: The closest point between the two lines in 3D space.
    """
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)

    diff = point1 - point2
    dot_a = np.dot(direction1, direction1)
    dot_b = np.dot(direction1, direction2)
    dot_c = np.dot(direction2, direction2)
    dot_d = np.dot(direction1, diff)
    dot_e = np.dot(direction2, diff)

    denom = dot_a * dot_c - dot_b ** 2
    distance_s = (dot_b * dot_e - dot_c * dot_d) / denom
    distance_t = (dot_a * dot_e - dot_b * dot_d) / denom

    point_on_line1 = point1 + distance_s * direction1
    point_on_line2 = point2 + distance_t * direction2

    return (point_on_line1 + point_on_line2) / 2
