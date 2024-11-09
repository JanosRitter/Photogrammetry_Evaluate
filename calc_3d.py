"""
This module calcualtes a 3D structure from 2D Pixelkoordinates by using the camera positions
and resolution to convert Pixelkoordinates to real ones and then calculate the intersection
"""
import numpy as np



def calculate_angles(indices):
    """
    Berechnet die X- und Y-Winkel basierend auf den Gitterindizes.
    Der Winkel für den Index (0,0) ist 0, positive Indices folgen
    dem Muster: Index * (1/3°) - (1/6°) und negative Indices:
    Index * (1/3°) + (1/6°).

    Parameters:
        - indices (np.ndarray): (n, 2) Array der Indizes (x_index, y_index).

    Returns:
        - angles_x (np.ndarray): Array der X-Achsen-Winkel in Bogenmaß.
        - angles_y (np.ndarray): Array der Y-Achsen-Winkel in Bogenmaß.
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


def triangulate_3d(camera1_data, camera2_data, camera_stats):
    """
    Calculates the 3D coordinates of points based on image data from two cameras.

    Parameters:
        - camera1_data (np.ndarray): Array of image points from camera 1 in pixel coordinates (n, 2)
        - camera2_data (np.ndarray): Array of image points from camera 2 in pixel coordinates (n, 2)
        - camera_stats (dict): Dictionary containing camera parameters:
            - 'a': Distance from the cameras to the origin plane along the x-axis.
            - 'f': Focal length of the cameras.
            - 'pixel_size': Size of a pixel in meters.
            - 'resolution': Tuple representing camera resolution in pixels (width, height).

    Returns:
        - np.ndarray: Array of reconstructed 3D coordinates (n, 3).
    """
    cam_distance = camera_stats['a']
    focal_lenght = camera_stats['f']
    pixel_size = camera_stats['pixel_size']
    resolution = camera_stats['resolution']

    n_points = camera1_data.shape[0]
    points_3d = np.zeros((n_points, 3))

    cam1_pos = np.array([cam_distance, 0, 0])
    cam2_pos = np.array([-cam_distance, 0, 0])

    offset_x = resolution[0] / 2
    offset_y = resolution[1] / 2

    for i in range(n_points):
        x1_img = (camera1_data[i, 0] - offset_x) * pixel_size
        y1_img = (camera1_data[i, 1] - offset_y) * pixel_size
        x2_img = (camera2_data[i, 0] - offset_x) * pixel_size
        y2_img = (camera2_data[i, 1] - offset_y) * pixel_size

        dir_cam1 = np.array([x1_img, y1_img, focal_lenght])
        dir_cam2 = np.array([x2_img, y2_img, focal_lenght])

        points_3d[i] = find_closest_point(cam1_pos, dir_cam1, cam2_pos, dir_cam2)

    return points_3d


def find_closest_point(point1, direction1, point2, direction2):
    """
    Finds the closest point between two lines that pass through points `point1` and `point2`
    and have directions `direction1` and `direction2`, respectively.

    Parameters:
        - point1, point2 (np.ndarray): Origin points of the two lines.
        - direction1, direction2 (np.ndarray): Direction vectors of the two lines.

    Returns:
        - np.ndarray: The nearest point between the two lines.
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
