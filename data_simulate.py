import numpy as np
import matplotlib.pyplot as plt

from file_io import plot_3d_points
from lpc_indexing import find_outlier_point, analyze_coordinates
from calc_3d import triangulate_3D

def generate_laser_projection_on_rotated_plane(d, n, alpha, beta=None, rho=0, phi=0):
    """
    Generates a matrix of laser points projected onto a rotated plane in 3D space.

    Parameters:
    - d (float): Distance from the laser source to the default (non-rotated) plane.
    - n (int): Size of the laser matrix in each quadrant.
    - alpha (float): Angle of splitting in the x-direction (in degrees).
    - beta (float, optional): Angle of splitting in the y-direction (in degrees).
                              If None, beta will be set equal to alpha.
    - rho (float): Rotation angle of the plane around the x-axis (in degrees).
    - phi (float): Rotation angle of the plane around the y-axis (in degrees).

    Returns:
    - np.ndarray: Array of shape ((2n)^2 + 1, 3) containing the (x, y, z) coordinates
                  of the laser points projected onto the rotated plane.
    """
    if beta is None:
        beta = alpha

    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    rho_rad = np.deg2rad(rho)
    phi_rad = np.deg2rad(phi)

    # Define the normal vector of the default plane (z = d) and rotate it
    normal = np.array([0, 0, 1])
    
    # Rotation matrices for the plane
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(rho_rad), -np.sin(rho_rad)],
        [0, np.sin(rho_rad), np.cos(rho_rad)]
    ])
    
    rotation_y = np.array([
        [np.cos(phi_rad), 0, np.sin(phi_rad)],
        [0, 1, 0],
        [-np.sin(phi_rad), 0, np.cos(phi_rad)]
    ])
    
    # Apply rotations to the normal vector
    rotated_normal = rotation_y @ (rotation_x @ normal)

    # A point on the rotated plane (initially at (0, 0, d))
    point_on_plane = np.array([0, 0, d])

    # Collect intersection points
    points = [point_on_plane]  # Add the central point (0, 0, d)

    def intersection_line_plane(P0, d_vec, Q, n_vec):
        d_dot_n = np.dot(d_vec, n_vec)
        if np.isclose(d_dot_n, 0):
            return None  # No intersection if parallel
        t = np.dot(n_vec, Q - P0) / d_dot_n
        return P0 + t * d_vec

    # Compute laser points
    for i in range(-n+1, n+1):
        for j in range(-n+1, n+1):
            x_direction = np.tan(i * alpha_rad - alpha_rad / 2)
            y_direction = np.tan(j * beta_rad - beta_rad / 2)
            direction_vector = np.array([x_direction, y_direction, 1])
            direction_vector /= np.linalg.norm(direction_vector)
            
            # Intersection with the rotated plane
            intersection_point = intersection_line_plane(
                np.array([0, 0, 0]), direction_vector, point_on_plane, rotated_normal
            )
            
            if intersection_point is not None:
                points.append(intersection_point)

    return np.array(points)

# Example usage
d = 10.0
n = 8
alpha = 1/3
rho = 45
phi = 0
points = generate_laser_projection_on_rotated_plane(d, n, alpha, rho=rho, phi=phi)
print(points)


plot_3d_points(points)


def project_points_to_cameras(laser_points, a=0.2, f=0.04, pixel_size=2.74e-6, resolution=(4096, 3000)):
    """
    Projects laser points onto two camera screens, providing pixel coordinates.

    Parameters:
    - laser_points (np.ndarray): Array of shape (n, 3) containing the (x, y, z) coordinates of laser points.
    - a (float): Horizontal offset of each camera from the origin (default is 0.2 m).
    - f (float): Focal length of the cameras (default is 0.04 m).
    - pixel_size (float): Size of each pixel in meters (default is 2.74 µm).
    - resolution (tuple): Resolution of the cameras in pixels (default is (4096, 3000)).

    Returns:
    - np.ndarray: Two arrays of shape (n, 2), representing the pixel (x, y) coordinates for each laser point
                  on each camera.
    """
    n_points = laser_points.shape[0]
    cam1_coords = np.zeros((n_points, 2))  # Camera on the right side (a, 0, 0)
    cam2_coords = np.zeros((n_points, 2))  # Camera on the left side (-a, 0, 0)

    # Camera center points
    cam1_position = np.array([a, 0, 0])
    cam2_position = np.array([-a, 0, 0])

    # Loop through each laser point and calculate projection
    for i, point in enumerate(laser_points):
        # Vector from camera to laser point
        vec_cam1 = point - cam1_position
        vec_cam2 = point - cam2_position

        # Find intersection with camera screen at z = f
        # Intersection parameter t = (f - cam_z) / vec_z for each camera
        t_cam1 = f / vec_cam1[2]
        t_cam2 = f / vec_cam2[2]

        # Calculate the intersection point on the camera screens
        intersect_cam1 = cam1_position + t_cam1 * vec_cam1
        intersect_cam2 = cam2_position + t_cam2 * vec_cam2

        # Adjust x-coordinates by subtracting camera position, then convert to pixel coordinates
        cam1_coords[i, 0] = (intersect_cam1[0] - a) / pixel_size + resolution[0] / 2
        cam1_coords[i, 1] = intersect_cam1[1] / pixel_size + resolution[1] / 2
        cam2_coords[i, 0] = (intersect_cam2[0] + a) / pixel_size + resolution[0] / 2
        cam2_coords[i, 1] = intersect_cam2[1] / pixel_size + resolution[1] / 2

    return cam1_coords, cam2_coords


cam1_coords, cam2_coords = project_points_to_cameras(points)


def subtract_coordinate(points_2d):
    """
    Subtracts a given (x, y) coordinate from each point in a 2D points array.

    Parameters:
    - points_2d (np.ndarray): A (n, 2) array of 2D points.
    - coordinate (tuple): A tuple (x, y) representing the coordinate to subtract.

    Returns:
    - np.ndarray: A new (n, 2) array with the coordinate subtracted from each point.
    """
    coordinate = find_outlier_point(points_2d)[0]
    return points_2d - np.array(coordinate)

#cam1_coords = subtract_coordinate(cam1_coords)
cam1_coords= analyze_coordinates(cam1_coords, tolerance=29.0)
#cam2_coords = subtract_coordinate(cam2_coords)
cam2_coords= analyze_coordinates(cam2_coords, tolerance=29.0)

def plot_2d_points(points_2d, title='2D Scatter Plot'):
    """
    Plots a 2D array of points using matplotlib.

    Parameters:
    - points_2d (np.ndarray): A (n, 2) array containing the 2D coordinates (x, y) of the points.
    - title (str): The title of the plot.
    """
    x_coords = points_2d[:, 0]
    y_coords = points_2d[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, c='blue', marker='o', s=30, alpha=0.7)
    
    plt.xlim(0,4096)
    plt.ylim(0,3000)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    
def plot_2d_points_pair(points_2d_1, points_2d_2, labels=('Dataset 1', 'Dataset 2'), title='2D Scatter Plot of Two Datasets'):
    """
    Plots two sets of 2D points on the same plot using matplotlib.

    Parameters:
    - points_2d_1 (np.ndarray): A (n, 2) array containing the 2D coordinates (x, y) of the first set of points.
    - points_2d_2 (np.ndarray): A (n, 2) array containing the 2D coordinates (x, y) of the second set of points.
    - labels (tuple): Labels for the two datasets, used in the legend.
    - title (str): The title of the plot.
    """
    x_coords_1, y_coords_1 = points_2d_1[:, 0], points_2d_1[:, 1]
    x_coords_2, y_coords_2 = points_2d_2[:, 0], points_2d_2[:, 1]

    plt.figure(figsize=(12, 8))
    plt.scatter(x_coords_1, y_coords_1, c='blue', marker='o', s=30, alpha=0.7, label=labels[0])
    plt.scatter(x_coords_2, y_coords_2, c='red', marker='x', s=30, alpha=0.7, label=labels[1])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    
    
print(cam1_coords)
print(cam2_coords)
plot_2d_points_pair(cam1_coords, cam2_coords)



three_d_points = triangulate_3D(cam1_coords, cam2_coords)

plot_3d_points(three_d_points)

#print(three_d_points)

#print(cam1_coords)























