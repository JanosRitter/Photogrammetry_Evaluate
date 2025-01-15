import numpy as np
import matplotlib.pyplot as plt
import os
from file_io.npy_processing import load_all_npy_files



def generate_laser_projection_on_rotated_plane(d, n, alpha, beta=None, angle=(0, 0)):
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

    rho, phi = angle
    if beta is None:
        beta = alpha

    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    rho_rad = np.deg2rad(rho)
    phi_rad = np.deg2rad(phi)

    normal = np.array([0, 0, 1])

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

    rotated_normal = rotation_y @ (rotation_x @ normal)

    point_on_plane = np.array([0, 0, d])

    points = [point_on_plane]

    def intersection_line_plane(P0, d_vec, Q, n_vec):
        d_dot_n = np.dot(d_vec, n_vec)
        if np.isclose(d_dot_n, 0):
            return None
        t = np.dot(n_vec, Q - P0) / d_dot_n
        return P0 + t * d_vec

    for i in range(-n+1, n+1):
        for j in range(-n+1, n+1):
            x_direction = np.tan(i * alpha_rad - alpha_rad / 2)
            y_direction = np.tan(j * beta_rad - beta_rad / 2)
            direction_vector = np.array([x_direction, y_direction, 1])
            direction_vector /= np.linalg.norm(direction_vector)

            intersection_point = intersection_line_plane(
                np.array([0, 0, 0]), direction_vector, point_on_plane, rotated_normal
            )

            if intersection_point is not None:
                points.append(intersection_point)

    return np.array(points)


def project_points_to_cameras(laser_points, camera_stats, translation_cam2=np.array([0.4, 0.0, 0.0]), rotation_cam2=np.eye(3)):
    """
    Projects laser points onto two camera screens, providing pixel coordinates.

    Parameters:
    - laser_points (np.ndarray): Array of shape (n, 3) containing the (x, y, z)
      coordinates of laser points in 3D space.
    - camera_stats (dict): Dictionary containing camera parameters:
        - 'f' (float): Focal length of the cameras in meters.
        - 'pixel_size' (float): Size of a pixel in meters.
        - 'resolution' (tuple): Camera resolution in pixels (width, height).
    - translation_cam2 (np.ndarray, optional): Translation vector of the second camera (3,).
      Defaults to np.array([0.4, 0.0, 0.0]).
    - rotation_cam2 (np.ndarray, optional): Rotation matrix of the second camera (3, 3).
      Defaults to np.eye(3) (no rotation).

    Returns:
    - tuple: Two arrays of shape (n, 2), representing the pixel (x, y) coordinates
      for each laser point on each camera.
    """
    focal_length = camera_stats['f']
    pixel_size = camera_stats['pixel_size']
    resolution = camera_stats['resolution']

    n_points = laser_points.shape[0]
    cam1_coords = np.zeros((n_points, 2))
    cam2_coords = np.zeros((n_points, 2))

    cam1_position = np.array([0, 0, 0])  # Kamera 1 im Ursprung

    for i, point in enumerate(laser_points):
        # Projektionsvektor zur Kamera 1
        vec_cam1 = point - cam1_position
        t_cam1 = focal_length / vec_cam1[2]
        intersect_cam1 = cam1_position + t_cam1 * vec_cam1
        cam1_coords[i, 0] = intersect_cam1[0] / pixel_size + resolution[0] / 2
        cam1_coords[i, 1] = intersect_cam1[1] / pixel_size + resolution[1] / 2

        # Kamera 2: Translations- und Rotationsmatrix berücksichtigen
        point_cam2_space = rotation_cam2 @ (point - translation_cam2)
        vec_cam2 = point_cam2_space
        t_cam2 = focal_length / vec_cam2[2]
        intersect_cam2 = t_cam2 * vec_cam2
        cam2_coords[i, 0] = intersect_cam2[0] / pixel_size + resolution[0] / 2
        cam2_coords[i, 1] = intersect_cam2[1] / pixel_size + resolution[1] / 2

    return cam1_coords, cam2_coords


def plot_2d_points_pair(points_2d_1, points_2d_2, labels=('Dataset 1', 'Dataset 2'), title='2D Scatter Plot of Two Datasets'):
    """
    Plots two sets of 2D points on the same plot using matplotlib.

    Parameters:
    - points_2d_1 (np.ndarray): A (n, 2) array containing the
    2D coordinates (x, y) of the first set of points.
    - points_2d_2 (np.ndarray): A (n, 2) array containing the
    2D coordinates (x, y) of the second set of points.
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


def plot_coordinates(coordinates, save_path):
    """
    Plots (x, y) coordinates from an (n, 2) array and saves the plot to a given path.

    Parameters:
        - coordinates (np.ndarray): A (n, 2) array with x and y values.
        - save_path (str): Path where the plot image should be saved.
    """
    if not isinstance(coordinates, (list, tuple, np.ndarray)):
        raise ValueError("The coordinates must be a list, tuple, or numpy array.")

    if len(coordinates) == 0 or len(coordinates[0]) != 2:
        raise ValueError("The input array must have shape (n, 2).")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(coordinates[:, 0], coordinates[:, 1], 'rx', label='Coordinates')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot of Coordinates")
    plt.grid(True)
    plt.xlim(0, 4096)
    plt.ylim(0, 3000)

    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    print(f"Plot saved at: {save_path}")


def simulate_background_noise(array_shape=(3000, 4096), mean=0, std=1):
    """
    Simuliert Hintergrundrauschen als normalverteiltes Zufalls-Array,
    wobei die Werte ganzzahlig und positiv sind.

    Parameter:
    - array_shape (tuple): Die Form des Ausgangsarrays, z. B. (3000, 4096).
    - mean (float): Der Mittelwert der Normalverteilung.
    - std (float): Die Standardabweichung der Normalverteilung.

    Rückgabewert:
    - np.ndarray: Array mit der Form `array_shape`, das simuliertes,
    ganzzahliges, positives Rauschen enthält.
    """
    noise_array = np.random.normal(loc=mean, scale=std, size=array_shape)

    noise_array = np.maximum(noise_array, 0)

    noise_array = np.round(noise_array).astype(int)

    return noise_array



def add_noise_to_data(data, noise, offset=0, value_range=(0, 255)):
    """
    Combines measurement data with noise while handling saturation for 8-bit data.

    Parameters:
    - data (np.ndarray): Original measurement data (e.g., shape (3000, 4096)).
    - noise (np.ndarray): Simulated noise to be added (same shape as `data`).
    - offset (int): Value to subtract from the original data before adding noise. Minimum value after subtraction is 0.
    - value_range (tuple): Tuple specifying the valid range of values (default: (0, 255)).

    Returns:
    - np.ndarray: Combined data with noise, clipped to `value_range`.
    """
    min_val, max_val = value_range

    data_float = data.astype(np.float32)

    print(f"Original Data Statistics (before offset):\nMin: {data_float.min()}\nMax: {data_float.max()}\nMean: {data_float.mean()}")

    adjusted_data = np.clip(data_float - offset, min_val, None)

    print(f"Adjusted Data Statistics (after offset):\nMin: {adjusted_data.min()}\nMax: {adjusted_data.max()}\nMean: {adjusted_data.mean()}")

    noisy_data = adjusted_data + noise

    print(f"Noisy Data Statistics (before clipping):\nMin: {noisy_data.min()}\nMax: {noisy_data.max()}\nMean: {noisy_data.mean()}")

    noisy_data_clipped = np.clip(noisy_data, min_val, max_val)

    print(f"Noisy Data Statistics (after clipping):\nMin: {noisy_data_clipped.min()}\nMax: {noisy_data_clipped.max()}\nMean: {noisy_data_clipped.mean()}")

    noisy_data_uint8 = noisy_data_clipped.astype(np.uint8)

    return noisy_data_uint8


def add_noise_to_folder(input_folder, output_folder_name, mean, std, offset):
    """
    Adds background noise to all `.npy` files in a folder and saves the noisy arrays
    in a new folder.

    Parameters:
    - input_folder (str): Path to the folder containing `.npy` files.
    - output_folder_name (str): Name of the new folder to save noisy data.
    - mean (float): Mean of the noise distribution.
    - std (float): Standard deviation of the noise distribution.
    - offset (float): Value to subtract from data before adding noise.

    Returns:
    - None
    """
    npy_files, arrays = load_all_npy_files(input_folder)

    base_folder = os.path.dirname(input_folder)
    output_folder = os.path.join(base_folder, output_folder_name)

    os.makedirs(output_folder, exist_ok=True)

    for filename, array in zip(npy_files, arrays):
        array_shape = array.shape
        print("Original Data Statistics:")
        print("Min:", np.min(array))
        print("Max:", np.max(array))
        print("Mean:", np.mean(array))

        noise = simulate_background_noise(array_shape=array_shape, mean=mean, std=std)
        print("Simulated Noise Statistics:")
        print("Min:", np.min(noise))
        print("Max:", np.max(noise))
        print("Mean:", np.mean(noise))

        noisy_array = add_noise_to_data(data=array, noise=noise, offset=offset)

        output_path = os.path.join(output_folder, filename)
        np.save(output_path, noisy_array)

        print(f"Processed and saved: {filename} -> {output_path}")
