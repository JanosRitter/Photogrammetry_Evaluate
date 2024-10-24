import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def triangulate_3D(camera1_data, camera2_data, baseline=0.4, focal_length=1.0, distance=10.0):
    """
    Triangulates 3D points from two sets of 2D points obtained from two cameras.
    The system assumes that the laser matrix is projected onto a flat surface with
    the center of the matrix as the origin of the 3D coordinate system.

    The angle between laser points is based on the grid index, with each step corresponding
    to a 1/6° or 1/3° deviation from the central point (0,0) at the laser origin.

    Parameters:
        - camera1_data (np.ndarray): (n, 4) array with 2D points and their grid indices (x_coord, y_coord, x_index, y_index) for camera 1.
        - camera2_data (np.ndarray): (n, 4) array with 2D points and their grid indices (x_coord, y_coord, x_index, y_index) for camera 2.
        - baseline (float): Distance between the two cameras in meters (default is 0.4 meters).
        - focal_length (float): The focal length of the cameras in meters (default is 1.0 meter).
        - distance (float): The initial distance from the cameras to the object plane in meters (default is 10 meters).

    Returns:
        - np.ndarray: (n, 3) array of triangulated 3D points (X, Y, Z).
    """
    
    def calculate_angles(indices):
        """
        Calculates the X and Y angles based on the grid indices.
        The central point (0,0) is the reference, with deviations of 1/6° for the first row/column
        and 1/3° for subsequent rows/columns.
        
        Parameters:
            - indices (np.ndarray): (n, 2) array of indices (x_index, y_index).
        
        Returns:
            - angles_x (np.ndarray): Array of X-axis angles in radians.
            - angles_y (np.ndarray): Array of Y-axis angles in radians.
        """
        angle_per_index = 1/3  # 1/3° deviation for all indices except the central ones
        # First row/column deviates by 1/6°
        near_center_angle = 1/6

        # Calculate angles based on the indices
        angles_x = np.zeros(indices.shape[0])
        angles_y = np.zeros(indices.shape[0])

        for i, (x_idx, y_idx) in enumerate(indices):
            if abs(x_idx) == 1:  # For the first row/column, use 1/6°
                angles_x[i] = near_center_angle * np.sign(x_idx)
            else:
                angles_x[i] = abs(x_idx) * angle_per_index * np.sign(x_idx)

            if abs(y_idx) == 1:  # For the first row/column, use 1/6°
                angles_y[i] = near_center_angle * np.sign(y_idx)
            else:
                angles_y[i] = abs(y_idx) * angle_per_index * np.sign(y_idx)

        # Convert degrees to radians
        angles_x = np.deg2rad(angles_x)
        angles_y = np.deg2rad(angles_y)

        return angles_x, angles_y

    # Get the indices from the camera data
    indices1 = camera1_data[:, 2:4]
    indices2 = camera2_data[:, 2:4]

    # Calculate angles for both sets of indices (same for both cameras since it's the same grid)
    angles_x1, angles_y1 = calculate_angles(indices1)
    angles_x2, angles_y2 = calculate_angles(indices2)

    # X and Y positions are determined by the angles at the given distance
    x_positions = distance * np.tan(angles_x1)  # X positions based on angles from camera1
    y_positions = distance * np.tan(angles_y1)  # Y positions based on angles from camera1

    # Now handle the Z calculation based on the disparity between camera1 and camera2
    disparity = camera1_data[:, 0] - camera2_data[:, 0]  # Disparity in x-coordinates (pixel difference)
    
    # Prevent divide-by-zero by setting a minimum disparity threshold
    min_disparity = 1e-6
    disparity = np.where(disparity == 0, min_disparity, disparity)

    # Calculate Z positions using the disparity and triangulation formula
    z_positions = baseline * focal_length / disparity

    # Stack the X, Y, and Z coordinates into a single (n, 3) array
    points_3D = np.vstack((x_positions, y_positions, z_positions)).T

    return points_3D


camera1_data = np.array([
    [850, 470, -2, -2],   # Punkt links oben
    [900, 480, -1, -2],   # Punkt daneben (rechts oben)
    [950, 490,  1, -2],   # Punkt daneben (weiter rechts oben)
    [1000, 500,  2, -2],  # Punkt rechts oben

    [860, 520, -2, -1],   # Punkt links mitte oben
    [910, 530, -1, -1],   # Punkt daneben
    [960, 540,  1, -1],   # Punkt daneben
    [1010, 550,  2, -1],  # Punkt rechts mitte oben

    [870, 570, -2,  1],   # Punkt links mitte unten
    [920, 580, -1,  1],   # Punkt daneben
    [970, 590,  1,  1],   # Punkt daneben
    [1020, 600,  2,  1],  # Punkt rechts mitte unten
    
    [880, 620, -2,  2],   # Punkt links unten
    [930, 630, -1,  2],   # Punkt daneben
    [980, 640,  1,  2],   # Punkt daneben
    [1030, 650,  2,  2],  # Punkt rechts unten

    [970, 590,  0,  0]    # Zentraler Punkt (Ursprung)
])


camera2_data = np.array([
    [830, 460, -2, -2],   # Punkt links oben
    [880, 470, -1, -2],   # Punkt daneben (rechts oben)
    [930, 480,  1, -2],   # Punkt daneben (weiter rechts oben)
    [980, 490,  2, -2],   # Punkt rechts oben

    [840, 510, -2, -1],   # Punkt links mitte oben
    [890, 520, -1, -1],   # Punkt daneben
    [940, 530,  1, -1],   # Punkt daneben
    [990, 540,  2, -1],   # Punkt rechts mitte oben

    [850, 560, -2,  1],   # Punkt links mitte unten
    [900, 570, -1,  1],   # Punkt daneben
    [950, 580,  1,  1],   # Punkt daneben
    [1000, 590,  2,  1],  # Punkt rechts mitte unten
    
    [860, 610, -2,  2],   # Punkt links unten
    [910, 620, -1,  2],   # Punkt daneben
    [960, 630,  1,  2],   # Punkt daneben
    [1010, 640,  2,  2],  # Punkt rechts unten

    [950, 580,  0,  0]    # Zentraler Punkt (Ursprung)
])



points_3D = triangulate_3D(camera1_data, camera2_data)

print(points_3D)


def plot_3d_points(points_3D, title='3D Scatter Plot', save_as_file=False, filename='3d_points_plot.png'):
    """
    Plots 3D points using matplotlib and saves the plot to a file if specified.

    Parameters:
        - points_3d (np.ndarray): A (n, 3) array containing the 3D coordinates of the points (X, Y, Z).
        - title (str): The title of the plot.
        - save_as_file (bool): If True, the plot is saved as a file.
        - filename (str): The filename for the plot image (if save_as_file is True).
    """
    # Create a new figure for the 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, Z coordinates from the input array
    x_coords = points_3D[:, 0]
    y_coords = points_3D[:, 1]
    z_coords = points_3D[:, 2]

    # Scatter plot of the 3D points
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis', s=50)

    # Add color bar for Z-axis
    fig.colorbar(scatter, ax=ax, label='Z (Depth)')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Optionally save the plot as a file
    if save_as_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"3D plot saved as '{filename}'")

    # Display the plot
    plt.show()
    
plot_3d_points(points_3D)

