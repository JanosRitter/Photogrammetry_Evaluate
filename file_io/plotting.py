"""
This module contains different plotting routines for contour, 2D and 3D plots
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from file_io.utility import construct_output_path
from mpl_toolkits.mplot3d import Axes3D

os.chdir('C:/Users/Janos/Documents/Masterarbeit/3D_scanner/Pythoncode')

def plot_brightness_array(brightness_array, x_limit=None, y_limit=None, save_path=None):
    """
    Creates a 3D PNG-Plot of a given array, where the color indicates the z-value

    parameter:
        - brightness_array: (n,m) array you want to visualize
        - x_limit: (1,2) array that limits the plot in x-direction
        - y_limit: (1,2) array that limits the plot in y-direction
        - save_path: path where the plot is saved

    returns:
        -no returns, except the savec image
    """

    if x_limit is None:
        x_limit = [0, brightness_array.shape[1]]
    if y_limit is None:
        y_limit = [0, brightness_array.shape[0]]

    x_max = min(x_limit[1], brightness_array.shape[1])
    y_max = min(y_limit[1], brightness_array.shape[0])
    x_min, y_min = x_limit[0], y_limit[0]

    width, height = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    brightness_subarray = brightness_array[y_min:y_max, x_min:x_max]

    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.plot_surface(width, height, brightness_subarray, cmap='gray')

    axis.set_xlabel('X (Pixel)')
    axis.set_ylabel('Y (Pixel)')
    axis.set_zlabel('Brightness')
    axis.set_title('3D Brightness Distribution')

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")
    plt.close()




def create_contour_plot(data, save_path=None):
    """
    Generates a contour plot from a .npy file with values representing z-coordinates.
    The x and y axes reflect the shape of the array.

    Parameters:
        file_path (str): Directory path where the .npy file is located.
        file_name (str): Name of the .npy file (without the extension).

    Returns:
        None: Saves the contour plot as a .png file in the same directory as the input file.
    """
    x_value = np.arange(data.shape[1])
    y_value = np.arange(data.shape[0])

    plt.contourf(x_value, y_value, data, levels=20, cmap='viridis')
    plt.colorbar(label='Brightness')
    plt.xlabel('X')
    plt.ylabel('Y')
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Contour plot saved at: {save_path}")
    plt.close()




def create_image_plot(data, peaks=None, mean=None, output_path=None, output_filename="plot.png"):
    """
    Creates an image plot from a 2D array representing pixel brightness and overlays points.

    Parameters:
        - data (np.ndarray): 2D array with pixel brightness values.
        - peaks (np.ndarray): (n, 2) array with the estimated laser point centers (optional).
        - mean (np.ndarray): (n, 2) array with calculated laser point centers (optional).
        - output_path (str): Path to the output directory (for saving the plot).
        - output_filename (str): Name of the output file (default: "plot.png").
    """

    plt.figure(figsize=(10, 6))
    colormap = 'viridis'
    plt.imshow(data, cmap=colormap, interpolation='nearest')
    plt.colorbar(label='Brightness')

    if peaks is not None:
        plt.plot(peaks[:, 0], peaks[:, 1], 'ro', markersize=1, label='Peaks')

    if mean is not None:
        plt.plot(mean[:, 0], mean[:, 1], 'bo', markersize=1, label='Mean Points')

    if peaks is not None:
        buffer = 100
        x_min, x_max = peaks[:, 0].min() - buffer, peaks[:, 0].max() + buffer
        y_min, y_max = peaks[:, 1].min() - buffer, peaks[:, 1].max() + buffer
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
    plt.xlim(180, 320)
    plt.ylim(180, 320)
    
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = os.path.join(output_path, output_filename)
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Image plot saved at: {save_path}")
    else:
        print("Output path not provided. Plot not saved.")
        plt.show()

    plt.close()




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

    plt.xlim(0, 4096)
    plt.ylim(0, 3000)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()




def plot_2d_points_pair(points_2d_1, points_2d_2, title='2D Scatter Plot of Two Datasets'):
    """
    Plots two sets of 2D points on the same plot using matplotlib.

    Parameters:
    - points_2d_1 (np.ndarray): A (n, 2) array containing the 2D coordinates (x, y)
    of the first set of points.
    - points_2d_2 (np.ndarray): A (n, 2) array containing the 2D coordinates (x, y)
    of the second set of points.
    - labels (tuple): Labels for the two datasets, used in the legend.
    - title (str): The title of the plot.
    """

    labels = ('Dataset 1', 'Dataset 2')

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




def plot_3d_points(points_3d, path=None, dateiname="3d_plot.png"):
    """
    Plots 3D points using matplotlib and returns the Figure object.

    Parameters:
        - points_3d (np.ndarray): A (n, 3) array containing the 3D coordinates (X, Y, Z).
        - path (str): The directory path where the plot will be saved.
        - dateiname (str): The name of the file to save the plot (default: "3d_plot.png").
    
    Returns:
        - fig (matplotlib.figure.Figure): The figure object containing the 3D plot.
    """
    fig = plt.figure(figsize=(6, 6))  # Größe anpassen
    axis = fig.add_subplot(111, projection='3d')

    x_coords, y_coords, z_coords = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    scatter = axis.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis', s=50)

    fig.colorbar(scatter, ax=axis, label='Z (Depth)')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    axis.set_title("3D Scatter Plot")

    if path:
        save_path = os.path.join(path, dateiname)
        os.makedirs(path, exist_ok=True)
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"3D plot saved at: {save_path}")

    plt.close(fig)  # Verhindert, dass sich ein extra Fenster öffnet
    return fig  # Jetzt wird das Figure-Objekt zurückgegeben




def plot_differences_as_bar_chart(differences, output_path, plot_type="all", abs_values=False, output_filename="differences_plot.png"):
    """
    Plots the differences (x, y, and optionally z) as bar charts and/or the Euclidean norm.

    Parameters:
    - differences (np.ndarray): Array of shape (n, 2) or (n, 3) containing
    the (x, y, z) differences.
    - output_path (str): Path to the output directory for saving the plots.
    - plot_type (str): Type of differences to plot. Options:
        "x", "y", "z", "norm", "all" (default: "all").
    - abs_values (bool): Whether to plot the absolute values
    of the differences (default: False).
    - output_filename (str): Name of the output file
    (default: "differences_plot.png").
    """
    if differences.shape[1] not in {2, 3}:
        raise ValueError("Input array must have shape (n, 2) or (n, 3).")

    data = np.abs(differences) if abs_values else differences

    labels = ["x", "y", "z"][:data.shape[1]]
    available_plots = {label: data[:, i] for i, label in enumerate(labels)}
    available_plots["norm"] = np.linalg.norm(differences, axis=1)

    if plot_type == "all":
        plot_keys = list(available_plots.keys())
    elif plot_type in available_plots:
        plot_keys = [plot_type]
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Choose from {list(available_plots.keys()) + ['all']}.")

    os.makedirs(output_path, exist_ok=True)

    def save_bar_chart(data, label, filename):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(data)), data, color='blue', alpha=0.7)
        plt.xlabel("Index")
        plt.ylabel(f"{label.capitalize()} Differences")
        plt.title(f"{label.capitalize()} Differences (absolute)" if abs_values else f"{label.capitalize()} Differences")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Plot saved as {filename}")

    for key in plot_keys:
        save_bar_chart(available_plots[key], key, os.path.join(output_path, f"{key}_{output_filename}"))

    if plot_type == "all":
        _, axes = plt.subplots(len(plot_keys), 1, figsize=(10, 4 * len(plot_keys)), sharex=True)

        if len(plot_keys) == 1:
            axes = [axes]

        for idx, key in enumerate(plot_keys):
            axes[idx].bar(range(len(available_plots[key])), available_plots[key], color='blue', alpha=0.7)
            axes[idx].set_ylabel(f"{key.capitalize()} Differences")
            axes[idx].set_title(f"{key.capitalize()} Differences (absolute)" if abs_values else f"{key.capitalize()} Differences")
            axes[idx].grid(True, linestyle='--', alpha=0.6)

        axes[-1].set_xlabel("Index")
        plt.tight_layout()

        save_path = os.path.join(output_path, f"all_{output_filename}")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Combined plot saved as {save_path}")




def analyze_background_noise(data, bins=256, hist_title="Value Frequency Histogram"):
    """
    Analyzes the background noise in a (n, m) array by generating a histogram and boxplot.

    Parameters:
    - data (np.ndarray): Input array of shape (n, m) containing the data to analyze.
    - bins (int): Number of bins for the histogram. Default is 50.
    - hist_title (str): Title for the histogram plot.
    - boxplot_title (str): Title for the boxplot.

    Returns:
    - None
    """
    flattened_data = data.flatten()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(flattened_data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(hist_title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    sns.boxplot(x=flattened_data, color='lightgreen', orient="h")
    plt.title("Boxplot of Values")
    plt.xlabel("Value")


    plt.tight_layout()
    plt.show()

def plot_all_methods(results, x_values=None, input_folder=None, save_plot=True):
    """
    Plot mean values with standard deviations for multiple methods.

    Parameters:
    - results (dict): Dictionary containing mean and standard deviation values for each method.
    - x_values (np.ndarray or None): X-values for the plot.
    - input_folder (str): Folder for saving the plot.
    - save_plot (bool): Whether to save the plot as an image file.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for method, data in results.items():
        means = data["means"]
        stds = data["stds"]

        if x_values is None:
            x_values = np.arange(len(means))

        plt.errorbar(
            x=x_values,
            y=means,
            yerr=stds,
            label=method,
            fmt='o',
            capsize=9
        )

    plt.xlabel("Laser spot size in pixels")
    plt.ylabel("Mean deviation from input structure in m")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_plot and input_folder:
        output_path = os.path.join(construct_output_path(input_folder), "plots")
        os.makedirs(output_path, exist_ok=True)
        plot_filename = "comparison_plot_all_methods.png"
        plt.savefig(os.path.join(output_path, plot_filename), dpi=300)
        print(f"Plot saved in: {output_path}")

    plt.show()
