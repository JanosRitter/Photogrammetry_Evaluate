import os
import numpy as np
from file_io import *
from calc_3d import triangulate_3d
from lpc_indexing import analyze_coordinates


input_folder = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\spot_scale_1\spots_with_backgroundnoise"

camera_stats = {
    'a': 0.2,                  # Distance from the cameras to the origin plane along the x-axis
    'f': 0.04,                 # Focal length of the cameras in meters
    'pixel_size': 2.74e-6,     # Pixel size in meters
    'resolution': (4096, 3000) # Resolution of the cameras in pixels (width, height)
}

def process_and_triangulate(input_folder, methode="center_of_mass", camera_stats=None):
    """
    Processes 2D LPC data from two cameras to generate 3D structures, save them as .npy files,
    and create plots.

    Parameters:
    - input_folder (str): Path to the folder containing input data.
    - methode (str): Method used to calculate LPC. Options: "center_of_mass", "gauss_fit", "skewed_gauss_fit".
    - camera_stats (dict): Dictionary containing camera parameters:
        - 'a': Distance from the cameras to the origin plane along the x-axis.
        - 'f': Focal length of the cameras.
        - 'pixel_size': Size of a pixel in meters.
        - 'resolution': Tuple representing camera resolution in pixels (width, height).
    """
    if camera_stats is None:
        raise ValueError("Camera stats must be provided for triangulation.")

    # Construct paths
    output_folder = construct_output_path(input_folder)
    method_folder = os.path.join(output_folder, methode)
    if not os.path.exists(method_folder):
        raise ValueError(f"The specified method folder does not exist: {method_folder}")

    structure_folder = os.path.join(method_folder, "3d_structure")
    if not os.path.exists(structure_folder):
        os.makedirs(structure_folder)

    # Load all .npy files from the method folder
    lpc_files, lpc_arrays = load_all_npy_files(method_folder)

    # Separate files by camera
    cam1_files = [file for file in lpc_files if "cam1" in file]
    cam2_files = [file for file in lpc_files if "cam2" in file]
    cam1_files.sort()  # Ensure files are in order
    cam2_files.sort()

    if len(cam1_files) != len(cam2_files):
        raise ValueError("Mismatch in the number of cam1 and cam2 files.")

    # Process file pairs
    for cam1_file, cam2_file in zip(cam1_files, cam2_files):
        print(f"Processing pair: {cam1_file} and {cam2_file}")

        # Load data
        cam1_data = np.load(os.path.join(method_folder, cam1_file))
        cam2_data = np.load(os.path.join(method_folder, cam2_file))

        # Analyze and sort coordinates
        cam1_data = analyze_coordinates(cam1_data)
        cam2_data = analyze_coordinates(cam2_data)

        # Triangulate 3D points
        points_3d = triangulate_3d(cam1_data, cam2_data, camera_stats)

        # Save 3D data using the filename of the first file in the pair
        base_name = os.path.splitext(cam1_file)[0]  # Use the name of the first file
        base_name = base_name.replace("cam1", "")  # Remove 'cam1' if present for simplicity

        npy_filename = f"{base_name}_3d.npy"
        npy_filepath = os.path.join(structure_folder, npy_filename)
        np.save(npy_filepath, points_3d)
        print(f"3D points saved to: {npy_filepath}")

        # Plot 3D points
        plot_filename = f"{base_name}_plot.png"
        plot_filepath = os.path.join(structure_folder, plot_filename)
        plot_3d_points(points_3d, path=plot_filepath)
        print(f"3D plot saved to: {plot_filepath}")

    print(f"Processing completed. 3D data and plots saved in: {structure_folder}")

#process_and_triangulate(input_folder, methode="center_of_mass_with_threshold", camera_stats=camera_stats)
#process_and_triangulate(input_folder, methode="gauss_fit", camera_stats=camera_stats)
#process_and_triangulate(input_folder, methode="skewed_gauss_fit", camera_stats=camera_stats)
#process_and_triangulate(input_folder, methode="non_linear_center_of_mass", camera_stats=camera_stats)


def compare_3d_structures_to_theoretical(input_folder, methode=None, filenames=None, abs_values=True, plot_type="norm"):
    """
    Compares 3D structure data from one or multiple methods to theoretical values for specific filenames.

    Parameters:
    - input_folder (str): Path to the folder containing input data.
    - methode (str or None): Method(s) to calculate the 3D structures. Options: 
                              "center_of_mass", "gauss_fit", "skewed_gauss_fit", 
                              "non_linear_center_of_mass", "center_of_mass_with_threshold".
                              If None, all methods are processed.
    - filenames (list or None): List of 3D structure filenames to process. If None, all `.npy` files are processed.
    - abs_values (bool): Whether to compute absolute differences (default: False).
    - plot_type (str): Type of plots to generate. Options: "x", "y", "z", "norm", "all" (default: "all").

    Returns:
    - None
    """
    import os

    # Hard-coded path to theoretical 3D data
    theoretical_path = r"C:\\Users\\Janos\\Documents\\Masterarbeit\\3D_scanner\\input_output\\output\\spot_scale_1\\averaged_noise_spots"
    theoretical_filename = "projection__scale_1_projection__scale_1_3d.npy"

    # Dictionary mapping methods to their shorthand folder names
    methods_dict = {
        "center_of_mass": "com",
        "gauss_fit": "gf",
        "skewed_gauss_fit": "sgf",
        "non_linear_center_of_mass": "nlcom",
        "center_of_mass_with_threshold": "comwt"
    }

    # Determine methods to process
    if methode is None:
        methods_to_process = methods_dict.keys()
    elif isinstance(methode, str):
        if methode not in methods_dict:
            raise ValueError(f"Unknown method '{methode}'. Available methods: {list(methods_dict.keys())}")
        methods_to_process = [methode]
    else:
        raise ValueError("Invalid 'methode' parameter. Must be None or a valid method name.")

    # Load theoretical data
    theoretical_data = load_npy_file(theoretical_path, theoretical_filename)

    # Process each method
    for method in methods_to_process:
        print(f"Processing method: {method}")
        # Construct the method folder path
        output_folder = construct_output_path(input_folder)
        method_folder = os.path.join(output_folder, method, "3d_structure")

        if not os.path.exists(method_folder):
            print(f"Warning: The folder for the method '{method}' does not exist: {method_folder}")
            continue

        # Load 3D structure data
        structure_files, structure_arrays = load_all_npy_files(method_folder, filenames=filenames)

        # Ensure output folder for results exists
        comparison_folder = os.path.join(method_folder, "3d_comparison")
        if not os.path.exists(comparison_folder):
            os.makedirs(comparison_folder)

        # Process each 3D structure file
        for structure_file, structure_array in zip(structure_files, structure_arrays):
            print(f"Processing 3D structure file: {structure_file}")

            # Check shape compatibility
            if structure_array.shape != theoretical_data.shape:
                raise ValueError(f"Shape mismatch: {structure_file} (shape {structure_array.shape}) "
                                 f"and theoretical data (shape {theoretical_data.shape}).")

            # Compute differences
            differences = calculate_differences(theoretical_data, structure_array)
            if abs_values:
                differences = np.abs(differences)

            # Save differences
            diff_filename = f"diff_{structure_file}"
            diff_filepath = os.path.join(comparison_folder, diff_filename)
            np.save(diff_filepath, differences)
            print(f"Differences saved to: {diff_filepath}")

            # Plot differences
            plot_filename = f"plot_{os.path.splitext(diff_filename)[0]}.png"
            plot_differences_as_bar_chart(
                differences,
                output_path=comparison_folder,
                plot_type=plot_type,
                abs_values=abs_values,
                output_filename=plot_filename
            )
            print(f"Plot created: {plot_filename}")

            # Compute and log mean differences
            mean_diff = np.mean(differences, axis=0)
            print(f"Mean differences for {structure_file}: {mean_diff}")

        print(f"Finished processing method: {method}")

    print("3D structure comparison completed for all selected methods.")
    
#compare_3d_structures_to_theoretical(input_folder, methode="center_of_mass_with_threshold")
    
def compare_3d_structures_between_folders(input_folder, scale_factor=1, methode=None, filenames=None, abs_values=True, plot_type="norm"):
    """
    Compares 3D structure data from one folder to theoretical values in another folder for specific filenames.

    Parameters:
    - input_folder (str): Path to the folder containing measured 3D structure data.
    - theoretical_folder (str): Path to the folder containing theoretical 3D structure data.
    - scale_factor (float): Factor by which to scale the theoretical data (default: 1).
    - methode (str or None): Method(s) to calculate the 3D structures. Options: 
                              "center_of_mass", "gauss_fit", "skewed_gauss_fit", 
                              "non_linear_center_of_mass", "center_of_mass_with_threshold".
                              If None, all methods are processed.
    - filenames (list or None): List of 3D structure filenames to process. If None, all `.npy` files are processed.
    - abs_values (bool): Whether to compute absolute differences (default: True).
    - plot_type (str): Type of plots to generate. Options: "x", "y", "z", "norm", "all" (default: "norm").

    Returns:
    - None
    """
    import os

    # Dictionary mapping methods to their shorthand folder names
    methods_dict = {
        "center_of_mass": "com",
        "gauss_fit": "gf",
        "skewed_gauss_fit": "sgf",
        "non_linear_center_of_mass": "nlcom",
        "center_of_mass_with_threshold": "comwt"
    }

    # Determine methods to process
    if methode is None:
        methods_to_process = methods_dict.keys()
    elif isinstance(methode, str):
        if methode not in methods_dict:
            raise ValueError(f"Unknown method '{methode}'. Available methods: {list(methods_dict.keys())}")
        methods_to_process = [methode]
    else:
        raise ValueError("Invalid 'methode' parameter. Must be None or a valid method name.")

    # Process each method
    for method in methods_to_process:
        print(f"Processing method: {method}")
        # Construct the method folder paths
        output_folder = construct_output_path(input_folder)
        method_folder_measured = os.path.join(output_folder, method, "3d_structure")
        theoretical_folder = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\spot_scale_1\diff_dis_10_add_noise\center_of_mass\3d_structure"


        if not os.path.exists(method_folder_measured):
            print(f"Warning: The folder for the measured data '{method}' does not exist: {method_folder_measured}")
            continue
        if not os.path.exists(theoretical_folder):
            print(f"Warning: The folder for the theoretical data '{method}' does not exist: {theoretical_folder}")
            continue

        # Load measured and theoretical 3D structure data
        measured_files, measured_arrays = load_all_npy_files(method_folder_measured, filenames=filenames)
        theoretical_files, theoretical_arrays = load_all_npy_files(theoretical_folder, filenames=filenames)

        if len(measured_files) != len(theoretical_files):
            raise ValueError("Mismatch in the number of files between measured and theoretical folders.")

        # Ensure output folder for results exists
        comparison_folder = os.path.join(method_folder_measured, "3d_comparison")
        if not os.path.exists(comparison_folder):
            os.makedirs(comparison_folder)

        # Process each pair of 3D structure files
        for measured_file, measured_array, theoretical_file, theoretical_array in zip(measured_files, measured_arrays, theoretical_files, theoretical_arrays):
            print(f"Processing file pair: Measured={measured_file}, Theoretical={theoretical_file}")

            # Check shape compatibility
            if measured_array.shape != theoretical_array.shape:
                raise ValueError(f"Shape mismatch between {measured_file} (shape {measured_array.shape}) "
                                 f"and {theoretical_file} (shape {theoretical_array.shape}).")

            # Scale the theoretical data
            theoretical_array = theoretical_array / scale_factor

            # Compute differences
            differences = calculate_differences(theoretical_array, measured_array)
            if abs_values:
                differences = np.abs(differences)

            # Save differences
            diff_filename = f"diff_{measured_file}"
            diff_filepath = os.path.join(comparison_folder, diff_filename)
            np.save(diff_filepath, differences)
            print(f"Differences saved to: {diff_filepath}")

            # Plot differences
            plot_filename = f"plot_{os.path.splitext(diff_filename)[0]}.png"
            plot_differences_as_bar_chart(
                differences,
                output_path=comparison_folder,
                plot_type=plot_type,
                abs_values=abs_values,
                output_filename=plot_filename
            )
            print(f"Plot created: {plot_filename}")

            # Compute and log mean differences
            mean_diff = np.mean(differences, axis=0)
            print(f"Mean differences for {measured_file}: {mean_diff}")

        print(f"Finished processing method: {method}")

    print("3D structure comparison completed for all selected methods.")

#compare_3d_structures_between_folders(input_folder)



def process_and_plot_comparison_results(input_folder, methods=None, plot_type="norm", x_values=None):
    """
    Process comparison results for multiple methods, compute norms, and plot mean values with standard deviations.
    
    Parameters:
    - input_folder (str): Base folder containing the results.
    - methods (list or None): List of methods to process. If None, all methods in the dictionary will be processed.
    - plot_type (str): Type of plot to create, defaults to "norm".
    - x_values (np.ndarray or None): X-values for the plot.
    
    Returns:
    - dict: Dictionary containing means and standard deviations for each method.
    """
    import os

    # Dictionary of methods and their shorthand folder names
    methods_dict = {
        #"center_of_mass": "com",
        "gauss_fit": "gf",
        "skewed_gauss_fit": "sgf",
        "non_linear_center_of_mass": "nlcom",
        "center_of_mass_with_threshold": "comwt"
    }

    # If methods are not specified, process all methods
    if methods is None:
        methods_to_process = methods_dict.keys()
    elif isinstance(methods, list):
        methods_to_process = methods
    else:
        raise ValueError("Invalid 'methods' parameter. Must be None or a list of valid method names.")

    # Initialize a dictionary to store results for all methods
    results = {}

    for method in methods_to_process:
        print(f"Processing method: {method}")

        # Construct the comparison results folder path
        comparison_folder = os.path.join(
            construct_output_path(input_folder),
            method,
            "3d_structure",
            "3d_comparison"
        )

        if not os.path.exists(comparison_folder):
            print(f"Warning: Comparison folder not found for method '{method}'. Skipping...")
            continue

        # Load all .npy files from the folder
        comparison_files, comparison_arrays = load_all_npy_files(comparison_folder)

        # Initialize lists to store means and standard deviations for norms
        means = []
        stds = []

        for array in comparison_arrays:
            # Ensure the array is a (n, 3) structure
            if array.shape[1] != 3:
                raise ValueError(f"Array shape mismatch: expected (n, 3), got {array.shape} in file {comparison_files}")
        
            # Compute the Euclidean norm for each row
            norms = np.linalg.norm(array, axis=1)
        
            # Filter out values greater than 0.2
            filtered_norms = norms[norms <= 1]
        
            if len(filtered_norms) == 0:
                raise ValueError(f"All values were filtered out in file {comparison_files}")
        
            # Calculate mean and standard deviation of the filtered norms
            means.append(np.mean(filtered_norms))
            stds.append(np.std(filtered_norms))

        # Store the results for the current method
        results[method] = {
            "means": np.array(means),
            "stds": np.array(stds)
        }

    # Plot results for all methods
    plot_all_methods(results=results, x_values=x_values, input_folder=input_folder, save_plot=True)

    print("Processing and plotting completed for all methods.")
    return results


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

    # Loop through the results dictionary and plot each method's data
    for method, data in results.items():
        means = data["means"]
        stds = data["stds"]

        # If x_values are not provided, create them from the length of means
        if x_values is None:
            x_values = np.arange(len(means))

        # Plot with error bars
        plt.errorbar(
            x=x_values,
            y=means,
            yerr=stds,
            label=method,  # Use method name as label
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


#x_values = np.array([10, 20, 30, 40, 50])
x_values = np.array([5, 25, 50/3, 50/4, 10, 50/6, 50/8])

process_and_plot_comparison_results(input_folder, plot_type="norm", x_values=x_values)

