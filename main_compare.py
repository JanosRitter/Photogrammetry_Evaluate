import file_io
import intensity_analysis
from intensity_analysis import compute_center_of_mass_with_uncertainty, fit_gaussian_3d, fit_skewed_gaussian_3d
from peak_find import find_peaks, peak_filter
from data_compare import calculate_differences

input_folder = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\spot_scale_1\averaged_noise_spots"

#filenames = ("Noise_projection_cam1_scale_1.npy","Noise_projection_cam2_scale_1.npy")

def detect_and_verify_peaks_batch(input_folder, filenames=None, factor=6, threshold=None):
    """
    Detects and processes peaks from `.npy` files in a batch.

    Parameters:
    - input_folder (str): Path to the folder containing `.npy` files.
    - filenames (list or None): List of specific `.npy` filenames to process. If None, all `.npy` files are processed.
    - factor (int): Block averaging factor for peak detection.
    - threshold (int or None): Threshold for peak detection. If None, it will be calculated.

    Returns:
    - None
    """
    import os

    # Load `.npy` files (all or filtered by filenames)
    npy_files, arrays = load_all_npy_files(input_folder, filenames=filenames)

    # Create output folder
    output_folder = construct_output_path(input_folder)

    for i, (filename, array) in enumerate(zip(npy_files, arrays)):
        print(f"\nProcessing file: {filename}")

        # Detect peaks
        peaks = find_peaks(array, factor=factor, threshold=threshold)
        print(f"Number of peaks found in {filename}: {peaks.shape[0]}")

        # Construct output path for the peaks
        peaks_output_path = os.path.join(output_folder, f"peaks_{os.path.splitext(filename)[0]}.npy")
        np.save(peaks_output_path, peaks)
        print(f"Saved peaks to: {peaks_output_path}")

        # Construct plot filename and path
        plot_filename = f"peaks_{os.path.splitext(filename)[0]}.png"

        # Plot the array with peaks
        create_image_plot(data=array,peaks=peaks,output_path=output_folder, output_filename=plot_filename)
        print(f"Created plot: {plot_filename}")


#detect_and_verify_peaks_batch(input_folder, factor=8)

def calc_lpc_batch(input_folder, methode="center_of_mass", filenames=None):
    """
    Batch calculation of LPC coordinates for `.npy` peak files.

    Parameters:
    - input_folder (str): Path to the folder containing `.npy` files.
    - methode (str): Method to calculate LPC. Options are "center_of_mass", "gauss_fit", "skewed_gauss_fit".
    - filenames (list or None): List of filenames to process. If None, all `.npy` files are processed.

    Returns:
    - None
    """
    import os

    # Map method names to functions and abbreviations
    lpc_methods = {
        "center_of_mass": ("com", lambda data: compute_center_of_mass_with_uncertainty(data)),
        "gauss_fit": ("gf", lambda data: fit_gaussian_3d(data)),
        "skewed_gauss_fit": ("sgf", lambda data: fit_skewed_gaussian_3d(data))
    }

    if methode not in lpc_methods:
        raise ValueError(f"Unbekannte Methode '{methode}'. Verfügbare Methoden: {list(lpc_methods.keys())}")

    method_abbr, method_function = lpc_methods[methode]

    # Load `.npy` files (all or filtered by filenames)
    npy_files, arrays = load_all_npy_files(input_folder, filenames=filenames)
    output_folder = construct_output_path(input_folder)

    # Create a subfolder for the selected method
    method_folder = os.path.join(output_folder, methode)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    # Load peak files
    peak_files, peaks_list = load_all_npy_files(output_folder, filenames=filenames)

    for i, (array_filename, array) in enumerate(zip(npy_files, arrays)):
        print(f"\nProcessing brightness array: {array_filename}")

        # Corresponding peaks file
        peak_filename = peak_files[i]
        peaks = peaks_list[i]
        print(f"Using peaks from file: {peak_filename}")

        # Create subarrays for LPC calculation
        bsc = brightness_subarray_creator(array, peaks)

        # Calculate LPC coordinates using the selected method
        mean_values = method_function(bsc)[0]
        lpc_coordinates = lpc_calc(mean_values, peaks)

        # Save LPC coordinates as `.npy`
        lpc_output_filename = f"lpc_{os.path.splitext(array_filename)[0]}_{method_abbr}.npy"
        lpc_output_path = os.path.join(method_folder, lpc_output_filename)
        np.save(lpc_output_path, lpc_coordinates)
        print(f"LPC coordinates saved to: {lpc_output_path}")

        # Plot the brightness array with LPC coordinates
        plot_filename = f"lpc_{os.path.splitext(array_filename)[0]}_{method_abbr}.png"
        create_image_plot(array, peaks, lpc_coordinates, output_path=method_folder, output_filename=plot_filename)
        print(f"Plot created for LPC coordinates: {plot_filename}")
        
#calc_lpc_batch(input_folder, methode="skewed_gauss_fit")

filenames_1 = ("lpc_Noise_projection_cam1_scale_2_com.npy",
               "lpc_Noise_projection_cam1_scale_3_com.npy",
               "lpc_Noise_projection_cam1_scale_4_com.npy",
               "lpc_Noise_projection_cam1_scale_5_com.npy",
               "lpc_Noise_projection_cam1_scale_6_com.npy",
               "lpc_Noise_projection_cam1_scale_8_com.npy",
               "lpc_Noise_projection_cam1_scale_10_com.npy")

filenames_2 = ("lpc_Noise_projection_cam2_scale_2_com.npy",
               "lpc_Noise_projection_cam2_scale_3_com.npy",
               "lpc_Noise_projection_cam2_scale_4_com.npy",
               "lpc_Noise_projection_cam2_scale_5_com.npy",
               "lpc_Noise_projection_cam2_scale_6_com.npy",
               "lpc_Noise_projection_cam2_scale_8_com.npy",
               "lpc_Noise_projection_cam2_scale_10_com.npy")

filenames_3 = ("lpc_Noise_projection_cam1_scale_2_gf.npy",
               "lpc_Noise_projection_cam1_scale_3_gf.npy",
               "lpc_Noise_projection_cam1_scale_4_gf.npy",
               "lpc_Noise_projection_cam1_scale_5_gf.npy",
               "lpc_Noise_projection_cam1_scale_6_gf.npy",
               "lpc_Noise_projection_cam1_scale_8_gf.npy",
               "lpc_Noise_projection_cam1_scale_10_gf.npy")

filenames_4 = ("lpc_Noise_projection_cam2_scale_2_gf.npy",
               "lpc_Noise_projection_cam2_scale_3_gf.npy",
               "lpc_Noise_projection_cam2_scale_4_gf.npy",
               "lpc_Noise_projection_cam2_scale_5_gf.npy",
               "lpc_Noise_projection_cam2_scale_6_gf.npy",
               "lpc_Noise_projection_cam2_scale_8_gf.npy",
               "lpc_Noise_projection_cam2_scale_10_gf.npy")

filenames_5 = ("lpc_Noise_projection_cam1_scale_2_sgf.npy",
               "lpc_Noise_projection_cam1_scale_3_sgf.npy",
               "lpc_Noise_projection_cam1_scale_4_sgf.npy",
               "lpc_Noise_projection_cam1_scale_5_sgf.npy",
               "lpc_Noise_projection_cam1_scale_6_sgf.npy",
               "lpc_Noise_projection_cam1_scale_8_sgf.npy",
               "lpc_Noise_projection_cam1_scale_10_sgf.npy")

filenames_6 = ("lpc_Noise_projection_cam2_scale_2_sgf.npy",
               "lpc_Noise_projection_cam2_scale_3_sgf.npy",
               "lpc_Noise_projection_cam2_scale_4_sgf.npy",
               "lpc_Noise_projection_cam2_scale_5_sgf.npy",
               "lpc_Noise_projection_cam2_scale_6_sgf.npy",
               "lpc_Noise_projection_cam2_scale_8_sgf.npy",
               "lpc_Noise_projection_cam2_scale_10_sgf.npy")
        
def compare_lpc_to_theoretical(input_folder, methode="center_of_mass", filenames=None, abs_values=False, plot_type="all"):
    """
    Compares LPC data from a specific method to theoretical values for specific filenames.
    
    Parameters:
    - input_folder (str): Path to the folder containing input data.
    - methode (str): Method used to calculate LPC. Options: "center_of_mass", "gauss_fit", "skewed_gauss_fit".
    - filenames (list or None): List of LPC filenames to process. If None, all `.npy` files are processed.
    - abs_values (bool): Whether to compute absolute differences (default: False).
    - plot_type (str): Type of plots to generate. Options: "x", "y", "z", "norm", "all" (default: "all").
    
    Returns:
    - None
    """
    import os

    # Hard-coded path to theoretical data
    theoretical_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\spot_scale_1\averaged_noise_spots"
    theoretical_filename = "projection_cam2_scale_1.npy"

    # Construct the method folder path
    output_folder = construct_output_path(input_folder)
    method_folder = os.path.join(output_folder, methode)

    if not os.path.exists(method_folder):
        raise ValueError(f"Der Pfad für die Methode '{methode}' existiert nicht: {method_folder}")

    # Load LPC data
    lpc_files, lpc_arrays = load_all_npy_files(method_folder, filenames=filenames)
    

    # Load theoretical data
    theoretical_data = load_npy_file(theoretical_path, theoretical_filename)

    # Ensure output folder for results exists
    result_folder = os.path.join(method_folder, "comparison_results")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Process each LPC file
    for lpc_file, lpc_array in zip(lpc_files, lpc_arrays):
        print(f"Processing LPC file: {lpc_file}")

        # Check shape compatibility
        if lpc_array.shape != theoretical_data.shape:
            raise ValueError(f"Shape mismatch: {lpc_file} (shape {lpc_array.shape}) "
                             f"and theoretical data (shape {theoretical_data.shape}).")

        # Compute differences
        differences = calculate_differences(theoretical_data, lpc_array)
        if abs_values:
            differences = np.abs(differences)

        # Save differences
        diff_filename = f"diff_{lpc_file}"
        diff_filepath = os.path.join(result_folder, diff_filename)
        np.save(diff_filepath, differences)
        print(f"Differences saved to: {diff_filepath}")

        # Plot differences
        plot_filename = f"plot_{os.path.splitext(diff_filename)[0]}.png"
        plot_differences_as_bar_chart(
            differences,
            output_path=result_folder,
            plot_type=plot_type,
            abs_values=abs_values,
            output_filename=plot_filename
        )
        print(f"Plot created: {plot_filename}")

        # Compute and log mean differences
        mean_diff = np.mean(differences, axis=0)
        print(f"Mean differences for {lpc_file}: {mean_diff}")

    print(f"Comparison completed. Results saved in: {result_folder}")


#compare_lpc_to_theoretical(input_folder, methode="skewed_gauss_fit", filenames=filenames_6, plot_type="norm")



import matplotlib.pyplot as plt
import numpy as np

def plot_means_with_error(results, input_folder, x_values=None, title="Means with Standard Deviation", xlabel="X-Axis", ylabel="Mean Norm", save_plot=True):
    """
    Plots means with standard deviations as error bars for multiple methods and optionally saves the plot.
    
    Parameters:
    - results (dict): Dictionary containing means and standard deviations for each method.
                      Example: {"center_of_mass": {"means": [...], "stds": [...]}, ...}
    - input_folder (str): Base input folder used to construct the output path for saving the plot.
    - x_values (np.ndarray or None): Array of x-values for the plot. If None, a default range is used.
    - title (str): Title of the plot (default: "Means with Standard Deviation").
    - xlabel (str): Label for the x-axis (default: "X-Axis").
    - ylabel (str): Label for the y-axis (default: "Mean Norm").
    - save_plot (bool): Whether to save the plot as a PNG file. Default is False.
    
    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    
    print(f"Number of x_values: {len(x_values)}")
    print(f"Number of means per method: {len(next(iter(results.values()))['means'])}")

    for method, data in results.items():
        means = np.array(data["means"])
        stds = np.array(data["stds"])

        # Flatten means and stds if necessary (in case they are lists of arrays)
        if means.ndim > 1:
            means = np.concatenate(means)
        if stds.ndim > 1:
            stds = np.concatenate(stds)

        # Check or create x_values
        if x_values is None:
            x_values = np.arange(len(means))
        elif len(x_values) != len(means):
            raise ValueError("Length of x_values must match the number of means.")

        # Plot with error bars
        plt.errorbar(
            x=x_values,
            y=means,
            yerr=stds,
            label=method,
            fmt='o', capsize=5
        )

    # Plot customization
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Methods")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()  # Ensures the plot fits nicely

    # Save the plot if requested
    if save_plot:
        # Generate the output path for saving the plot
        output_path = os.path.join(construct_output_path(input_folder), 'plots')
        os.makedirs(output_path, exist_ok=True)  # Create directory if it doesn't exist
        plot_filename = os.path.join(output_path, "comparison_plot.png")
        plt.savefig(plot_filename)
        print(f"Plot saved to: {plot_filename}")
    
    # Display the plot
    plt.show()



def process_comparison_results(input_folder):
    """
    Processes comparison results for LPC data across methods and computes means and standard deviations of the norms.
    
    Parameters:
    - input_folder (str): Base input folder containing LPC method subfolders.

    Returns:
    - dict: Dictionary with means and standard deviations for each method.
    """
    # Fest definierte LPC-Methoden
    lpc_methods = {
        "center_of_mass": ("com", lambda data: compute_center_of_mass_with_uncertainty(data)),
        "gauss_fit": ("gf", lambda data: fit_gaussian_3d(data)),
        "skewed_gauss_fit": ("sgf", lambda data: fit_skewed_gaussian_3d(data))
    }

    # Speicherstruktur für die Ergebnisse
    results = {}

    for method in lpc_methods.keys():
        # Konstruktion des Pfads zum Ordner "comparison_results"
        comparison_folder = os.path.join(
            construct_output_path(input_folder),
            method,
            "comparison_results"
        )

        if not os.path.exists(comparison_folder):
            print(f"Comparison results folder not found for method '{method}': {comparison_folder}")
            continue

        # Lade alle `.npy` Dateien aus dem Ordner
        comparison_files, comparison_arrays = load_all_npy_files(comparison_folder)

        # Mittelwerte und Standardabweichungen berechnen
        means = []
        stds = []
        for array in comparison_arrays:
            # Berechne die Norm jeder Zeile (r_i = sqrt(x^2 + y^2))
            r_values = np.linalg.norm(array, axis=1)
            # Berechne den Mittelwert der Normen
            means.append(np.mean(r_values))
            # Berechne die Standardabweichung der Normen
            stds.append(np.std(r_values))

        # Ergebnisse für die Methode speichern
        results[method] = {
            "means": np.array(means),  # Mittelwert der Normen für das ganze Array
            "stds": np.array(stds)     # Standardabweichung der Normen
        }

        print(f"Processed method '{method}': {len(means)} datasets computed in {comparison_folder}.")

    return results


def process_comparison_results_with_plot(input_folder, x_values=None):
    """
    Processes comparison results and plots norms with error bars.
    
    Parameters:
    - input_folder (str): Base input folder containing LPC method subfolders.
    - lpc_methods (dict): Dictionary mapping method names to their abbreviations and functions.
    - x_values (np.ndarray or None): Array of x-values for the plot. If None, a placeholder will be used.
    
    Returns:
    - dict: Dictionary with norms and standard deviations for each method.
    """
    results = process_comparison_results(input_folder)

    # Prüfe und erstelle x_values
    if x_values is None:
        max_length = max(len(data["means"]) for data in results.values())
        x_values = np.empty((max_length, 1))
        x_values.fill(np.nan)  # Initialisiere mit NaN, damit der Benutzer sie anpassen kann

    # Plotten
    plot_means_with_error(results, x_values=x_values, input_folder=input_folder)
    return results


x_values = np.array([5, 25, 50/3, 50/4, 10, 50/6, 50/8, 5, 25, 50/3, 50/4, 10, 50/6, 50/8])

process_comparison_results_with_plot(input_folder, x_values=x_values)
