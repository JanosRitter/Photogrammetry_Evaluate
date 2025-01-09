from file_io import *
from intensity_analysis import *
#from intensity_analysis import compute_center_of_mass_with_uncertainty, fit_gaussian_3d, fit_skewed_gaussian_3d, non_linear_center_of_mass, center_of_mass_with_threshold
from peak_find import find_peaks, combined_filter
from data_compare import calculate_differences

input_folder = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\spot_scale_1\spots_with_backgroundnoise"

filenames = ("Noise_projection_cam1_scale_4_bgn_50.npy",
             "Noise_projection_cam1_scale_8.npy")

def detect_and_verify_peaks_batch(input_folder, filenames=None, factor=15, threshold=None):
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
        
        if peaks.shape[0] != 257:
            peaks = combined_filter(peaks)
            print(f"Number of peaks after filtering: {peaks.shape[0]}")

        # Construct output path for the peaks
        peaks_output_path = os.path.join(output_folder, f"peaks_{os.path.splitext(filename)[0]}.npy")
        np.save(peaks_output_path, peaks)
        print(f"Saved peaks to: {peaks_output_path}")

        # Construct plot filename and path
        plot_filename = f"peaks_{os.path.splitext(filename)[0]}.png"

        # Plot the array with peaks
        create_image_plot(data=array,peaks=peaks,output_path=output_folder, output_filename=plot_filename)
        print(f"Created plot: {plot_filename}")


#detect_and_verify_peaks_batch(input_folder, factor=8,threshold=None, filenames=None)

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
        "skewed_gauss_fit": ("sgf", lambda data: fit_skewed_gaussian_3d(data)),
        "non_linear_center_of_mass": ("nlcom", lambda data: non_linear_center_of_mass(data)),
        "center_of_mass_with_threshold": ("comwt", lambda data: center_of_mass_with_threshold(data))
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

#calc_lpc_batch(input_folder, methode="center_of_mass")
#calc_lpc_batch(input_folder, methode="non_linear_center_of_mass")     
#calc_lpc_batch(input_folder, methode="center_of_mass_with_threshold")
#calc_lpc_batch(input_folder, methode="gauss_fit")
#calc_lpc_batch(input_folder, methode="skewed_gauss_fit")
        
import os
import numpy as np
import matplotlib.pyplot as plt

def combined_plot_all_methods(input_folder):
    """
    Combines contour plots of input intensity data with LPC coordinates 
    from multiple methods and saves the results.

    Parameters:
    - input_folder (str): Path to the folder containing `.npy` input files.

    Returns:
    - None
    """
    # Festes Dictionary mit den Methoden und Kürzeln
    methods_dict = {
        "center_of_mass": "com",
        "gauss_fit": "gf",
        "skewed_gauss_fit": "sgf",
        "non_linear_center_of_mass": "nlcom",
        "center_of_mass_with_threshold": "comwt"
    }

    # Hilfsfunktion: Lade alle `.npy`-Dateien aus einem Ordner
    def load_all_npy_from_folder(folder):
        npy_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])
        arrays = [np.load(f) for f in npy_files]
        return npy_files, arrays

    # Lade Input-Intensitätsdateien
    input_files, input_arrays = load_all_npy_from_folder(input_folder)

    # Konstruktion des Output-Ordners
    output_folder = construct_output_path(input_folder)
    combined_folder = os.path.join(output_folder, "combined_plots")
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    # Lade alle `.npy`-Dateien aus den Methodenordnern
    method_data = {}
    for method_name, method_abbr in methods_dict.items():
        method_folder = os.path.join(output_folder, method_name)
        if os.path.exists(method_folder):
            method_files, method_arrays = load_all_npy_from_folder(method_folder)
            method_data[method_name] = method_arrays
            print(f"Loaded {len(method_files)} files from method '{method_name}'")
        else:
            print(f"Method folder '{method_name}' not found. Skipping.")

    # Erstelle kombinierte Plots
    for idx, (input_file, input_array) in enumerate(zip(input_files, input_arrays)):
        print(f"\nCreating combined plot for: {input_file}")
        base_filename = os.path.splitext(os.path.basename(input_file))[0]

        # Initialize the figure
        plt.figure(figsize=(10, 8))

        # Plot the intensity data as contour
        plt.imshow(input_array, cmap='viridis', interpolation='nearest')
        plt.colorbar(label="Intensity")
        plt.title(f"Combined Plot for all methods")
        plt.gca().invert_yaxis()
        plt.xlim(2145, 2165)
        plt.ylim(1570, 1595)

        # Add LPC coordinates from all available methods
        for method_name, method_arrays in method_data.items():
            if idx < len(method_arrays):  # Verhindere Index-Fehler
                lpc_coordinates = method_arrays[idx]
                plt.scatter(lpc_coordinates[:, 0], lpc_coordinates[:, 1], label=method_name, s=10)
            else:
                print(f"Warning: Missing data for method '{method_name}' at index {idx}")

        # Add legend and save the plot
        plt.legend()
        combined_plot_path = os.path.join(combined_folder, f"{base_filename}_combined.png")
        plt.savefig(combined_plot_path, dpi=300)
        plt.close()
        print(f"Combined plot saved to: {combined_plot_path}")
        
#combined_plot_all_methods(input_folder)
        
def combined_plot_all_methods_with_slices(input_folder):
    """
    Combines sliced brightness data with LPC coordinates from multiple methods,
    sums up the slices, and creates combined plots.

    Parameters:
    - input_folder (str): Path to the folder containing `.npy` input files.

    Returns:
    - None
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Festes Dictionary mit den Methoden und Kürzeln
    methods_dict = {
        "center_of_mass": "com",
        "gauss_fit": "gf",
        "skewed_gauss_fit": "sgf",
        "non_linear_center_of_mass": "nlcom",
        "center_of_mass_with_threshold": "comwt"
    }

    # Hilfsfunktion: Lade alle `.npy`-Dateien aus einem Ordner
    def load_all_npy_from_folder(folder):
        npy_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])
        arrays = [np.load(f) for f in npy_files]
        return npy_files, arrays

    # Lade Input-Intensitätsdateien
    input_files, input_arrays = load_all_npy_from_folder(input_folder)

    # Konstruktion des Output-Ordners
    output_folder = construct_output_path(input_folder)
    combined_folder = os.path.join(output_folder, "combined_plots")
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    # Lade alle `.npy`-Dateien aus den Methodenordnern
    method_data = {}
    for method_name, method_abbr in methods_dict.items():
        method_folder = os.path.join(output_folder, method_name)
        if os.path.exists(method_folder):
            method_files, method_arrays = load_all_npy_from_folder(method_folder)
            method_data[method_name] = method_arrays
            print(f"Loaded {len(method_files)} files from method '{method_name}'")
        else:
            print(f"Method folder '{method_name}' not found. Skipping.")

    # Erstelle kombinierte Plots
    for idx, (input_file, input_array) in enumerate(zip(input_files, input_arrays)):
        print(f"\nProcessing input file: {input_file}")
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
        for method_name, method_arrays in method_data.items():
            if idx >= len(method_arrays):
                print(f"Warning: Missing data for method '{method_name}' at index {idx}")
                continue
    
            # Peaks für die aktuelle Methode und Datei
            peaks = method_arrays[idx]
            
            # Koordinaten runden
            rounded_peaks = np.round(peaks).astype(int)
    
            # Helligkeitsscheiben erstellen
            slices = brightness_subarray_creator(input_array, rounded_peaks)
    
            # Summiere alle Scheiben, um ein 2D-Array zu erstellen
            # Summiere alle Scheiben, um ein 2D-Array zu erstellen
            def get_zoomed_slice(array, target_size):
                h, w = array.shape
                if h < target_size or w < target_size:
                    raise ValueError(f"Array shape {array.shape} is smaller than the target zoom size {target_size}")
                
                center_x, center_y = w // 2, h // 2  # Mittelpunkt des Arrays
                half_size = target_size // 2
                
                # Berechnung der Grenzen
                x_start, x_end = center_x - half_size, center_x + half_size
                y_start, y_end = center_y - half_size, center_y + half_size
                
                return array[y_start:y_end, x_start:x_end]
            
            # Summiere alle Scheiben, um ein 2D-Array zu erstellen
            summed_slices = slices.sum(axis=0)
            
            # Sicherstellen, dass das Array numerisch ist
            summed_slices = np.nan_to_num(summed_slices, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
            
            # Dynamisch auf die Mitte mit 50x50 zoomen (oder gewünschter Größe)
            zoomed_slices = get_zoomed_slice(summed_slices, target_size=30)
            
            # Debugging: Ausgabe der Shape und Werte
            print(f"Zoomed slices shape: {zoomed_slices.shape}")
            print(f"Zoomed slices dtype: {zoomed_slices.dtype}, min: {np.min(zoomed_slices)}, max: {np.max(zoomed_slices)}")
            
            # Plotten
            plt.figure(figsize=(8, 6))
            plt.imshow(zoomed_slices, cmap="viridis", interpolation="nearest")
            plt.colorbar(label="Summed Intensity")
            #plt.title(f"Zoomed Summed Intensity for {base_filename} - {method_name}")
            plt.gca().invert_yaxis()
    
            # Plot speichern
            plot_filename = f"{base_filename}_{method_name}_summed.png"
            plot_path = os.path.join(combined_folder, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved plot for {method_name}: {plot_path}")

#combined_plot_all_methods_with_slices(input_folder)

filenames_1 = ("lpc_Noise_projection_cam1_scale_2_nlcom.npy",
               "lpc_Noise_projection_cam1_scale_3_nlcom.npy",
               "lpc_Noise_projection_cam1_scale_4_nlcom.npy",
               "lpc_Noise_projection_cam1_scale_5_nlcom.npy",
               "lpc_Noise_projection_cam1_scale_6_nlcom.npy",
               "lpc_Noise_projection_cam1_scale_8_nlcom.npy",
               "lpc_Noise_projection_cam1_scale_10_nlcom.npy")

filenames_2 = ("lpc_Noise_projection_cam1_scale_2_comwt.npy",
               "lpc_Noise_projection_cam1_scale_3_comwt.npy",
               "lpc_Noise_projection_cam1_scale_4_comwt.npy",
               "lpc_Noise_projection_cam1_scale_5_comwt.npy",
               "lpc_Noise_projection_cam1_scale_6_comwt.npy",
               "lpc_Noise_projection_cam1_scale_8_comwt.npy",
               "lpc_Noise_projection_cam1_scale_10_comwt.npy")

filenames_3 = ("lpc_Noise_projection_cam1_scale_2_gf.npy",
               "lpc_Noise_projection_cam1_scale_3_gf.npy",
               "lpc_Noise_projection_cam1_scale_4_gf.npy",
               "lpc_Noise_projection_cam1_scale_5_gf.npy",
               "lpc_Noise_projection_cam1_scale_6_gf.npy",
               "lpc_Noise_projection_cam1_scale_8_gf.npy",
               "lpc_Noise_projection_cam1_scale_10_gf.npy")

filenames_4 = ("lpc_Noise_projection_cam1_scale_2_com.npy",
               "lpc_Noise_projection_cam1_scale_3_com.npy",
               "lpc_Noise_projection_cam1_scale_4_com.npy",
               "lpc_Noise_projection_cam1_scale_5_com.npy",
               "lpc_Noise_projection_cam1_scale_6_com.npy",
               "lpc_Noise_projection_cam1_scale_8_com.npy",
               "lpc_Noise_projection_cam1_scale_10_com.npy")

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
    theoretical_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\simulated_data\simulated_data"
    theoretical_filename = "projection_rho_0_phi_0_cam1.npy"

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

filenames_7 = ("lpc_tcam_11_image_1_comwt.npy")
filenames_8 = ("lpc_tcam_11_image_1_nlcom.npy")
filenames_9 = ("lpc_tcam_11_image_1_com.npy")
filenames_10 = ("lpc_tcam_11_image_1_gf.npy")
filenames_11 = ("lpc_tcam_11_image_1_sgf.npy")

#compare_lpc_to_theoretical(input_folder, methode="center_of_mass_with_threshold", filenames=filenames_2, plot_type="norm")
#compare_lpc_to_theoretical(input_folder, methode="non_linear_center_of_mass", filenames=filenames_1, plot_type="norm")
#compare_lpc_to_theoretical(input_folder, methode="center_of_mass", filenames=filenames_4, plot_type="norm")
#compare_lpc_to_theoretical(input_folder, methode="gauss_fit", filenames=filenames_3, plot_type="norm")
#compare_lpc_to_theoretical(input_folder, methode="skewed_gauss_fit", filenames=filenames_5, plot_type="norm")
#compare_lpc_to_theoretical(input_folder, methode="skewed_gauss_fit", filenames=filenames_6, plot_type="norm")

def compare_folders_lpc(input_folder, theoretical_folder=None, scale_factor=5, methode="center_of_mass", filenames=None, abs_values=True, plot_type="norm"):
    """
    Vergleicht LPC-Daten aus einem Messordner mit theoretischen Werten aus einem zweiten Ordner.

    Parameters:
    - input_folder (str): Pfad zum Ordner mit Messdaten.
    - theoretical_folder (str): Pfad zum Ordner mit theoretischen Daten (falls None, ein Standardpfad wird verwendet).
    - scale_factor (int): Skalierungsfaktor, um theoretische Daten anzupassen.
    - methode (str): Methode zur LPC-Berechnung. Optionen: "center_of_mass", "gauss_fit", "skewed_gauss_fit".
    - filenames (list or None): Liste von LPC-Dateinamen. Wenn None, werden alle `.npy`-Dateien verarbeitet.
    - abs_values (bool): Ob absolute Differenzen berechnet werden sollen (Standard: False).
    - plot_type (str): Art der Plots, die erzeugt werden sollen. Optionen: "x", "y", "z", "norm", "all" (Standard: "all").

    Returns:
    - None
    """
    import os

    # Hard-coded path to theoretical data
    if theoretical_folder is None:
        theoretical_folder = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\spot_scale_1\diff_dis_10_add_noise\center_of_mass"

    # Construct the method folder path
    output_folder = construct_output_path(input_folder)
    method_folder = os.path.join(output_folder, methode)

    if not os.path.exists(method_folder):
        raise ValueError(f"Der Pfad für die Methode '{methode}' existiert nicht: {method_folder}")

    # Load LPC files and theoretical files
    lpc_files, lpc_arrays = load_all_npy_files(method_folder, filenames=filenames)
    theoretical_files, theoretical_arrays = load_all_npy_files(theoretical_folder, filenames=filenames)

    # Ensure the number of files matches
    if len(lpc_files) != len(theoretical_files):
        raise ValueError(f"Anzahl der Dateien stimmt nicht überein: {len(lpc_files)} Messdaten vs. {len(theoretical_files)} theoretische Daten.")

    # Ensure output folder for results exists
    result_folder = os.path.join(method_folder, "comparison_results")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Process each file pair
    for lpc_file, lpc_array, theoretical_file, theoretical_array in zip(lpc_files, lpc_arrays, theoretical_files, theoretical_arrays):
        print(f"Processing LPC file: {lpc_file} with theoretical file: {theoretical_file}")

        # Check shape compatibility
        if lpc_array.shape != theoretical_array.shape:
            raise ValueError(f"Shape mismatch: {lpc_file} (shape {lpc_array.shape}) "
                             f"and {theoretical_file} (shape {theoretical_array.shape}).")

        # Scale theoretical data
        scaled_theoretical = theoretical_array / scale_factor

        # Compute differences
        differences = calculate_differences(scaled_theoretical, lpc_array)
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

#compare_folders_lpc(input_folder, methode="skewed_gauss_fit")

import matplotlib.pyplot as plt
import numpy as np

def plot_means_with_error(results, input_folder, x_values=None, save_plot=True):
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
            fmt='o', capsize=9
        )

    # Plot customization
    #plt.title(r"Mean deviation from theretical values over laser Spo")
    plt.xlabel(r"Measuring distance in m")
    plt.ylabel(r"Mean deviation from input values in pixels")
    plt.legend(title="Methods")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()  # Ensures the plot fits nicely

    # Save the plot if requested
    if save_plot:
        # Generate the output path for saving the plot
        output_path = os.path.join(construct_output_path(input_folder), 'plots')
        os.makedirs(output_path, exist_ok=True)  # Create directory if it doesn't exist
        plot_filename = os.path.join(output_path, "comparison_plot.png")
        plt.savefig(plot_filename, dpi=300)
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
        "skewed_gauss_fit": ("sgf", lambda data: fit_skewed_gaussian_3d(data)),
        "non_linear_center_of_mass": ("nlcom", lambda data: non_linear_center_of_mass(data)),
        "center_of_mass_with_threshold": ("comwt", lambda data: center_of_mass_with_threshold(data))
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


x_values = np.array([5, 25, 50/3, 50/4, 10, 50/6, 50/8])
#x_values = np.array([10, 20, 30, 40, 50, 10, 20, 30, 40, 50])
#x_values = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

process_comparison_results_with_plot(input_folder, x_values=x_values)
