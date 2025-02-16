from file_io import *
from main_compare import detect_and_verify_peaks_batch, calc_lpc_batch

folder_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\Spotsize"


#detect_and_verify_peaks_batch(folder_path)

#calc_lpc_batch(folder_path, methode="circle_fit")

import numpy as np
import os

def filter_npy_by_range_and_points(folder_path, file_name, xlim=None, ylim=None, points_to_remove=None):
    """
    Loads a .npy file containing (n,2) coordinate data and removes:
    1. Points outside the given x and y limits.
    2. The closest points to user-specified coordinates.

    The filtered data overwrites the original .npy file.

    Parameters:
    - folder_path (str): Directory where the .npy file is located.
    - file_name (str): Name of the .npy file.
    - xlim (tuple, optional): (min_x, max_x) range for filtering.
    - ylim (tuple, optional): (min_y, max_y) range for filtering.
    - points_to_remove (np.ndarray, optional): Array of shape (m,2) with specific points to remove.

    Returns:
    - None: Saves the filtered array back to the original file location.
    """
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path)
    
    if data.shape[1] != 2:
        raise ValueError("Expected input array of shape (n,2).")

    initial_count = data.shape[0]

    x, y = data[:, 0], data[:, 1]
    
    # Apply x and y limits
    if xlim:
        mask_x = (x >= xlim[0]) & (x <= xlim[1])
    else:
        mask_x = np.ones_like(x, dtype=bool)

    if ylim:
        mask_y = (y >= ylim[0]) & (y <= ylim[1])
    else:
        mask_y = np.ones_like(y, dtype=bool)

    mask = mask_x & mask_y
    filtered_data = data[mask]

    removed_outside_bounds = initial_count - filtered_data.shape[0]

    # Remove nearest points to specified locations
    removed_points = []
    if points_to_remove is not None:
        remaining_indices = np.ones(filtered_data.shape[0], dtype=bool)
        
        for px, py in points_to_remove:
            distances = np.sqrt((filtered_data[:, 0] - px) ** 2 + (filtered_data[:, 1] - py) ** 2)
            nearest_idx = np.argmin(distances)
            removed_points.append(filtered_data[nearest_idx])
            remaining_indices[nearest_idx] = False
        
        filtered_data = filtered_data[remaining_indices]

    removed_specific_points = len(removed_points)
    total_removed = removed_outside_bounds + removed_specific_points

    # Save the filtered data, overwriting the original file
    np.save(file_path, filtered_data)

    # Print statistics
    print(f"Original number of points: {initial_count}")
    print(f"Removed out-of-bounds points: {removed_outside_bounds}")
    if removed_specific_points > 0:
        print(f"Removed specific points: {removed_specific_points}")
        for point in removed_points:
            print(f"  - Removed closest point to ({point[0]:.2f}, {point[1]:.2f})")
    print(f"Final number of points: {filtered_data.shape[0]}")

#path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\Spotsize" 
#file_name = r"peaks_tcam-capture-13320612-aravis-GRAY8_2048x1536_11m.npy"  

#filter_npy_by_range_and_points(path, file_name, xlim=(250,1600), ylim=None, points_to_remove=np.array([[500, 520], [1150, 450]]))
    
def calc_lpc_circle_fitting(input_folder, thresholds, filenames=None):
    """
    Batch calculation of LPC coordinates using Circle Fitting for `.npy` peak files 
    with varying threshold values. Results are stored in separate subfolders per threshold.

    Parameters:
    - input_folder (str): Path to the folder containing `.npy` files.
    - thresholds (list or np.ndarray): List of threshold values to apply.
    - filenames (list or None): List of filenames to process.
      If None, all `.npy` files in the folder are processed.

    Returns:
    - None: Saves (n,3) `.npy` files containing x, y, and radius for each threshold.
    """
    if not isinstance(thresholds, (list, np.ndarray)):
        raise ValueError("Thresholds must be a list or numpy array.")

    # Lade alle `.npy` Dateien
    npy_files, arrays = load_all_npy_files(input_folder, filenames=filenames)
    output_folder = construct_output_path(input_folder)

    # Hauptordner für circle fitting Ergebnisse
    base_folder = os.path.join(output_folder, "circle_fitting")
    os.makedirs(base_folder, exist_ok=True)

    peak_files, peaks_list = load_all_npy_files(output_folder, filenames=filenames)

    for i, (array_filename, array) in enumerate(zip(npy_files, arrays)):
        print(f"\nProcessing brightness array: {array_filename}")

        peak_filename = peak_files[i]
        peaks = peaks_list[i]
        print(f"Using peaks from file: {peak_filename}")

        bsc = brightness_subarray_creator(array, peaks)

        for threshold in thresholds:
            print(f"Applying circle fitting with threshold: {threshold}")

            # Erstelle einen separaten Ordner für diesen Threshold
            threshold_folder = os.path.join(base_folder, f"threshold_{threshold:.2f}")
            os.makedirs(threshold_folder, exist_ok=True)

            mean_values, radii = circle_fitting_with_threshold(bsc, threshold)
            lpc_coordinates = lpc_calc(mean_values, peaks)
            coords_and_radius = np.hstack((lpc_coordinates, radii[:, np.newaxis]))  # Shape (n,3)

            # Speichern der Ergebnisse
            lpc_output_filename = f"lpc_{os.path.splitext(array_filename)[0]}_cf.npy"
            lpc_output_path = os.path.join(threshold_folder, lpc_output_filename)
            np.save(lpc_output_path, coords_and_radius)
            print(f"LPC results saved to: {lpc_output_path}")

            # Speichern des Plots
            plot_filename = f"lpc_{os.path.splitext(array_filename)[0]}_cf.png"
            #create_image_plot(array, peaks, lpc_coordinates, output_path=threshold_folder, output_filename=plot_filename)
            print(f"Plot created for LPC coordinates: {plot_filename}")


#calc_lpc_circle_fitting(folder_path, thresholds= [60,80,100,120,140,160,180,200])
            
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_radius_vs_threshold(input_folder, x_values):
    """
    Plots the mean radius for each file across different threshold values in two separate plots:
    - One in pixels
    - One converted to millimeters using the optical system parameters.

    Parameters:
    - input_folder (str): Path to the folder containing `.npy` files (before processing).
    - x_values (list or np.ndarray): X-axis values (distances in meters).

    Returns:
    - None: Saves the plots in the "circle_fitting" folder.
    """
    if not isinstance(x_values, (list, np.ndarray)):
        raise ValueError("x_values must be a list or numpy array.")

    # Korrekte Pfadkonstruktion wie in den vorherigen Funktionen
    output_folder = construct_output_path(input_folder)
    base_folder = os.path.join(output_folder, "circle_fitting")

    # Alle Threshold-Ordner sammeln
    threshold_folders = sorted([f for f in os.listdir(base_folder) if f.startswith("threshold_")])

    if not threshold_folders:
        raise ValueError("No threshold folders found in the circle_fitting directory.")

    # Optische Parameter
    pixel_size = 2.74e-6  # Pixelgröße auf dem Sensor in m
    focal_length = 0.05    # Brennweite des Systems in m

    # Erster Plot: Radius in Pixeln
    plt.figure(figsize=(10, 6))
    for folder in threshold_folders:
        threshold_value = float(folder.split("_")[1])  # Extrahiere den Threshold-Wert
        folder_path = os.path.join(base_folder, folder)

        mean_radii = []
        filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

        for filename in filenames:
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path)  # (n, 3) Array mit x, y, radius
            mean_radius = np.nanmean(data[:, 2])  # Mittelwert des Radius berechnen
            mean_radii.append(mean_radius)

        if len(mean_radii) != len(x_values):
            raise ValueError(f"Number of x_values ({len(x_values)}) does not match number of files ({len(mean_radii)}) in {folder}")

        plt.plot(x_values, mean_radii, marker="o", linestyle="-", label=f"Threshold {threshold_value:.2f}")

    plt.xlabel("Distance (m)")
    plt.ylabel("Mean Radius (Pixels)")
    plt.title("Mean Radius vs. Distance (Pixels)")
    plt.legend()
    plt.grid(True)

    # Plot speichern eine Ebene höher im "circle_fitting"-Ordner
    pixel_plot_path = os.path.join(base_folder, "radius_vs_threshold_pixels.png")
    plt.savefig(pixel_plot_path, dpi=300)
    plt.show()
    
    print(f"Pixel-based plot saved to: {pixel_plot_path}")

    # Zweiter Plot: Radius in Millimetern
    plt.figure(figsize=(10, 6))
    for folder in threshold_folders:
        threshold_value = float(folder.split("_")[1])
        folder_path = os.path.join(base_folder, folder)

        mean_radii_mm = []
        filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

        for i, filename in enumerate(filenames):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path)  # (n, 3) Array mit x, y, radius
            mean_radius_pixels = np.nanmean(data[:, 2])  # Mittelwert des Radius berechnen

            # Umrechnung in mm: Skalierungsfaktor pro Abstand
            scale_factor = (x_values[i] / focal_length) * pixel_size * 1000  # in mm
            mean_radius_mm = mean_radius_pixels * scale_factor
            mean_radii_mm.append(mean_radius_mm)

        plt.plot(x_values, mean_radii_mm, marker="o", linestyle="-", label=f"Threshold {threshold_value:.2f}")

    plt.xlabel("Distance (m)")
    plt.ylabel("Mean Radius (mm)")
    plt.title("Mean Radius vs. Distance (mm)")
    plt.legend()
    plt.grid(True)

    mm_plot_path = os.path.join(base_folder, "radius_vs_threshold_mm.png")
    plt.savefig(mm_plot_path, dpi=300)
    plt.show()
    
    print(f"Millimeter-based plot saved to: {mm_plot_path}")

    
    
plot_radius_vs_threshold(folder_path, x_values=[5, 6, 7, 8, 9, 10, 11])