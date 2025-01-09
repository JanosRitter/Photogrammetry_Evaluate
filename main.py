import file_io
import intensity_analysis
from peak_find import find_peaks, combined_filter


inputpath = r"example_set_1\example_6"

def detect_and_verify_peaks(inputpath, factor=8, threshold=None):
    """
    Detects peaks in two brightness arrays, compares their shapes, and applies filtering if needed.
    Allows the user to force filtering even if the shapes match or mismatch.

    Parameters:
        - inputpath (str): Path to the input data containing the brightness arrays.
        - factor (int): Block averaging factor for peak detection.
        - threshold (int or None): Threshold for peak detection. If None, it will be calculated.

    Returns:
        None
    """
    brightness_array_1, brightness_array_2 = load_brightness_arrays(inputpath)

    peaks_1 = find_peaks(brightness_array_1, factor=factor, threshold=threshold)
    print("Number of peaks found in Array 1:", peaks_1.shape[0])
    
    peaks_2 = find_peaks(brightness_array_2, factor=factor, threshold=threshold)
    print("Number of peaks found in Array 2:", peaks_2.shape[0])

    if peaks_1.shape[0] == peaks_2.shape[0]:
        print("Number of peaks in both arrays match.")
    else:
        print("Mismatch in the number of peaks.")

    filter_prompt = input("Do you want to apply filtering? (yes/no): ").strip().lower()
    if filter_prompt == "yes":
        print("Applying filtering...")
        peaks_1 = combined_filter(peaks_1)
        peaks_2 = combined_filter(peaks_2)
        print("Filtering complete.")
        print("Number of peaks found in Array 1:", peaks_1.shape[0])
        print("Number of peaks found in Array 2:", peaks_2.shape[0])
    else:
        print("Skipping filtering as per user request.")

    if peaks_1.shape[0] == peaks_2.shape[0]:
        print("Final check: Number of peaks in both arrays match.")
    else:
        print("Final check: Mismatch still exists.")
        
    output_path = construct_output_path(inputpath)
    
    save_array_as_npy(peaks_1, output_path, "peaks_1.npy")
    save_array_as_npy(peaks_2, output_path, "peaks_2.npy")
    
    print("Peaks were saved.")
        
    #create_image_plot(brightness_array_1, peaks_1, input_path=inputpath, peaks_name="peaks_1")
    #create_image_plot(brightness_array_2, peaks_2, input_path=inputpath, peaks_name="peaks_2")
    
detect_and_verify_peaks(inputpath)
    
    
camera_stats = {
    'a': 0.2,                  # Distance from the cameras to the origin plane along the x-axis
    'f': 0.04,                 # Focal length of the cameras in meters
    'pixel_size': 2.74e-6,     # Pixel size in meters
    'resolution': (4096, 3000) # Resolution of the cameras in pixels (width, height)
}


def calc_lpc_and_3d(inputpath, methode="center_of_mass", camera_stats=None):
    brightness_array_1, brightness_array_2 = load_brightness_arrays(inputpath)
    
    peak_path = construct_output_path(inputpath)
    
    peaks_1 = load_npy_file(peak_path, "peaks_1.npy")
    peaks_2 = load_npy_file(peak_path, "peaks_2.npy")
    
    bsc_1 = brightness_subarray_creator(brightness_array_1, peaks_1)
    bsc_2 = brightness_subarray_creator(brightness_array_2, peaks_2)
    
    lpc_methods = {
            "gauss_fit": lambda data: fit_gaussian_3d(data),
            "center_of_mass": lambda data: compute_center_of_mass_with_uncertainty(data)
    }
    
    if methode not in lpc_methods:
        raise ValueError(f"Unbekannte Methode '{methode}'. Verfügbare Methoden: {list(methoden.keys())}")
    
    
    mean_1 = lpc_methods[methode](bsc_1)[0]
    mean_2 = lpc_methods[methode](bsc_2)[0]
    
    
    lpc_1 = lpc_calc(mean_1, peaks_1)
    lpc_2 = lpc_calc(mean_2, peaks_2)
    
    create_image_plot(brightness_array_1, peaks_1, lpc_1, inputpath, peaks_name="peaks_lpc_1")
    create_image_plot(brightness_array_2, peaks_2, lpc_2, inputpath, peaks_name="peaks_lpc_2")
    
    lpc_sortet_1 = analyze_coordinates(lpc_1)
    lpc_sortet_2 = analyze_coordinates(lpc_2)
    
    save_array_as_npy(lpc_sortet_1, peak_path, "lpc_1.npy")
    save_array_as_npy(lpc_sortet_2, peak_path, "lpc_2.npy")
    
    points_3d = triangulate_3d(lpc_sortet_1, lpc_sortet_2, camera_stats)
    save_array_as_npy(points_3d, peak_path, "points_3d.npy")
    plot_3d_points(points_3d, path=peak_path)
    
    

    
    
#calc_lpc_and_3d(inputpath, camera_stats=camera_stats)

