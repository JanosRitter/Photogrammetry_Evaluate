from file_io import load_brightness_arrays, create_c_plot_with_points, create_image_plot, plot_3d_points, create_contour_plot, load_npy_file
from peak_find import find_peaks, brightness_subarray_creator, lpc_calc, peak_filter
from intensity_analysis import fit_gaussian_3d
from lpc_indexing import find_outlier_point, analyze_coordinates, rotate_coordinates
from calc_3d import triangulate_3d
import os
from data_compare import calculate_differences

import numpy as np

np.set_printoptions(threshold=np.inf)


image_name_1 = 'tcam_11_image_1.bmp'
image_name_2 = 'tcam_12_image_1.bmp'

folder_name = "example_1"

camera_stats = {
    'a': 0.2,                  # Distance from the cameras to the origin plane along the x-axis
    'f': 0.04,                 # Focal length of the cameras in meters
    'pixel_size': 2.74e-6,     # Pixel size in meters
    'resolution': (1600, 1600) # Resolution of the cameras in pixels (width, height)
}


file_path_1 = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\simulated_data_with_noise\simulated_data_with_noise\added_noise"
file_name_1 = r"noisy_projection_rho_0_phi_0_cam1.npy"

file_path_2 = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\simulated_data"
file_name_2 = r"projection_rho_0_phi_0_cam1.npy"

ba_1, ba_2 = load_brightness_arrays(folder_name)






























