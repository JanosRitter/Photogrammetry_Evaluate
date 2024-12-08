import file_io
from peak_find import find_peaks_7, lpc_calc, peak_filter
import intensity_analysis
from lpc_indexing import find_outlier_point, analyze_coordinates, rotate_coordinates
from calc_3d import triangulate_3d
from data_compare import calculate_differences
import os
#from data_compare import calculate_differences

import numpy as np

np.set_printoptions(threshold=np.inf)


image_name_1 = 'tcam_11_image_1.bmp'
image_name_2 = 'tcam_12_image_1.bmp'

folder_name = "example_1"

camera_stats = {
    'a': 0.2,                  # Distance from the cameras to the origin plane along the x-axis
    'f': 0.04,                 # Focal length of the cameras in meters
    'pixel_size': 2.74e-6,     # Pixel size in meters
    'resolution': (4096, 3000) # Resolution of the cameras in pixels (width, height)
}

path_image = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\simulated_data\simulated_data\new_data"
path_coordinates = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\simulated_data"

file_name_image_1 = r"Noise_projection_rho_0_phi_30_cam1.npy"
file_name_image_2 = r"Noise_projection_rho_0_phi_30_cam2.npy"
file_name_coords_1 = r"projection_rho_0_phi_30_cam1.npy"
file_name_coords_2 = r"projection_rho_0_phi_30_cam2.npy"

br_ar_1 = load_npy_file(path_image, file_name_image_1)
br_ar_2 = load_npy_file(path_image, file_name_image_2)

coords_1 = load_npy_file(path_coordinates, file_name_coords_1)
coords_2 = load_npy_file(path_coordinates, file_name_coords_2)

print(coords_1.shape)

save_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\output\simulated_data\images"
file_name = r"Noise_projection_rho_0_phi_30_cam1.png"
fullpath = os.path.join(save_path, file_name)

peaks_1 = find_peaks_7(br_ar_1, factor=8)
peaks_2 = find_peaks_7(br_ar_2, factor=8)

print(peaks.shape)

array_1 = brightness_subarray_creator(br_ar_1, peaks_1)
array_2 = brightness_subarray_creator(br_ar_2, peaks_2)

mean_1 = compute_center_of_mass_with_uncertainty(array_1)[0]
mean_2 = compute_center_of_mass_with_uncertainty(array_2)[0]

laser_point_centers_1 = lpc_calc(mean_1, peaks_1)
laser_point_centers_2 = lpc_calc(mean_2, peaks_2)

differences_1 = calculate_differences(coords_1, laser_point_centers_1)
differences_2 = calculate_differences(coords_1, laser_point_centers_1)


plot_differences_as_bar_chart(differences_1, abs_values=True)































