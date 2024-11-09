from file_io import bmp_to_brightness_array, create_c_plot_with_points, create_image_plot, plot_3d_points
from peak_find import find_peaks, brightness_subarray_creator, lpc_calc, peak_filter
from gauß_fit import fit_gaussian_3d
from lpc_indexing import find_outlier_point, analyze_coordinates, rotate_coordinates
from calc_3d import triangulate_3d

import numpy as np

np.set_printoptions(threshold=np.inf)


image_name_1 = 'tcam_11_image_1.bmp'
image_name_2 = 'tcam_12_image_1.bmp'

camera_stats = {
    'a': 0.2,                  # Distance from the cameras to the origin plane along the x-axis
    'f': 0.05,                 # Focal length of the cameras in meters
    'pixel_size': 2.74e-6,     # Pixel size in meters
    'resolution': (4096, 3000) # Resolution of the cameras in pixels (width, height)
}

brightness_array_1 = bmp_to_brightness_array(image_name_1)
brightness_array_2 = bmp_to_brightness_array(image_name_2)



peaks_1 = find_peaks(brightness_array_1, filename=image_name_1, factor=8)
peaks_2 = find_peaks(brightness_array_2, factor=8, filename=image_name_2)

filtered_peaks_1 = peak_filter(peaks_1)
filtered_peaks_2 = peak_filter(peaks_2, boundary_factor=2)

print(filtered_peaks_1.shape)
print(filtered_peaks_2.shape)


create_c_plot_with_points(brightness_array_2, filtered_peaks_2)



subarray_1 = brightness_subarray_creator(brightness_array_1, filtered_peaks_1)
subarray_2 = brightness_subarray_creator(brightness_array_2, filtered_peaks_2)

print(subarray_1.shape)
print(subarray_2.shape)


mean_values_1, deviations_1, fitted_data_1 = fit_gaussian_3d(subarray_1)
mean_values_2, deviations_2, fitted_data_2 = fit_gaussian_3d(subarray_2)

laser_point_centers_1 = lpc_calc(mean_values_1, filtered_peaks_1)
laser_point_centers_2 = lpc_calc(mean_values_2, filtered_peaks_2)

print(laser_point_centers_1.shape)
print(laser_point_centers_2.shape)

points_3D = triangulate_3d(laser_point_centers_2, laser_point_centers_1, camera_stats)


plot_3d_points(points_3D)




























