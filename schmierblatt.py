from file_io import bmp_to_brightness_array, create_c_plot_with_points, create_image_plot
from peak_find import find_peaks, brightness_subarray_creator, lpc_calc, compute_centroids_with_uncertainty_limited, peak_filter
from gauß_fit import fit_gaussian_3d
from lpc_indexing import find_outlier_point, analyze_coordinates, rotate_coordinates
from calc_3d import triangulate_3D, plot_3d_points

import numpy as np

np.set_printoptions(threshold=np.inf)


image_name_1 = 'tcam_11_image_1.bmp'
image_name_2 = 'tcam_12_image_1.bmp'
brightness_array_1 = bmp_to_brightness_array(image_name_1)
brightness_array_2 = bmp_to_brightness_array(image_name_2)



peaks_1 = find_peaks(brightness_array_1, filename=image_name_1, factor=8)
peaks_2 = find_peaks(brightness_array_2, factor=8, filename=image_name_2)

filtered_peaks_1 = peak_filter(peaks_1)
filtered_peaks_2 = peak_filter(peaks_2, boundary_factor=2)





#create_c_plot_with_points(brightness_array_2, filtered_peaks_2, filename=image_name_2)

subarray_1 = brightness_subarray_creator(brightness_array_1, filtered_peaks_1)
subarray_2 = brightness_subarray_creator(brightness_array_2, filtered_peaks_2)


mean_values_1, deviations_1, fitted_data_1 = fit_gaussian_3d(subarray_1)
mean_values_2, deviations_2, fitted_data_2 = fit_gaussian_3d(subarray_2)

laser_point_centers_1 = lpc_calc(brightness_array_1, mean_values_1, filtered_peaks_1)
laser_point_centers_2 = lpc_calc(brightness_array_2, mean_values_2, filtered_peaks_2)

print(laser_point_centers_1.shape)
print(laser_point_centers_2.shape)

#points_3D = triangulate_3D(laser_point_centers_1, laser_point_centers_2)

outlier_1, outlier_index = find_outlier_point(laser_point_centers_1)
outlier_2, outlier_index = find_outlier_point(laser_point_centers_2)

lpc_aligned_1 = rotate_coordinates(laser_point_centers_1)
lpc_aligned_2 = rotate_coordinates(laser_point_centers_2)


lpc_indexed_1 = analyze_coordinates(lpc_aligned_1, tolerance=20.0)
lpc_indexed_2 = analyze_coordinates(lpc_aligned_2, tolerance=20.0)


points_3D = triangulate_3D(lpc_indexed_1, lpc_indexed_2)
print(points_3D)

plot_3d_points(points_3D)




























