from file_io import bmp_to_brightness_array, create_c_plot_with_points
from peak_find import find_peaks, brightness_subarray_creator, fit_gaussian_3d, lpc_calc, compute_centroids_with_uncertainty_limited
#from lpc_indexing import find_outlier_point, analyze_coordinates


image_name = '\laser_spots.png'

brightness_array = bmp_to_brightness_array(image_name)

peaks = find_peaks(brightness_array, factor=15)

subarray = brightness_subarray_creator(brightness_array, peaks)

mean_values, deviations, fitted_data = fit_gaussian_3d(subarray)

laser_point_center = lpc_calc(brightness_array, mean_values, peaks)

print(laser_point_center)

create_c_plot_with_points(brightness_array, peaks, laser_point_center, filename='c_plot.png', colormap='viridis')








