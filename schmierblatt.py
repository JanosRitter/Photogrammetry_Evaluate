from file_io import bmp_to_brightness_array, create_c_plot_with_points
from peak_find import find_peaks, brightness_subarray_creator, fit_gaussian_3d, lpc_calc, compute_centroids_with_uncertainty_limited
#from lpc_indexing import find_outlier_point, analyze_coordinates


image_name_1 = 'tcam_11_image_1.bmp'
#image_name_2 = '\\tcam_12_image_1.bmp'
brightness_array_1 = bmp_to_brightness_array(image_name_1)
#brightness_array_2 = bmp_to_brightness_array(image_name_2)



peaks = find_peaks(brightness_array_1)

print(peaks.shape)

#create_c_plot_with_points(brightness_array_1, peaks)










