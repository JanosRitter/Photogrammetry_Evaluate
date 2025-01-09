import file_io

input_folder = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\noise_spots_20pixels_D2\noise_spots"

file_name = r"Noise_projection_cam1_scale_3.npy"

array = load_npy_file(input_folder, file_name)

create_image_plot(array, output_path=input_folder)































