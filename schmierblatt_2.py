import file_io


filepath_old = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\simulated_data\simulated_data\noise_spots"
filepath_new = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\simulated_data\simulated_data\new_data"
filename_1 = r"Noise_projection_rho_0_phi_0_cam1.npy"
filename_2 = r"Noise_projection_rho_0_phi_0_cam2.npy"

data_old_1 = load_npy_file(filepath_old, filename_1)
data_old_2 = load_npy_file(filepath_old, filename_2)
data_new_1 = load_npy_file(filepath_new, filename_1)
data_new_2 = load_npy_file(filepath_new, filename_2)

save_path = os.path.join(filepath_new, r"what_it_should_look_like_2.png")

create_contour_plot(data_new_2, save_path=save_path)
