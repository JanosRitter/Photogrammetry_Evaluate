import sys
import os

# Projektpfad hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_path)

from file_io import add_noise_to_folder
from data_simulate import simulate_background_noise, add_noise_to_data

input_folder = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\spot_scale_1\example"
output_folder_name = r"varying_bg_noise"

add_noise_to_folder(input_folder, output_folder_name, mean=50, std=16, offset=45)