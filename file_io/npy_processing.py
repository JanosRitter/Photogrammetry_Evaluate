"""
This module contains functions to load, save and process .npy files
"""

import os
import numpy as np
from PIL import Image


def load_brightness_arrays(folder_name):
    """
    Reads BMP image files in a specified folder, ensuring exactly two are present,
    and converts each to a brightness array. Loads from existing .npy files if available.

    Parameters:
        - folder_name: Name of the final directory containing the BMP files.

    Returns:
        - brightness_array_1, brightness_array_2: Two np.ndarrays with brightness values (0-255).

    Raises:
        - ValueError: If there are not exactly two BMP files in the folder.
    """

    base_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input"
    folder_path = os.path.join(base_path, folder_name)
    bmp_files = sorted([file for file in os.listdir(folder_path) if file.lower().endswith('.bmp')])

    if len(bmp_files) != 2:
        raise ValueError(f"Expected exactly 2 BMP files, but found {len(bmp_files)}.")

    brightness_arrays = []

    for bmp_file in bmp_files:
        bmp_path = os.path.join(folder_path, bmp_file)
        npy_file_path = os.path.join(folder_path, bmp_file.replace('.bmp', '.npy'))

        if os.path.exists(npy_file_path):
            print(f".npy file found: Loading {npy_file_path}")
            brightness_array = np.load(npy_file_path)
        else:
            print(f".npy file not found: Processing {bmp_file} and saving as {npy_file_path}")
            img = Image.open(bmp_path).convert('RGB')
            width, height = img.size
            brightness_array = np.zeros((height, width), dtype=np.uint8)

            for y_value in range(height):
                for x_value in range(width):
                    red, green, blue = img.getpixel((x_value, y_value))
                    brightness = (red + green + blue) / 3
                    brightness_array[y_value, x_value] = brightness

            np.save(npy_file_path, brightness_array)
            print(f"Brightness array saved as {npy_file_path}")

        brightness_arrays.append(brightness_array)

    return brightness_arrays[0], brightness_arrays[1]





def save_array_as_npy(array, file_path, file_name):
    """
    Saves a given NumPy array to a .npy file.

    Parameters:
        - array (np.ndarray): The NumPy array to save.
        - file_path (str): The directory path where the file will be saved.
        - file_name (str): The name of the file (can include any extension; will be saved as .npy).
    """

    supported_extension = '.npy'
    if not file_name.lower().endswith(supported_extension):
        file_name = os.path.splitext(file_name)[0] + supported_extension

    full_path = os.path.join(file_path, file_name)

    try:
        np.save(full_path, array)
        print(f"Array saved successfully to {full_path}")
    except OSError as error:
        print(f"OS error occurred: {error}")
    except ValueError as error:
        print(f"Value error: {error}")




def load_npy_file(filepath, filename):
    """
    Loads a .npy file and returns its contents as a numpy array.

    Parameters:
    - filepath (str): The path to the directory containing the .npy file.
    - filename (str): The name of the .npy file to be loaded.

    Returns:
    - np.ndarray: Array containing the data from the .npy file.
    """
    full_path = os.path.join(filepath, filename)
    return np.load(full_path)



def transpose_with_padding(array):
    """
    Makes a 2D array square by trimming or padding with zeros, transposes it,
    and then restores the original shape by padding with zeros.

    Parameters:
        array (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Transposed array restored to the original shape with zeros padded.
    """
    original_shape = array.shape

    square_size = min(original_shape)

    squared_array = array[:square_size, :square_size]

    transposed_array = squared_array.T

    padded_array = np.pad(
        transposed_array,
        pad_width=((0, original_shape[0] - transposed_array.shape[0]),  # Rows
                   (0, original_shape[1] - transposed_array.shape[1])),  # Columns
        mode='constant',
        constant_values=0
    )

    return padded_array
        