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

    bmp_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.bmp')])
    if len(bmp_files) != 2:
        raise ValueError(f"Expected exactly 2 BMP files, but found {len(bmp_files)}.")

    def load_or_process_brightness(file_name):
        npy_path = os.path.join(folder_path, file_name.replace('.bmp', '.npy'))
        if os.path.exists(npy_path):
            print(f".npy file found: Loading {npy_path}")
            return np.load(npy_path)
        print(f".npy file not found: Processing {file_name} and saving as {npy_path}")
        img = Image.open(os.path.join(folder_path, file_name)).convert('RGB')
        brightness_array = np.mean(np.array(img), axis=2).astype(np.uint8)  # Mittelwert der Kanäle
        np.save(npy_path, brightness_array)
        print(f"Brightness array saved as {npy_path}")
        return brightness_array

    brightness_arrays = [load_or_process_brightness(f) for f in bmp_files]
    return brightness_arrays[0], brightness_arrays[1]




def load_all_npy_files(folder_path, filenames=None):
    """
    Loads all `.npy` files from a folder or a specified subset of files.

    Parameters:
    - folder_path (str): Path to the folder containing `.npy` files.
    - filenames (list or None): A list of specific `.npy` filenames to load. If None, all
    `.npy` files are loaded.

    Returns:
    - tuple: (npy_files, arrays)
        - npy_files: List of loaded `.npy` filenames.
        - arrays: List of corresponding arrays loaded from the `.npy` files.
    """
    all_npys = sorted([file for file in os.listdir(folder_path) if file.lower().endswith('.npy')])

    if filenames is not None:
        npy_files = [file for file in all_npys if file in filenames]
    else:
        npy_files = all_npys

    arrays = [np.load(os.path.join(folder_path, file)) for file in npy_files]

    return npy_files, arrays





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



def convert_png_to_npy(folder_path):
    """
    Converts all PNG images in the specified folder to NumPy arrays representing brightness values (0-255) and
    saves them as .npy files in the same directory.

    Parameters:
        - folder_path (str): Path to the directory containing PNG files.
    """
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    for file_name in png_files:
        img_path = os.path.join(folder_path, file_name)
        npy_path = img_path.replace('.png', '.npy')
        
        img = Image.open(img_path).convert('RGB')
        brightness_array = np.mean(np.array(img), axis=2).astype(np.uint8)
        
        print(brightness_array.shape)
        
        np.save(npy_path, brightness_array)




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
        