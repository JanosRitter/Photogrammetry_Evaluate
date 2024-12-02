"""
This module contains
"""
import os

def ensure_output_path(base_output_dir, project_name, measurement_name, filename):
    """
    Ensures the output path exists and returns the full path for saving a file.

    Parameters:
        - base_output_dir (str): Base directory for output files.
        - project_name (str): Name of the project.
        - measurement_name (str): Name of the measurement.
        - filename (str): Name of the file to save.

    Returns:
        - str: Full path to the output file.
    """
    output_dir = os.path.join(base_output_dir, project_name, measurement_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured: {output_dir}")
    return os.path.join(output_dir, filename)




def construct_output_path(input_path, base_folder="output", filename=None):
    """
    Constructs an output path by replacing 'input' with a specified base folder (e.g., 'output').

    Parameters:
        - input_path (str): Relative input path (e.g., to the data folder within the base directory).
        - base_folder (str): Name of the folder to replace 'input' with (default: "output").
        - filename (str): Name of the file to append to the path (optional).

    Returns:
        - str: Full output path, with the directory structure mirrored from the input.
    """
    # Define the base path for input
    base_input_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input"
    
    # Construct the full input path
    full_input_path = os.path.join(base_input_path, input_path)

    # Replace 'input' with the base folder name in the path
    output_path = full_input_path.replace("\\input\\", f"\\{base_folder}\\")
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory ensured: {output_path}")

    # Append the filename if provided
    if filename:
        return os.path.join(output_path, filename)
    return output_path
