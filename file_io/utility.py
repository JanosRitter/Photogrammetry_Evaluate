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
