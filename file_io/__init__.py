"""
The `file_io` package provides functionality for managing file input and output operations,
including handling `.npy` files and generating various plots.

Modules:
- `npy_processing`: Includes functions for loading and saving `.npy` files and
manipulating brightness arrays.
- `plotting`: Provides tools for visualizing data, including brightness arrays,
contour plots, 3D point plots,
  and bar charts for differences.
- `utility`: Contains helper functions for ensuring valid output paths and
constructing file paths.

Functions available for direct import:
- From `npy_processing`:
  - `load_brightness_arrays`: Load brightness data from `.bmp` or `.npy` files.
  - `save_array_as_npy`: Save an array as a `.npy` file.
  - `load_npy_file`: Load a single `.npy` file.
  - `load_all_npy_files`: Load all `.npy` files from a folder.
- From `plotting`:
  - `plot_brightness_array`: Plot a 2D brightness array.
  - `create_contour_plot`: Generate a contour plot from data.
  - `create_image_plot`: Create an image plot from array data.
  - `plot_3d_points`: Plot 3D points in a scatter plot.
  - `plot_differences_as_bar_chart`: Visualize differences as bar charts.
- From `utility`:
  - `ensure_output_path`: Ensure the existence of an output path.
  - `construct_output_path`: Construct a valid file path for saving output.

Usage:
Import functions or modules as needed, for example:

    from file_io import load_brightness_arrays, plot_3d_points
    from file_io.utility import ensure_output_path

This package is designed to simplify file handling and visualization tasks,
especially for `.npy` files and related plotting workflows.
"""

from .npy_processing import load_brightness_arrays, save_array_as_npy, load_npy_file, load_all_npy_files, convert_png_to_npy
from .plotting import plot_brightness_array, create_contour_plot, create_image_plot, plot_3d_points, plot_differences_as_bar_chart
from .utility import ensure_output_path, construct_output_path, construct_flex_op_path
