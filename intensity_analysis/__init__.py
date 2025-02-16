"""
This package contains a set of functions for analyzing intensity distributions,
fitting models to the data, and calculating the centers of mass. The modules
offer various methods for processing and analyzing 3D data arrays, including
Gaussian fitting, skewed Gaussian fitting, and center of mass calculations.
Additionally, utility functions for data preprocessing and subarray extraction
are also included.

Modules:
- gauss_fit: Contains the function `fit_gaussian_3d` for fitting a 2D Gaussian
  model to each slice of a 3D intensity array.
- skew_gauss_fit: Contains the function `fit_skewed_gaussian_3d` for fitting
  a skewed 2D Gaussian model to each slice of a 3D intensity array.
- center_of_mass: Contains the function `compute_center_of_mass_with_uncertainty`
  for computing the center of mass and its uncertainty for each 2D slice in a
  3D array.
- non_linear_center_of_mass: Contains the function `non_linear_center_of_mass`
  for calculating a non-linear center of mass, offering an alternative approach.
- center_of_mass_with_threshold: Contains the function `center_of_mass_with_threshold`
  for computing the center of mass with an intensity threshold applied to the data.
- utility: Contains utility functions such as `brightness_subarray_creator`,
  `subtract_mean_background`, and `lpc_calc` for preprocessing data, extracting
  subarrays around peaks, and calculating laser point centers.

This package facilitates the extraction of meaningful features from intensity
distributions in various scientific applications, such as laser scanning, image
analysis, and other intensity-based measurement tasks.

Usage:
    - Use the functions in this package for performing model fitting, calculating
      centers of mass, and processing intensity data arrays.
    - Utility functions can help with data preprocessing, background subtraction,
      and peak detection.

Note:
    All functions expect numpy arrays as input and may raise errors if the input
    does not match the expected format.
"""
from .gauss_fit import fit_gaussian_3d
from .skew_gauss_fit import fit_skewed_gaussian_3d
from .center_of_mass import compute_center_of_mass_with_uncertainty
from .non_linear_center_of_mass import non_linear_center_of_mass
from .center_of_mass_with_threshold import center_of_mass_with_threshold
from .circle_fit import circle_fitting_with_threshold
from .utility import brightness_subarray_creator, lpc_calc
