import numpy as np
from scipy.optimize import leastsq

def fit_circle_2d(x, y):
    """ Least-squares fit of a circle to 2D data points. """
    def calc_R(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x_m, y_m = x.mean(), y.mean()
    center, _ = leastsq(f_2, (x_m, y_m))
    radius = calc_R(*center).mean()
    return center[0], center[1], radius

def circle_fitting_with_threshold(data_3d, threshold=None, max_radius=30, border_margin=5):
    """
    Fits a circle to each 2D slice of a 3D array using a specified threshold.

    Parameters:
    - data_3d (np.ndarray): Input 3D array of shape (num, height, width).
    - threshold (float or None): Intensity threshold for filtering points.
      If None, it is automatically set to mean + 2.5 * std.
    - max_radius (float): Maximum allowed radius.
    - border_margin (int): Minimum distance of the center from the array edges.

    Returns:
    - fitted_centers (np.ndarray): Array of shape (n, 2) containing x, y of the fitted center.
    - uncertainties (np.ndarray): Array of shape (n,) containing radius as uncertainty.
    """
    if not isinstance(data_3d, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if data_3d.ndim != 3:
        raise ValueError("Input array must be 3D.")

    num_slices, height, width = data_3d.shape
    fitted_centers = np.full((num_slices, 2), np.nan)  # (x_center, y_center)
    uncertainties = np.full(num_slices, np.nan)  # Radius as uncertainty

    for i in range(num_slices):
        slice_2d = data_3d[i]

        # Falls kein Threshold übergeben wurde, berechne Standard-Threshold
        if threshold is None:
            mean_val = np.mean(slice_2d)
            std_val = np.std(slice_2d)
            threshold = mean_val + 2.5 * std_val

        # Threshold anwenden
        filtered_slice = np.where(slice_2d >= threshold, slice_2d, 0)

        # Indizes der nicht-null Werte (Punkte für das Circle Fitting)
        y_indices, x_indices = np.nonzero(filtered_slice)

        if len(x_indices) < 3:  # Mindestens 3 Punkte für das Fitting notwendig
            continue  # Überspringe diesen Slice, bleibt NaN

        # Kreis anpassen
        x_center, y_center, radius = fit_circle_2d(x_indices, y_indices)

        # Bedingungen prüfen: Nicht am Rand & Radius maximal 30
        if (
            border_margin <= x_center <= width - border_margin and
            border_margin <= y_center <= height - border_margin and
            radius <= max_radius
        ):
            fitted_centers[i] = [x_center, y_center]
            uncertainties[i] = radius

    return fitted_centers, uncertainties


