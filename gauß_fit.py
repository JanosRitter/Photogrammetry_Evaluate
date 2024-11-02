import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

def gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude):
    """
    2D-Gaußmodell zur Erzeugung einer Gauß-Verteilung für gegebene Parameter.
    """
    x, y = xy
    return amplitude * np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +
                                ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))

def gaussian_2d_residuals(params, xy, slice_2d):
    """
    Berechnet die Residuen für das 2D-Gauß-Fitting.
    """
    mu_x, mu_y, sigma_x, sigma_y, amplitude = params
    model = gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude)
    return np.sum((model - slice_2d.ravel()) ** 2)  # Summe der quadrierten Fehler

def prepare_meshgrid(width, height):
    """
    Erstellt ein Meshgrid für gegebene Breite und Höhe.
    """
    x = np.arange(0, width)
    y = np.arange(0, height)
    return np.meshgrid(x, y)

def fit_single_slice(slice_2d, width, height):
    # Überprüfen Sie den Datentyp
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError("slice_2d muss ein numpy Array sein.")
    
    # Stellen Sie sicher, dass das Array numerisch ist
    slice_2d = np.asarray(slice_2d, dtype=np.float64)

    # Leichtes Glätten des Slices zur Rauschminderung
    smoothed_slice = gaussian_filter(slice_2d, sigma=2)
    
    # Maximale Intensität und deren Koordinaten im geglätteten Slice
    y_max, x_max = np.unravel_index(np.argmax(smoothed_slice), smoothed_slice.shape)

    # Initialer Startwert nahe am Zentrum
    initial_guess = [x_max, y_max, width / 8, height / 8, smoothed_slice.max()]

    # Grenzen zur Stabilisierung des Fits
    bounds = [(0, width), (0, height), (1, width), (1, height), (0.1, None)]
    
    # Meshgrid und Fitting starten
    x_value, y_value = prepare_meshgrid(width, height)
    xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])
    
    result = minimize(gaussian_2d_residuals, initial_guess, args=(xy_grid, slice_2d),
                      bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        return result.x[:4], result.x[4]  # Erwartungswerte, Standardabweichungen, Amplitude
    else:
        return [np.nan] * 4, np.nan

def fit_gaussian_3d(intensity_array):
    """
    Führt das Gauß-Fitting auf jedem 2D-Slice eines 3D-Arrays durch.
    """
    num_slices, height, width = intensity_array.shape

    mean_values = np.zeros((num_slices, 2))
    deviations = np.zeros((num_slices, 2))
    fitted_data = np.zeros((num_slices, height, width))

    for i in range(num_slices):
        slice_2d = intensity_array[i]
        (mu_x, mu_y, sigma_x, sigma_y), amplitude = fit_single_slice(slice_2d, width, height)

        mean_values[i] = [mu_x, mu_y]
        deviations[i] = [sigma_x, sigma_y]

        if not np.isnan(mu_x):
            x_value, y_value = prepare_meshgrid(width, height)
            xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])
            fitted_slice = gaussian_2d_model(xy_grid, mu_x, mu_y, sigma_x, sigma_y, amplitude).reshape(height, width)
            fitted_data[i] = fitted_slice
        else:
            fitted_data[i] = np.full((height, width), np.nan)  # Bei Fit-Fehler NaNs einfügen

    return mean_values, deviations, fitted_data




def compute_centroids_with_uncertainty_limited(intensity_array, estimated_positions, max_limits=(60, 60)):
    """
    Computes the centroids of laser points and their uncertainties in a given intensity array,
    with limits on window size around each point.

    Parameters:
    - intensity_array (np.ndarray): 2D array of intensity values (0-255).
    - estimated_positions (np.ndarray): (n, 2) array of estimated (x, y) positions of laser points.
    - max_limits (tuple): Maximum window size for (x_limit, y_limit).

    Returns:
    - centroids (np.ndarray): (n, 2) array of centroid positions (x, y) for each laser point.
    - uncertainties (np.ndarray): (n, 2) array of uncertainties (sigma_x, sigma_y) for each centroid.
    """

    num_points = estimated_positions.shape[0]
    centroids = np.zeros((num_points, 2))
    uncertainties = np.zeros((num_points, 2))

    # Find the window size for each point using the find_limit function
    window_x_limit, window_y_limit = find_limit(intensity_array.shape, estimated_positions, max_limits)

    for i, (x_est, y_est) in enumerate(estimated_positions):
        # Define the window around the estimated position
        x_min = max(int(x_est - window_x_limit), 0)
        x_max = min(int(x_est + window_x_limit), intensity_array.shape[1] - 1)
        y_min = max(int(y_est - window_y_limit), 0)
        y_max = min(int(y_est + window_y_limit), intensity_array.shape[0] - 1)

        # Extract the window around the estimated point
        sub_array = intensity_array[y_min:y_max+1, x_min:x_max+1]

        # Create a grid of x and y positions
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))

        # Calculate the weighted sum of the positions using the intensity as weights
        total_intensity = np.sum(sub_array)

        if total_intensity > 0:
            x_centroid = np.sum(x_grid * sub_array) / total_intensity
            y_centroid = np.sum(y_grid * sub_array) / total_intensity

            # Calculate uncertainty (standard deviation) based on weighted variance
            x_var = np.sum(((x_grid - x_centroid) ** 2) * sub_array) / total_intensity
            y_var = np.sum(((y_grid - y_centroid) ** 2) * sub_array) / total_intensity

            centroids[i] = [x_centroid, y_centroid]
            uncertainties[i] = [np.sqrt(x_var), np.sqrt(y_var)]
        else:
            # If no intensity is found, return NaN for the centroid and uncertainty
            centroids[i] = [np.nan, np.nan]
            uncertainties[i] = [np.nan, np.nan]

    return centroids, uncertainties

