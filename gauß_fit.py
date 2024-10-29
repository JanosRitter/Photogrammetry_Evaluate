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

