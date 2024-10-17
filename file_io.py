"""
This module contains all functions that save and read in files to evaluate the
3D structure from a given 2D image using triangulation.
The main functions read in the given image and same different arrays as
.dat files after each computation. It also saves the end results in
form of a 3D plot and some images during the computation progress to
check if all is working correctly
"""

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt



def bmp_to_brightness_array(image_name):
    """
    Reads in a bmp image file and asigns each value its intensity value

    parameter:
        - image_name

    return:
        - array shape equivalent to the resolution of the image, usually (1920,1080)
        the intensity values are in between 0 and 255
    """
    base_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner"
    image_path = base_path + image_name
    img = Image.open(image_path)
    img = img.convert('RGB')


    width, height = img.size

    brightness_array = np.zeros((height, width))

    for y_value in range(height):
        for x_value in range(width):
            red, green, blue = img.getpixel((x_value, y_value))

            brightness = (red + green + blue) / 3

            brightness_array[y_value, x_value] = brightness

    return brightness_array




def plot_brightness_array(brightness_array, x_limit=None, y_limit=None, save_path=None):
    """
    Creates a 3D PNG-Plot of a given array, where the color indicates the z-value

    parameter:
        - brightness_array: (n,m) array you want to visualize
        - x_limit: (1,2) array that limits the plot in x-direction
        - y_limit: (1,2) array that limits the plot in y-direction
        - save_path: path where the plot is saved

    returns:
        -no returns, except the savec image
    """

    if x_limit is None:
        x_limit = [0, brightness_array.shape[1]]
    if y_limit is None:
        y_limit = [0, brightness_array.shape[0]]

    x_max = min(x_limit[1], brightness_array.shape[1])
    y_max = min(y_limit[1], brightness_array.shape[0])
    x_min = x_limit[0]
    y_min = y_limit[0]


    widht, height = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))


    brightness_subarray = brightness_array[y_min:y_max, x_min:x_max]

    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')

    axis.plot_surface(widht, height, brightness_subarray, cmap='gray')

    axis.set_xlabel('X (Pixel)')
    axis.set_ylabel('Y (Pixel)')
    axis.set_zlabel('Brightness (Helligkeit)')
    axis.set_title('3D-Helligkeitsverteilung (Teilbereich)')

    if save_path:
        try:
            plt.savefig(save_path, format='png', dpi=300)  # Format und DPI anpassen
            print(f"Plot gespeichert unter: {save_path}")
        except PermissionError as error:
            print(f"PermissionError: {error}")

    plt.show()




def create_c_plot_with_points(data, peaks, mean_values, filename='c_plot.png', colormap='viridis'):
    """
    Creates a contour plot from a 2D array and puts points on it

    Parameters:
        - data (np.ndarray): 2D array with data for the contour plot
        - points (np.ndarray): (n, 2) array with an estimation of the laser point centers
        - mean_values: (n, 2) array with calculated laser point centers
        - filename (str): The file name where the plot is saved
          (Standard: 'c_plot.png').
        - colormap (str)
    """

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(data, cmap=colormap)
    plt.colorbar(contour)

    plt.title('Konturplot mit Punkten')
    plt.xlabel('X-Achse')
    plt.ylabel('Y-Achse')


    plt.plot(peaks[:, 0], peaks[:, 1], 'ro', markersize=1, label='Punkte')
    plt.plot(mean_values[:, 0], mean_values[:, 1], 'yo', markersize=1, label='Punkte')
    plt.legend()

    plt.gca().invert_yaxis()

    base_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner"
    image_path = os.path.join(base_path, filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Konturplot mit Punkten wurde als '{filename}' gespeichert.")




def save_array_as_dat(array, file_path, file_name):
    """
    Saves a given NumPy array to a .dat file.

    Parameters:
        - array (np.ndarray): The NumPy array to save.
        - file_path (str): The directory path where the file will be saved.
        - file_name (str): The name of the .dat file (including extension).
    """

    full_path = file_path + "/" + file_name

    try:
        np.savetxt(full_path, array, delimiter=' ', fmt='%f')
        print(f"Array saved successfully to {full_path}")
    except OSError as error:
        print(f"OS error occurred: {error}")
    except ValueError as error:
        print(f"Value error: {error}")
        