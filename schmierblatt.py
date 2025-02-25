import numpy as np
import os
from file_io import *
from moduls.data_simulate import simulate_background_noise, add_noise_to_data

def generate_xy_pattern(
    size_x, size_y=None, 
    scale=1.0, scale_factors=None, 
    save_dir="output/"
):
    """
    Generates an x, y pattern of coordinates and saves each pattern as an .npy file.

    Parameters:
    - size_x (int): Number of points in the x-direction.
    - size_y (int, optional): Number of points in the y-direction. Defaults to size_x if None.
    - scale (float, optional): General scaling factor for the grid. Default is 1.0.
    - scale_factors (list or np.ndarray, optional): Additional scaling factors to loop over.
    - save_dir (str, optional): Directory to save the generated .npy files. Default is "output/".

    Returns:
    - None
    """
    if size_y is None:
        size_y = size_x

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Loop through scale_factors and save each pattern
    if scale_factors is None:
        scale_factors = [1.0]

    for sf in scale_factors:
        # Generate a grid of points for the current scale factor
        x_coords = np.linspace(scale * sf, scale * sf * size_x, size_x)
        y_coords = np.linspace(scale * sf, scale * sf * size_y, size_y)
        x, y = np.meshgrid(x_coords, y_coords)

        pattern = np.column_stack((x.ravel(), y.ravel()))
        
        # Debug output for verification
        print(f"Scale Factor {sf}:")
        print(pattern)
        plot_2d_points(pattern)
        
        # Save pattern with the scale factor as a suffix in the file name
        file_name = f"xy_pattern_scale_{int(sf)}.npy"
        file_path = os.path.join(save_dir, file_name)
        np.save(file_path, pattern)
        print(f"Pattern with scale factor {sf:.2f} saved to {file_path}")

# Example usage
save_dir = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\sim_dat_large_scale"

#scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50]


import os
import numpy as np
from PIL import Image

import os
import numpy as np
from PIL import Image

def npy_to_bmp(npy_path, filename):
    """
    Wandelt eine .npy-Datei mit Intensitätswerten (0-255) in eine .bmp-Datei um und speichert sie im gleichen Ordner.

    Parameter:
        npy_path (str): Pfad zum Ordner mit der .npy-Datei oder direkt zur .npy-Datei.
        filename (str): Name der Datei ohne Erweiterung.

    Rückgabewert:
        str: Pfad zur gespeicherten .bmp-Datei.
    """
    # Prüfe, ob npy_path eine Datei oder ein Ordner ist
    if os.path.isdir(npy_path):  
        npy_file = os.path.join(npy_path, f"{filename}.npy")  # Falls nur der Ordner übergeben wurde
    else:
        npy_file = npy_path  # Falls der gesamte Pfad inkl. Datei angegeben wurde

    # Prüfe, ob die Datei existiert
    if not os.path.isfile(npy_file):
        raise FileNotFoundError(f"Die Datei existiert nicht: {npy_file}")

    # Lade das .npy-Array
    img_array = np.load(npy_file)

    # Wertebereich sicherstellen (0-255)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Speichern im gleichen Ordner wie die .npy-Datei
    output_folder = os.path.dirname(npy_file)
    output_path = os.path.join(output_folder, f"{filename}.bmp")

    # Bild speichern
    Image.fromarray(img_array).save(output_path)
    
    print(f"Bild gespeichert: {output_path}")
    return output_path


#npy_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\spot_scale_1\spots_with_backgroundnoise"
#filename = r"Noise_projection_cam2_scale_10"

#npy_to_bmp(npy_path, filename)


import os
import numpy as np

def compress_and_save_npy(folder_path):
    """
    Komprimiert alle `.npy` Dateien in einem Ordner, mittelt die Werte entsprechend ihrer Größe, 
    skaliert das Ergebnis auf den Bereich 0-255, fügt thermisches Rauschen hinzu und speichert sie als 8-Bit `.npy` Dateien.
    
    Parameters:
    - folder_path (str): Pfad zum Ordner mit den `.npy` Dateien.
    """
    # Dateien einlesen
    npy_files, arrays = load_all_npy_files(folder_path)

    if not npy_files:
        print("Keine `.npy` Dateien gefunden.")
        return

    # Neuen Ordner für die komprimierten Daten erstellen
    output_folder = os.path.join(folder_path, "compressed_2")
    os.makedirs(output_folder, exist_ok=True)

    for filename, array in zip(npy_files, arrays):
        # Originalgröße auslesen
        h, w = array.shape
        if h % 550 != 0 or w % 550 != 0:
            print(f"Überspringe {filename}, da die Größe nicht durch 550 teilbar ist ({h}x{w}).")
            continue

        # Berechnung von n
        n_h = h // 550
        n_w = w // 550

        # Array in Blöcke der Größe 550x550 mitteln
        compressed_array = array.reshape(550, n_h, 550, n_w).mean(axis=(1, 3))

        # **Skalierung auf den Bereich 0-255**
        max_value = compressed_array.max()
        if max_value > 0:  # Vermeidung von Division durch 0
            compressed_array = compressed_array * (245 / max_value)

        # Werte runden und auf Ganzzahlen setzen
        compressed_array = np.rint(compressed_array).astype(np.uint8)

        # **Rauschen hinzufügen**
        noise = simulate_background_noise(array_shape=(550, 550), mean=10, std=3)
        noisy_array = add_noise_to_data(compressed_array, noise, offset=10)

        # **Speichern**
        output_path = os.path.join(output_folder, filename)
        np.save(output_path, noisy_array)

        print(f"Gespeichert: {output_path}")

    print("Alle Dateien wurden erfolgreich verarbeitet und gespeichert.")


folder_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input\noisy_550x550_set1\shifted_noisy"

compress_and_save_npy(folder_path)

















