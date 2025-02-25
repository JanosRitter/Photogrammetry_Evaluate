import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_beam(size=50, center=None, sigma_x=10, sigma_y=10):
    """
    Erzeugt ein 2D-Array mit einer Normalverteilung (kreisförmig oder elliptisch).
    
    Parameters:
    - size (int): Die Breite und Höhe des quadratischen Arrays.
    - center (tuple): Die Koordinaten des Zentrums der Normalverteilung (x, y).
                      Wenn None, wird das Zentrum automatisch in die Mitte gelegt.
    - sigma_x (float): Die Standardabweichung in x-Richtung.
    - sigma_y (float): Die Standardabweichung in y-Richtung.
    
    Returns:
    - np.ndarray: Ein 2D-Array mit der Normalverteilung.
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Formel für eine 2D-Normalverteilung
    gaussian = np.exp(-(((X - center[0])**2 / (2 * sigma_x**2)) + ((Y - center[1])**2 / (2 * sigma_y**2))))
    
    # Normiere auf den Bereich 0-255
    gaussian = 255 * (gaussian / np.max(gaussian))
    
    return gaussian

def plot_array(array, title="2D Gaussian Beam", cmap="viridis", save_path=None):
    """
    Plottet ein 2D-Array mit einer Farbskala und speichert es optional.
    
    Parameters:
    - array (np.ndarray): Das 2D-Array, das geplottet werden soll.
    - title (str): Der Titel des Plots.
    - cmap (str): Das Farbskalen-Schema.
    - save_path (str): Wenn angegeben, wird der Plot unter diesem Pfad gespeichert.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap=cmap, interpolation="nearest", vmin=0, vmax=255)
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def generate_egg_shaped_gaussian_beam(size=50, center=None, sigma_x=10, sigma_y=10, asymmetry_factor=1.5):
    """
    Erzeugt ein 2D-Array mit einer asymmetrischen (eiförmigen) Normalverteilung.
    
    Parameters:
    - size (int): Die Breite und Höhe des quadratischen Arrays.
    - center (tuple): Die Koordinaten des Zentrums der Normalverteilung (x, y).
                      Wenn None, wird das Zentrum automatisch in die Mitte gelegt.
    - sigma_x (float): Die Standardabweichung in x-Richtung.
    - sigma_y (float): Die Standardabweichung in y-Richtung.
    - asymmetry_factor (float): Ein Faktor zur Erzeugung der Eiform.
                                Werte >1 machen die obere Seite "dicker".
    
    Returns:
    - np.ndarray: Ein 2D-Array mit der asymmetrischen Normalverteilung.
    """
    if center is None:
        center = (size // 2, size // 2)
    
    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Berechne eine asymmetrische Skalierung in y-Richtung
    asymmetry = 1 + asymmetry_factor * ((Y - center[1]) > 0)
    adjusted_sigma_y = sigma_y * asymmetry
    
    # Formel für die eiförmige Normalverteilung
    gaussian = np.exp(-(((X - center[0])**2 / (2 * sigma_x**2)) + ((Y - center[1])**2 / (2 * adjusted_sigma_y**2))))
    
    # Normiere auf den Bereich 0-255
    gaussian = 255 * (gaussian / np.max(gaussian))
    
    return gaussian

def add_speckle_noise(array, speckle_intensity=0.5):
    """
    Fügt einem 2D-Array Speckle-Noise hinzu.
    
    Parameters:
    - array (np.ndarray): Das Eingangsarray (z. B. eine 2D-Gaussian-Verteilung).
    - speckle_intensity (float): Die Stärke des Speckles (0 bis 1).
    
    Returns:
    - np.ndarray: Das Array mit Speckle-Noise.
    """
    # Erzeuge zufälliges Speckle-Noise
    noise = np.random.normal(loc=1.0, scale=speckle_intensity, size=array.shape)
    
    # Multipliziere das Array mit dem Noise
    noisy_array = array * noise
    
    # Normiere den Wertebereich zurück auf 0-255
    noisy_array = 255 * (noisy_array / np.max(noisy_array))
    
    return noisy_array

def add_background_noise(array, noise_mean=0, noise_std=10):
    """
    Fügt einem gegebenen 2D-Array normalverteiltes Hintergrundrauschen hinzu.
    
    Parameters:
    - array (np.ndarray): Das Eingangsarray.
    - noise_mean (float): Mittelwert des Hintergrundrauschens.
    - noise_std (float): Standardabweichung des Hintergrundrauschens.
    
    Returns:
    - np.ndarray: Ein neues Array mit Hintergrundrauschen.
    """
    noise = np.random.normal(noise_mean, noise_std, array.shape)
    noisy_array = array + noise
    
    # Clipping, um Werte im gültigen Bereich zu halten (0 bis 255)
    noisy_array = np.clip(noisy_array, 0, 255)
    return noisy_array

# Erzeuge die fünf Arrays
circular_gaussian = generate_gaussian_beam(size=50, sigma_x=6, sigma_y=6)
elliptical_gaussian = generate_gaussian_beam(size=50, sigma_x=6, sigma_y=12)
egg_shaped_gaussian = generate_egg_shaped_gaussian_beam(size=50, sigma_x=6, sigma_y=8, asymmetry_factor=1.2)
speckled_egg_gaussian = add_speckle_noise(egg_shaped_gaussian, speckle_intensity=0.05)
final_noisy_array = add_background_noise(speckled_egg_gaussian, noise_mean=25, noise_std=7)

# Liste der Arrays und Titel
arrays = [
    circular_gaussian,
    elliptical_gaussian,
    egg_shaped_gaussian,
    speckled_egg_gaussian,
    final_noisy_array
]

titles = [
    "Circular Gaussian Beam",
    "Elliptical Gaussian Beam",
    "Egg-shaped Gaussian Beam",
    "Speckle Noise Added",
    "With Background Noise"
]

# Erstelle eine gemeinsame Figur
fig, axes = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)

# Gemeinsame Farbskala bestimmen
vmin, vmax = 0, 255

# Plots erstellen
for ax, array, title in zip(axes, arrays, titles):
    im = ax.imshow(array, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")  # Entferne Achsenbeschriftungen

# Gemeinsame Farbskala hinzufügen
cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
cbar.set_label("Intensity")

# Haupttitel der Figur
#plt.suptitle("Comparison of Gaussian Beam Disturbances", fontsize=14)

# Speichern der Abbildung als PNG
save_path = r"C:\Users\Janos\Documents\Masterarbeit\Presentation\gaussian_beam_disturbances.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot wurde gespeichert unter: {save_path}")

# Zeige die Figur
plt.show()