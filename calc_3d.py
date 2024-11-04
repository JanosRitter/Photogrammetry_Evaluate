import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_angles(indices):
    """
    Berechnet die X- und Y-Winkel basierend auf den Gitterindizes.
    Der Winkel für den Index (0,0) ist 0, positive Indices folgen 
    dem Muster: Index * (1/3°) - (1/6°) und negative Indices: 
    Index * (1/3°) + (1/6°).
    
    Parameters:
        - indices (np.ndarray): (n, 2) Array der Indizes (x_index, y_index).
        
    Returns:
        - angles_x (np.ndarray): Array der X-Achsen-Winkel in Bogenmaß.
        - angles_y (np.ndarray): Array der Y-Achsen-Winkel in Bogenmaß.
    """
    angles_x = np.zeros(indices.shape[0])
    angles_y = np.zeros(indices.shape[0])

    for i, (x_idx, y_idx) in enumerate(indices):
        if (x_idx, y_idx) == (0, 0):
            angles_x[i] = 0  # Winkel bei (0, 0) ist 0 Grad
        else:
            # Berechnung für X-Winkel
            angles_x[i] = (x_idx * (1/3) - (1/6)) if x_idx > 0 else (x_idx * (1/3) + (1/6))
        
        # Berechnung für Y-Winkel
        if (y_idx, y_idx) == (0, 0):
            angles_y[i] = 0  # Winkel bei (0, 0) ist 0 Grad
        else:
            angles_y[i] = (y_idx * (1/3) - (1/6)) if y_idx > 0 else (y_idx * (1/3) + (1/6))

    # Umrechnung von Grad in Bogenmaß
    angles_x = np.deg2rad(angles_x)
    angles_y = np.deg2rad(angles_y)

    return angles_x, angles_y

import numpy as np

def triangulate_3D(camera1_data, camera2_data, a=0.2, f=0.04, pixel_size=2.74e-6, resolution=(4096, 3000)):
    """
    Berechnet die 3D-Koordinaten der Punkte basierend auf den Bilddaten von zwei Kameras.
    
    Parameters:
        - camera1_data (np.ndarray): Array der Bildpunkte auf Kamera 1 in Pixelkoordinaten (n, 2).
        - camera2_data (np.ndarray): Array der Bildpunkte auf Kamera 2 in Pixelkoordinaten (n, 2).
        - a (float): Abstand der Kameras zur Ursprungsebene entlang der x-Achse.
        - f (float): Brennweite der Kameras.
        - pixel_size (float): Größe eines Pixels in Metern.
        - resolution (tuple): Auflösung der Kameras in Pixeln (Breite, Höhe).
    
    Returns:
        - np.ndarray: Array der rekonstruierten 3D-Koordinaten (n, 3).
    """
    n_points = camera1_data.shape[0]
    points_3D = np.zeros((n_points, 3))

    # Kamerapositionen
    cam1_pos = np.array([a, 0, 0])
    cam2_pos = np.array([-a, 0, 0])
    
    # Offset zur Zentrierung auf dem Bild
    offset_x = resolution[0] / 2
    offset_y = resolution[1] / 2
    
    for i in range(n_points):
        # Pixelkoordinaten in reale Bildkoordinaten umrechnen
        x1_img = (camera1_data[i, 0] - offset_x) * pixel_size
        y1_img = (camera1_data[i, 1] - offset_y) * pixel_size
        x2_img = (camera2_data[i, 0] - offset_x) * pixel_size
        y2_img = (camera2_data[i, 1] - offset_y) * pixel_size

        # Richtungsvektoren für die Geraden von den Kameras zu den Punkten auf der Bildebene
        dir_cam1 = np.array([x1_img, y1_img, f])
        dir_cam2 = np.array([x2_img, y2_img, f])

        # Berechne den nächsten Punkt zwischen den beiden Geraden
        points_3D[i] = find_closest_point(cam1_pos, dir_cam1, cam2_pos, dir_cam2)

    return points_3D


def find_closest_point(p1, d1, p2, d2):
    """
    Findet den nächsten Punkt zwischen zwei Geraden, die durch die Punkte p1 und p2 verlaufen
    und in die Richtungen d1 und d2 zeigen.
    
    Parameters:
        - p1, p2 (np.ndarray): Ausgangspunkte der beiden Geraden.
        - d1, d2 (np.ndarray): Richtungsvektoren der beiden Geraden.
    
    Returns:
        - np.ndarray: Der nächstgelegene Punkt zwischen den beiden Geraden.
    """
    d1 = d1 / np.linalg.norm(d1)  # Normiere Richtungsvektoren
    d2 = d2 / np.linalg.norm(d2)
    
    # Definiere das Gleichungssystem
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    # Löse für die Parameter s und t, die die kürzesten Distanzen beschreiben
    denom = a * c - b * b
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    # Berechne die Punkte auf den beiden Geraden
    point_on_line1 = p1 + s * d1
    point_on_line2 = p2 + t * d2

    # Der Mittelwert der beiden Punkte ergibt den bestmöglichen Schnittpunkt
    return (point_on_line1 + point_on_line2) / 2




