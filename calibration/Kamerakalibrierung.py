import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def analyze_brightness_histogram(image_files):
    """
    Analysiert die Helligkeitsverteilung der Bilder anhand des Histogramms.
    
    Parameter:
        image_files (list): Liste der Bilddateien.
    
    Rückgabewert:
        list: Liste der Helligkeitshistogramme.
    """
    histograms = []
    for img_file in image_files:
        frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Fehler beim Laden des Bildes: {img_file}")
            continue
        
        histogram = cv2.calcHist([frame], [0], None, [256], [0, 256])
        histograms.append(histogram)
        
        plt.plot(histogram, label=f'Bild: {os.path.basename(img_file)}')
    
    plt.title("Helligkeitshistogramme")
    plt.xlabel("Pixelwert")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.show()
    
    return histograms

def mono_calibrate(obj_points_all, img_points, img_size):
    """
    Führt die Monokalibrierung einer einzelnen Kamera durch.
    
    1. Intrinsische Kameraparameter
        - Brennweite
        - Lage des Hauptpunktes
        - Verzerrungskoeffizienten
    2. Ermittung der Kameramatrix und der Verzerrungskoeffizienten (radial und tangential)
        - Wahl eines Kalibrierungsmusters (hier: Schachbrett)
        - Konvertieren der verwendeten Bilder in schwarz-weiß
        - Suche nach den Ecken im Kalibrierungsmuster (Festlegung von chessboard_size und square_size)
        - Berechnung der gewünschten Werte
    3. Zuordnung der einzelnen Werte
        - Kameramatrix: 
            intrinsische Parameter der Kamera (Pixelkoordinaten)
            Fokuslänge (Pixel)
            Hauptpunktkoordinaten des optischen Zentrums (Pixelkoordinaten)
        - Verzerrungskoeffizienten:
            Verzerrung des Bildes durch die Linse
            Einheitslos
    
    Returns:
        ret: RMS-Fehler (Root Mean Square Error)
        mtx: Kameramatrix
        dist: Verzerrungskoeffizienten
        rvecs: Rotationsvektoren
        tvecs: Translationsvektoren
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_all, img_points, img_size, None, None)
    
    return ret, mtx, dist, rvecs, tvecs

def stereo_calibrate(obj_points_all, img_points1, img_points2, mtx1, dist1, mtx2, dist2, img_size):
    """
    Führt die Stereokalibrierung beider Kameras durch.
    
    Voraussetzungen für Stereokalibrierung:
        - feste und relative Positionierung der Kameras
        - gleiche Blickrichtung aus unterschiedlichen Positionen (parallel)
        - stabiler und stationärer Untergrund
    
    Rotationsmatrix:
        - Drehung von Kamera 2 relativ zu Kamera 1
        - einheitslos
    
    Translationsmatrix: 
        - Verschiebung von Kamera 2 relativ zu Kamera 1
        - mm (gleiche physikalische Einheit wie 3D-Punkte)
        
    Essentielle Matrix:
        - Beschreibung der euklidischen Transformation von Kamera 2 nach Kamera 1
        - Sensorkoordinaten
        - kalibrierter Fall
        
    Fundamentalmatrix: 
        - Beschreibung der geometrischen Beziehungen der Kameras im Stereoaufbau
        - mathematische Darstellung der Epipolargeometrie
        - Bildkoordinaten
    
    Returns:
        ret: RMS-Fehler (Root Mean Square Error)
        mtx1, dist1: Kalibrierungsergebnisse der ersten Kamera
        mtx2, dist2: Kalibrierungsergebnisse der zweiten Kamera
        R: Rotation zwischen den Kameras
        T: Translation zwischen den Kameras
        E: Essenzielle Matrix
        F: Fundamentalmatrix
    """
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points_all, img_points1, img_points2,
        mtx1, dist1, mtx2, dist2,
        img_size,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return ret, mtx1, dist1, mtx2, dist2, R, T, E, F

#Speicherung der Kalibrierungsdaten
def save_calibration(mtx, dist, R, T, camera_name):
    """
    Speicherung der Kalibrierungsdaten als .npz-Datei.

    Parameter:
        mtx (ndarray): Kameramatrix.
        dist (ndarray): Verzerrungskoeffizienten.
        R (ndarray): Rotationsmatrix zwischen den Kameras.
        T (ndarray): Translationsvektor zwischen den Kameras.
        camera_name (str): Name der Kamera (z.B. Kamera_1 oder Kamera_2).
    
    Rückgabewert:
        str: absoluter Pfad der gespeicherten Datei.
    """
    #Zielverzeichnis für die Kalibrierungsdaten
    output_dir = os.path.join(os.getcwd(), "Kalibrierungsdaten")
    os.makedirs(output_dir, exist_ok=True) #Erstellen eines Verzeichnisses falls keines existiert
    
    #Absoluter Pfad zur Datei
    file_name = os.path.join(output_dir, f'{camera_name}_Kalibrierung.npz')

    try: 
        #Speicherung der Kalibrierungsdaten als .npz-Datei
        np.savez(file_name, mtx=mtx, dist=dist, R=R, T=T) 
        print(f"Kalibrierungsdaten für {camera_name} gespeichert unter {file_name}.")
        return file_name
    except Exception as e:
        #Fehlerbehandlung bei Problem mit dem Speichern
        print(f"Fehler beim Speichern der Kalibierungsdaten: {e}")
        return None

#Visualisierung des Kamera-Setups
def visualize_camera_setup(R, T):
    """
    Visualisiert die Kamerapositionen und Achsen in 3D.

    Parameter:
        R (ndarray): Rotationsmatrix.
        T (ndarray): Translationsvektor.
    """
    #Kamera 1
    origin = np.array([0, 0, 0]) #Position der Kamera
    x_axis = np.array([1, 0, 0]) #horizontale Bildrichtung
    y_axis = np.array([0, 1, 0]) #vertikale Bildrichtung
    z_axis = np.array([0, 0, 1]) #Sichtachse der Kamera
    print(f"Position der Kamera 1: {origin}")

    #Kamera 2
    cam2_origin = T.flatten()
    cam2_x_axis = R @ x_axis
    cam2_y_axis = R @ y_axis
    cam2_z_axis = R @ z_axis
    print(f"Position der Kamera 2: {cam2_origin}")

    #Erstellen der 3D-Achsen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Achsen von Kamera 1
    ax.quiver(*origin, *x_axis, color='r', length=1, label="Kamera 1 X-Achse")
    ax.quiver(*origin, *y_axis, color='g', length=1, label="Kamera 1 Y-Achse")
    ax.quiver(*origin, *z_axis, color='b', length=1, label="Kamera 1 Z-Achse")

    #Achsen von Kamera 2
    ax.quiver(*cam2_origin, *cam2_x_axis, color='r', linestyle='--', length=1, label="Kamera 2 X-Achse")
    ax.quiver(*cam2_origin, *cam2_y_axis, color='g', linestyle='--', length=1, label="Kamera 2 Y-Achse")
    ax.quiver(*cam2_origin, *cam2_z_axis, color='b', linestyle='--', length=1, label="Kamera 2 Z-Achse")
    
    #Markierung der Kamerapositionen als Punkte
    ax.scatter(*origin, color='black', s=50, label="Kamera 1 Position", marker='o')
    ax.scatter(*cam2_origin, color='orange', s=50, label="Kamera 2 Position", marker='o')
    
    #Visualisierung der Kamera-Position (T) im 3D-Raum
    ax.scatter(T[0], T[1], T[2], color='r', s=100, label='Kamera-Position')

    #Angenommene Richtung, in die die Kamera zeigt (durch Rotation R)
    camera_direction = R @ np.array([0, 0, 1])  # Annahme: Kamera zeigt in Richtung der Z-Achse
    ax.quiver(T[0], T[1], T[2], camera_direction[0], camera_direction[1], camera_direction[2], length=0.5, color='b', label='Kamera-Richtung')

    #Plot-Einstellungen
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Kamera-Setup in 3D")
    
    plt.legend()
    plt.show()
    
# =============================================================================
# Fehlerberechnung und -analyse, Validierung der Ergebnisse:
#     - Berechnung des Epipolarfehlers
#     - Analyse der Reprojektionspunkte
#     - Berechnung des Reprojektionsfehlers
#     - Validierung der intrinsischen Kameraparameter
#     - Validierung der extrinsischen Kameraparameter
#     - Konsistenzprüfung der Kalibrierungsergebnisse
# =============================================================================

#Berechnung des Epipolarfehlers
def compute_epipolar_error(F, img_points1, img_points2):
    """
    Berechnet den Epipolarfehler zwischen Punktpaaren aus zwei Bildern.
    
    Epipolarfehler:
        - Maß, wie gut die Epipolargeometrie zwischen zwei Bildern/Kamerasystemen eingehalten wird
        - Genauigkeitsangabe der Korresponzenzpunkte auf den berechneten Epipolarlinien
        - niedrig bei übereinstimmender Kameraposition, Orientierung und Verzerrungsmodelle
        - hoch bei Problemen mit der Kalibrierung durch schlechte Korrespondenzen, Verzerrungen oder Kamerabewegungen
        
    Parameter:
        F (ndarray): Fundamentalmatrix.
        imgpoints1 (list): Liste von 2D-Punkten im Bild 1.
        imgpoints2 (list): Liste von 2D-Punkten im Bild 2.

    Rückgabewert:
        float: Mittlerer Epipolarfehler.
    """
    total_error = 0
    num_points = 0

    for p1, p2 in zip(img_points1, img_points2):
        for pt1, pt2 in zip(p1, p2):
            # Umwandlung in homogene Koordinaten
            pt1_h = np.array([pt1[0][0], pt1[0][1], 1]).reshape(3, 1)
            pt2_h = np.array([pt2[0][0], pt2[0][1], 1]).reshape(3, 1)

            # Berechnung des Epipolarfehlers
            error1 = np.abs(pt2_h.T @ F @ pt1_h)  # Punkt in Bild 1 auf Epipolare in Bild 2
            error2 = np.abs(pt1_h.T @ F.T @ pt2_h)  # Punkt in Bild 2 auf Epipolare in Bild 1

            total_error += (error1[0][0] + error2[0][0])
            num_points += 1

    return total_error / num_points if num_points > 0 else float('inf')

    #Berechnung des mittleren Epipolarfehlers
    epipolar_error = compute_epipolar_error(F, img_points1, img_points2)
    
    return epipolar_error
        
def compute_reprojection_error(obj_points_all, img_points1, img_points2,
                               rvecs1, tvecs1, rvecs2, tvecs2,
                               mtx1, dist1, mtx2, dist2):
    """
    Berechnet den mittleren Reprojektionsfehler für beide Kameras.
    
    Reprojektionsfehler:
        - geometrischer Fehler
        - Messung des Abstandes zwischen einem projizierten und einem gemessenen Punkt
        - Angabe, wie gut die 3D-Punkte auf die Bildpunkte zurückprojiziert werden können
        - hoher Wert bei Fehlern in der Kalibrierung (Ziel: unter 1 Pixel)
        - Einheit in Pixeln
    
    Returns:
        mean_error: Mittlerer Fehler über alle Bilder und Kameras
    """
    total_error = 0
    total_points = 0

    for i in range(len(obj_points_all)):
        # Projiziere die 3D-Punkte in beide Bildräume
        img_points_proj1, _ = cv2.projectPoints(obj_points_all[i], rvecs1[i], tvecs1[i], mtx1, dist1)
        img_points_proj2, _ = cv2.projectPoints(obj_points_all[i], rvecs2[i], tvecs2[i], mtx2, dist2)
        
        # Berechne den Fehler pro Bild
        error1 = cv2.norm(img_points1[i], img_points_proj1, cv2.NORM_L2) / len(img_points_proj1)
        error2 = cv2.norm(img_points2[i], img_points_proj2, cv2.NORM_L2) / len(img_points_proj2)
        
        total_error += (error1 + error2)
        total_points += len(obj_points_all[i])

    mean_error = total_error / total_points
    
    return mean_error

#Validierung der intrinsischen Kameraparameter
def validate_intrinsic_parameter(mtx, dist, img_size):
    """
    Validierung der intrinsischen Kameraparameter auf physikalisch plausible Werte.

    Parameter:
        mtx1, mtx2 (ndarray): Kameramatrizen der beiden Kameras.
        dist1, dist2 (ndarray): Verzerrungskoeffizienten der beiden Kameras.
        img_size (tuple): Bildgröße als (Breite, Höhe).

    Rückgabewert:
        bool: True, wenn die Parameter plausibel sind, sonst False.
    """
    if mtx is None:
        print("Keine gültige Kameramatrix zur Validierung vorhanden.")
        return False

    fx, fy = mtx[0, 0], mtx[1, 1]  # Brennweite
    cx, cy = mtx[0, 2], mtx[1, 2]  # Lage des Hauptpunktes (optisches Zentrum)
    width, height = img_size

    errors = []  # Liste, um alle Fehler zu speichern

    # Pixelgröße in mm, basierend auf den Spezifikationen des Sensors
    pixel_size_mm = 0.00274  # Pixelgröße in mm (2.74 µm)
    
    # Brennweite in Pixeln umrechnen (fx, fy in Pixel)
    fx_pixels = 50 / pixel_size_mm  # Brennweite fx in Pixel
    fy_pixels = 50 / pixel_size_mm  # Brennweite fy in Pixel
    
    # Wertebereiche basierend auf den Spezifikationen der Kamera und des Objektivs
    fx_min, fx_max = fx_pixels * 0.8, fx_pixels * 1.2  # ±20% der Brennweite
    fy_min, fy_max = fy_pixels * 0.8, fy_pixels * 1.2  # ±20% der Brennweite
    cx_min, cx_max = 0.4 * width, 1.6 * width  # ±20% von Bildmitte
    cy_min, cy_max = 0.4 * height, 1.6 * height  # ±20% von Bildmitte
    k_min, k_max = -0.1, 0.1  # Radialverzerrung
    p_min, p_max = -0.1, 0.1  # Tangentialverzerrung
    
    # Prüfung der Brennweiten
    if not (fx_min <= fx <= fx_max):
        errors.append(f"Brennweite fx ({fx}) liegt außerhalb des Bereichs [{fx_min:.1f}, {fx_max:.1f}].")
    if not (fy_min <= fy <= fy_max):
        errors.append(f"Brennweite fy ({fy}) liegt außerhalb des Bereichs [{fy_min:.1f}, {fy_max:.1f}].")

    # Prüfung des optischen Zentrums
    if not (cx_min <= cx <= cx_max):
        errors.append(f"Optisches Zentrum cx ({cx}) liegt außerhalb des Bereichs [{cx_min:.1f}, {cx_max:.1f}].")
    if not (cy_min <= cy <= cy_max):
        errors.append(f"Optisches Zentrum cy ({cy}) liegt außerhalb des Bereichs [{cy_min:.1f}, {cy_max:.1f}].")

    # Prüfung der Verzerrungskoeffizienten
    if len(dist) >= 3:
        if not (k_min <= dist[0] <= k_max):
            errors.append(f"Radialverzerrung k1 ({dist[0]}) liegt außerhalb des Bereichs [{k_min}, {k_max}].")
        if not (k_min <= dist[1] <= k_max):
            errors.append(f"Radialverzerrung k2 ({dist[1]}) liegt außerhalb des Bereichs [{k_min}, {k_max}].")
        if not (k_min <= dist[4] <= k_max):
            errors.append(f"Radialverzerrung k3 ({dist[4]}) liegt außerhalb des Bereichs [{k_min}, {k_max}].")
    if len(dist) >= 5:
        if not (p_min <= dist[2] <= p_max):
            errors.append(f"Tangentialverzerrung p1 ({dist[2]}) liegt außerhalb des Bereichs [{p_min}, {p_max}].")
        if not (p_min <= dist[3] <= p_max):
            errors.append(f"Tangentialverzerrung p2 ({dist[3]}) liegt außerhalb des Bereichs [{p_min}, {p_max}].")

    # Fehler zusammenfassen und Ergebnis zurückgeben
    if errors:
        print("Validierung der intrinsischen Kameraparameter fehlgeschlagen:")
        for error in errors:
            print(f" - {error}")
        return False

    print("Alle intrinsischen Kameraparameter sind plausibel.")
    return True

#Validierung der extrinsischen Kameraparameter
def validate_extrinsic_parameters(R, T, threshold_angle=5.0, threshold_translation=100.0):
    """
    Validierung der extrinsischen Kameraparameter auf physikalisch plausible Werte.

    Parameter:
        R (ndarray): Rotationsmatrix (3x3).
        T (ndarray): Translationsvektor (3x1).
        threshold_angle (float): Maximal erlaubte Abweichung der Rotation (in Grad).
        threshold_translation (float): Maximal erlaubte Abweichung der Translation (in mm).

    Rückgabewert:
        bool: True, wenn die Parameter plausibel sind, sonst False.
    """
    errors = []  # Liste für Fehler
    results = []  # Liste für Ergebnisse

    # 1. Prüfung: Ist R eine gültige Rotationsmatrix?
    det_R = np.linalg.det(R)
    if not (0.99 <= det_R <= 1.01):
        errors.append(f"Die Determinante der Rotationsmatrix ist {det_R:.3f}, sollte jedoch nahe 1 liegen.")
    else:
        results.append(f"Die Determinante der Rotationsmatrix ist {det_R:.3f} (gültig).")

    orthogonality_error = np.linalg.norm(np.dot(R.T, R) - np.eye(3))
    if orthogonality_error > 1e-6:
        errors.append(f"Die Rotationsmatrix ist nicht orthogonal (Fehler: {orthogonality_error:.6e}).")
    else:
        results.append(f"Die Rotationsmatrix ist orthogonal (Fehler: {orthogonality_error:.6e}).")

    # 2. Prüfung: Rotationswinkel
    rotation_angle = np.arccos((np.trace(R) - 1) / 2) * (180 / np.pi)  # Winkel in Grad
    if rotation_angle > threshold_angle:
        errors.append(f"Der Rotationswinkel ({rotation_angle:.2f}°) überschreitet den Schwellenwert von {threshold_angle}°.")
    else:
        results.append(f"Der Rotationswinkel beträgt {rotation_angle:.2f}° (innerhalb des Schwellenwerts).")

    # 3. Prüfung: Translation
    translation_norm = np.linalg.norm(T)
    if translation_norm > threshold_translation:
        errors.append(f"Die Translation ({translation_norm:.2f} mm) überschreitet den Schwellenwert von {threshold_translation} mm.")
    else:
        results.append(f"Die Translation beträgt {translation_norm:.2f} mm (innerhalb des Schwellenwerts).")

    # 4. Ausgabe der Ergebnisse
    print("Ergebnisse der Validierung der extrinsischen Kameraparameter:")
    for result in results:
        print(f" - {result}")

    if errors:
        print("Fehler bei der Validierung der extrinsischen Kameraparameter:")
        for error in errors:
            print(f" - {error}")
        return False

    print("Alle extrinsischen Kameraparameter sind plausibel.")
    return True

#Konsistenzprüfung der Kalibierungsergebnisse
def check_consistency(R, T, reprojection_error, epipolar_error, reprojection_threshold=2.0, epipolar_threshold=1.0):
    """
    Überprüft die Konsistenz der Kalibrierungsergebnisse.
    Kein Abbrechen des Codes beim Auftreten von Fehlern.
    
    Parameter:
        R (ndarray): Rotationsmatrix.
        T (ndarray): Translationsvektor.
        reprojection_error (float): Reprojektionsfehler.
        epipolar_error (float): Epipolarfehler.
        reprojection_threshold (float): Akzeptanzschwelle für Reprojektionsfehler.
        epipolar_threshold (float): Akzeptanzschwelle für Epipolarfehler.
    
    Rückgabewert:
        bool: True, wenn die Ergebnisse konsistent sind, sonst False.
    """
    #Überprüfung der Rotationsmatrix auf Gültigkeit
    is_rotation_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-3)
    is_rotation_determinant_one = np.isclose(np.linalg.det(R), 1.0, atol=1e-3)
    consistent_rotation = is_rotation_orthogonal and is_rotation_determinant_one
    
    #Überprüfung der Plausibilität des Translatiionsvektors (nicht zu klein)
    consistent_translation = np.linalg.norm(T) > 1e-6 
    
    #Überprüfung der Fehlerwerte
    reprojection_within_threshold = reprojection_error < reprojection_threshold
    epipolar_within_threshold = epipolar_error < epipolar_threshold
    
    #Konsistenzprüfung
    all_consistent = (
        consistent_rotation and
        consistent_translation and
        reprojection_within_threshold and
        epipolar_within_threshold
    )
    
    #Ausgabe
    if all_consistent:
        print("Konsistenzprüfung der Kalibrierungsergebnisse bestanden.")
    else:
        print("Konsistenzprüfung der Kalibrierungsergebnisse fehlgeschlagen.")
        if not consistent_rotation:
            print(" - Rotationsmatrix ist inkonsistent.")
            if not is_rotation_orthogonal:
                print("   - R ist nicht orthogonal.")
            if not is_rotation_determinant_one:
                print("   - Determinante von R ist nicht 1.")
        if not consistent_translation:
            print(" - Translationsvektor ist inkonsistent (möglicherweise zu klein).")
        if not reprojection_within_threshold:
            print(f" - Reprojektionsfehler ({reprojection_error:.4f}) über Schwelle ({reprojection_threshold}).")
        if not epipolar_within_threshold:
            print(f" - Epipolarfehler ({epipolar_error:.4f}) über Schwelle ({epipolar_threshold}).")
    
    return all_consistent

def main():
    
    #Festlegung der Parameter für das verwendete Schachbrettmuster
    chessboard_size = (5, 8)  #Anzahl der inneren Ecken (Spalten, Reihen)
    square_size = 25.0        #Größe eines Schachbrettquadrats (in mm) 
    
    #Erzeugung der 3D-Koordinaten für die Ecken des Schachbrettmusters
# =============================================================================
#     Definition des erzeugten Koordinatensystems:
#         Ursprung: erste Ecke des Schachbretts
#         x-Achse: horizontal entlang der Schachbrettfelder
#         y-Achse: vertikal entlang der Schachbrettfelder
#         z-Achse: orthogonal zur Schachbrettebene (von der Kamera weg)
# =============================================================================
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32) #Erzeugung eines Arrays mit Nullen für die 3D-Koordinaten
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) #Erzeugung der 2D-Koordinaten für die Ecken in x- und y-Ebene
    objp *= square_size #Anpassung (Multiplikation) mit der Größe der Schachbrettquadrate (Skalierung in mm)

    #Listen der Objekt- und Bildpunkte der Kameras
    obj_points_all = []
    img_points_camera1 = []
    img_points_camera2 = []

    #Dateipfade (bei Bedarf anpassen)
    images_camera1_path = r'C:/Users/LMI/Pictures/Kalibrierungsbilder_Kamera_12/*.png'
    images_camera2_path = r'C:/Users/LMI/Pictures/Kalibrierungsbilder_Kamera_11/*.png'
    
    images_camera1 = glob.glob(images_camera1_path)
    images_camera2 = glob.glob(images_camera2_path)
    
# =============================================================================
#     Anforderung an die aufgenommenen Bilder für eine erfolgreiche Kalibrierung:
#         1. Hardwareausrichtung:
#             - feste Montage der Kameras (konstante Position und Orientierung)
#             - parallele Ausrichtung (Basislinie) der Kameras 
#             - Aufnahme des gleichen Kalibrierungsmusters
#         2. Kalibrierungsmuster:
#             - scharf und gut erkennbar
#             - ausreichende Anzahl an Kontrollpunkten
#         3. Aufnahmeszenarien:
#             - vielfältige Blickwinkel, Positionen und Entfernungen
#             - Kalibrierungsmuster in beiden Bildern vollständig erkennbar
#             - gleichmäßige Beleuchtung
#         4. Bildqualität:
#             - ausreichende Bildschärfe
#             - Vermeidung von Bildrauschen
#             - höchstmögliche Auflösung i.V.m. mit einer angemessenen Dateigröße
#         5. Anzahl der Bilder: min 10
#         6. Vermeidung von Verzerrungen durch angemessenen Abstand
#         7. Konstante Kameraeinstellungen (Deaktivieren von Autofokus und AutoROI)
# =============================================================================

    #Überprüfung, ob beide Kameras die gleiche Anzahl an Bildern haben
    assert len(images_camera1) == len(images_camera2), "Die Anzahl der Bilder für beide Kameras muss übereinstimmen."

    if not images_camera1:
        print("Keine Bilder für Kamera 1 gefunden. Bitte den Pfad überprüfen.")
    if not images_camera2:
        print("Keine Bilder für Kamera 2 gefunden. Bitte den Pfad überprüfen.")

    img_size = None
    
    #Histogramme der Helligkeit der aufgenommenen Bilder
    print("Helligkeitshistogramme der Bilder.")
    analyze_brightness_histogram(images_camera1)
    analyze_brightness_histogram(images_camera2)

    #Durchlaufen der Bildpaare und Suche nach dem Schachbrettmuster
    for img1_path, img2_path in zip(images_camera1, images_camera2):
        img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1_gray is None or img2_gray is None:
            print(f"Fehler beim Laden eines der Bilder: {img1_path} oder {img2_path}")
            continue

        #Festlegung der Bildgröße
        if img_size is None:
            img_size = img1_gray.shape[::-1]  #(Breite, Höhe)

        ret1, corners1 = cv2.findChessboardCorners(img1_gray, chessboard_size)
        ret2, corners2 = cv2.findChessboardCorners(img2_gray, chessboard_size)
        
        if ret1 and ret2:
            # Verfeinerung der erkannten Ecken
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners1 = cv2.cornerSubPix(img1_gray, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(img2_gray, corners2, (11, 11), (-1, -1), criteria)
            
            img_points_camera1.append(corners1)
            img_points_camera2.append(corners2)
            obj_points_all.append(objp)

    #Monokalibrierung der einzelnen Kameras
    
    #Durchführung der Monokalibrierung für Kamera 1
    print("Durchführung der Monokalibrierung für Kamera 1.")
    ret1, mtx1, dist1, rvecs1, tvecs1 = mono_calibrate(obj_points_all, img_points_camera1, img_size)
    print("Kameramatrix für Kamera 1:\n", mtx1)
    print("Verzerrungskoeffizienten für Kamera 1:\n", dist1)
    print("RMS-Fehler für Kamera 1:\n", ret1)
    if not obj_points_all or not img_points_camera1:
        print("Fehler: Kamera 1 konnte nicht kalibriert werden.")
        return
    print("Monokalibrierung für Kamera 1 abgeschlossen.")
    
    #Durchführung der Monokalibrierung für Kamera 2
    print("Durchführung der Monokalibrierung für Kamera 2.")
    ret2, mtx2, dist2, rvecs2, tvecs2 = mono_calibrate(obj_points_all, img_points_camera2, img_size)
    print("Kameramatrix für Kamera 2:\n", mtx2)
    print("Verzerrungskoeffizienten für Kamera 2:\n", dist2)
    print("RMS-Fehler für Kamera 2:\n", ret2)
    if not obj_points_all or not img_points_camera2:
        print("Fehler: Kamera 2 konnte nicht kalibriert werden.")
        return
    print("Monokalibrierung für Kamera 2 abgeschlossen.")

    #Stereokalibrierung 
    
    #Sicherstellung, dass Kalibrierungspunkte für beide Kameras vorhanden sind
    if not obj_points_all or not img_points_camera1 or not obj_points_all or not img_points_camera2:
        print("Nicht genügend Kalibrierungsdaten für die Stereo-Kalibrierung.")
        return
    
    #Ausgabe der Anzahl der erkannten Objekt- und Bildpunkte
    print(f"Anzahl der 3D-Punkte Kamera 1: {len(obj_points_all)}")
    print(f"Anzahl der 2D-Punkte Kamera 1: {len(img_points_camera1)}")
    if len(obj_points_all) == len(img_points_camera1):
        print ("Die Anzahl der Bild- und Objektpunkte für Kamera 1 stimmt überein.")
        if not len(obj_points_all) == len(img_points_camera1):
            print("Achtung: Die Anzahl der Bild- und Objektpunkte für Kamera 1 stimmt nicht überein.")
    print(f"Anzahl der 3D-Punkte Kamera 2: {len(obj_points_all)}")
    print(f"Anzahl der 2D-Punkte Kamera 2: {len(img_points_camera2)}")
    if len(obj_points_all) == len(img_points_camera2):
        print ("Die Anzahl der Bild- und Objektpunkte für Kamera 2 stimmt überein.")
        if not len(obj_points_all) == len(img_points_camera2):
            print("Achtung: Die Anzahl der Bild- und Objektpunkte für Kamera 2 stimmt nicht überein.")
            
    #DEBUGGING: Ausgabe der erkannten Objekt- und Bildpunkte
    #print("DEBUGGING: 3D-Objektpunkte:\n", obj_points_all[0][:])
    #print("DEBUGGING: 2D-Bildpunkte für Kamera 1:\n", img_points_camera1[0][:])
    #print("DEBUGGING: 2D-Bildpunkte für Kamera 2:\n", img_points_camera2[0][:])
    
    #Durchführung der Stereokalibrierung
    print("Durchführung der Stereokalibrierung.")
    ret_stereo, mtx1, dist1, mtx2, dist2, R, T, E, F = stereo_calibrate(
        obj_points_all, img_points_camera1, img_points_camera2,
        mtx1, dist1, mtx2, dist2, img_size)
    print(f"Rotationsmatrix (R):\n{R}")
    print(f"Translationsmatrix (T):\n{T}") 
    print(f"Essentielle Matrix (E):\n{E}") 
    print(f"Fundamentalmatrix (F):\n{F}")
    print("RMS-Fehler für Stereokalibrierung:\n", ret_stereo)    
    print("Stereokalibrierung abgeschlossen.")

    #Speichern der Kalibrierungsergebnisse
    print("Speichern der Kalibrierungsergebnisse.")
    save_calibration(mtx1, dist1, R, T, "Kamera_1")
    save_calibration(mtx2, dist2, R, T, "Kamera_2")
    
    #Visualisierung
    print("Kamerapositionen (in mm).")
    visualize_camera_setup(R, T)
    
    print("Kamerakalibrierung abgeschlossen.")
    print("Beginn der Fehlerberechnung und -analyse.")
    
    #Berechnung des Epipolarfehlers
    epipolar_error = compute_epipolar_error(F, img_points_camera1, img_points_camera2)
    print(f"Mittlerer Epipolarfehler: {epipolar_error}")

    #Berechnung des Reprojektionsfehlers
    mean_error = compute_reprojection_error(
        obj_points_all, img_points_camera1, img_points_camera2,
        rvecs1, tvecs1, rvecs2, tvecs2,
        mtx1, dist1, mtx2, dist2)
    print(f"Mittlerer Reprojektionsfehler: {mean_error}")
    
    #Überprüfung der intrinschen Kameraparameter für Kamera 1
    print("Validierung der intrinsischen Parameter für Kamera 1.")
    validate_intrinsic_parameter(mtx1, dist1, img_size)

    #Überprüfung der intrinschen Kameraparameter für Kamera 2
    print("Validierung der intrinsischen Parameter für Kamera 2.")
    validate_intrinsic_parameter(mtx2, dist2, img_size)
    
    #Überprüfung der extrinsischen Kameraparameter
    validate_extrinsic_parameters(R, T)
    
    #Konsistenzprüfung der Kalibrierungsergebnisse
    check_consistency(R, T, mean_error, epipolar_error)
    
    print("Fehlerberechnung und -analyse abgeschlossen.")
    print("Programm abgeschlossen.")
    
if __name__ == "__main__":
    main()