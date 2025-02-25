import numpy as np
import os
import tkinter as tk
from PIL import ImageDraw, ImageTk
from moduls.peak_find import find_peaks
from moduls.calc_3d import triangulate_3d
from moduls.lpc_indexing import analyze_coordinates
from file_io import *
from intensity_analysis import *
from gui.gui_helpers import log_message, display_image_with_peaks


def detect_peaks(gui):
    """Erkennt Peaks in beiden Bildern und aktualisiert die Anzeige."""

    # Hole den Wert des Window Size Sliders
    window_size_value = gui.slider_window_size.get()

    # Stelle sicher, dass der Wert ungerade ist
    if window_size_value % 2 == 0:
        window_size_value += 1  # Falls der Wert gerade ist, wird er auf die nächste ungerade Zahl gesetzt

    # Hole den Wert des Threshold Sliders
    threshold_value = gui.slider_threshold.get()

    if threshold_value == 0:
        threshold_value = None  # Wird automatisch in der Funktion `find_peaks` berechnet

    for i in range(2):
        if gui.images[i] is None:
            continue

        img_gray = np.array(gui.images[i].convert("L"))

        # Verwende den ungeraden Wert von window_size_value und threshold_value in der Funktion find_peaks
        peaks, calculated_threshold = find_peaks(img_gray, factor=gui.slider_factor.get(), threshold=threshold_value, window_size=window_size_value, gui=gui)

        # Wenn der threshold automatisch berechnet wurde, setze ihn auf den Slider
        if threshold_value is None:
            gui.slider_threshold.set(calculated_threshold)  # Setze den berechneten Wert auf den Slider

        img_width, img_height = gui.images[i].size
        gui.scale_x[i] = 500 / img_width
        gui.scale_y[i] = 500 / img_height
        gui.peaks[i] = peaks

        display_image_with_peaks(gui.canvas1 if i == 0 else gui.canvas2, 
                         gui.images[i], 
                         gui.peaks[i], 
                         gui.lpc[i],  # Hier lpc übergeben
                         i + 1, 
                         gui)
        log_message(gui.text_log, f"Found {len(peaks)} peaks in Image {i + 1}.")

    gui.btn_save["state"] = "normal"
    gui.btn_refresh["state"] = "normal"
    gui.btn_remove_mode["state"] = "normal"
    gui.btn_add_mode["state"] = "normal"

    gui.selected_peaks = [[], []]
    log_message(gui.text_log, "Selected points list cleared after detecting new peaks.")




def start_fit(gui):
    """Startet den Fit-Prozess für die aktuell geladenen Bilder und zeigt die LPC-Koordinaten an."""
    methode = gui.method_var.get()

    if not any(gui.image_paths):  # Mindestens ein Bild muss existieren
        log_message(gui.text_log, "Bitte mindestens ein Bild laden.")
        return

    for i, image_path in enumerate(gui.image_paths):
        if not image_path:  # Falls kein Bild geladen wurde, überspringen
            continue

        input_dir = os.path.dirname(gui.image_paths[i])
        filename_no_ext = os.path.splitext(os.path.basename(gui.image_paths[i]))[0]

        intensity_file = os.path.join(input_dir, f"{filename_no_ext}_intensity.npy")
        output_dir = construct_flex_op_path(input_dir, "output")
        peaks_file = os.path.join(output_dir, f"{filename_no_ext}_peaks.npy")

        if not os.path.exists(intensity_file):
            log_message(gui.text_log, f"Fehlende Intensitätsdatei für Bild {i+1}: {intensity_file}")
            continue
        if not os.path.exists(peaks_file):
            log_message(gui.text_log, f"Fehlende Peaks-Datei für Bild {i+1}: {peaks_file}")
            continue

        log_message(gui.text_log, f"Starte Fit für Bild {i+1} mit Methode {methode}...")

        intensity_data = np.load(intensity_file)
        peaks_data = np.load(peaks_file)

        bsc = brightness_subarray_creator(intensity_data, peaks_data)

        lpc_methods = {
            "center_of_mass": ("com", lambda data: compute_center_of_mass_with_uncertainty(data)),
            "gauss_fit": ("gf", lambda data: fit_gaussian_3d(data)),
            "skewed_gauss_fit": ("sgf", lambda data: fit_skewed_gaussian_3d(data)),
            "non_linear_center_of_mass": ("nlcom", lambda data: non_linear_center_of_mass(data)),
            "center_of_mass_with_threshold": ("comwt", lambda data: center_of_mass_with_threshold(data)),
            "circle_fit": ("cf", lambda data: circle_fitting_with_threshold(data))
        }

        if methode not in lpc_methods:
            log_message(gui.text_log, f"Unbekannte Methode: {methode}")
            return

        method_abbr, method_function = lpc_methods[methode]
        mean_values, uncertainties = method_function(bsc)

        lpc_coordinates = lpc_calc(mean_values, peaks_data)

        # **Sortierung mit analyze_coordinates**
        sorted_lpc_coordinates = analyze_coordinates(lpc_coordinates) # Nur x, y zurückgeben

        method_folder = os.path.join(output_dir, methode)
        os.makedirs(method_folder, exist_ok=True)

        lpc_output_filename = f"lpc_{filename_no_ext}_{method_abbr}.npy"
        lpc_output_path = os.path.join(method_folder, lpc_output_filename)
        np.save(lpc_output_path, sorted_lpc_coordinates)

        log_message(gui.text_log, f"LPC-Koordinaten gespeichert: {lpc_output_path}")

        # **LPC-Koordinaten speichern und direkt visualisieren**
        gui.lpc[i] = sorted_lpc_coordinates
        display_image_with_peaks(gui.canvas1 if i == 0 else gui.canvas2, gui.images[i], gui.peaks[i], gui.lpc[i], i + 1, gui)

    log_message(gui.text_log, "Fit abgeschlossen.")



def calculate_3d_structure(self):
    """Berechnet die 3D-Struktur aus den gefitteten LPC-Daten der beiden Bilder."""
    
    if len([p for p in self.image_paths if p]) != 2:  # Prüft, ob genau zwei Bilder geladen sind
        log_message(self.text_log, "Bitte genau zwei Bilder laden, um 3D zu berechnen.")
        return
    
    methode = self.method_var.get()
    if not methode:
        log_message(self.text_log, "Keine Fit-Methode ausgewählt.")
        return

    # Methode in die Kurzform umwandeln
    lpc_methods = {
        "center_of_mass": "com",
        "gauss_fit": "gf",
        "skewed_gauss_fit": "sgf",
        "non_linear_center_of_mass": "nlcom",
        "center_of_mass_with_threshold": "comwt",
        "circle_fit": "cf"
    }

    if methode not in lpc_methods:
        log_message(self.text_log, f"Unbekannte Methode: {methode}")
        return

    method_abbr = lpc_methods[methode]
    lpc_data = []
    
    for i, image_path in enumerate(self.image_paths):
        if not image_path:
            continue

        input_dir = os.path.dirname(image_path)
        filename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = construct_flex_op_path(input_dir, "output")
        method_folder = os.path.join(output_dir, methode)

        # Erstelle den korrekten Dateinamen mit der Kurzform der Methode
        lpc_file = os.path.join(method_folder, f"lpc_{filename_no_ext}_{method_abbr}.npy")
        
        if not os.path.exists(lpc_file):
            log_message(self.text_log, f"Fehlende LPC-Datei für Bild {i+1}: {lpc_file}")
            return
        
        # Laden der LPC-Daten und nur die ersten 3 Spalten (x, y, z) extrahieren
        lpc_coordinates = np.load(lpc_file)
        lpc_data.append(lpc_coordinates[:, :2])  # Wir nehmen nur die ersten 2 Spalten (x, y)

    if len(lpc_data) != 2:
        log_message(self.text_log, "Fehler: Es müssen genau zwei LPC-Dateien geladen werden.")
        return

    # **3D-Triangulation durchführen**
    log_message(self.text_log, "Starte 3D-Triangulation...")
    
    calibration_file = self.calibration_file.get() if self.calibration_file.get() else None
    points_3d = triangulate_3d(lpc_data[0], lpc_data[1], calibration_file)

    # **3D-Daten speichern**
    output_3d_file = os.path.join(output_dir, f"3d_points_{method_abbr}.npy")
    np.save(output_3d_file, points_3d)
    log_message(self.text_log, f"3D-Koordinaten gespeichert: {output_3d_file}")
    
    # **3D-Plot speichern**
    plot_filename = f"3d_plot_{method_abbr}.png"
    plot_3d_points(points_3d, path=output_dir, dateiname=plot_filename)

    # **3D-Plot erstellen und anzeigen**
    log_message(self.text_log, "Plotte 3D-Punkte...")
    self.display_3d_plot(points_3d)

    log_message(self.text_log, "3D-Berechnung abgeschlossen.")



from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def display_3d_plot(self, points_3d):
    """Zeigt den 3D-Plot im Canvas an."""
    fig = plot_3d_points(points_3d)  # Holt das Figure-Objekt

    if fig is None:
        print("Fehler: Keine gültige Figure erhalten!")  # Debugging-Hilfe
        return

    # **Alten Plot im Canvas entfernen**
    for widget in self.canvas3d.winfo_children():
        widget.destroy()

    # **Matplotlib-Canvas für Tkinter erstellen**
    canvas = FigureCanvasTkAgg(fig, master=self.canvas3d)
    canvas.draw()
    
    # **Canvas in das Tkinter-Widget einbetten**
    widget = canvas.get_tk_widget()
    widget.pack(fill=tk.BOTH, expand=True)











def save_results(gui):
    """Speichert die erkannten Peaks beider Bilder und die Intensitätswerte als .npy-Dateien."""
    for i in range(2):
        if len(gui.peaks[i]) == 0:
            log_message(gui.text_log, f"Keine Peaks für Bild {i + 1}.")
            continue

        if not gui.image_paths[i]:
            log_message(gui.text_log, f"Kein Input-Bild {i + 1} vorhanden, um den Speicherpfad zu bestimmen.")
            continue

        # **Pfad für Peaks (deine Methode)**
        input_dir = os.path.dirname(gui.image_paths[i])
        filename_no_ext = os.path.splitext(os.path.basename(gui.image_paths[i]))[0]
        output_path = construct_flex_op_path(input_dir, "output")
        peaks_file = os.path.join(output_path, f"{filename_no_ext}_peaks.npy")

        np.save(peaks_file, gui.peaks[i])
        log_message(gui.text_log, f"Gespeichert: {peaks_file}")
        log_message(gui.text_log, f"Shape der gespeicherten Peaks für Bild {i + 1}: {gui.peaks[i].shape}")

        # **Pfad für Intensitätswerte (meine Methode)**
        img_dir = os.path.dirname(gui.image_paths[i])
        intensity_file = os.path.join(img_dir, f"{filename_no_ext}_intensity.npy")

        img_gray = np.array(gui.images[i].convert("L"))
        np.save(intensity_file, img_gray)
        log_message(gui.text_log, f"Gespeichert: {intensity_file}")
        log_message(gui.text_log, f"Shape der gespeicherten Intensitätsdaten für Bild {i + 1}: {img_gray.shape}")

    log_message(gui.text_log, "Speicherung abgeschlossen.")


def refresh_peaks(gui):
    """Entfernt die nächstgelegenen Peaks zu den ausgewählten Punkten und setzt den Modus zurück."""
    if not any(gui.selected_peaks):  # Prüfen, ob überhaupt Punkte zum Entfernen markiert sind
        log_message(gui.text_log, "No points selected for removal.")
        return

    for i in range(2):  # Beide Bilder durchgehen
        if not gui.selected_peaks[i]:
            continue  # Falls für dieses Bild nichts markiert ist, skippen

        initial_count = len(gui.peaks[i]) if gui.peaks[i] is not None else 0
        if initial_count == 0:
            log_message(gui.text_log, f"No peaks to remove in image {i + 1}.")
            continue

        removed_peaks = []

        for selected_peak in gui.selected_peaks[i]:
            if len(gui.peaks[i]) == 0:  # Falls alle Peaks entfernt wurden, abbrechen
                break

            closest_index = -1
            min_dist = float("inf")

            for j, peak in enumerate(gui.peaks[i]):
                dist = np.sqrt((peak[0] - selected_peak[0]) ** 2 + (peak[1] - selected_peak[1]) ** 2)
                if dist < min_dist:
                    closest_index = j
                    min_dist = dist

            if closest_index != -1:
                removed_peaks.append(gui.peaks[i][closest_index])
                gui.peaks[i] = np.delete(gui.peaks[i], closest_index, axis=0)

        removed_count = len(removed_peaks)
        if removed_count > 0:
            log_message(gui.text_log, f"Removed {removed_count} peaks from image {i + 1}. {removed_peaks}")
        else:
            log_message(gui.text_log, f"No peaks removed in image {i + 1}.")

        # **Bild mit aktualisierten Peaks neu anzeigen**
        display_image_with_peaks(
            gui.canvas1 if i == 0 else gui.canvas2,
            gui.images[i],
            gui.peaks[i],
            gui.lpc[i],
            i + 1,
            gui
        )

        # **Zurücksetzen der Auswahl**
        gui.selected_peaks[i] = []
        log_message(gui.text_log, f"Cleared selected points for image {i + 1}.")

    # **Setze den Modus zurück auf "drag"**
    gui.set_mode("drag")
    log_message(gui.text_log, "Mode reset to 'drag' after refreshing peaks.")




def add_peak_manually(gui, x, y, image_index):
    """Fügt einen Peak manuell hinzu."""
    if image_index not in [0, 1]:
        log_message(gui.text_log, "Invalid image index for peak addition.")
        return

    if len(gui.peaks[image_index]) > 0:
        gui.peaks[image_index] = np.vstack([gui.peaks[image_index], [x, y]])
    else:
        gui.peaks[image_index] = np.array([[x, y]])

    log_message(gui.text_log, f"Added new peak to image {image_index + 1}: ({x}, {y})")

    # **Bild mit aktualisierten Peaks neu anzeigen**
    display_image_with_peaks(gui.canvas1 if image_index == 0 else gui.canvas2, gui.images[image_index], gui.peaks[image_index], gui.lpc[image_index], image_index + 1, gui)







