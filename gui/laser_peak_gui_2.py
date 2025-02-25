import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import cv2
from moduls.peak_find import find_peaks
from file_io import construct_flex_op_path


class LaserPointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Point Detector")
    
        # Buttons
        self.btn_open = tk.Button(root, text="Open Image 1", command=lambda: self.load_image(1))
        self.btn_open.pack()
    
        self.btn_open2 = tk.Button(root, text="Open Image 2", command=lambda: self.load_image(2))
        self.btn_open2.pack()
    
        self.btn_run = tk.Button(root, text="Run", command=self.detect_peaks, state=tk.DISABLED)
        self.btn_run.pack()
    
        self.btn_save = tk.Button(root, text="Save", command=self.save_results, state=tk.DISABLED)
        self.btn_save.pack()
    
        self.btn_refresh = tk.Button(root, text="Refresh", command=self.refresh_peaks, state=tk.DISABLED)
        self.btn_refresh.pack()
    
        # Neue Buttons für Modi
        self.btn_remove_mode = tk.Button(root, text="Remove Points", command=lambda: self.set_mode("remove"), state=tk.DISABLED)
        self.btn_remove_mode.pack()
    
        self.btn_add_mode = tk.Button(root, text="Add Points", command=lambda: self.set_mode("add"), state=tk.DISABLED)
        self.btn_add_mode.pack()
    
        # Canvas für Bilder (zwei nebeneinander)
        self.canvas1 = tk.Canvas(root)
        self.canvas1.pack(side=tk.LEFT)
        self.canvas1.bind("<Button-1>", lambda event: self.on_canvas_click(event, 1))
    
        self.canvas2 = tk.Canvas(root)
        self.canvas2.pack(side=tk.RIGHT)
        self.canvas2.bind("<Button-1>", lambda event: self.on_canvas_click(event, 2))
    
        # Textfeld für Logs
        self.text_log = tk.Text(root, height=10, width=50, wrap=tk.WORD)
        self.text_log.pack()
    
        # Bilddaten
        self.image_paths = [None, None]
        self.images = [None, None]
        self.processed_images = [None, None]
        self.peaks = [[], []]  # Für jedes Bild eine Liste von Peaks
        self.selected_peaks = [[], []]  # Separate Liste für "zu entfernende" Peaks
        self.scale_x = [1, 1]
        self.scale_y = [1, 1]
        self.mode = "remove"  # Standardmodus ist "remove"


    def log_message(self, message):
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.yview(tk.END)

    def set_mode(self, mode):
        """Ändert den aktuellen Modus."""
        self.mode = mode
        self.log_message(f"Mode changed to: {mode.upper()}")

    def load_image(self, image_index):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.bmp")])
        if not file_path:
            return
    
        self.image_paths[image_index - 1] = file_path
        self.images[image_index - 1] = Image.open(file_path)
        self.display_image_with_peaks(image_index)  # Bild anzeigen
        self.btn_run["state"] = tk.NORMAL


    def detect_peaks(self):
        for i in range(2):  # Für beide Bilder (Index 0 und 1)
            if self.images[i] is None:
                continue
    
            img_gray = np.array(self.images[i].convert("L"))
    
            peaks = find_peaks(img_gray, factor=10, threshold=None, window_size=9)
    
            img_width, img_height = self.images[i].size
            self.scale_x[i] = 500 / img_width  # Skaliert für beide Bilder
            self.scale_y[i] = 500 / img_height
    
            self.peaks[i] = peaks
    
            self.display_image_with_peaks(i + 1)  # Bild aktualisieren
            self.log_message(f"Found {len(self.peaks[i])} peaks in Image {i + 1}.")
    
        self.btn_save["state"] = tk.NORMAL
        self.btn_refresh["state"] = tk.NORMAL
        self.btn_remove_mode["state"] = tk.NORMAL
        self.btn_add_mode["state"] = tk.NORMAL
    
        # Vor der erneuten Erkennung löschen wir die bisherigen ausgewählten Peaks
        self.selected_peaks = [[], []]
        self.log_message("Selected points list cleared after detecting new peaks.")



    def save_results(self):
        """Speichert die Peaks beider Bilder im automatisch erkannten Output-Ordner."""
        for i in range(2):  # Für beide Bilder
            if self.peaks[i] is None or len(self.peaks[i]) == 0:
                self.log_message(f"Keine Peaks zum Speichern für Bild {i + 1}.")
                continue
    
            if not self.image_paths[i]:
                self.log_message(f"Kein Input-Bild {i + 1} vorhanden, um den Speicherpfad zu bestimmen.")
                continue
    
            # Extrahiere den übergeordneten Ordner und Dateinamen
            input_dir = os.path.dirname(self.image_paths[i])
            filename = os.path.basename(self.image_paths[i])
            filename_no_ext, _ = os.path.splitext(filename)
    
            # Generiere den Output-Ordner
            output_path = construct_flex_op_path(input_dir, "output")
    
            # Erstelle den Dateipfad für das jeweilige Bild
            file_path = os.path.join(output_path, f"peaks_{i+1}_{filename_no_ext}.npy")
    
            # Speichern der Peaks
            np.save(file_path, self.peaks[i])
    
            # Ausgabe der Form der gespeicherten Peaks
            self.log_message(f"Speicherung abgeschlossen für Bild {i + 1}: {file_path}")
            self.log_message(f"Shape der gespeicherten Peaks für Bild {i + 1}: {self.peaks[i].shape}")



    def refresh_peaks(self):
        """Entfernt die nächstgelegenen Peaks zu den ausgewählten Punkten für beide Bilder."""
        if self.mode != "remove":
            self.log_message("Switch to 'Remove Points' mode to delete peaks.")
            return
        
        for i in range(2):  # Über beide Bilder iterieren
            if not self.selected_peaks[i]:
                self.log_message(f"No points selected for removal in image {i + 1}.")
                continue
    
            initial_peak_count = len(self.peaks[i])
    
            for selected_peak in self.selected_peaks[i]:
                closest_peak = None
                min_dist = float("inf")
                closest_index = -1
    
                # Suche nach dem nächstgelegenen Peak
                for j, peak in enumerate(self.peaks[i]):
                    px, py = peak
                    dist = np.sqrt((px - selected_peak[0]) ** 2 + (py - selected_peak[1]) ** 2)
    
                    if dist < min_dist:
                        closest_peak = peak
                        closest_index = j
                        min_dist = dist
    
                # Entferne den nächstgelegenen Peak, wenn gefunden
                if closest_peak is not None and closest_index != -1:
                    # Verwende np.delete, um das Element zu entfernen
                    self.peaks[i] = np.delete(self.peaks[i], closest_index, axis=0)  # Achse 0 bedeutet Zeilen löschen
                    self.log_message(f"Removed peak from image {i + 1}: {closest_peak}")
    
            # Anzahl der entfernten Peaks berechnen
            removed_count = initial_peak_count - len(self.peaks[i])
            if removed_count > 0:
                self.log_message(f"Removed {removed_count} peaks from image {i + 1}.")
            else:
                self.log_message(f"No peaks removed in image {i + 1}.")
    
            # **Änderung hier: Beide Bilder gleichzeitig aktualisieren**
            for j in range(2):  # Beide Bilder nach der Entfernung der Punkte aktualisieren
                self.display_image_with_peaks(j)
    
            self.selected_peaks[i] = []  # Auswahl zurücksetzen
            self.log_message(f"Selected points list cleared for image {i + 1}.")



    def add_peak_manually(self, x, y, image_index):
        """Fügt einen Peak an den angeklickten Koordinaten zum entsprechenden Bild hinzu."""
        if image_index not in [0, 1]:  # Sicherstellen, dass das Bild existiert
            self.log_message("Invalid image index for peak addition.")
            return
        
        # Fügt den Punkt zu der jeweiligen Peaks-Liste hinzu (mit numpy.vstack)
        if len(self.peaks[image_index]) > 0:
            self.peaks[image_index] = np.vstack([self.peaks[image_index], [x, y]])
        else:
            self.peaks[image_index] = np.array([[x, y]])
        
        # Log-Nachricht, dass der Peak hinzugefügt wurde
        self.log_message(f"Added new peak to image {image_index + 1}: ({x}, {y})")
        
        # Direkt das Bild mit den neuen Peaks anzeigen (Canvas wird neu gezeichnet)
        self.display_image_with_peaks(image_index + 1)




    def on_canvas_click(self, event, image_index):
        """Verarbeitet Klicks je nach Modus für das angeklickte Bild."""
        # Log-Nachricht für Debugging
        self.log_message(f"Canvas clicked, image_index: {image_index}")
        
        if image_index not in [1, 2]:
            self.log_message(f"Invalid image index: {image_index}")
            return
        
        # Berechne die Originalkoordinaten des Klicks im Bild
        canvas = self.canvas1 if image_index == 1 else self.canvas2
        
        scale_x = canvas.winfo_width() / self.images[image_index - 1].size[0]
        scale_y = canvas.winfo_height() / self.images[image_index - 1].size[1]
        
        original_x = int(event.x / scale_x)
        original_y = int(event.y / scale_y)
        
        # Log-Nachricht zu den berechneten Koordinaten
        self.log_message(f"Original coordinates: ({original_x}, {original_y})")
        
        if self.mode == "remove":
            # Peaks zur Liste für das angeklickte Bild hinzufügen
            self.selected_peaks[image_index - 1].append((original_x, original_y))
            self.log_message(f"Selected for removal: ({original_x}, {original_y})")
        elif self.mode == "add":
            # Manuell Peak hinzufügen
            self.add_peak_manually(original_x, original_y, image_index - 1)








    def display_image_with_peaks(self, image_index):
        """Zeigt das Bild mit Peaks an."""
        image = self.images[image_index - 1]
        if image is None:
            return
        
        img_width, img_height = image.size
        max_size = 500
        scale = min(max_size / img_width, max_size / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        img_resized = image.resize((new_width, new_height), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(img_resized)
        
        # Wähle das richtige Canvas basierend auf dem Bildindex
        if image_index == 1:
            canvas = self.canvas1
        else:
            canvas = self.canvas2
        
        # Setze die Größe des Canvas auf die neue Größe des Bildes
        canvas.config(width=new_width, height=new_height)
        
        # Lösche das Canvas, um das neue Bild zu zeichnen
        canvas.delete("all")
        
        # Füge das Bild zum Canvas hinzu
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        
        # Speichere die Bildreferenz, um zu verhindern, dass das Bild von der Garbage Collection gelöscht wird
        if image_index == 1:
            self.canvas1.image = tk_image
        else:
            self.canvas2.image = tk_image
        
        # Berechne die Skalierung, um die Peaks korrekt anzuzeigen
        scale_x = new_width / img_width
        scale_y = new_height / img_height
        
        # Zeige Peaks auf dem Bild an
        for (x, y) in self.peaks[image_index - 1]:
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5, outline="red", width=2)






if __name__ == "__main__":
    root = tk.Tk()
    app = LaserPointGUI(root)
    root.mainloop()

