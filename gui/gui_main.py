import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import cv2
from gui.gui_helpers import log_message, display_image_with_peaks, zoom_image, start_drag, drag_image, stop_drag, update_canvas_view
from gui.gui_processing import detect_peaks, save_results, refresh_peaks, add_peak_manually, start_fit, calculate_3d_structure, display_3d_plot
from moduls.peak_find import find_peaks
from file_io import construct_flex_op_path

class LaserPointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Point Detector")

        # 🔹 Haupt-Frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 🔹 Kontrollbereich (Oben links, 2 Spalten)
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=0, sticky="nw", padx=10)

        # **Spalte 1: Buttons**
        self.button_frame = tk.Frame(self.control_frame)
        self.button_frame.grid(row=0, column=0, padx=5)

        self.btn_open = tk.Button(self.button_frame, text="Open Image 1", command=lambda: self.load_image(1))
        self.btn_open.grid(row=0, column=0, pady=2)

        self.btn_open2 = tk.Button(self.button_frame, text="Open Image 2", command=lambda: self.load_image(2))
        self.btn_open2.grid(row=1, column=0, pady=2)

        self.btn_run = tk.Button(self.button_frame, text="Run", command=self.detect_peaks, state=tk.DISABLED)
        self.btn_run.grid(row=2, column=0, pady=2)

        self.btn_save = tk.Button(self.button_frame, text="Save", command=self.save_results, state=tk.DISABLED)
        self.btn_save.grid(row=3, column=0, pady=2)

        self.btn_refresh = tk.Button(self.button_frame, text="Refresh", command=self.refresh_peaks, state=tk.DISABLED)
        self.btn_refresh.grid(row=4, column=0, pady=2)

        self.btn_remove_mode = tk.Button(self.button_frame, text="Remove Points (Off)", command=self.toggle_remove_points)
        self.btn_remove_mode.grid(row=5, column=0, pady=2)

        self.btn_add_mode = tk.Button(self.button_frame, text="Add Points (Off)", command=self.toggle_add_points)
        self.btn_add_mode.grid(row=6, column=0, pady=2)

        self.btn_calc_3d = tk.Button(self.button_frame, text="Calc 3D", command=self.calculate_3d_structure)
        self.btn_calc_3d.grid(row=7, column=0, pady=5)

        self.btn_start_fit = tk.Button(self.button_frame, text="Start Fit", command=lambda: self.start_fit())
        self.btn_start_fit.grid(row=8, column=0, pady=2)
        
        # Neuer Button für Calibration
        self.calibration_frame = tk.Frame(self.control_frame)
        self.calibration_frame.grid(row=0, column=2, padx=5)
        self.btn_load_calibration = tk.Button(self.calibration_frame, text="Load Calibration", command=self.load_calibration_file)
        self.btn_load_calibration.grid(row=0, column=0, pady=2)
        
        # Variable für Kalibrierungsdatei
        self.calibration_file = tk.StringVar(value="")  # Standardwert: keine Datei geladen


        # **Spalte 2: Dropdown & Sliders**
        self.slider_frame = tk.Frame(self.control_frame)
        self.slider_frame.grid(row=0, column=1, padx=5)

        self.label_method = tk.Label(self.slider_frame, text="Fitting-Methode:")
        self.label_method.grid(row=0, column=0, pady=2)

        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(self.slider_frame, textvariable=self.method_var)
        self.method_dropdown["values"] = [
            "center_of_mass",
            "gauss_fit",
            "skewed_gauss_fit",
            "non_linear_center_of_mass",
            "center_of_mass_with_threshold",
            "circle_fit"
        ]
        self.method_dropdown.current(0)
        self.method_dropdown.grid(row=1, column=0, pady=2)

        self.slider_factor = tk.Scale(self.slider_frame, from_=1, to=50, orient=tk.HORIZONTAL, label="Factor")
        self.slider_factor.set(10)
        self.slider_factor.grid(row=2, column=0, pady=2)

        self.slider_window_size = tk.Scale(self.slider_frame, from_=3, to=15, orient=tk.HORIZONTAL, label="Window Size", resolution=1)
        self.slider_window_size.set(9)
        self.slider_window_size.grid(row=3, column=0, pady=2)

        self.slider_threshold = tk.Scale(self.slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold", resolution=0.1)
        self.slider_threshold.set(0)
        self.slider_threshold.grid(row=4, column=0, pady=2)

        # **🔹 Log-Box (oben rechts)**
        self.log_frame = tk.Frame(self.main_frame)
        self.log_frame.grid(row=0, column=1, sticky="ne", padx=10)

        self.text_log = tk.Text(self.log_frame, height=10, width=50, wrap=tk.WORD)
        self.text_log.grid(row=0, column=0, padx=10, pady=10)

        # **🔹 Canvas-Bereich (unten, Reihenfolge 1-3-2)**
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.canvas1 = tk.Canvas(self.canvas_frame, width=300, height=300, bg="gray")
        self.canvas1.grid(row=0, column=0, padx=10)
        self.canvas1.bind("<Button-1>", lambda event: self.on_canvas_click(event, 1))

        self.canvas3d = tk.Canvas(self.canvas_frame, width=300, height=300, bg="white")  # 3D-Plot in der Mitte
        self.canvas3d.grid(row=0, column=1, padx=10)

        self.canvas2 = tk.Canvas(self.canvas_frame, width=300, height=300, bg="gray")
        self.canvas2.grid(row=0, column=2, padx=10)
        self.canvas2.bind("<Button-1>", lambda event: self.on_canvas_click(event, 2))

        # 🔹 Bild-Attribute
        self.image_paths = [None, None]
        self.images = [None, None]
        self.peaks = [[], []]
        self.lpc = [None, None]
        self.scale = [1.0, 1.0] 
        self.scale_x = [1.0, 1.0]  # Skalierungsfaktoren für die Anzeige
        self.scale_y = [1.0, 1.0]
        
        # ➡️ Diese Zeilen sind neu:
        self.offset_x = [0, 0]
        self.offset_y = [0, 0]

        
        # Set initial mode to drag
        self.set_mode("drag")


        
    def debug_event(self, event):
        print(f"Event received: {event}")


    def log_message(self, message):
        log_message(self.text_log, message)

    def load_image(self, image_index):
        """Lädt ein Bild und passt die Canvas-Größe basierend auf dem Seitenverhältnis an."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.bmp")])
        if not file_path:
            return
    
        self.image_paths[image_index - 1] = file_path
        self.images[image_index - 1] = Image.open(file_path)
    
        img_width, img_height = self.images[image_index - 1].size
    
        # Setze konstante Canvas-Breite und berechne die Höhe basierend auf dem Seitenverhältnis
        canvas_width = 500
        aspect_ratio = img_height / img_width
        canvas_height = int(canvas_width * aspect_ratio)
    
        # Setze die neue Größe des Canvas
        canvas = self.canvas1 if image_index == 1 else self.canvas2
        canvas.config(width=canvas_width, height=canvas_height)
    
        # Berechne die anfängliche Skalierung
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        initial_scale = min(scale_x, scale_y)
    
        self.scale[image_index - 1] = initial_scale
    
        # Zeige das Bild mit Peaks an
        display_image_with_peaks(canvas,
                                 self.images[image_index - 1],
                                 self.peaks[image_index - 1],
                                 self.lpc[image_index - 1],
                                 image_index,
                                 self)
    
        self.btn_run["state"] = tk.NORMAL
    
    def load_calibration_file(self):
        """Öffnet einen Dateidialog zur Auswahl einer Kalibrierungsdatei (.npz)."""
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npz")])
        if file_path:  # Falls eine Datei ausgewählt wurde
            self.calibration_file.set(file_path)
            log_message(self.text_log, f"Kalibrierungsdatei geladen: {file_path}")
        else:
            log_message(self.text_log, "Keine Kalibrierungsdatei ausgewählt.")



    
    def on_canvas_click(self, event, image_index):
        """Verarbeitet Klicks im Canvas abhängig vom Modus."""
        if self.mode == "drag":
            log_message(self.text_log, f"Drag mode activated, canvas clicked, image_index: {image_index}")
            
            if image_index not in [1, 2]:
                log_message(self.text_log, f"Invalid image index: {image_index}")
                return
            
            # Starte den Drag-Prozess
            canvas = self.canvas1 if image_index == 1 else self.canvas2
            self.start_drag(event, image_index)
    
        elif self.mode not in ["remove", "add"]:
            return  # Falls nicht im Remove- oder Add-Modus, nichts tun
        
        log_message(self.text_log, f"Canvas clicked, image_index: {image_index}")
        
        if image_index not in [1, 2]:
            log_message(self.text_log, f"Invalid image index: {image_index}")
            return
    
        canvas = self.canvas1 if image_index == 1 else self.canvas2
    
        # **Korrekte Skalierung mit Zoom berücksichtigen**
        zoom_factor = self.scale[image_index - 1]  # Aktueller Zoomfaktor
        offset_x = self.offset_x[image_index - 1]  # Aktuelle Verschiebung X
        offset_y = self.offset_y[image_index - 1]  # Aktuelle Verschiebung Y
    
        # **Korrekte Originalkoordinaten berechnen**
        original_x = int((event.x + offset_x) / zoom_factor)
        original_y = int((event.y + offset_y) / zoom_factor)
    
        log_message(self.text_log, f"Original coordinates: ({original_x}, {original_y})")
    
        if self.mode == "remove":
            self.selected_peaks[image_index - 1].append((original_x, original_y))
            log_message(self.text_log, f"Selected for removal: ({original_x}, {original_y})")
        elif self.mode == "add":
            add_peak_manually(self, original_x, original_y, image_index - 1)



    
    def set_mode(self, mode):
        """Setzt den Modus und aktualisiert die Buttons."""
        self.mode = mode
    
        # Button-Status aktualisieren
        if mode == "remove":
            self.btn_remove_mode.config(text="Remove Points (On)")
            self.btn_add_mode.config(text="Add Points (Off)", state=tk.NORMAL)
    
            # **Klick-Events für Remove-Modus binden**
            self.canvas1.bind("<Button-1>", lambda event: self.on_canvas_click(event, 1))
            self.canvas2.bind("<Button-1>", lambda event: self.on_canvas_click(event, 2))
    
        elif mode == "add":
            self.btn_add_mode.config(text="Add Points (On)")
            self.btn_remove_mode.config(text="Remove Points (Off)")
    
            # **Klick-Events für Add-Modus binden**
            self.canvas1.bind("<Button-1>", lambda event: self.on_canvas_click(event, 1))
            self.canvas2.bind("<Button-1>", lambda event: self.on_canvas_click(event, 2))
    
        else:  # Drag-Modus
            self.btn_remove_mode.config(text="Remove Points (Off)")
            self.btn_add_mode.config(text="Add Points (Off)", state=tk.NORMAL)
    
            # **Mouse-Events für Drag-Modus binden**
            self.canvas1.bind("<ButtonPress-1>", lambda event: self.start_drag(event, 1))
            self.canvas1.bind("<B1-Motion>", self.drag_image)
            self.canvas1.bind("<ButtonRelease-1>", self.stop_drag)
    
            self.canvas2.bind("<ButtonPress-1>", lambda event: self.start_drag(event, 2))
            self.canvas2.bind("<B1-Motion>", self.drag_image)
            self.canvas2.bind("<ButtonRelease-1>", self.stop_drag)
    
        log_message(self.text_log, f"Switched to '{mode}' mode.")







     
    def toggle_remove_points(self):
        """Schaltet den 'Remove Points'-Modus an/aus und deaktiviert 'Add Points'."""
        if self.mode == "remove":
            self.set_mode("drag")  # Zurück in den Drag-Modus
        else:
            self.set_mode("remove")


    def toggle_add_points(self):
        """Schaltet den 'Add Points'-Modus an/aus und deaktiviert 'Remove Points'."""
        if self.mode == "add":
            self.set_mode("drag")  # Zurück in den Drag-Modus
        else:
            self.set_mode("add")




    
    def update_labels(self, event=None):
        """Aktualisiert die Label-Anzeigen bei der Slider-Verschiebung."""
        self.label_factor.config(text=f"Factor: {self.slider_factor.get()}")
        self.label_window_size.config(text=f"Window Size: {self.slider_window_size.get()}")
        self.label_threshold.config(text=f"Threshold: {self.slider_threshold.get():.1f}")


    def detect_peaks(self):
        detect_peaks(self)
    
    def start_fit(self):
        start_fit(self)

    def save_results(self):
        save_results(self)

    def refresh_peaks(self):
        refresh_peaks(self)
    
    def calculate_3d_structure(self):
        calculate_3d_structure(self)
    
    def display_3d_plot(self, points_3d):
        display_3d_plot(self, points_3d)
    
    def zoom_image(self, event, canvas_index):
        zoom_image(self, event, canvas_index)
    
    def start_drag(self, event, canvas_index):
        start_drag(self, event, canvas_index)
    
    def drag_image(self, event):
        drag_image(self, event)
    
    def update_canvas_view(self, canvas_index):
        update_canvas_view(self, canvas_index)
        
    def stop_drag(self, event):
        stop_drag(self, event)
