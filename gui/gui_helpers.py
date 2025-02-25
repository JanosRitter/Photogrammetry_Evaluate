from PIL import Image, ImageTk
import tkinter as tk

def log_message(text_widget, message):
    text_widget.insert("end", message + "\n")
    text_widget.yview("end")
    

def display_image_with_peaks(canvas, image, peaks, lpc_coords, canvas_index, gui_instance):
    if image is None:
        return

    # Bildgröße und Canvas-Größe berechnen
    img_width, img_height = image.size
    canvas_width = 500  # Fixierte Breite
    canvas_height = int(canvas_width * (img_height / img_width))  # Höhe aus Seitenverhältnis

    # **Canvas-Größe bleibt konstant**
    canvas.config(width=canvas_width, height=canvas_height)

    # Bild skalieren, um es in das Canvas zu passen
    resized_image = image.resize((canvas_width, canvas_height), Image.LANCZOS)
    tk_image = ImageTk.PhotoImage(resized_image)

    # Canvas leeren und Bild zeichnen
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

    # **Speichere Bildreferenz**
    if canvas_index == 1:
        gui_instance.canvas1.image = tk_image
    else:
        gui_instance.canvas2.image = tk_image

    # **Peaks anzeigen**
    for (x, y) in peaks:
        # Skalierung der Punkte basierend auf der Bildgröße
        scaled_x = x * (canvas_width / img_width)
        scaled_y = y * (canvas_height / img_height)
        canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5, outline="red", width=2)

    # **LPC-Koordinaten anzeigen**
    if lpc_coords is not None:
        # Wenn lpc_coords ein (n, 4)-Array ist, dann die ersten beiden Spalten (x, y) extrahieren
        lpc_coords = lpc_coords[:, :2]

        for (x, y) in lpc_coords:
            # Skalierung der LPC-Koordinaten basierend auf der Bildgröße
            scaled_x = x * (canvas_width / img_width)
            scaled_y = y * (canvas_height / img_height)
            canvas.create_oval(scaled_x - 3, scaled_y - 3, scaled_x + 3, scaled_y + 3, outline="yellow", width=2)





def zoom_image(self, event, canvas_index):
    """Zoomt das Bild auf die Mausposition, ohne die Canvas-Größe zu verändern."""
    
    scale_factor = 1.1 if event.delta > 0 else 0.9  # Scroll-Up: Vergrößern, Scroll-Down: Verkleinern

    # Originalbildgröße
    img_width, img_height = self.images[canvas_index - 1].size

    # Aktuelle Skalierung
    current_scale = self.scale[canvas_index - 1]
    new_scale = max(0.1, min(current_scale * scale_factor, 5))  # Begrenzung des Zooms

    # **Canvas hat feste Dimensionen**
    canvas = self.canvas1 if canvas_index == 1 else self.canvas2

    canvas_width = 500
    canvas_height = int(canvas_width * (img_height / img_width))

    # **Debug-Prints vor Zoom**
    print(f"[ZOOM] Canvas {canvas_index}: Breite={canvas_width}, Höhe={canvas_height}")
    print(f"[ZOOM] Bildgröße vor Zoom: Breite={img_width}, Höhe={img_height}, Zoom-Stufe={current_scale}")

    # Mausposition relativ zum Canvas
    mouse_x = canvas.winfo_pointerx() - canvas.winfo_rootx()
    mouse_y = canvas.winfo_pointery() - canvas.winfo_rooty()
    rel_x = mouse_x / canvas_width
    rel_y = mouse_y / canvas_height

    # **Exakte Bildkoordinate unter der Maus**
    img_x_at_cursor = self.view_left[canvas_index - 1] + (rel_x * img_width / current_scale)
    img_y_at_cursor = self.view_top[canvas_index - 1] + (rel_y * img_height / current_scale)

    # **Debug-Prints für Maus-Position**
    print(f"[ZOOM] Mausposition: x={mouse_x}, y={mouse_y}, Relativ: x={rel_x:.2f}, y={rel_y:.2f}")
    print(f"[ZOOM] Bildkoordinate unter Maus: x={img_x_at_cursor:.2f}, y={img_y_at_cursor:.2f}")

    # Neue linke obere Ecke berechnen
    new_left = img_x_at_cursor - (rel_x * img_width / new_scale)
    new_top = img_y_at_cursor - (rel_y * img_height / new_scale)

    # Begrenzung der linken oberen Ecke
    max_left = img_width - (canvas_width / new_scale)
    max_top = img_height - (canvas_height / new_scale)
    new_left = max(0, min(new_left, max_left))
    new_top = max(0, min(new_top, max_top))

    # **Debug-Prints für neue Bildposition**
    print(f"[ZOOM] Neue Bild-Offsets: left={new_left:.2f}, top={new_top:.2f}")

    # **Neue Ansicht speichern**
    self.view_left[canvas_index - 1] = new_left
    self.view_top[canvas_index - 1] = new_top
    self.scale[canvas_index - 1] = new_scale

    # **Bild mit neuer Skalierung zuschneiden**
    new_width = int(img_width * new_scale)
    new_height = int(img_height * new_scale)
    resized_image = self.images[canvas_index - 1].resize((new_width, new_height), Image.LANCZOS)

    cropped_image = resized_image.crop((new_left, new_top, new_left + canvas_width, new_top + canvas_height))

    # **Debug-Prints nach Zoom**
    print(f"[ZOOM] Bildgröße nach Zoom: Breite={new_width}, Höhe={new_height}, Zoom-Stufe={new_scale}")

    # **Peaks und LPC-Koordinaten anpassen**
    adjusted_peaks = [
        ((x * new_scale) - new_left, (y * new_scale) - new_top)
        for (x, y) in self.peaks[canvas_index - 1]
        if 0 <= (x * new_scale) - new_left < canvas_width and 0 <= (y * new_scale) - new_top < canvas_height
    ]

    adjusted_lpc = [
        ((x * new_scale) - new_left, (y * new_scale) - new_top)
        for (x, y) in (self.lpc[canvas_index - 1] or [])
        if 0 <= (x * new_scale) - new_left < canvas_width and 0 <= (y * new_scale) - new_top < canvas_height
    ]

    # **Bild mit Peaks anzeigen und Canvas-Größe dynamisch anpassen**
    display_image_with_peaks(canvas, cropped_image, adjusted_peaks, adjusted_lpc, canvas_index, self)

    # **Dynamische Anpassung der Canvas-Größe nach Zoom**
    canvas.config(width=canvas_width, height=canvas_height)


















def start_drag(self, event, canvas_index):
    """Speichert die Startposition der Maus und den aktuellen Offset."""
    if self.mode != "drag":
        return

    self.dragging = True
    self.last_x = event.x
    self.last_y = event.y

    self.canvas_being_dragged = canvas_index  # Speichert, welches Canvas bewegt wird


def drag_image(self, event):
    """Verschiebt das Bild basierend auf der Mausbewegung."""
    if not self.dragging or self.canvas_being_dragged is None:
        return

    canvas_index = self.canvas_being_dragged
    dx = event.x - self.last_x
    dy = event.y - self.last_y

    # **Fix: Berechnung direkt mit aktuellem Offset**
    self.offset_x[canvas_index - 1] -= dx
    self.offset_y[canvas_index - 1] -= dy

    # Begrenzung der Offsets, damit das Bild nicht aus dem Canvas "herausgezogen" werden kann
    max_x_offset = max(0, self.zoomed_width - self.canvas1.winfo_width())
    max_y_offset = max(0, self.zoomed_height - self.canvas1.winfo_height())

    self.offset_x[canvas_index - 1] = max(0, min(self.offset_x[canvas_index - 1], max_x_offset))
    self.offset_y[canvas_index - 1] = max(0, min(self.offset_y[canvas_index - 1], max_y_offset))

    # **Fix: Sofortige Anzeige der Änderungen**
    self.update_canvas_view(canvas_index)

    # **Fix: Mausposition für das nächste Event speichern**
    self.last_x = event.x
    self.last_y = event.y


def update_canvas_view(self, canvas_index):
    """Zeigt das Bild mit den aktuellen Offsets & Zoom-Stufen an, ohne den Zoom zurückzusetzen.
       Passt die Peaks und LPC-Koordinaten korrekt an.
    """
    canvas = self.canvas1 if canvas_index == 1 else self.canvas2
    img = self.images[canvas_index - 1]

    zoom_factor = self.scale[canvas_index - 1]
    img_width, img_height = img.size

    # **Fix: Zoomed-Werte richtig speichern**
    self.zoomed_width = int(img_width * zoom_factor)
    self.zoomed_height = int(img_height * zoom_factor)

    # **Fix: Maximal erlaubte Offsets neu berechnen**
    max_x_offset = max(0, self.zoomed_width - canvas.winfo_width())
    max_y_offset = max(0, self.zoomed_height - canvas.winfo_height())

    # **Fix: Offsets korrekt begrenzen**
    self.offset_x[canvas_index - 1] = min(self.offset_x[canvas_index - 1], max_x_offset)
    self.offset_y[canvas_index - 1] = min(self.offset_y[canvas_index - 1], max_y_offset)

    # **Fix: Bildausschnitt korrekt berechnen**
    left = self.offset_x[canvas_index - 1]
    top = self.offset_y[canvas_index - 1]
    right = min(left + canvas.winfo_width(), self.zoomed_width)
    bottom = min(top + canvas.winfo_height(), self.zoomed_height)

    cropped_image = img.resize((self.zoomed_width, self.zoomed_height), Image.NEAREST).crop((left, top, right, bottom))


    # **Anpassung der Peaks**
    adjusted_peaks = []
    for (x, y) in self.peaks[canvas_index - 1]:
        new_x = (x * zoom_factor) - left
        new_y = (y * zoom_factor) - top
        if 0 <= new_x < canvas.winfo_width() and 0 <= new_y < canvas.winfo_height():
            adjusted_peaks.append((new_x, new_y))

    # **Anpassung der LPC-Koordinaten (Fit-Punkte)**
    adjusted_lpc = []
    if self.lpc[canvas_index - 1] is not None:
        for (x, y) in self.lpc[canvas_index - 1]:
            new_x = (x * zoom_factor) - left
            new_y = (y * zoom_factor) - top
            if 0 <= new_x < canvas.winfo_width() and 0 <= new_y < canvas.winfo_height():
                adjusted_lpc.append((new_x, new_y))

    # **Fix: Bild, Peaks und Fit-Punkte aktualisieren**
    display_image_with_peaks(
        canvas,
        cropped_image,
        adjusted_peaks,
        adjusted_lpc,
        canvas_index,
        self
    )



def stop_drag(self, event):
    """Beendet das Ziehen des Bildes."""
    self.dragging = False
    self.canvas_being_dragged = None





















