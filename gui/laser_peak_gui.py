import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from gui.gui_main import LaserPointGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = LaserPointGUI(root)
    root.mainloop()

