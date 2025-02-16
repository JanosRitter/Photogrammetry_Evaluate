"""
This module contains
"""
import os

def ensure_output_path(base_output_dir, project_name, measurement_name, filename):
    """
    Ensures the output path exists and returns the full path for saving a file.

    Parameters:
        - base_output_dir (str): Base directory for output files.
        - project_name (str): Name of the project.
        - measurement_name (str): Name of the measurement.
        - filename (str): Name of the file to save.

    Returns:
        - str: Full path to the output file.
    """
    output_dir = os.path.join(base_output_dir, project_name, measurement_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured: {output_dir}")
    return os.path.join(output_dir, filename)




def construct_output_path(input_path, base_folder="output", filename=None):
    """
    Constructs an output path by replacing 'input' with a specified base folder (e.g., 'output').

    Parameters:
        - input_path (str): Relative input path (e.g., to the data
        folder within the base directory).
        - base_folder (str): Name of the folder to replace 'input' with (default: "output").
        - filename (str): Name of the file to append to the path (optional).

    Returns:
        - str: Full output path, with the directory structure mirrored from the input.
    """
    base_input_path = r"C:\Users\Janos\Documents\Masterarbeit\3D_scanner\input_output\input"

    full_input_path = os.path.join(base_input_path, input_path)

    output_path = full_input_path.replace("\\input\\", f"\\{base_folder}\\")

    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory ensured: {output_path}")

    if filename:
        return os.path.join(output_path, filename)
    return output_path



def construct_flex_op_path(input_path, base_folder="output", filename=None):
    """
    Erzeugt einen Output-Pfad basierend auf einem gegebenen Input-Pfad.
    Die Funktion erkennt automatisch den übergeordneten 'input_output'-Ordner
    und spiegelt die Unterordnerstruktur unter einem neuen Basisordner ('output').

    Parameters:
        - input_path (str): Absoluter Pfad zur Input-Datei.
        - base_folder (str): Name des neuen Basisordners (default: "output").
        - filename (str): Optionaler Dateiname für den Output.

    Returns:
        - str: Generierter Output-Pfad mit der gleichen Ordnerstruktur.
    """
    input_path = os.path.abspath(input_path)  # Absoluten Pfad sicherstellen

    # Suche nach dem 'input_output'-Ordner in der Pfadhierarchie
    path_parts = input_path.split(os.sep)
    if "input_output" not in path_parts:
        raise ValueError("Der 'input_output'-Ordner konnte im Pfad nicht gefunden werden.")

    idx = path_parts.index("input_output")  # Index des 'input_output'-Ordners
    base_dir = os.sep.join(path_parts[:idx+1])  # Der gesamte Pfad bis 'input_output'

    # Ersetze 'input' durch den gewünschten Basisordner (z. B. 'output')
    rel_path = os.sep.join(path_parts[idx+2:])  # Relativer Pfad nach 'input'
    output_path = os.path.join(base_dir, base_folder, rel_path)

    # Falls der Pfad eine Datei war, das letzte Element als Dateiname extrahieren
    if filename is None and os.path.splitext(output_path)[1]:  # Falls eine Dateiendung existiert
        output_path, filename = os.path.split(output_path)

    # Sicherstellen, dass das Output-Verzeichnis existiert
    os.makedirs(output_path, exist_ok=True)

    # Falls ein Dateiname angegeben ist, füge ihn zum Pfad hinzu
    if filename:
        return os.path.join(output_path, filename)
    
    return output_path
