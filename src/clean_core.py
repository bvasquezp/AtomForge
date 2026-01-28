import os
import numpy as np
import sys

def read_xyz(filepath):
    """Lee el archivo XYZ y devuelve atomos y coordenadas."""
    atoms = []
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Saltamos las 2 primeras lineas de encabezado
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords)

def write_xyz(filepath, atoms, coords, comment="Cleaned Core"):
    """Escribe un archivo XYZ."""
    with open(filepath, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atoms, coords):
            f.write(f"{atom:<2} {x:.5f} {y:.5f} {z:.5f}\n")

def clean_inner_core(filename, cutoff_radius=3.0):
    """Elimina atomos dentro de un radio especifico del centro de masa."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, 'top_candidates', filename)
    output_filename = filename.replace(".xyz", "_hollow.xyz")
    output_path = os.path.join(project_root, 'top_candidates', output_filename)
    
    if not os.path.exists(input_path):
        print(f"Error: No se encuentra {input_path}")
        return

    atoms, coords = read_xyz(input_path)
    
    # 1. Calcular el Centro de Masa (Geometrico)
    center = np.mean(coords, axis=0)
    
    # 2. Calcular distancias de cada atomo al centro
    distances = np.linalg.norm(coords - center, axis=1)
    
    # 3. Filtrar: Quedarse solo con los que estan LEJOS del centro
    # cutoff_radius: Radio del agujero que vamos a taladrar (en Angstroms)
    mask = distances > cutoff_radius
    
    new_atoms = [atoms[i] for i in range(len(atoms)) if mask[i]]
    new_coords = coords[mask]
    
    print(f"--- LIMPIEZA DE NUCLEO ---")
    print(f"Original: {len(atoms)} atomos")
    print(f"Eliminados (Lana): {len(atoms) - len(new_atoms)} atomos")
    print(f"Final: {len(new_atoms)} atomos")
    
    write_xyz(output_path, new_atoms, new_coords, f"Hollowed {filename}")
    print(f"Archivo guardado: {output_filename}")

if __name__ == "__main__":
    # Asegurate de poner el nombre correcto de tu archivo actual
    target_file = "crystal_gen_007.xyz"
    clean_inner_core(target_file, cutoff_radius=4.0)