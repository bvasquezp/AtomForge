import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import LogEuclideanExp

# Tabla de numeros atomicos a simbolos (Z -> Symbol)
Z_TO_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 44: "Ru",
    45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
}

def export_trajectory_to_cinematic_xyz(trajectory_path: str, output_path: str):
    print(f" Cargando trayectoria desde: {trajectory_path}")
    traj = torch.load(trajectory_path, weights_only=False)
    
    types = traj["types"]          # (T, N)
    coords = traj["coords"]        # (T, N, 3)
    lattice_log = traj["lattice_log"]  # (T, 1, 3, 3)
    
    num_steps = traj["num_steps"]
    num_atoms = coords.shape[1]
    
    print(f" Pasos de integracion: {num_steps} | Atomos: {num_atoms}")
    
    with open(output_path, "w") as f:
        for t_idx in range(num_steps):
            # 1. Coordenadas fraccionales en el toroide T^3
            frac_t = coords[t_idx]
            frac_t = frac_t % 1.0  # Periodic Boundary Conditions
            
            # 2. Celda Unitaria desde el espacio Log-Euclidiano
            A_t = lattice_log[t_idx]
            L_t = LogEuclideanExp.apply(A_t).squeeze(0)  # (3, 3)
            matrix = L_t.numpy()
            
            # 3. Mapeo a Cartesianas Reales
            cart_t = frac_t.numpy() @ matrix
            
            # 4. Tipos atomicos
            types_t = types[t_idx].numpy().astype(int)
            
            # 5. Estado de cristalizacion (c_state) en [0, 1]
            # Sirve para que Blender anime la opacidad, color o brillo de cada atomo
            c_state = t_idx / max(1, num_steps - 1)
            
            f.write(f"{num_atoms}\n")
            f.write(f"Properties=species:S:1:pos:R:3:c_state:R:1 Time={c_state:.4f}\n")
            
            for i in range(num_atoms):
                elem = Z_TO_SYMBOL.get(types_t[i], "C")
                x, y, z = cart_t[i]
                # Escribir Elemento, X, Y, Z, c_state
                f.write(f"{elem:<2} {x:10.5f} {y:10.5f} {z:10.5f} {c_state:10.5f}\n")
                
    print(f" Cinematic XYZ guardado en: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta al tensor .pt")
    parser.add_argument("--output", required=True, help="Ruta al xyz de salida")
    args = parser.parse_args()
    
    export_trajectory_to_cinematic_xyz(args.input, args.output)
