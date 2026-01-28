import os
import torch
import numpy as np
from ase.io import read
from scipy.spatial import distance_matrix

# --- CONFIGURACIÓN DE RUTAS ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)

# CAMBIO AQUÍ: Ahora apuntamos a "raw_cif"
RAW_DIR = os.path.join(project_root, "data", "raw_cifs") 
PROCESSED_DIR = os.path.join(project_root, "data", "processed")

# Distancia para considerar enlace (Angstroms)
BOND_THRESHOLD = 2.0 

def create_graph_data():
    if not os.path.exists(RAW_DIR):
        print(f" Error: No encuentro la carpeta: {RAW_DIR}")
        print("Asegúrate de que el nombre sea exacto.")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    cif_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.cif')]
    
    if len(cif_files) == 0:
        print(f"La carpeta {RAW_DIR} existe, pero está vacía de archivos .cif")
        return

    print(f" Procesando {len(cif_files)} archivos desde '{RAW_DIR}'...")

    count = 0
    for file_name in cif_files:
        try:
            file_path = os.path.join(RAW_DIR, file_name)
            atoms = read(file_path)
            
            # 1. Tipos de átomos
            atom_types = atoms.get_atomic_numbers()
            
            # 2. Posiciones (Vital para que la IA aprenda geometría)
            pos = atoms.get_positions()
            
            # 3. Calcular enlaces
            dist_mat = distance_matrix(pos, pos)
            adj = (dist_mat < BOND_THRESHOLD) & (dist_mat > 0)
            rows, cols = np.where(adj)
            edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
            
            # 4. Tensores
            x = torch.tensor(atom_types, dtype=torch.float).unsqueeze(1)
            pos_tensor = torch.tensor(pos, dtype=torch.float)

            data_object = {
                'x': x,
                'edge_index': edge_index,
                'pos': pos_tensor,
                'num_nodes': len(atoms)
            }
            
            save_name = file_name.replace('.cif', '.pt')
            torch.save(data_object, os.path.join(PROCESSED_DIR, save_name))
            count += 1
            
        except Exception as e:
            print(f"Error en {file_name}: {e}")

    print(f" ¡Listo! Se convirtieron {count} cristales a grafos en data/processed")

if __name__ == "__main__":
    create_graph_data()