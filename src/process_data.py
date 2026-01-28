# src/process_data.py
import os
import torch
import warnings
from pymatgen.core import Structure
from torch_geometric.data import Data
from pymatgen.analysis.local_env import CrystalNN

# Ignorar advertencias de pymatgen sobre ocupación parcial
warnings.filterwarnings("ignore")

# Configuración de directorios
RAW_DIR = os.path.join("data", "raw_cifs")
PROCESSED_DIR = os.path.join("data", "processed")

# Diccionario simple para convertir símbolo atómico a número
# Esto ayuda a la IA a saber que "C" es diferente de "Zn"
ATOM_CODES = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 
    'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Zr': 40
}

def get_atom_feature(element_symbol):
    """Convierte el símbolo del átomo en un vector numérico (One-Hot encoding simplificado)."""
    atomic_num = ATOM_CODES.get(element_symbol, 0) # 0 si es un elemento raro no listado
    return [atomic_num]

def cif_to_graph(cif_path):
    """Lee un CIF y devuelve un objeto Data de PyTorch Geometric."""
    try:
        # 1. Leer estructura con Pymatgen
        struct = Structure.from_file(cif_path)
        
        # 2. Obtener Nodos (Átomos)
        node_features = []
        for site in struct:
            node_features.append(get_atom_feature(site.specie.symbol))
        
        x = torch.tensor(node_features, dtype=torch.float)

        # 3. Obtener Aristas (Enlaces) usando CrystalNN (encuentra vecinos cercanos)
        # Esto es mejor que usar solo distancia porque entiende química
        cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0)
        
        edge_indices = []
        edge_attrs = []
        
        # Iterar sobre todos los átomos para encontrar sus vecinos
        for i, _ in enumerate(struct):
            # Obtener vecinos hasta 4 Angstroms (radio típico de interacción)
            neighbors = cnn.get_nn_info(struct, i)
            
            for n in neighbors:
                j = n['site_index'] # Índice del vecino
                dist = n['site'].distance(struct[i]) # Distancia real
                
                edge_indices.append([i, j])
                edge_attrs.append([dist])

        if len(edge_indices) == 0:
            return None # Estructura aislada sin enlaces, descartar

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # 4. Empaquetar en objeto Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    except Exception as e:
        # Si el archivo está corrupto o es muy raro, lo ignoramos
        return None

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    cif_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".cif")]
    print(f"--- PROCESANDO {len(cif_files)} ESTRUCTURAS ---")

    success_count = 0
    
    for i, filename in enumerate(cif_files):
        cif_path = os.path.join(RAW_DIR, filename)
        
        # Convertir a Grafo
        graph_data = cif_to_graph(cif_path)
        
        if graph_data:
            # Guardar como archivo .pt (Formato nativo de PyTorch)
            save_path = os.path.join(PROCESSED_DIR, f"graph_{success_count}.pt")
            torch.save(graph_data, save_path)
            success_count += 1
        
        # Barra de progreso simple
        if i % 10 == 0:
            print(f"Procesando: {i}/{len(cif_files)} (Exitosos: {success_count})", end="\r")

    print(f"\n--- COMPLETADO ---")
    print(f"Se crearon {success_count} grafos validos en '{PROCESSED_DIR}'.")
    print("Los datos estan listos para entrenar la IA.")

if __name__ == "__main__":
    main()