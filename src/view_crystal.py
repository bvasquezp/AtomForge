import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def view_file(filename):
    # Rutas relativas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    filepath = os.path.join(project_root, "generated_crystals", filename)

    if not os.path.exists(filepath):
        print(f"Error: No encuentro el archivo {filename}")
        print(f"Busqué en: {filepath}")
        return

    print(f"Cargando: {filename} ...")
    
    # Cargar datos
    data = torch.load(filepath, weights_only=False)
    
    # Extraer informacion
    adj = data['adjacency'].numpy()
    num_atoms = data['num_atoms']
    density = data['density']

    # Visualizar
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj, cmap="Greens", cbar=False, xticklabels=False, yticklabels=False)
    
    plt.title(f"Archivo: {filename}\nDensidad: {density:.4f} | Atomos: {num_atoms}", fontsize=14)
    plt.xlabel("Atomos Conectados (Estructura de Red)", fontsize=12)
    
    print(f"Mostrando estructura con densidad {density:.4f}")
    plt.show()

if __name__ == "__main__":
    # Puedes cambiar el nombre del archivo aqui para ver otros
    archivo_a_ver = "crystal_gen_001.pt" 
    
    view_file(archivo_a_ver)