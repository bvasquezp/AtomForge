import torch
import networkx as nx
import numpy as np
import os

# Mapeo de reglas: Grado (vecinos) -> Elemento Quimico
def get_element_by_degree(degree):
    if degree >= 6:
        return "Zr" # Zirconio (Para los Super Clusters)
    elif 4 <= degree < 6:
        return "Cu" # Cobre (Para nodos metalicos estandar)
    elif degree == 3:
        return "C"  # Carbono (Ramificaciones)
    elif degree == 2:
        return "O"  # Oxigeno (Puentes lineales)
    else:
        return "H"  # Hidrogeno (Terminaciones)

def export_crystal(filename):
    # Rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, "generated_crystals", filename)
    
    # Nombre del archivo de salida (cambiamos .pt por .xyz)
    output_filename = filename.replace(".pt", ".xyz")
    output_path = os.path.join(project_root, "generated_crystals", output_filename)

    if not os.path.exists(input_path):
        print(f"Error: No encuentro {filename}")
        return

    print(f"Convirtiendo {filename} a formato 3D real (.xyz)...")
    
    # 1. Cargar Grafo
    data = torch.load(input_path, weights_only=False)
    adj_matrix = data['adjacency'].numpy()
    rows, cols = adj_matrix.nonzero()
    
    G = nx.Graph()
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    
    # Importante: Añadir nodos que puedan estar aislados pero existan en la matriz
    # (Aunque nuestro filtro suele limpiar esto)
    for i in range(data['num_atoms']):
        if i not in G:
            G.add_node(i)

    # 2. GENERAR COORDENADAS 3D (La "Inflacion")
    # Usamos un algoritmo de resortes en 3 dimensiones (dim=3)
    print("Simulando fisicas para encontrar coordenadas 3D estables...")
    # pos es un diccionario {id_atomo: [x, y, z]}
    pos = nx.spring_layout(G, dim=3, iterations=100, seed=42, scale=10.0)

    # 3. ESCRIBIR ARCHIVO XYZ
    # Formato estandar:
    # [Numero de atomos]
    # [Comentario]
    # [Elemento] [X] [Y] [Z]
    
    num_atoms_export = len(G.nodes())
    
    with open(output_path, "w") as f:
        # Cabecera
        f.write(f"{num_atoms_export}\n")
        f.write(f"Generado por IA - MOF Sintetico desde {filename}\n")
        
        # Atomos
        for node in G.nodes():
            degree = G.degree[node]
            element = get_element_by_degree(degree)
            x, y, z = pos[node]
            
            # Escribir linea: Elemento  X      Y      Z
            f.write(f"{element:<2} {x:.5f} {y:.5f} {z:.5f}\n")

    print(f"--- EXITO ---")
    print(f"Archivo creado: {output_path}")
    print("Ahora puedes abrir este archivo en Avogadro, VESTA o cualquier visor 3D.")

if __name__ == "__main__":
    # Usamos el cristal 004 que te gusto
    export_crystal("crystal_gen_004.pt")