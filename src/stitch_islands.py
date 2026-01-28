import os
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import sys

# Radio maximo para considerar que dos atomos ya estan conectados (Enlace quimico largo)
BOND_THRESHOLD = 2.0 

def read_xyz(filepath):
    atoms = []
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords)

def write_xyz(filepath, atoms, coords):
    with open(filepath, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"Stitched MOF\n")
        for atom, (x, y, z) in zip(atoms, coords):
            f.write(f"{atom:<2} {x:.5f} {y:.5f} {z:.5f}\n")

def stitch_structure(filename):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, 'top_candidates', filename)
    output_filename = filename.replace(".xyz", "_stitched.xyz")
    output_path = os.path.join(project_root, 'top_candidates', output_filename)
    
    if not os.path.exists(input_path):
        print(f"Error: No se encuentra {input_path}")
        return

    atoms, coords = read_xyz(input_path)
    n_atoms = len(atoms)
    
    # 1. Construir el Grafo actual basado en distancias
    print("Construyendo grafo de conectividad...")
    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))
    
    # Matriz de distancias
    dists = distance_matrix(coords, coords)
    
    # Agregar enlaces existentes
    rows, cols = np.where((dists < BOND_THRESHOLD) & (dists > 0))
    for i, j in zip(rows, cols):
        G.add_edge(i, j)
        
    # 2. Identificar Islas (Componentes Conectados)
    components = list(nx.connected_components(G))
    print(f"Islas detectadas: {len(components)}")
    
    if len(components) == 1:
        print("La estructura ya esta conectada. No se necesitan cambios.")
        return

    # 3. Zurcir las islas (Conectar cada isla a la mas cercana)
    # Convertimos a listas para poder modificar atoms/coords
    atoms_list = list(atoms)
    coords_list = coords.tolist()
    
    # Estrategia: Conectar componente 0 con 1, 1 con 2, etc. en cadena
    # Esto asegura que todo quede unido al final.
    
    sorted_components = sorted(components, key=len, reverse=True)
    main_comp = list(sorted_components[0])
    
    for i in range(1, len(sorted_components)):
        target_comp = list(sorted_components[i])
        
        # Buscar los dos atomos mas cercanos entre la Isla Principal y la Isla Objetivo
        # Extraemos coordenadas de ambos grupos
        main_indices = main_comp
        target_indices = target_comp
        
        main_coords = np.array([coords_list[idx] for idx in main_indices])
        target_coords = np.array([coords_list[idx] for idx in target_indices])
        
        # Calcular distancias entre grupos
        d_matrix = distance_matrix(main_coords, target_coords)
        
        # Encontrar el minimo absoluto
        min_idx_flat = np.argmin(d_matrix)
        # Convertir indice a coordenadas de matriz (row, col)
        r, c = np.unravel_index(min_idx_flat, d_matrix.shape)
        
        # Indices reales en la lista global
        atom_a_idx = main_indices[r]
        atom_b_idx = target_indices[c]
        
        dist = d_matrix[r, c]
        
        # 4. Accion de Conexion
        if dist < 4.0:
            # Si estan cerca (< 4 A), agregamos un atomo de CARBONO puente en el medio
            # para no estirar un enlace imposible.
            midpoint = (np.array(coords_list[atom_a_idx]) + np.array(coords_list[atom_b_idx])) / 2
            
            atoms_list.append("C")
            coords_list.append(midpoint.tolist())
            print(f"   -> Puente C creado entre isla {i} y principal (Dist: {dist:.2f} A)")
            
        else:
            # Si estan muy lejos, ponemos una cadena de 2 carbonos para relajar
            p1 = np.array(coords_list[atom_a_idx])
            p2 = np.array(coords_list[atom_b_idx])
            
            v = p2 - p1
            step = v / 3.0
            
            c1_pos = p1 + step
            c2_pos = p1 + (step * 2)
            
            atoms_list.append("C")
            coords_list.append(c1_pos.tolist())
            atoms_list.append("C")
            coords_list.append(c2_pos.tolist())
            print(f"   -> Puente C-C largo creado (Dist: {dist:.2f} A)")

        # Expandimos la isla principal para incluir la nueva isla (logica simplificada para la iteracion)
        main_comp.extend(target_comp)

    # Guardar
    write_xyz(output_path, atoms_list, np.array(coords_list))
    print(f"Estructura reparada guardada en: {output_filename}")

if __name__ == "__main__":
    # Pon aqui el nombre de tu archivo con las islas (el hollow)
    target_file = "crystal_gen_007_hollow_stitched.xyz"
    stitch_structure(target_file)