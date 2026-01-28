import torch
import networkx as nx
import numpy as np
import os

# --- TABLA PERIODICA SIMPLIFICADA ---
# Masas atomicas (g/mol)
ATOMIC_MASS = {
    "Zr": 91.224,  # Zirconio (Nodos grandes)
    "Cu": 63.546,  # Cobre (Nodos medianos)
    "C":  12.011,  # Carbono (Estructura)
    "O":  15.999,  # Oxigeno (Puentes)
    "H":  1.008    # Hidrogeno (Terminaciones)
}

def get_element_by_degree(degree):
    """Misma logica que usamos para visualizar"""
    if degree >= 6: return "Zr"
    elif 4 <= degree < 6: return "Cu"
    elif degree == 3: return "C"
    elif degree == 2: return "O"
    else: return "H"

def calculate_properties(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    filepath = os.path.join(project_root, "generated_crystals", filename)

    if not os.path.exists(filepath):
        return None

    # 1. Cargar Datos
    data = torch.load(filepath, weights_only=False)
    adj = data['adjacency'].numpy()
    
    # 2. Reconstruir Grafo
    rows, cols = adj.nonzero()
    G = nx.Graph()
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    # Asegurar que esten todos los nodos
    for i in range(data['num_atoms']):
        if i not in G: G.add_node(i)

    # 3. Inflar en 3D (Necesario para calcular volumen)
    # Usamos seed fija para que el calculo sea reproducible
    pos = nx.spring_layout(G, dim=3, iterations=100, seed=42, scale=10.0)
    
    # 4. CALCULOS QUIMICOS
    total_mass = 0.0
    coords = []

    atom_counts = {"Zr": 0, "Cu": 0, "C": 0, "O": 0, "H": 0}

    for node in G.nodes():
        degree = G.degree[node]
        element = get_element_by_degree(degree)
        
        # Sumar masa
        total_mass += ATOMIC_MASS[element]
        atom_counts[element] += 1
        
        # Guardar coordenada
        coords.append(pos[node])

    # 5. CALCULO DE DENSIDAD
    coords = np.array(coords)
    
    # Metodo de la "Caja Envolvente" (Bounding Box)
    # Imaginamos una caja que encierra al cristal
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    box_dims = max_xyz - min_xyz
    
    # Volumen en Angstroms cubicos (A^3)
    # Evitamos volumen 0 sumando un pequeñisimo margen (radio atomico aprox)
    volume_A3 = (box_dims[0] + 2) * (box_dims[1] + 2) * (box_dims[2] + 2)
    
    # Factor de conversion: (g/mol / A^3) -> g/cm^3
    # 1.66054e-24 es la conversion de Dalton a Gramos, normalizado por cm3
    conversion_factor = 1.66054 
    density = (total_mass / volume_A3) * conversion_factor

    return {
        "filename": filename,
        "mass": total_mass,
        "volume": volume_A3,
        "density": density,
        "formula": atom_counts,
        "num_atoms": data['num_atoms']
    }

def run_lab():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    gen_dir = os.path.join(project_root, "generated_crystals")
    
    print(f"{'ARCHIVO':<20} | {'MASA (g/mol)':<12} | {'VOLUMEN (A^3)':<15} | {'DENSIDAD (g/cm3)':<15} | {'FORMULA'}")
    print("-" * 100)

    results = []
    
    # Analizar del 001 al 010
    for i in range(1, 11):
        fname = f"crystal_gen_{i:03d}.pt"
        res = calculate_properties(fname)
        if res:
            results.append(res)
            # Crear string de formula resumida
            f_str = f"Zr{res['formula']['Zr']} C{res['formula']['C']}..."
            
            print(f"{res['filename']:<20} | {res['mass']:<12.2f} | {res['volume']:<15.2f} | {res['density']:<15.4f} | {f_str}")

    # Analisis final
    avg_density = np.mean([r['density'] for r in results])
    print("-" * 100)
    print(f"PROMEDIO LOTE: Densidad = {avg_density:.4f} g/cm3")
    
    if avg_density < 1.2:
        print("CONCLUSION: ¡Materiales ultra-ligeros! Excelentes candidatos para adsorcion de gases.")
    elif avg_density < 2.0:
        print("CONCLUSION: Densidad media. Estructuras estables tipo MOF estandar.")
    else:
        print("CONCLUSION: Materiales densos. Probablemente muy compactos.")

if __name__ == "__main__":
    run_lab()