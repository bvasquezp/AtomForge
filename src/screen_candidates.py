import os
import shutil
import torch
import networkx as nx
import sys

# Configurar rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    from lab_analysis import calculate_properties
except ImportError:
    sys.path.append(os.path.join(current_dir, 'src'))
    from src.lab_analysis import calculate_properties

def get_element_export(degree):
    # Elementos basados en valencia
    if degree >= 6: return "Zr"
    elif 4 <= degree < 6: return "Cu"
    elif degree == 3: return "C"
    elif degree == 2: return "O"
    else: return "H"

def export_balanced_crystal(filename, source_folder, target_folder):
    input_path = os.path.join(source_folder, filename)
    output_filename = filename.replace(".pt", ".xyz")
    output_path = os.path.join(target_folder, output_filename)

    if not os.path.exists(input_path): return

    data = torch.load(input_path, weights_only=False)
    adj_matrix = data['adjacency'].numpy()
    rows, cols = adj_matrix.nonzero()
    
    G = nx.Graph()
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))

    # 1. LIMPIEZA
    if len(G.nodes()) > 0:
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

    # 2. GEOMETRIA EQUILIBRADA (NI MUY CERCA, NI MUY LEJOS)
    # Calculamos una escala basada en la raiz cubica del numero de atomos.
    # Esto mantiene la densidad constante sea cual sea el tamaño del cristal.
    try:
        n_atoms = len(G.nodes())
        # Formula empirica: Escala = 1.8 * (N_atomos)^(1/3)
        # Esto suele dejar los enlaces cerca de 1.5 Angstroms
        optimal_scale = 1.8 * (n_atoms ** (1/3))
        
        pos = nx.kamada_kawai_layout(G, dim=3, scale=optimal_scale)
    except:
        pos = nx.spring_layout(G, dim=3, seed=42, scale=8.0)

    # 3. ESCRITURA
    with open(output_path, "w") as f:
        f.write(f"{len(G.nodes())}\n")
        f.write(f"MOF Equilibrado - {filename}\n")
        
        for node in G.nodes():
            degree = G.degree[node]
            element = get_element_export(degree)
            
            if node in pos:
                # Aseguramos 3D
                coords = pos[node]
                if len(coords) == 2: 
                    x, y = coords; z = 0.0
                else: 
                    x, y, z = coords
                
                f.write(f"{element:<2} {x:.5f} {y:.5f} {z:.5f}\n")
    
    print(f"   -> ✨ 3D Generado (Escala Quimica): {output_filename}")

def screen_and_save():
    project_root = os.path.dirname(current_dir)
    source_dir = os.path.join(project_root, "generated_crystals")
    target_dir = os.path.join(project_root, "top_candidates")
    
    if os.path.exists(target_dir): shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"--- GENERANDO ESTRUCTURAS DE ESCALA REAL ---")
    
    files = [f for f in os.listdir(source_dir) if f.endswith(".pt")]
    
    count = 0
    for f in files:
        try:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(target_dir, f))
            export_balanced_crystal(f, source_dir, target_dir)
            count += 1
        except Exception as e:
            pass
            
    print(f"\n--- LISTO ---")
    print(f"Se generaron {count} archivos. Prueba el 007 en Avogadro ahora.")

if __name__ == "__main__":
    screen_and_save()