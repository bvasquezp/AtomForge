import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

def get_atom_type_pro(degree):
    """
    Clasificacion avanzada basada en tus nuevos resultados (Top-2 strategy).
    """
    if degree >= 6:
        # Los anclajes maestros (Clusters de Zr, Fe, etc.)
        return "Super Cluster (Metal)", "#800080", 350  # MORADO OSCURO (Gigante)
    elif 4 <= degree < 6:
        # Metales conectores estandar
        return "Nodo Metalico", "#FF4500", 200        # NARANJA (Grande)
    elif degree == 3:
        # Ramificaciones organicas
        return "Carbono (Ramificado)", "#333333", 100 # GRIS (Mediano)
    elif degree == 2:
        # Puentes lineales
        return "Linker (C/O)", "#A9A9A9", 80          # GRIS CLARO (Pequeno)
    else: 
        # Terminaciones
        return "Terminal (H)", "#ADD8E6", 50          # AZUL CIELO (Muy pequeno)

def visualize_chem(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    filepath = os.path.join(project_root, "generated_crystals", filename)

    if not os.path.exists(filepath):
        print(f"Error: No encuentro {filename}")
        return

    print(f"--- ANALISIS DE ARQUITECTURA: {filename} ---")
    data = torch.load(filepath, weights_only=False)
    adj_matrix = data['adjacency'].numpy()
    
    rows, cols = adj_matrix.nonzero()
    G = nx.Graph()
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    G.remove_nodes_from(list(nx.isolates(G)))

    node_colors = []
    node_sizes = []
    
    # Analisis de componentes
    hub_count = 0
    
    for node in G.nodes():
        degree = G.degree[node]
        name, color, size = get_atom_type_pro(degree)
        node_colors.append(color)
        node_sizes.append(size)
        
        if degree >= 6:
            hub_count += 1

    # --- DIBUJAR CON FISICA ---
    plt.figure(figsize=(12, 12))
    
    # Kamada-Kawai es mejor para separar clusters densos que Spring
    print("Calculando disposicion fisica (esto toma un segundo)...")
    pos = nx.kamada_kawai_layout(G) 
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors="white")
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.3, edge_color="#555555")
    
    # Leyenda
    legend_patches = [
        mpatches.Patch(color='#800080', label='Super Cluster (>=6 enlaces)'),
        mpatches.Patch(color='#FF4500', label='Nodo Metalico (4-5 enlaces)'),
        mpatches.Patch(color='#333333', label='Carbono Ramificado (3 enlaces)'),
        mpatches.Patch(color='#A9A9A9', label='Puente Lineal (2 enlaces)'),
        mpatches.Patch(color='#ADD8E6', label='Terminal (1 enlace)')
    ]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.title(f"Estructura Generada: {filename}\nNucleos Densos: {hub_count} | Densidad Global: {data['density']:.4f}", fontsize=14)
    plt.axis("off")
    
    print(f"Se encontraron {hub_count} Super-Clusters masivos.")
    print("Abriendo visor...")
    plt.show()

if __name__ == "__main__":
    # Prueba con el cristal 4 o 5, que tenian muchos vecinos
    visualize_chem("crystal_gen_004.pt")