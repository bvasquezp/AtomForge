import torch
import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_graph_structure(filename):
    # 1. Configurar rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    filepath = os.path.join(project_root, "generated_crystals", filename)

    if not os.path.exists(filepath):
        print(f"Error: No encuentro {filename}")
        return

    # 2. Cargar la matriz
    print(f"Cargando estructura: {filename}...")
    data = torch.load(filepath, weights_only=False)
    adj_matrix = data['adjacency'].numpy()
    
    # 3. Convertir Matriz a Grafo (NetworkX)
    # NetworkX es una libreria experta en conectar puntos
    rows, cols = adj_matrix.nonzero()
    edges = zip(rows.tolist(), cols.tolist())
    
    G = nx.Graph()
    G.add_nodes_from(range(data['num_atoms']))
    G.add_edges_from(edges)
    
    # Eliminar atomos que quedaron huerfanos (sin enlaces) para limpiar la vista
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Atomos conectados visibles: {G.number_of_nodes()} (Se ocultaron {len(isolated_nodes)} huerfanos)")

    # 4. DIBUJAR (La Magia)
    plt.figure(figsize=(10, 10))
    
    # Usamos "kamada_kawai_layout" o "spring_layout"
    # Son algoritmos fisicos que separan los nodos para que se vea bonito
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="#6A0DAD", alpha=0.8) # Morado
    
    # Dibujar enlaces
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color="gray")
    
    plt.title(f"Estructura del Cristal: {filename}\n(Topologia Generada )", fontsize=15)
    plt.axis("off") # Quitar ejes X/Y feos
    
    print("Ventana grafica abierta. ¡Mira tu molecula!")
    plt.show()

if __name__ == "__main__":
    # Prueba con el 001, o cambia al 002, 003...
    visualize_graph_structure("crystal_gen_001.pt")