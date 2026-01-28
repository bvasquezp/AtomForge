import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
from model import MOFVAE

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/vae_full_v1.pth"
HIDDEN_DIM = 32
LATENT_DIM = 16
NUM_ATOMS = 20
THRESHOLD = 0.75  # <--- CAMBIO: Solo aceptamos enlaces MUY seguros (antes 0.5)

def visualize_material():
    print(f"--- 🎨 PINTANDO CON FILTRO DE CALIDAD ({THRESHOLD*100}%) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MOFVAE(input_dim=1, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Generar
    z = torch.randn(NUM_ATOMS, LATENT_DIM).to(device)
    row, col = torch.meshgrid(torch.arange(NUM_ATOMS), torch.arange(NUM_ATOMS), indexing='ij')
    edge_index_all = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0).to(device)
    mask = edge_index_all[0] != edge_index_all[1]
    edge_index_all = edge_index_all[:, mask]

    with torch.no_grad():
        probs = model.decoder(z, edge_index_all)

    # Filtrar con el nuevo umbral más estricto
    edges = edge_index_all[:, probs > THRESHOLD].cpu().numpy()

    # Construir Grafo
    G = nx.Graph()
    G.add_nodes_from(range(NUM_ATOMS))
    edge_list = list(zip(edges[0], edges[1]))
    G.add_edges_from(edge_list)

    # --- PARTE ESTÉTICA NUEVA ---
    plt.figure(figsize=(10, 8))
    
    # 1. Colores Químicos Falsos (Simulación)
    # Asignamos aleatoriamente roles: Azul (Metal), Gris (Carbono), Rojo (Oxígeno)
    colors = []
    sizes = []
    for _ in range(NUM_ATOMS):
        r = random.random()
        if r < 0.2: # 20% probabilidad de ser Metal (Nodos grandes)
            colors.append('#1f77b4') # Azul
            sizes.append(800)
        elif r < 0.5: # 30% Oxígeno/Nitrógeno
            colors.append('#d62728') # Rojo
            sizes.append(400)
        else: # 50% Carbono (Estructura)
            colors.append('#7f7f7f') # Gris
            sizes.append(300)

    # 2. Layout "Kamada-Kawai"
    # Este algoritmo es mejor para visualizar estructuras químicas que el anterior
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)

    # 3. Dibujar
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.4, edge_color='#333333')
    
    # Calcular densidad nueva
    density = len(edge_list) / (NUM_ATOMS * (NUM_ATOMS - 1))
    
    plt.title(f"Estructura MOF Generada (Threshold: {THRESHOLD})\nDensidad: {density:.1%}", fontsize=14)
    plt.axis('off')
    
    print(f"✨ Densidad bajada a: {density:.1%}")
    print("✨ Abriendo ventana...")
    plt.show()

if __name__ == "__main__":
    visualize_material()