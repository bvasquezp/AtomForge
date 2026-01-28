import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gnn_model import GraphVAE

# --- CONFIGURACION ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
MODEL_PATH = os.path.join(project_root, "models", "gnn_mof_v3.pth")

HIDDEN_DIM = 64
LATENT_DIM = 32
ATOM_TYPES = 118

# CONFIGURACION DE LIMPIEZA
NUM_ATOMS = 100         # Tamano del cristal
MAX_NEIGHBORS = 4       # Maximo de vecinos permitidos por atomo (fisica real)
HARD_THRESHOLD = 0.5    # Filtro de respaldo

def generate_smart_crystal():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar Modelo
    model = GraphVAE(num_atom_types=ATOM_TYPES, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    if not os.path.exists(MODEL_PATH):
        print("Error: No hay modelo entrenado.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"Generando estructura inteligente (Max {MAX_NEIGHBORS} vecinos)...")

    # 2. GENERAR RUIDO LATENTE
    z_random = torch.randn(NUM_ATOMS, LATENT_DIM).to(device)

    # 3. DECODIFICAR
    with torch.no_grad():
        adj_logits = model.decode(z_random)
        probs = torch.sigmoid(adj_logits)

    # 4. LIMPIEZA AVANZADA (Simetria + Top-K)
    
    # A. Simetria
    probs_sym = (probs + probs.t()) / 2.0
    probs_sym.fill_diagonal_(0)
    
    # B. Estrategia Top-K (Solo los K enlaces mas fuertes por atomo)
    # Creamos una matriz de ceros
    clean_adj = torch.zeros_like(probs_sym)
    
    # Para cada fila (atomo), buscamos los valores mas altos
    values, indices = torch.topk(probs_sym, k=MAX_NEIGHBORS, dim=1)
    
    # Filtro adicional: Solo mantenemos el Top-K si supera el umbral minimo
    # Esto evita conectar atomos si la probabilidad es muy baja
    mask = values > HARD_THRESHOLD
    
    # Rellenamos la matriz limpia
    # Esto es un poco tecnico, pero basicamente pone un 1 donde estan los indices fuertes
    rows = torch.arange(NUM_ATOMS, device=device).unsqueeze(1).expand_as(indices)
    clean_adj[rows[mask], indices[mask]] = 1
    
    # Hacemos simetrica la matriz binaria final (si A elige a B, B conecta con A)
    clean_adj = ((clean_adj + clean_adj.t()) > 0).float()

    # Convertir a Numpy para visualizar
    probs_np = probs_sym.cpu().numpy()
    adj_final_np = clean_adj.cpu().numpy()

    # Diagnostico de Densidad
    density = adj_final_np.mean()
    print(f"--- Resultado ---")
    print(f"Densidad Final: {density:.4f} (Ideal: 0.02 - 0.08)")
    print(f"Atomo promedio tiene: {density * NUM_ATOMS:.1f} vecinos")

    # 5. VISUALIZAR
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Lo que la IA imaginaba (Probabilidad)
    sns.heatmap(probs_np, ax=axes[0], cmap="Reds", vmin=0, vmax=1, cbar=True)
    axes[0].set_title("1. Probabilidad Cruda (La 'Mancha')", fontsize=12)

    # Panel 2: La Estructura Cristalina (Limpia)
    sns.heatmap(adj_final_np, ax=axes[1], cmap="Greens", cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title(f"2. CRISTAL GENERADO (Top-{MAX_NEIGHBORS})", fontsize=14, fontweight='bold')

    plt.suptitle(f"Generacion Inteligente de Material (N={NUM_ATOMS})")
    plt.tight_layout()
    
    print("Ventana grafica abierta.")
    plt.show()

if __name__ == "__main__":
    generate_smart_crystal()