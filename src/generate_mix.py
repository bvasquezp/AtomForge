import torch
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gnn_model import GraphVAE

# --- CONFIGURACION ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
DATA_DIR = os.path.join(project_root, "data", "processed")
MODEL_PATH = os.path.join(project_root, "models", "gnn_mof_v3.pth")

HIDDEN_DIM = 64
LATENT_DIM = 32
ATOM_TYPES = 118

# --- NIVEL DE FILTRO ---
# Aqui definimos que tan exigente somos.
# 0.90 significa: "Solo muestra el enlace si estas 90% segura"
STRICT_THRESHOLD = 0.65

def load_crystal(file_path, device):
    """Carga un cristal y lo prepara para la IA"""
    data = torch.load(file_path, weights_only=False)
    
    if isinstance(data, dict):
        x = data['x'].long().to(device)
        edge_index = data['edge_index'].to(device)
        pos = data['pos'].float().to(device)
        num_nodes = data['num_nodes']
    else:
        x = data.x.long().to(device)
        edge_index = data.edge_index.to(device)
        pos = data.pos.float().to(device)
        num_nodes = data.num_nodes

    x = torch.clamp(x, max=ATOM_TYPES-1)
    pos_max = pos.abs().max() + 1e-6
    pos = pos / pos_max
    
    return x, pos, edge_index, num_nodes

def mix_crystals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar Modelo
    model = GraphVAE(num_atom_types=ATOM_TYPES, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    if not os.path.exists(MODEL_PATH):
        print("Error: No hay modelo.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 2. Seleccionar 2 cristales
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if len(files) < 2:
        print("Error: Necesitas al menos 2 archivos .pt.")
        return
    
    file_A = files[0]
    file_B = files[-1] 
    
    print("Mezclando ADN de:")
    print(f"  A: {os.path.basename(file_A)}")
    print(f"  B: {os.path.basename(file_B)}")

    # 3. Obtener Latentes
    xA, posA, edgeA, nodesA = load_crystal(file_A, device)
    xB, posB, edgeB, nodesB = load_crystal(file_B, device)

    with torch.no_grad():
        mu_A, _ = model.encode(xA, posA, edgeA) 
        mu_B, _ = model.encode(xB, posB, edgeB) 

    # --- ESTRATEGIA: CROP & MIX ---
    min_size = min(mu_A.size(0), mu_B.size(0))
    print(f"Ajustando a {min_size} atomos...")
    
    mu_A_cut = mu_A[:min_size, :]
    mu_B_cut = mu_B[:min_size, :]
    
    # Mezclamos
    z_new = (mu_A_cut + mu_B_cut) / 2.0
    
    # Decodificar
    adj_logits_new = model.decode(z_new)
    probs_new = torch.sigmoid(adj_logits_new).cpu().detach().numpy()
    
    # --- DIAGNOSTICO Y FILTRO ---
    max_prob = probs_new.max()
    print(f"--- Diagnostico ---")
    print(f"Confianza maxima: {max_prob:.4f}")
    
    # Logica de seguridad: Si pedimos 0.90 pero la IA solo llega a 0.80, ajustamos
    # para no mostrar una pantalla en blanco.
    final_threshold = STRICT_THRESHOLD
    if max_prob < final_threshold:
        print(f"Aviso: La confianza ({max_prob:.4f}) es menor que el filtro ({final_threshold}).")
        final_threshold = max_prob * 0.95
        print(f"Ajustando filtro automaticamente a {final_threshold:.4f}")

    adj_new_clean = (probs_new > final_threshold).astype(int)

    # 4. Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # A
    with torch.no_grad():
        rec_A = torch.sigmoid(model.decode(mu_A)).cpu().numpy()
        clean_A = (rec_A > STRICT_THRESHOLD).astype(int)
    sns.heatmap(clean_A, ax=axes[0], cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    axes[0].set_title(f"Padre A ({nodesA} atomos)\nFiltro > {STRICT_THRESHOLD}")

    # HIJO
    sns.heatmap(adj_new_clean, ax=axes[1], cmap="Purples", cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title(f"HIBRIDO ({min_size} atomos)\nFiltro > {final_threshold:.2f}", fontsize=14, fontweight='bold')

    # B
    with torch.no_grad():
        rec_B = torch.sigmoid(model.decode(mu_B)).cpu().numpy()
        clean_B = (rec_B > STRICT_THRESHOLD).astype(int)
    sns.heatmap(clean_B, ax=axes[2], cmap="Reds", cbar=False, xticklabels=False, yticklabels=False)
    axes[2].set_title(f"Madre B ({nodesB} atomos)\nFiltro > {STRICT_THRESHOLD}")

    plt.tight_layout()
    print("Graficos generados.")
    plt.show()

if __name__ == "__main__":
    mix_crystals()