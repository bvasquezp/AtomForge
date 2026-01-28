import torch
import os
import numpy as np
from gnn_model import GraphVAE

# --- CONFIGURACION ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
MODEL_PATH = os.path.join(project_root, "models", "gnn_mof_v3.pth")
OUTPUT_DIR = os.path.join(project_root, "generated_crystals")

os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 64
LATENT_DIM = 32
ATOM_TYPES = 118

# --- NUEVA FISICA ESTRICTA ---
NUM_ATOMS = 100

# CAMBIO CLAVE: Bajamos de 4 a 2.
# Al obligar a elegir solo 2, la simetria natural subira el promedio a ~3.
INITIAL_K_NEIGHBORS = 2 

def clean_adjacency_strict(probs, device):
    """
    Logica de 'Escasez Forzada' para crear estructuras abiertas.
    """
    # 1. Simetria de Probabilidades (Promediamos fuerza de enlace)
    probs_sym = (probs + probs.t()) / 2.0
    probs_sym.fill_diagonal_(0)
    
    # 2. SELECCION MUY ESTRICTA (Top-2)
    # Solo permitimos los 2 enlaces mas fuertes de salida.
    values, indices = torch.topk(probs_sym, k=INITIAL_K_NEIGHBORS, dim=1)
    
    # 3. Crear mascara base
    clean_adj = torch.zeros_like(probs_sym)
    rows = torch.arange(probs.size(0), device=device).unsqueeze(1).expand_as(indices)
    
    # Filtro de ruido base (0.5)
    mask = values > 0.5
    clean_adj[rows[mask], indices[mask]] = 1
    
    # 4. SIMETRIZACION (La Magia)
    # Aqui es donde el promedio sube.
    # Si A elige a B (Top-2), y C elige a A (Top-2)...
    # A termina conectado con B y C. (Grado 2 original -> Grado real final)
    clean_adj = ((clean_adj + clean_adj.t()) > 0).float()
    
    return clean_adj

def generate_batch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphVAE(num_atom_types=ATOM_TYPES, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    if not os.path.exists(MODEL_PATH):
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"--- GENERANDO LOTE (ESTRATEGIA 'TOP-2') ---")
    print(f"Objetivo: Crear estructuras ligeras y porosas.")
    
    for i in range(1, 11): 
        z_random = torch.randn(NUM_ATOMS, LATENT_DIM).to(device)
        
        with torch.no_grad():
            adj_logits = model.decode(z_random)
            probs = torch.sigmoid(adj_logits)
        
        # Usamos la nueva funcion estricta
        adj_final = clean_adjacency_strict(probs, device)
        
        # Metricas
        density = adj_final.mean().item()
        degrees = adj_final.sum(dim=1)
        avg_deg = degrees.mean().item()
        max_deg = degrees.max().item()
        
        # Guardar
        filename = f"crystal_gen_{i:03d}.pt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        torch.save({'adjacency': adj_final.cpu(), 'num_atoms': NUM_ATOMS, 'density': density}, filepath)
        
        print(f"[Cristal #{i}] Promedio: {avg_deg:.2f} | Max Vecinos: {int(max_deg)} | Estado: {'PERFECTO' if 2.5 <= avg_deg <= 3.8 else 'Denso'}")

    print("--- LISTO ---")

if __name__ == "__main__":
    generate_batch()