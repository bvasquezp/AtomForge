import torch
import torch.optim as optim
import os
import glob
import torch.nn.functional as F
from gnn_model import GraphVAE
import numpy as np

# --- CONFIGURACIÓN ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
DATA_DIR = os.path.join(project_root, "data", "processed")
MODELS_DIR = os.path.join(project_root, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "gnn_mof_v3.pth")

EPOCHS = 200        
HIDDEN_DIM = 64
LATENT_DIM = 32
LR = 0.001          # Bajamos un poco el LR para que sea más estable
ATOM_TYPES_COUNT = 118 

def get_beta(epoch, total_epochs):
    # KL Annealing más suave
    ratio = min(1.0, epoch / (total_epochs * 0.5))
    beta_max = 0.0001 # Beta MUY baja para priorizar que aprenda a reconstruir primero
    return ratio * beta_max

def loss_function(recon_logits, target_adj, mu, log_std, pos_weight, beta):
    # 1. Reconstruction Loss (Weighted BCE)
    loss_rec = F.binary_cross_entropy_with_logits(
        recon_logits, 
        target_adj, 
        pos_weight=pos_weight, 
        reduction='mean'
    )
    
    # 2. KL Divergence
    kl_loss = -0.5 * torch.mean(torch.sum(1 + log_std - mu.pow(2) - log_std.exp(), dim=1))
    
    # Priorizamos MUCHO la reconstrucción multiplicando Beta por un factor pequeño
    total_loss = loss_rec + (beta * kl_loss)
    return total_loss, loss_rec, kl_loss

def load_data():
    search_path = os.path.join(DATA_DIR, "*.pt")
    file_list = glob.glob(search_path)
    dataset = []
    print(f"Cargando datos desde {DATA_DIR}...")
    for file_path in file_list:
        try:
            data = torch.load(file_path, weights_only=False)
            if isinstance(data, list): dataset.extend(data)
            else: dataset.append(data)
        except: pass
    return dataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando V3 (Normalizado) en: {device}")

    dataset = load_data()
    if not dataset: return

    model = GraphVAE(num_atom_types=ATOM_TYPES_COUNT, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

    model.train()
    
    print(f"{'Epoch':^5} | {'Total Loss':^12} | {'Rec Loss':^10} | {'KL Loss':^10} | {'Beta':^8}")
    print("-" * 55)

    for epoch in range(EPOCHS):
        total_loss_accum = 0
        rec_loss_accum = 0
        kl_loss_accum = 0
        
        beta = get_beta(epoch, EPOCHS)

        for data in dataset:
            # 1. Átomos y Adyacencia
            if hasattr(data, 'x'): x = data.x.long().to(device)
            else: x = data['x'].long().to(device)
            x = torch.clamp(x, max=ATOM_TYPES_COUNT-1)

            if hasattr(data, 'edge_index'): edge_index = data.edge_index.to(device)
            else: edge_index = data['edge_index'].to(device)
            
            if hasattr(data, 'num_nodes'): num_nodes = data.num_nodes
            else: num_nodes = data['num_nodes']

            # 2. POSICIONES (Normalización crítica)
            if hasattr(data, 'pos'): pos = data.pos.float().to(device)
            else: pos = data['pos'].float().to(device)
            
            # --- TRUCO 1: NORMALIZACIÓN ---
            # Dividimos por el valor máximo absoluto para tener rango -1 a 1 aprox
            # Añadimos 1e-6 para evitar división por cero
            pos_max = pos.abs().max() + 1e-6
            pos = pos / pos_max
            # ------------------------------

            # 3. Target y Peso Dinámico
            target_adj = torch.zeros((num_nodes, num_nodes)).to(device)
            target_adj[edge_index[0], edge_index[1]] = 1
            
            # PESO DINÁMICO 
            # Calculamos cuántos 0s y cuántos 1s hay en ESTE cristal
            num_ones = target_adj.sum()
            num_zeros = target_adj.numel() - num_ones
            # Si hay muchos ceros, el peso de los unos debe ser alto
            weight_val = num_zeros / (num_ones + 1e-6)
            pos_weight = torch.tensor([weight_val]).to(device)
            # ------------------------------

            optimizer.zero_grad()
            
            recon_logits, mu, log_std = model(x, pos, edge_index)
            
            loss, l_rec, l_kl = loss_function(recon_logits, target_adj, mu, log_std, pos_weight, beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss_accum += loss.item()
            rec_loss_accum += l_rec.item()
            kl_loss_accum += l_kl.item()

        avg_total = total_loss_accum / len(dataset)
        avg_rec = rec_loss_accum / len(dataset)
        avg_kl = kl_loss_accum / len(dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:^5} | {avg_total:^12.4f} | {avg_rec:^10.4f} | {avg_kl:^10.4f} | {beta:^8.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()