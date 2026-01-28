import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from model import MOFVAE

# --- CONFIGURACIÓN ---
PROCESSED_DIR = os.path.join("data", "processed")
BATCH_SIZE = 4      # Bajamos un poco el batch para grafos complejos
HIDDEN_DIM = 32
LATENT_DIM = 16
EPOCHS = 50         # Más épocas porque ahora la tarea es más difícil
LR = 0.005

# --- DATASET (Igual que antes) ---
class MOFDataset(Dataset):
    def __init__(self, root):
        super(MOFDataset, self).__init__(root)
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith('.pt')]

    def len(self):
        return len(self.files)

    def get(self, idx):
        # Mantenemos el fix de seguridad
        data = torch.load(os.path.join(self.root, self.files[idx]), weights_only=False)
        return data

def main():
    print("--- INICIANDO ENTRENAMIENTO COMPLETO (VAE) ---")
    
    if not os.path.exists(PROCESSED_DIR):
        print("❌ Error: No hay datos procesados.")
        return

    dataset = MOFDataset(root=PROCESSED_DIR)
    # Importante: drop_last=True evita errores con lotes incompletos al final
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando: {device} | Grafos: {len(dataset)}")

    # Inicializar modelo completo
    model = MOFVAE(input_dim=1, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_kl = 0
        total_recon = 0
        
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # 1. Forward
            # La IA intenta predecir la probabilidad de existencia de los enlaces reales
            recon_edges, mu, logstd = model(data)
            
            # 2. CALCULAR PÉRDIDAS
            
            # A) Loss de Reconstrucción (Binary Cross Entropy)
            # Queremos que recon_edges sea 1 (porque estamos pasándole los enlaces reales)
            # También muestreamos enlaces negativos (donde NO hay conexión) para que aprenda a distinguir
            pos_loss = -torch.log(recon_edges + 1e-15).mean()
            
            # Simplificación: Asumimos que la IA debe aprender que los enlaces existen.
            # (En un modelo pro añadiríamos "negative sampling", pero esto basta para empezar)
            recon_loss = pos_loss

            # B) Loss de Regularización (KL Divergence)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))
            
            # Suma total
            loss = recon_loss + (0.01 * kl_loss) # Ponderamos KL para que no domine
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss Total: {total_loss/len(loader):.4f} (Recon: {total_recon/len(loader):.4f} | KL: {total_kl/len(loader):.4f})")

    # Guardar modelo final
    torch.save(model.state_dict(), "models/vae_full_v1.pth")
    print("\n✅ ENTRENAMIENTO COMPLETO FINALIZADO")
    print("Modelo guardado en 'models/vae_full_v1.pth'")

if __name__ == "__main__":
    main()