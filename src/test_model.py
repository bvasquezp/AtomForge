import torch
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gnn_model import GraphVAE

# --- CONFIGURACIÓN ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
DATA_DIR = os.path.join(project_root, "data", "processed")
MODEL_PATH = os.path.join(project_root, "models", "gnn_mof_v3.pth")

HIDDEN_DIM = 64
LATENT_DIM = 32
ATOM_TYPES = 118

def visualize_reconstruction():
    # 1. Cargar Modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphVAE(num_atom_types=ATOM_TYPES, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No encuentro el modelo en: {MODEL_PATH}")
        return

    try:
        # map_location asegura que cargue en CPU si no hay GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except Exception as e:
        print(f"⚠️ Error cargando pesos del modelo: {e}")
        return
        
    model.eval()
    print("🧠 Modelo cargado correctamente.")

    # 2. Cargar un cristal de prueba
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not files:
        print(f"❌ Error: No hay archivos .pt en {DATA_DIR}")
        return
    
    # Probamos con el primer archivo que encuentre
    test_file = files[0] 
    print(f"🧪 Probando con el cristal: {os.path.basename(test_file)}")
    
    try:
        data = torch.load(test_file, weights_only=False)
    except Exception as e:
         print(f"❌ Error cargando el archivo de datos: {e}")
         return

    # 3. Preparar datos (Compatible con diccionarios)
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

    # Clamp y Normalización
    x = torch.clamp(x, max=ATOM_TYPES-1) 
    pos_max = pos.abs().max() + 1e-6
    pos = pos / pos_max

    print("⚙️ Datos procesados. Realizando inferencia en la IA...")

    # 4. PASAR POR LA IA (Inferencia)
    with torch.no_grad():
        recon_logits, _, _ = model(x, pos, edge_index)
        # Probabilidades crudas (borrosas)
        probs = torch.sigmoid(recon_logits).cpu().numpy()

    # 5. Construir Matriz Real (Ground Truth)
    adj_real = np.zeros((num_nodes, num_nodes))
    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    adj_real[rows, cols] = 1

    print("🎨 Dibujando resultados...")

    # 6. DIBUJAR COMPARACIÓN (3 PANELES)
    # Creamos una figura con 1 fila y 3 columnas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # A. Real (Azul)
    sns.heatmap(adj_real, ax=axes[0], cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    axes[0].set_title("1. Realidad (Ground Truth)", fontsize=14)
    
    # B. Imaginación Borrosa (Rojo)
    sns.heatmap(probs, ax=axes[1], cmap="Reds", vmin=0, vmax=1, cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title("2. Imaginación (Probabilidad)", fontsize=14)

    # C. Decisión Final Nítida (Verde)
    # Filtro: Si probabilidad > 50% es un enlace (1), si no, es cero (0)
    adj_clean = (probs > 0.8).astype(int)
    sns.heatmap(adj_clean, ax=axes[2], cmap="Greens", cbar=False, xticklabels=False, yticklabels=False)
    axes[2].set_title("3. Decisión Final (Filtro > 80%)", fontsize=14)
    
    plt.suptitle(f"Análisis Visual de Reconstrucción: {os.path.basename(test_file)}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # ¡ESTA LÍNEA ES LA QUE MUESTRA LA VENTANA!
    print("✨ ¡Ventana de gráficos abierta! (Mira tu barra de tareas si no aparece)")
    plt.show() 
    
    # 7. Diagnóstico numérico
    print("\n--- Diagnóstico Rápido ---")
    max_prob = probs.max()
    mean_prob = probs.mean()
    print(f"Probabilidad Máxima predicha: {max_prob:.4f}")
    print(f"Probabilidad Promedio Global: {mean_prob:.4f}")
    
    if max_prob < 0.5:
        print("⚠️ ALERTA: La IA es muy insegura. Ningún enlace supera el 50% de certeza.")
    else:
        print("✅ La IA detecta estructuras con confianza (>50%).")

if __name__ == "__main__":
    visualize_reconstruction()