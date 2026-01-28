# src/generate.py
import torch
import os
from model import MOFVAE

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/vae_full_v1.pth"
LATENT_DIM = 16  # Debe ser igual al del entrenamiento
HIDDEN_DIM = 32  # Debe ser igual al del entrenamiento
NUM_ATOMS = 20   # Vamos a intentar generar una celda pequeña con 20 átomos

def generate_material():
    print("--- 🔮 INICIANDO GENERACIÓN DE NUEVO MATERIAL ---")

    # 1. Cargar el Cerebro (Modelo)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MOFVAE(input_dim=1, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    
    # Cargar los pesos entrenados
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print("Modelo cargado correctamente.")
    else:
        print("Error: No encuentro el modelo entrenado.")
        return

    model.eval() # Poner en modo evaluación (apaga el entrenamiento)

    # 2. Generar el "Alma" de los átomos (Espacio Latente)
    # Creamos vectores aleatorios (ruido normal) para simular átomos nuevos
    print(f"🧪 Inventando {NUM_ATOMS} átomos nuevos desde el espacio latente...")
    z = torch.randn(NUM_ATOMS, LATENT_DIM).to(device)

    # 3. Decodificar: ¿Quién se conecta con quién?
    # Para saber esto, tenemos que probar TODAS las combinaciones posibles de parejas
    # (Átomo 0 con 1, 0 con 2... hasta el final)
    
    # Crear todos los pares posibles
    row, col = torch.meshgrid(torch.arange(NUM_ATOMS), torch.arange(NUM_ATOMS), indexing='ij')
    edge_index_all = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0).to(device)
    
    # Quitar los auto-bucles (un átomo no se conecta consigo mismo en este modelo simplificado)
    mask = edge_index_all[0] != edge_index_all[1]
    edge_index_all = edge_index_all[:, mask]

    # 4. Preguntar al Oráculo (Decoder)
    with torch.no_grad():
        # La IA nos da la probabilidad de enlace para cada par (0 a 1)
        probs = model.decoder(z, edge_index_all)

    # 5. Filtrar: Solo nos quedamos con los enlaces fuertes (> 50% seguridad)
    threshold = 0.5
    edges_kept = edge_index_all[:, probs > threshold]

    # --- RESULTADOS ---
    num_bonds = edges_kept.size(1)
    print(" ¡MATERIAL GENERADO!")
    print(f"🔹 Átomos: {NUM_ATOMS}")
    print(f"🔹 Enlaces químicos predichos: {num_bonds}")
    print(f"🔹 Densidad de conexión: {num_bonds / (NUM_ATOMS * (NUM_ATOMS-1)):.2%}")
    
    if num_bonds > 0:
        print("\nEjemplo de algunos enlaces creados (Átomo A -> Átomo B):")
        for i in range(min(10, num_bonds)): # Mostrar solo los primeros 10
            print(f"  {edges_kept[0][i].item()} -- {edges_kept[1][i].item()}")
    else:
        print("\nLa IA decidió no conectar nada (el material se desmoronó). Intenta de nuevo.")

if __name__ == "__main__":
    generate_material()