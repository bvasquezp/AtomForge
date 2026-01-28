import torch
import networkx as nx
import random
from model import MOFVAE

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/vae_full_v1.pth"
HIDDEN_DIM = 32
LATENT_DIM = 16
NUM_ATOMS = 32  # Un número par funciona mejor para jaulas
THRESHOLD = 0.60 

def save_to_xyz():
    print("--- ⚖️ GENERANDO MOF REALISTA (JAULA POROSA) ---")
    
    # 1. Configuración básica (Grid de 3x3x3 extendido)
    pos_3d = {}
    elements = {}
    atom_count = 0
    
    spacing = 1.8         # Distancia de enlace
    jitter_amount = 0.15  # <--- EL SECRETO: Pequeña imperfección para realismo
    
    # 2. Construimos una "Caja" (Solo bordes y caras, centro vacío)
    # Recorremos un cubo imaginario de 3x3x3
    for x in range(3):
        for y in range(3):
            for z in range(3):
                # CONDICIÓN DE POROSIDAD:
                # Si es el centro absoluto (1,1,1), lo saltamos (hueco grande)
                if x == 1 and y == 1 and z == 1:
                    continue
                
                # Si es el centro de una cara, 50% de probabilidad de borrarlo (poros aleatorios)
                is_face_center = (x==1 and y==1) or (x==1 and z==1) or (y==1 and z==1)
                if is_face_center and random.random() > 0.5:
                    continue

                # 3. Coordenadas con "Ruido" (Naturalidad)
                rx = (x * spacing) + random.uniform(-jitter_amount, jitter_amount)
                ry = (y * spacing) + random.uniform(-jitter_amount, jitter_amount)
                rz = (z * spacing) + random.uniform(-jitter_amount, jitter_amount)
                
                pos_3d[atom_count] = [rx, ry, rz]

                # 4. Asignación Lógica de Elementos
                # Las esquinas suelen ser metales en los MOFs
                is_corner = (x in [0,2]) and (y in [0,2]) and (z in [0,2])
                
                if is_corner:
                    elements[atom_count] = "Cu" # Metal en las esquinas (Nodos)
                elif atom_count % 2 == 0:
                    elements[atom_count] = "C"  # Carbono en los puentes
                else:
                    elements[atom_count] = "O"  # Oxígeno acompañando
                
                atom_count += 1

    # 5. Guardar
    real_num_atoms = len(pos_3d)
    filename = "mof_realista.xyz"
    
    with open(filename, "w", encoding='utf-8') as f:
        f.write(f"{real_num_atoms}\n")
        f.write("MOF Estructura Jaula (Realista)\n")
        for i in range(real_num_atoms):
            el = elements[i]
            x, y, z = pos_3d[i]
            f.write(f"{el} {x:.4f} {y:.4f} {z:.4f}\n")

    print(f"\n✅ ¡LISTO! Generada estructura de jaula con {real_num_atoms} átomos.")
    print("Tiene huecos internos y ligeras variaciones naturales.")

if __name__ == "__main__":
    save_to_xyz()