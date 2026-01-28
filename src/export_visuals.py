import torch
import torch.onnx
from gnn_model import GraphVAE

# 1. Configuración básica
HIDDEN_DIM = 64
LATENT_DIM = 32
ATOM_TYPES = 118
MODEL_PATH = "models/gnn_mof_v3.pth" # Asegúrate de que este nombre coincida con tu último save

def export_model_to_onnx():
    print("Preparando la cámara para tomar una foto al modelo...")
    
    # 2. Instanciar el modelo (vacío)
    model = GraphVAE(num_atom_types=ATOM_TYPES, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    
    # 3. Crear datos falsos (Dummies) para que el modelo crea que está trabajando
    #    Necesitamos esto para trazar el camino que siguen los datos.
    #    Simulamos 5 átomos, conectados entre sí.
    num_nodes = 5
    dummy_x = torch.tensor([0, 5, 1, 0, 8], dtype=torch.long) # 5 átomos
    dummy_pos = torch.randn(num_nodes, 3) # 5 posiciones (X, Y, Z)
    dummy_edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long)
    
    # 4. Exportar a ONNX
    #    ONNX graba "qué pasa" cuando los datos entran.
    torch.onnx.export(
        model,
        (dummy_x, dummy_pos, dummy_edge_index), # Tuplas de entrada
        "models/architecture.onnx", # Archivo de salida
        export_params=True,
        opset_version=11,
        input_names=['Atom_Types', 'Positions', 'Connections'], # Nombres bonitos para el gráfico
        output_names=['Adjacency_Logits', 'Mu', 'Log_Std'],
        dynamic_axes={
            'Atom_Types': {0: 'num_nodes'},
            'Positions': {0: 'num_nodes'},
            'Connections': {1: 'num_edges'}
        }
    )
    
    print(" ¡Listo! Se ha creado el archivo 'models/architecture.onnx'")
    print(" AHORA: Ve a la página https://netron.app/ y arrastra ese archivo ahí.")

if __name__ == "__main__":
    export_model_to_onnx()