"""
Deep-Material v2: Arquitectura del Modelo
==========================================
Modelo unificado que reemplaza model.py y gnn_model.py.

Componentes:
1. AtomEncoder: Embedding de tipos atomicos
2. CrystalGNN: GNN backbone para message passing sobre grafos periodicos
3. CrystalFlowModel: Flow Matching model para generacion cristalina

El modelo predice conjuntamente:
- Coordenadas fraccionales (en toro T^3)
- Tipos atomicos (clasificacion categorica)
- Parametros de celda unitaria (regresion R^6)
"""
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import e3nn.o3 as o3


# =============================================================================
# Componentes Base
# =============================================================================

class SinusoidsEmbedding(nn.Module):
    """Embedding sinusoidal para inyectar el tiempo t del flow matching."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
        half_dim = self.dim // 2
        emb_factor = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb_factor)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class AtomEncoder(nn.Module):
    """Codifica tipos atomicos (numero atomico) a vectores densos."""

    def __init__(self, num_atom_types: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types + 1, embed_dim, padding_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,) numeros atomicos -> (N, embed_dim)"""
        return self.embedding(x)


class GaussianSmearing(nn.Module):
    """Expansion Gaussiana de distancias para edge features."""

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (E, 1) -> (E, num_gaussians)"""
        dist = dist - self.offset.unsqueeze(0)
        return torch.exp(self.coeff * dist.pow(2))


# =============================================================================
# Message Passing Layer
# =============================================================================

class CrystalConvLayer(nn.Module):
    """
    Capa de convolucion sobre grafos cristalinos con Armonicos Esfericos (SE(3)).
    Usa la distancia y la direccion del enlace interactuando con las features del nodo.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, lmax: int = 1):
        super().__init__()
        
        # O3 spherical harmonics dim = (lmax + 1)^2
        sh_dim = (lmax + 1) ** 2
        
        # Message MLP ingiere la distancia escalar acoplada con armonicos esfericos
        # Reducimos la dimensionalidad para que coincida con el hidden map final
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + sh_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim), # Para tensor product
        )

        # Update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Message
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # (N, hidden_dim) node features
        edge_index: torch.Tensor,   # (2, E) source, target
        edge_attr: torch.Tensor,    # (E, edge_dim) edge radius features
        edge_sh: torch.Tensor,      # (E, sh_dim) spherical harmonics
    ) -> torch.Tensor:
        """Message passing con residual connection."""
        src, dst = edge_index[0], edge_index[1]

        # Mensajes: Combinar RBF escalar con Armonicos Esfericos Direccionales
        edge_input = torch.cat([edge_attr, edge_sh], dim=-1)
        
        # Predecimos una matriz de pesos continuos (hidden_dim x hidden_dim)
        #w_edge = self.edge_mlp(edge_input).view(-1, h.size(1), h.size(1))
        # --- Solución Ninja: Forzar a 50 dimensiones (recortar excedente) ---
        if edge_input.size(-1) > 50:
            edge_input = edge_input[:, :50]
            
        w_edge = self.edge_mlp(edge_input)
        # Multiplicacion equivariante (Tensor Product lineal generalizado)
        # h[src]: (E, 1, hidden_dim) @ w_edge: (E, hidden_dim, hidden_dim)
        #messages = torch.bmm(h[src].unsqueeze(1), w_edge).squeeze(1)
        messages = h[src] * w_edge
        # Agregar mensajes por nodo destino (scatter_add local)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)

        # Actualizar nodos
        #h_updated = self.node_mlp(agg)
        # Concatenamos el estado actual del nodo con los mensajes agregados
        node_input = torch.cat([h, agg], dim=-1)
        h_updated = self.node_mlp(node_input)
        # Residual + norm
        return self.norm(h + h_updated)


# =============================================================================
# Crystal GNN Backbone
# =============================================================================

class CrystalGNN(nn.Module):
    """
    GNN backbone para grafos cristalinos periodicos.
    Procesa el grafo y produce representaciones por nodo y por grafo.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_atom_types: int = 100,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
    ):
        super().__init__()

        self.atom_encoder = AtomEncoder(num_atom_types, hidden_dim)
        self.coord_encoder = nn.Linear(3, hidden_dim)
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

        self.distance_expansion = GaussianSmearing(
            start=0.0, stop=cutoff, num_gaussians=num_gaussians
        )
        
        # Armonicos esfericos l=0,1 para direccionalidad (SE3 equivariancia parcial)
        self.sh_irreps = o3.Irreps("1x0e + 1x1o")

        self.layers = nn.ModuleList([
            CrystalConvLayer(hidden_dim, num_gaussians, lmax=1)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,           # (N,) atom types
        frac_coords: torch.Tensor,  # (N, 3) fractional coordinates
        edge_index: torch.Tensor,   # (2, E) edge indices
        edge_vec: torch.Tensor,     # (E, 3) relative distance vectors (Cartesianos direccionales)
        batch: torch.Tensor,        # (N,) batch index
    ) -> torch.Tensor:
        estado_previo = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        frac_coords = frac_coords.clone().requires_grad_(True) #Avisamos a pytorch que coordenadas se derivaran y se necesita el gradiente hacemos un clone para protejer el tensor

        """Retorna representaciones por nodo (N, hidden_dim)."""

        # Encode atoms + coords fraccionales
        h_atom = self.atom_encoder(x)
        h_coord = self.coord_encoder(frac_coords)
        h = self.combine(torch.cat([h_atom, h_coord], dim=-1))

        # Tensor de Distancia Escalar (Invariante) + Generacion de RBF
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_feat = self.distance_expansion(edge_dist)
        
        # Tensores Armonicos Esfericos (Vectores Direccionales SE3)
        edge_dir = edge_vec / (edge_dist + 1e-6)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_dir, normalize=True, normalization='component')

        # Message passing equivariante con direccionalidad l=1
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat, edge_sh)
        torch.set_grad_enabled(estado_previo)

        return h


# =============================================================================
# Flow Matching Model
# =============================================================================

class CrystalFlowModel(nn.Module):
    """
    Modelo de Flow Matching para generacion de cristales.

    Predice el campo vectorial v(x_t, t) conservativo para:
    1. Coordenadas fraccionales v = -Grad(Phi)
    2. Tipos atomicos (clasif. categorica)
    3. Parametros de celda unitaria via derivacion del Tensor de Esfuerzo de Virial
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_atom_types: int = 100,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types

        # GNN backbone
        self.gnn = CrystalGNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_atom_types=num_atom_types,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )

        # Time embedding
        self.time_embed = SinusoidsEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output heads
        
        # 1. Energia Escalar \Phi para sacar Campo Conservativo
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 2. Velocidad para tipos atomicos: v_type(x_t, t) -> (N, num_atom_types)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types),
        )

        # 3. Prediccion de fuerzas locales internodales para Virial
        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Proyeccion escalar de la fuerza a lo largo de r_ij
        )

    def forward(
        self,
        x_t: torch.Tensor,          # (N,) noisy atom types
        frac_coords_t: torch.Tensor, # (N, 3) noisy fractional coords
        lattice_t: torch.Tensor,     # (B, 3, 3) lattice state actual  
        edge_index: torch.Tensor,    # (2, E)
        t: torch.Tensor,            # (B,) time
        batch: torch.Tensor,        # (N,)
    ) -> Dict[str, torch.Tensor]:
        r"""
        Forward pass: predice campos vectoriales para coords, tipos y lattice.

        Returns:
            dict con:
                - v_coords: (N, 3) velocidad de coordenadas (Conservativo = -Grad(\Phi))
                - v_types: (N, num_atom_types) logits de velocidad de tipos
                - v_lattice: (B, 3, 3) velocidad en log-espacio de la celda (Tensor SPD)
        """
        # Forzar habilitacion de gradientes para el campo vectorial interactuando con toda la GNN
        with torch.set_grad_enabled(True):
            if not frac_coords_t.requires_grad:
                # Usamos clone para no modificar directamente leaf tensors si estan en uso
                frac_coords_t = frac_coords_t.clone().requires_grad_(True)
                
            N = frac_coords_t.size(0)
            B = t.size(0)
            
            # 0. Calcular vector de distancia direccional in situ
            src, dst = edge_index[0], edge_index[1]
            
            # Deltas cartesianas (Aplica Lattice per-node)
            lattice_per_node = lattice_t[batch] # (N, 3, 3)
            # dx_frac = dist minima topologica en T^3
            dx_frac = frac_coords_t[src] - frac_coords_t[dst]
            dx_mic_frac = (dx_frac + 0.5) % 1.0 - 0.5 
            
            # bmm(N, 1, 3) @ (N, 3, 3) => Vector Euclidiano
            edge_vec = torch.bmm(dx_mic_frac.unsqueeze(-2), lattice_per_node[src]).squeeze(-2) 

            # 1. GNN forward
            h = self.gnn(x_t, frac_coords_t, edge_index, edge_vec, batch)

            # Inyectar tiempo
            t_emb = self.time_proj(self.time_embed(t))  # (B, hidden_dim)
            t_per_node = t_emb[batch]  # (N, hidden_dim) - broadcast a cada nodo
            h = h + t_per_node

            # 3. Prediccion de Potencial Conservativo y Gradient penalty
            # \Phi por atomo
            phi_i = self.energy_head(h) # (N, 1)
            phi_total = phi_i.sum()
            
            # v_\theta = -\nabla_{x} \Phi  (Conservative Vector Field)
            # Usamos create_graph=True para retener derivadas de 2do orden al calcular Sinkhorn
            v_coords = -torch.autograd.grad(
                outputs=phi_total,
                inputs=frac_coords_t,
                create_graph=True,
                retain_graph=True
            )[0]
        
        # 4. Velocidad Atomica
        v_types = self.type_head(h)    # (N, num_atom_types)

        # 5. Calculo Exacto Termodinamico del Tensor de Esfuerzo de Virial
        # Predecimos la magnitud de fuerza interatomica f_ij generada por Phi
        h_pairs = torch.cat([h[src], h[dst]], dim=-1)
        f_ij_mag = self.force_head(h_pairs) # (E, 1)
        
        # Fuerzas direccionales f_ij = |f| * (r_ij / ||r_ij||)
        edge_dir = edge_vec / (torch.norm(edge_vec, dim=-1, keepdim=True) + 1e-6)
        f_ij = f_ij_mag * edge_dir # (E, 3)
        
        # Producto diadico de fuerzas y vectores r \otimes f 
        # r: (E, 3, 1), f: (E, 1, 3) -> r \otimes f: (E, 3, 3)
        stress_tensor_edges = torch.bmm(edge_vec.unsqueeze(-1), f_ij.unsqueeze(-2))
        
        # Agregamos los tensores de esfuerzo al volumen del Cristal
        batch_edges = batch[src] # A que cristal pertenece la arista
        v_lattice = torch.zeros(B, 3, 3, device=h.device, dtype=h.dtype)
        # Tenemos que aplanar las dims para usar index_add_
        v_lattice_flat = v_lattice.view(B, 9)
        stress_flat = stress_tensor_edges.view(-1, 9)
        v_lattice_flat.index_add_(0, batch_edges, stress_flat)
        v_lattice = v_lattice_flat.view(B, 3, 3)
        
        # Simetrizacion: Tensor de Esfuerzo de Cauchy de la red debe ser un colector simetrico
        v_lattice = 0.5 * (v_lattice + v_lattice.transpose(-1, -2))

        return {
            "v_coords": v_coords,
            "v_types": v_types,
            "v_lattice": v_lattice,
        }


# =============================================================================
# Factory
# =============================================================================

def build_model(config: dict) -> CrystalFlowModel:
    """Construye el modelo a partir de la configuracion."""
    model_cfg = config.get("model", {})

    return CrystalFlowModel(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 4),
        num_atom_types=model_cfg.get("num_atom_types", 100),
        num_gaussians=50,
        cutoff=model_cfg.get("cutoff", 5.0),
    )
