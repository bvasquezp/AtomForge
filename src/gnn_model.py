import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphVAE(nn.Module):
    def __init__(self, num_atom_types, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()
        
        # 1. Embedding para el TIPO de átomo (C, Cu, O...)
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        # 2. ENCODER
        # --- CAMBIO CLAVE: ---
        # La entrada ahora es: (Embedding del átomo) + (3 coordenadas X,Y,Z)
        input_dim = hidden_dim + 3 
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim) 
        
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)

    # --- CAMBIO CLAVE: Añadimos 'pos' a los argumentos ---
    def encode(self, x, pos, edge_index):
        # A. Convertimos tipo de átomo a vector
        h = self.embedding(x.squeeze().long())
        
        # B. Concatenamos la Posición (Geometry-Awareness)
        # Aquí unimos la info química (h) con la geométrica (pos)
        h = torch.cat([h, pos], dim=1) 
        
        # C. Pasamos por las capas GCN
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)

    def reparameterize(self, mu, log_std):
        if self.training:
            std = torch.exp(0.5 * log_std)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        # Producto punto para predecir si hay enlace
        adj_logits = torch.matmul(z, z.t())
        return adj_logits

    # --- CAMBIO CLAVE: Añadimos 'pos' aquí también ---
    def forward(self, x, pos, edge_index):
        # Ahora el modelo recibe las 3 cosas: Qué es (x), Dónde está (pos), Conexiones (edge_index)
        mu, log_std = self.encode(x, pos, edge_index)
        z = self.reparameterize(mu, log_std)
        adj_logits = self.decode(z)
        return adj_logits, mu, log_std