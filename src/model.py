import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

# 1. EL ENCODER (Ya lo conoces: Comprime)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        
        # Dos salidas: Media y Varianza
        self.conv_mu = GCNConv(hidden_dim * 2, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim * 2, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# 2. EL DECODER (Nuevo: Reconstruye enlaces)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, z, edge_index):
        # Producto punto: Si dos átomos están cerca en el espacio latente,
        # la IA predice que deberían tener un enlace químico.
        # z: Representación latente de los átomos
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

# 3. EL MODELO COMPLETO (VAE)
class MOFVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MOFVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder()

    def reparameterize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, data):
        # 1. Codificar
        mu, logstd = self.encoder(data.x, data.edge_index)
        
        # 2. Truco de reparametrización
        z = self.reparameterize(mu, logstd)
        
        # 3. Decodificar (Intentar predecir los enlaces originales)
        # Usamos los mismos índices de bordes para entrenar (Link Prediction)
        recon_edges = self.decoder(z, data.edge_index)
        
        return recon_edges, mu, logstd