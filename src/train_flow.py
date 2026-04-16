"""
==============================================
Entrenamiento del CrystalFlowModel usando Conditional Flow Matching.

Algoritmo:
1. Samplear dato real (x_1) del dataset
2. Samplear ruido (x_0) de distribucion base
3. Samplear tiempo t ~ U(0, 1)
4. Interpolar: x_t = (1-t)*x_0 + t*x_1
5. Target velocity: u_t = x_1 - x_0
6. Predecir v_theta(x_t, t)
7. Loss = ||v_theta - u_t||^2
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_config, resolve_paths, set_seed, get_device, save_checkpoint, LogEuclideanExp
from dataset import MOFDataset, collate_periodic_graphs
from crystal_model import build_model


# =============================================================================
# Flow Matching Utilities
# =============================================================================

def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    """Muestrea t uniformemente en (eps, 1-eps) para estabilidad."""
    eps = 1e-5
    return torch.rand(batch_size, device=device) * (1 - 2 * eps) + eps


def wrap_coords(coords: torch.Tensor) -> torch.Tensor:
    """Envuelve coordenadas fraccionales al rango [0, 1) (toro)."""
    return coords % 1.0


def logm_sym(L: torch.Tensor) -> torch.Tensor:
    """Mapeo del Espacio SPD(3) al Espacio Tangente Sym(3) (Log-Euclidiano)."""
    L_sym = 0.5 * (L + L.transpose(-1, -2))
    eigvals, U = torch.linalg.eigh(L_sym, UPLO='U')
    log_eigvals = torch.log(eigvals.clamp(min=1e-7))
    A = U @ torch.diag_embed(log_eigvals) @ U.transpose(-2, -1)
    return A


def sinkhorn_knopp(x: torch.Tensor, y: torch.Tensor, batch: torch.Tensor, species: torch.Tensor, epsilon: float = 0.1, n_iters: int = 5):
    """
    Distancia de Transporte Optimo (Sinkhorn) corregida.
    Optimizacion RAM: Procesa cada cristal de forma independiente (O(N_i^2) vs O(N_batch^2)).
    Retorna (loss_total, W_Zn, W_C, W_H).
    """
    batch_size = int(batch.max().item() + 1)
    total_loss = 0.0
    w_stats = {"Zn": [], "C": [], "H": []}
    
    for i in range(batch_size):
        mask = (batch == i)
        xi, yi, si = x[mask], y[mask], species[mask]
        Ni = xi.size(0)
        if Ni == 0: continue
        
        # Matriz de Costos Local (MIC)
        dx = xi.unsqueeze(1) - yi.unsqueeze(0)
        dx_mic = (dx + 0.5) % 1.0 - 0.5
        C = torch.sum(dx_mic**2, dim=-1)
        
        K = torch.exp(-C / epsilon)
        
        mu = torch.ones(Ni, 1, device=x.device) / Ni
        nu = torch.ones(Ni, 1, device=x.device) / Ni
        
        u = torch.ones(Ni, 1, device=x.device) / Ni
        v = torch.ones(Ni, 1, device=x.device) / Ni
        
        for _ in range(n_iters):
            u = mu / (K @ v + 1e-8)
            v = nu / (K.t() @ u + 1e-8)
            
        P = u * K * v.t()
        li = torch.sum(P * C)
        total_loss += li
        
        # Metricas por especie discretizadas
        def get_w(z):
            mz = (si == z)
            if not mz.any(): return 0.0
            # Costo normalizado por la fraccion de la masa de la especie
            return (torch.sum(P[mz, :] * C[mz, :]) / (mz.sum() / Ni)).item()
            
        w_stats["Zn"].append(get_w(30))
        w_stats["C"].append(get_w(6))
        w_stats["H"].append(get_w(1))

    return total_loss / batch_size, np.mean(w_stats["Zn"]), np.mean(w_stats["C"]), np.mean(w_stats["H"])


class PhysicsInformedLoss(nn.Module):
    """
    Calcula la perdida de Flow Matching con fisica informada (Fase 1).
    Contempla el calculo Log-Euclidiano y prepara la base para Sinkhorn/Divergencia.
    """
    def __init__(self, num_atom_types: int):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.lambda_coords = 5.0
        self.lambda_lattice = 0.5
        self.lambda_types = 0.1
        self.lambda_sym = 1.0
        self.lambda_div = 0.001  # Ajustado: Frenar colapso de densidad

    def forward(self, model: nn.Module, batch: dict, device: torch.device) -> dict:
        # Mover datos al dispositivo
        x_1 = batch["x"].to(device)                     # (N,) atom types target
        frac_1 = batch["frac_coords"].to(device)         # (N, 3) target coords
        edge_index = batch["edge_index"].to(device)      # (2, E)
        lattice_1 = batch["lattice"].to(device)          # (B, 3, 3) target lattice
        batch_idx = batch["batch"].to(device)            # (N,)

        batch_size = lattice_1.shape[0]

        # --- 1. Samplear ruido base x_0 ---
        frac_0 = torch.rand_like(frac_1)

        type_1_onehot = F.one_hot(x_1.long(), num_classes=self.num_atom_types).float()
        type_0 = torch.randn_like(type_1_onehot) * 0.1 + (1.0 / self.num_atom_types)

        A_1 = logm_sym(lattice_1)
        A_0 = torch.randn_like(A_1)
        A_0 = 0.5 * (A_0 + A_0.transpose(-1, -2))

        # --- 2. Samplear tiempo ---
        t = sample_time(batch_size, device)
        t_per_node = t[batch_idx].unsqueeze(-1)

        # --- 3. Interpolar ---
        target_v_coords = (frac_1 - frac_0 + 0.5) % 1.0 - 0.5
        frac_t = (frac_0 + t_per_node * target_v_coords) % 1.0

        target_v_types = type_1_onehot - type_0
        type_t = (1 - t_per_node) * type_0 + t_per_node * type_1_onehot

        target_v_lattice = A_1 - A_0
        t_lattice = t.view(-1, 1, 1)
        A_t = (1 - t_lattice) * A_0 + t_lattice * A_1
        
        lattice_t = LogEuclideanExp.apply(A_t)

        # Preparamos para Fase 2 (Divergencia de Liouville)
        frac_t.requires_grad_(True)

        # --- 4. Forward del modelo Conservativo ---
        x_t_discrete = type_t.argmax(dim=-1)

        predictions = model(
            x_t=x_t_discrete,
            frac_coords_t=frac_t,
            lattice_t=lattice_t,
            edge_index=edge_index,
            t=t,
            batch=batch_idx,
        )

        # --- 5. L_total: MSE ---
        loss_coords = F.mse_loss(predictions["v_coords"], target_v_coords)
        loss_types = F.mse_loss(predictions["v_types"], target_v_types)
        loss_lattice = F.mse_loss(predictions["v_lattice"], target_v_lattice)

        # --- 6. Topologia: L_sym (Transporte Optimo con Annealing) ---
        x_pred = (frac_t + predictions["v_coords"] * (1.0 - t_per_node)) % 1.0
        x_pred_sym = (1.0 - x_pred) % 1.0
        
        loss_sym_raw, W_Zn, W_C, W_H = sinkhorn_knopp(x_pred, x_pred_sym, batch_idx, x_1)
        gamma_t = (t ** 4).mean()
        loss_sym = gamma_t * loss_sym_raw

        # --- 7. Divergencia (Hutchinson) ---
        eps = torch.randn_like(predictions["v_coords"])
        e_dzdx = torch.autograd.grad(
            outputs=predictions["v_coords"],
            inputs=frac_t,
            grad_outputs=eps,
            create_graph=True,
            retain_graph=True
        )[0]
        div = torch.sum(e_dzdx * eps, dim=1)
        loss_div_raw = div.mean()
        # Clamping para evitar divergencia negativa creciente (observada en run de 500 epocas)
        loss_div = loss_div_raw.clamp(-10.0, 10.0)

        loss_total = (self.lambda_coords * loss_coords +
                      self.lambda_lattice * loss_lattice + 
                      self.lambda_types * loss_types + 
                      self.lambda_sym * loss_sym +
                      self.lambda_div * loss_div)

        return {
            "total": loss_total,
            "coords": loss_coords.item(),
            "types": loss_types.item(),
            "lattice": loss_lattice.item(),
            "sym": loss_sym.item(),
            "div": loss_div.item(),
            "W_Zn": W_Zn,
            "W_C": W_C,
            "W_H": W_H,
            "gamma": gamma_t.item()
        }


# =============================================================================
# Training Loop
# =============================================================================

from typing import Optional
def train(config: dict, resume_checkpoint: Optional[str] = None) -> None:
    """Loop principal de entrenamiento."""
    # Setup
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)
    
    device = get_device(config.get("project", {}).get("device", "auto"))
    print(f"Dispositivo: {device}")
    
    # Optimizar threads CPU
    torch.set_num_threads(10)
    torch.set_num_interop_threads(2)
    print(f"Usando {torch.get_num_threads()} CPU threads")
    print(f"Seed: {seed}")

    # Paths
    paths = config["paths"]
    os.makedirs(paths["models"], exist_ok=True)
    os.makedirs(os.path.join(paths["models"], "checkpoints"), exist_ok=True)
    os.makedirs(paths.get("logs", "logs"), exist_ok=True)

    # Dataset
    train_dataset = MOFDataset(paths["processed"], split="train")
    val_dataset = MOFDataset(paths["processed"], split="val")

    if len(train_dataset) == 0:
        print("No hay datos de entrenamiento. Ejecuta primero: python src/dataset.py")
        return

    # Logging Setup
    log_cfg = config.get("logging", {})
    tracker = log_cfg.get("tracker", "none")
    if tracker == "wandb":
        import wandb
        wandb.init(
            project=log_cfg.get("project_name", "deep-material-v2"),
            config=config,
            name="Final_Run_8_Cores_Recovered" if resume_checkpoint else "Final_Run_8_Cores",
            resume="allow" if resume_checkpoint else None
        )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_cfg = config.get("training", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_periodic_graphs,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_periodic_graphs,
        drop_last=False,
    ) if len(val_dataset) > 0 else None

    # Modelo
    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {num_params:,} parametros entrenables")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-6),
    )

    # Scheduler
    epochs = train_cfg.get("epochs", 300)
    warmup = train_cfg.get("warmup_epochs", 10)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup, eta_min=1e-7
    )

    # Mixed precision
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    num_atom_types = config.get("model", {}).get("num_atom_types", 100)

    # Criterion
    criterion = PhysicsInformedLoss(num_atom_types).to(device)

    # Training
    best_val_loss = float("inf")
    start_epoch = 0
    
    if resume_checkpoint:
        print(f"♻️ Cargando checkpoint: {resume_checkpoint}")
        from utils import load_checkpoint
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("loss", float("inf"))
        print(f"🚀 Reanudando desde época {start_epoch}")

    log_every = config.get("logging", {}).get("log_every_n_steps", 10)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Coords':>8} | {'Types':>8} | {'Lattice':>8} | {'Sym':>8} | {'Div':>8} | {'LR':>10} | {'Time':>6}")
    print("-" * 105)

    import csv
    metrics_path = os.path.join(paths.get("logs", "logs"), "training_metrics.csv")
    mode = "a" if resume_checkpoint else "w"
    with open(metrics_path, mode, newline="") as f:
        writer = csv.writer(f)
        if not resume_checkpoint:
            writer.writerow(["step", "W_Zn", "W_C", "W_H", "gamma", "divergence"])

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        # --- Train ---
        model.train()
        train_losses = {"total": 0.0, "coords": 0.0, "types": 0.0, "lattice": 0.0, "sym": 0.0, "div": 0.0, "W_Zn": 0.0, "W_C": 0.0, "W_H": 0.0, "gamma": 0.0}
        n_batches = 0

        for batch_data in train_loader:
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    losses = criterion(model, batch_data, device)
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = criterion(model, batch_data, device)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            for k in train_losses:
                train_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
            n_batches += 1

        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # --- Validate ---
        val_loss_avg = 0
        if val_loader:
            model.eval()
            val_losses_total = 0
            n_val = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    losses = criterion(model, batch_data, device)
                    val_losses_total += losses["total"].item()
                    n_val += 1
            val_loss_avg = val_losses_total / max(n_val, 1)

        # Scheduler (despues de warmup)
        if epoch >= warmup:
            scheduler.step()

        t_elapsed = time.time() - t_start

        # Log
        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_loss_avg)

        if tracker == "wandb":
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_total": train_losses["total"],
                "train/loss_coords": train_losses["coords"],
                "train/loss_types": train_losses["types"],
                "train/loss_lattice": train_losses["lattice"],
                "train/loss_sym": train_losses["sym"],
                "train/loss_div": train_losses["div"],
                "val/loss_total": val_loss_avg,
                "lr": optimizer.param_groups[0]["lr"]
            })

        if (epoch + 1) % log_every == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch+1:>5} | {train_losses['total']:>10.4f} | {val_loss_avg:>10.4f} | "
                f"{train_losses['coords']:>8.4f} | {train_losses['types']:>8.4f} | "
                f"{train_losses['lattice']:>8.4f} | {train_losses['sym']:>8.4f} | "
                f"{train_losses['div']:>8.4f} | {lr:>10.2e} | {t_elapsed:>5.1f}s"
            )

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                train_losses["W_Zn"], 
                train_losses["W_C"], 
                train_losses["W_H"], 
                train_losses["gamma"], 
                train_losses["div"]
            ])

        # Save best
        if val_loss_avg < best_val_loss and val_loader:
            best_val_loss = val_loss_avg
            save_checkpoint(
                model, optimizer, epoch, val_loss_avg,
                os.path.join(paths["models"], "checkpoints", "flow_best.pth"),
                extra={"config": config}
            )

        # Periodic save
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_losses["total"],
                os.path.join(paths["models"], "checkpoints", f"flow_epoch_{epoch+1}.pth"),
            )

    # Save final
    save_checkpoint(
        model, optimizer, epochs, train_losses["total"],
        os.path.join(paths["models"], "checkpoints", "flow_final.pth"),
        extra={"config": config, "history": history}
    )

    # Save history
    history_path = os.path.join(paths.get("logs", "logs"), "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if tracker == "wandb":
        wandb.finish()

    print(f"\nEntrenamiento completado. Mejor val loss: {best_val_loss:.4f}")
    print(f"Modelo guardado en {paths['models']}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenamiento Flow Matching - Deep-Material v2")
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    parser.add_argument("--resume", default=None, help="Ruta a checkpoint para reanudar")
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_paths(config)
    train(config, resume_checkpoint=args.resume)
